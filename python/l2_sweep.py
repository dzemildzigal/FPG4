"""L2 and L2a sweep: quantization-aware training, four configurations, ten seeds.

Run from python/:

    uv run python l2_sweep.py mnist            # all 10 seeds, weights only, rung L2
    uv run python l2_sweep.py mnist 1          # the first seed only, a smoke run
    uv run python l2_sweep.py mnist 10 wa      # all 10 seeds, weights and activations, rung L2a
    uv run python l2_sweep.py mnist1d

The third argument is the scope: "w" quantizes the weights only and is rung L2, "wa" quantizes the
weights and every activation tensor that enters a matrix multiply and is rung L2a. The bias is
quantized in neither, because the bias lives in the accumulator's fixed-point format.

What it does, in this order, for the chosen dataset:

    1. make sure the ten L0 weight files exist. Train any that are missing and save them.
    2. L0 row    evaluate the saved float weights.
    3. L1a row   the FP4 payload with a float64 scale, no 8-bit scale. One column, the same for
                 every configuration, because the configuration lives in the scale.
    4. ptq rows  quantize those weights once with the configuration's own 8-bit scale and evaluate.
                 This is rung L1b or L1c, and in scope "wa" it is rung L1d.
    5. L2 rows   train the network again with quantization inside the forward pass.

Every training run is written to out/<mode>/<dataset>/ as soon as it finishes. A finished run is
never repeated, so the sweep can be stopped and restarted.

The reported accuracy is always the DEPLOYED model, which is the quantized one. The float master
weight never reaches the fabric, so its accuracy is not the accuracy of any rung.
"""
import json
import os
import sys

import numpy as np

import quant as q
import qat
from model import evaluate
from train import DATASETS, DTYPE, load_dataset, load_weights, save_weights, weights_path

# The ten seeds, fixed in docs/experiment_L2.md. The first three are the L0 and L1 seeds.
SEEDS = [42, 4242, 424242, 123, 456, 789, 1010, 2020, 3030, 4040]
EPOCHS = 10
BATCH = 128

DATASET = sys.argv[1] if len(sys.argv) > 1 else "mnist"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else len(SEEDS)
MODE = sys.argv[3] if len(sys.argv) > 3 else "w"
assert DATASET in DATASETS, f"unknown dataset {DATASET}, choose from {list(DATASETS)}"
assert MODE in ("w", "wa"), f"unknown scope {MODE}, choose w or wa"
RUNG = {"w": "L2", "wa": "L2a"}[MODE]

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "" if quant.SCALE_ROUNDING == "nearest" else "_" + quant.SCALE_ROUNDING
OUT = os.path.join(HERE, "..", "out", MODE + TAG, DATASET)
CONFIG_NAMES = list(qat.CONFIGS)


def result_path(config, seed):
    return os.path.join(OUT, f"{MODE}_{config}_seed{seed}.json")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        obj["scale_rounding"] = quant.SCALE_ROUNDING
        json.dump(obj, f, indent=1)


def ensure_l0(seed, data):
    """Train the L0 weights for this seed if they are not on disk yet, and return them.

    The file holds float weights. Every later rung compares against this run, so it must be the
    unquantized network and nothing else.
    """
    path = weights_path(seed, DATASET) + ".npz"
    if os.path.exists(path):
        p = load_weights(seed, DATASET)
        if p["W1"].dtype == DTYPE:
            return p
        # A row must never mix widths. One float64 seed and nine float32 seeds give a mean that belongs
        # to no single run. So a weight file of the wrong width is retrained and overwritten.
        print(f"  L0 seed {seed}: stored weights are {p['W1'].dtype}, retraining in {np.dtype(DTYPE).name}",
              flush=True)
    else:
        print(f"  L0 seed {seed}: training", flush=True)
    save_weights(seed, DATASET, data, dtype=DTYPE)
    return load_weights(seed, DATASET)


def evaluate_params(p, X, y, quant=None):
    """Test loss and accuracy as percentages. evaluate returns a fraction, this prints points."""
    loss, acc = evaluate(p, X, y, quant=quant)
    return float(loss), 100.0 * float(acc)


def prequantize(A, table, rule):
    """Quantize a whole tensor once, outside the training loop. Returns (Aq, SQNR in dB).

    Why this is allowed: a block is 16 values inside ONE row of A, and the scale of that block
    depends on those 16 values only. Rows never mix. So quantizing the whole tensor at once gives
    the same result as quantizing it in batches of 128, value for value.
    check_prequant_equivalence() proves that claim instead of assuming it.

    Why it is needed: the input tensor is 784 wide, so the quantizer sees 49 blocks per row.
    One call on a batch of 128 costs 68 ms, against 0.6 ms for the 16-wide activation tensors.
    Quantized once, the input costs nothing during training.

    The chunk size keeps the temporary array of encode_scale small. That array is
    (rows, blocks, 255) float64, so 2,040 rows of a 784-wide tensor is 204 MB.
    """
    n, K = A.shape
    step = max(64, int(100_000 // (K // 16)))
    out = np.empty_like(A, dtype=np.float32)
    loss_sig = 0.0
    loss_err = 0.0
    for s in range(0, n, step):
        Aq, _, _, _, _ = q.quantize_blocks_act(np.asarray(A[s:s + step], dtype=np.float32), table, rule)
        out[s:s + step] = Aq
        d = np.asarray(A[s:s + step], dtype=np.float64) - np.asarray(Aq, dtype=np.float64)
        loss_sig += float(np.sum(np.asarray(A[s:s + step], dtype=np.float64) ** 2))
        loss_err += float(np.sum(d ** 2))
    good = 10 * np.log10(loss_sig / loss_err) if loss_sig > 0 else 0.0
    return out, float(good)


def check_prequant_equivalence(X, table, rule, rows=4096):
    """Prove that the block quantizer does not mix rows.

    One call on 4,096 rows against 32 calls on 128 rows. The two results must be bit-identical.
    If a block ever crossed a row boundary, the two would differ, and every L2a number in this
    project would be a measurement of a different function than the one the fabric runs.
    """
    A = np.asarray(X[:rows], dtype=np.float32)
    whole, _, _, _, _ = q.quantize_blocks_act(A, table, rule)
    parts = np.empty_like(whole)
    for s in range(0, len(A), BATCH):
        pq_, _, _, _, _ = q.quantize_blocks_act(A[s:s + BATCH], table, rule)
        parts[s:s + BATCH] = pq_
    return bool(np.array_equal(whole, parts)), float(np.max(np.abs(whole - parts)))


def run_l2(seed, config, data, X_test, y_test, X_train_q=None, X_test_q=None, input_sqnr=None):
    """One training run, or the stored result of one that already finished."""
    path = result_path(config, seed)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    X_train, y_train = (X_train_q, data[1]) if X_train_q is not None else (data[0], data[1])
    Xt = X_test_q if X_test_q is not None else X_test
    print(f"  {config} seed {seed}: training", flush=True)
    result = qat.train_qat(seed, EPOCHS, BATCH, X_train, y_train, Xt, y_test, config,
                           acts=(MODE == "wa"), X_pre_quantized=(X_train_q is not None),
                           input_sqnr=input_sqnr, verbose=False)
    # The history holds numpy scalars and dicts of them, so it is converted by hand before json.
    history = result["history"]
    obj = {
        "config": config,
        "seed": seed,
        "test_loss": history["test_loss"],
        "test_acc": [100.0 * a for a in history["test_acc"]],
        "train_acc": [100.0 * a for a in history["train_acc"]],
        "sqnr": [{k: float(v) for k, v in d.items()} for d in history["sqnr"]],
        "counters": history["counters"],
        "frozen_steps": history["frozen_steps"],
        "stale_elements": [{k: float(v) for k, v in d.items()} for d in history["stale_elements"]],
        "act_sqnr": [{k: float(v) for k, v in d.items()} for d in history["act_sqnr"]],
        "act_counters": history["act_counters"],
    }
    save_json(path, obj)
    return obj


def paired(a, b):
    """The pre-registered comparison of two runs that share a seed.

    a and b are per-seed accuracies in the same seed order. The difference is paired,
    which removes the seed noise that dominates an unpaired comparison.

    Condition 1: the sign of the paired difference agrees with the sign of its mean in at
    least 8 of 10 seeds. A mean that comes from two large opposite seeds is not a result.
    Condition 2: the absolute mean is larger than twice its standard error.

    Returns a dict. "winner" is "a", "b" or None. None means no decision, and no decision
    is a legal result that keeps the option open.
    """
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    agree = int(np.sum(np.sign(d) == np.sign(mean))) if mean != 0 else 0
    cond1 = (agree >= 8) if n >= 8 else None   # the rule is written for 10 seeds. Fewer is a smoke run.
    cond2 = bool(abs(mean) > 2 * se) if n > 1 else False
    winner = None
    if cond1 and cond2:
        winner = "a" if mean > 0 else "b"
    return {"mean": mean, "se": se, "agree": agree, "n": n,
            "cond1": cond1, "cond2": cond2, "winner": winner}


def _fmt(v):
    return "n/a" if v is None else str(v)


def main():
    data = load_dataset(DATASET)
    X_test, y_test = data[2], data[3]
    seeds = SEEDS[:N_SEEDS]
    print(f"dataset {DATASET}: test set {X_test.shape}, {len(seeds)} seeds, "
          f"{len(CONFIG_NAMES)} configurations, rung {RUNG} "
          f"({'weights and activations' if MODE == 'wa' else 'weights only'})", flush=True)

    l0, l1a, ptq, l2 = {}, {}, {}, {c: {} for c in CONFIG_NAMES}

    # Pass 1: the float baseline and the rung above it. Both need no new training beyond L0.
    masters = {}
    for seed in seeds:
        p = ensure_l0(seed, data)
        masters[seed] = p
        l0[seed] = evaluate_params(p, X_test, y_test)
        p_l1a = dict(p)
        for key in qat.WEIGHTS:
            p_l1a[key], _, _, _ = q.quantize_fp4_b_fp64(p[key])
        l1a[seed] = evaluate_params(p_l1a, X_test, y_test)

    # Pass 2: one configuration at a time, so that only one pre-quantized copy of the training set
    # is in memory. On MNIST-1D that copy is 1.25 GB.
    check = None
    for config in CONFIG_NAMES:
        table, rule = qat.CONFIGS[config]
        if MODE == "wa":
            if check is None:
                same, worst_diff = check_prequant_equivalence(X_test, table, rule)
                check = (same, worst_diff)
                print(f"  pre-quantization is row independent: {same} "
                      f"(worst difference {worst_diff:g})", flush=True)
                if not same:
                    raise SystemExit("the block quantizer mixes rows. L2a cannot use a pre-quantized input.")
            X_train_q, input_sqnr = prequantize(data[0], table, rule)
            X_test_q, _ = prequantize(X_test, table, rule)
            print(f"  {config}: input X quantized, SQNR {input_sqnr:.2f} dB", flush=True)
        else:
            X_train_q = X_test_q = None
            input_sqnr = None

        for seed in seeds:
            p = masters[seed]
            pq, _, _ = qat.quantize_master(p, table, rule)
            hook = qat.ActQuant(table, rule, pre_quantized_input=True) if MODE == "wa" else None
            ptq[(config, seed)] = evaluate_params(pq, X_test_q if X_test_q is not None else X_test,
                                                  y_test, quant=hook)
            l2[config][seed] = run_l2(seed, config, data, X_test, y_test,
                                      X_train_q=X_train_q, X_test_q=X_test_q, input_sqnr=input_sqnr)
        del X_train_q, X_test_q

    # ---- table 1: accuracy per seed, the raw evidence -------------------------------------
    print("\naccuracy, percent, test set. Every row below L0 is a rung of the ladder.")
    print(f"{'seed':>7} {'L0':>7} {'L1a':>7} " + " ".join(f"{c:>9}" for c in CONFIG_NAMES))
    print(f"{'':>7} {'':>7} {'':>7} " + " ".join(f"{'ptq':>9}" for _ in CONFIG_NAMES))
    print(f"{'':>7} {'':>7} {'':>7} " + " ".join(f"{'L2a' if MODE == 'wa' else 'L2':>9}"
                                               for _ in CONFIG_NAMES))
    for seed in seeds:
        print(f"{seed:>7} {l0[seed][1]:>7.2f} {l1a[seed][1]:>7.2f} "
              + " ".join(f"{ptq[(c, seed)][1]:>9.2f}" for c in CONFIG_NAMES))
        print(f"{'':>7} {'':>7} {'':>7} "
              + " ".join(f"{l2[c][seed]['test_acc'][-1]:>9.2f}" for c in CONFIG_NAMES))
    print(f"{'mean':>7} {np.mean([l0[s][1] for s in seeds]):>7.2f} "
          f"{np.mean([l1a[s][1] for s in seeds]):>7.2f} "
          + " ".join(f"{np.mean([ptq[(c, s)][1] for s in seeds]):>9.2f}" for c in CONFIG_NAMES))
    print(f"{'L2':>7} {'':>7} {'':>7} " + " ".join(
        f"{np.mean([l2[c][s]['test_acc'][-1] for s in seeds]):>9.2f}" for c in CONFIG_NAMES))

    # ---- table 2: what each rung costs, and what training gives back ---------------------
    base = float(np.mean([l0[s][1] for s in seeds]))
    l1a_drop = base - float(np.mean([l1a[s][1] for s in seeds]))
    print("\ndrop against L0, in points. A smaller number is better.")
    print(f"{'config':>10} {'L1a drop':>9} {'ptq drop':>9} {'trained drop':>13} {'training recovers':>18}")
    for c in CONFIG_NAMES:
        a = base - float(np.mean([ptq[(c, s)][1] for s in seeds]))
        b = base - float(np.mean([l2[c][s]["test_acc"][-1] for s in seeds]))
        print(f"{c:>10} {l1a_drop:>9.2f} {a:>9.2f} {b:>13.2f} {a - b:>18.2f}")

    # ---- table 3: the training-side measurements that L1 could not make -----------------
    print("\nend of training, mean over seeds")
    print(f"{'config':>10} {'SQNR W1':>8} {'SQNR W2':>8} {'SQNR W3':>8} "
          f"{'stale W1':>9} {'stale W2':>9} {'stale W3':>9} {'frozen steps':>13}")
    for c in CONFIG_NAMES:
        sq = {k: np.mean([l2[c][s]['sqnr'][-1][k] for s in seeds]) for k in qat.WEIGHTS}
        st = {k: 100 * np.mean([l2[c][s]['stale_elements'][-1][k] for s in seeds]) for k in qat.WEIGHTS}
        fz = np.mean([sum(l2[c][s]['frozen_steps'][-1].values()) for s in seeds])
        print(f"{c:>10} {sq['W1']:>8.2f} {sq['W2']:>8.2f} {sq['W3']:>8.2f} "
              f"{st['W1']:>8.2f}% {st['W2']:>8.2f}% {st['W3']:>8.2f}% {fz:>13.1f}")
    print("stale = percent of weight elements whose FP4 value did not move in the last epoch "
          "while the master weight was free to move")

    if MODE == "wa":
        # Which activation tensor carries the damage. A single pooled number over all three hides it.
        print("\nactivation SQNR, dB, mean over seeds, end of training")
        print(f"{'config':>10} " + " ".join(f"{n:>10}" for n in qat.ACT_SITES))
        for c in CONFIG_NAMES:
            vals = [np.mean([l2[c][s]["act_sqnr"][-1][n] for s in seeds]) for n in qat.ACT_SITES]
            print(f"{c:>10} " + " ".join(f"{v:>10.2f}" for v in vals))
        print("the input X is quantized once, outside the loop, and its SQNR is measured there")

    counters = {k: 0 for k in qat.COUNTERS}
    act_counters = {k: 0 for k in qat.COUNTERS}
    for c in CONFIG_NAMES:
        for s in seeds:
            for epoch in l2[c][s]["counters"]:
                for k in qat.COUNTERS:
                    counters[k] += epoch[k]
            for epoch in l2[c][s].get("act_counters", []):
                for k in qat.COUNTERS:
                    act_counters[k] += epoch[k]
    print("\nscale counters on the weights, summed over every epoch of every seed and configuration")
    for k, v in counters.items():
        print(f"  {k:>18}: {v}")
    if MODE == "wa":
        print("scale counters on the activations, same sum")
        for k, v in act_counters.items():
            print(f"  {k:>18}: {v}")

    # ---- the pre-registered decision rule ----------------------------------------------
    print("\npaired comparisons. Two conditions, both required: the sign agrees in 8 of 10 "
          "seeds, and the mean beats twice its standard error.")
    print(f"{'a':>10} {'b':>10} {'mean a-b':>9} {'se':>6} {'agree':>6} {'cond1':>6} {'cond2':>6} {'winner':>7}")
    pairs = []
    for c in CONFIG_NAMES:
        pairs.append((f"{RUNG} {c}", "L0", [l2[c][s]["test_acc"][-1] for s in seeds],
                      [l0[s][1] for s in seeds]))
        pairs.append((f"{RUNG} {c}", f"ptq {c}", [l2[c][s]["test_acc"][-1] for s in seeds],
                      [ptq[(c, s)][1] for s in seeds]))
    for rule, other in (("a6", "f6"), ("f6", "a6")):
        for fmt in ("e6m2", "e5m3"):
            a, b = f"{rule}-{fmt}", f"{other}-{fmt}"
            if a in CONFIG_NAMES and b in CONFIG_NAMES:
                pairs.append((f"{RUNG} {a}", f"{RUNG} {b}", [l2[a][s]["test_acc"][-1] for s in seeds],
                              [l2[b][s]["test_acc"][-1] for s in seeds]))
    for fmt, other in (("e6m2", "e5m3"), ("e5m3", "e6m2")):
        for rule in ("a6", "f6"):
            a, b = f"{rule}-{fmt}", f"{rule}-{other}"
            if a in CONFIG_NAMES and b in CONFIG_NAMES:
                pairs.append((f"{RUNG} {a}", f"{RUNG} {b}", [l2[a][s]["test_acc"][-1] for s in seeds],
                              [l2[b][s]["test_acc"][-1] for s in seeds]))
    seen = set()
    for a, b, va, vb in pairs:
        if (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        r = paired(va, vb)
        se = "n/a" if r["n"] < 2 else f"{r['se']:.2f}"
        print(f"{a:>10} {b:>10} {r['mean']:>+9.2f} {se:>6} {r['agree']:>4d}/{r['n']:<2d} "
              f"{_fmt(r['cond1']):>6} {str(r['cond2']):>6} {str(r['winner']):>7}")

    print("\nnote: docs/experiment_L2.md condition 1 reads \"the paired difference (same seed, "
          "L2 against L2)\". That phrase cannot be applied as written, so every pair above is "
          "shown and none is selected. Name the pair and the rule is used unchanged.")


if __name__ == "__main__":
    main()
