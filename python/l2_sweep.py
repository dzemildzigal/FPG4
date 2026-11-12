"""L2 sweep: quantization-aware training, weights only, four configurations, ten seeds.

Run from python/:

    uv run python l2_sweep.py mnist            # all 10 seeds
    uv run python l2_sweep.py mnist 1          # the first seed only, a smoke run
    uv run python l2_sweep.py mnist1d

What it does, in this order, for the chosen dataset:

    1. make sure the ten L0 weight files exist. Train any that are missing and save them.
       This is the same float64 run as rung L0, nothing new.
    2. L0 row    evaluate the saved float weights.
    3. L1a row   quantize those weights once and evaluate. This is rung L1a, the PTQ
                 reference. It needs no training, because L1a and L0 share the same weights.
    4. L2 rows   train the network again with quantization inside the forward pass,
                 once per configuration and seed.

Every L2 run is written to out/l2/<dataset>/l2_<config>_seed<seed>.json as soon as it
finishes. A finished run is never repeated, so the sweep can be stopped and restarted.
A four-hour run that loses everything on the last seed is not a tool, it is a trap.

The last section applies the decision rule that was written into docs/experiment_L2.md
before the run. It is printed and never abbreviated: sign agreement across the seeds
first, then the size of the mean against twice its standard error.
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
assert DATASET in DATASETS, f"unknown dataset {DATASET}, choose from {list(DATASETS)}"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "l2", DATASET)
CONFIG_NAMES = list(qat.CONFIGS)


def result_path(config, seed):
    return os.path.join(OUT, f"l2_{config}_seed{seed}.json")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def ensure_l0(seed, data):
    """Train the L0 weights for this seed if they are not on disk yet, and return them.

    The file holds float64 weights. Every later rung compares against this run, so it must
    be the unquantized network and nothing else.
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


def evaluate_params(p, X_test, y_test):
    """Test loss and accuracy as percentages. evaluate returns a fraction, this prints points."""
    loss, acc = evaluate(p, X_test, y_test)
    return float(loss), 100.0 * float(acc)


def run_l2(seed, config, data, X_test, y_test):
    """One L2 run, or the stored result of one that already finished."""
    path = result_path(config, seed)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    X_train, y_train, _, _ = data
    print(f"  {config} seed {seed}: training", flush=True)
    result = qat.train_qat(seed, EPOCHS, BATCH, X_train, y_train, X_test, y_test,
                           config, verbose=False)
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
          f"{len(CONFIG_NAMES)} configurations", flush=True)

    l0, l1a, l2 = {}, {}, {c: {} for c in CONFIG_NAMES}
    for seed in seeds:
        p = ensure_l0(seed, data)
        l0[seed] = evaluate_params(p, X_test, y_test)
        for config in CONFIG_NAMES:
            table, rule = qat.CONFIGS[config]
            pq, _, _ = qat.quantize_master(p, table, rule)
            l1a[(config, seed)] = evaluate_params(pq, X_test, y_test)
            l2[config][seed] = run_l2(seed, config, data, X_test, y_test)

    # ---- table 1: accuracy per seed, the raw evidence -------------------------------------
    print("\naccuracy, percent, test set")
    head = f"{'seed':>7} {'L0':>7} " + " ".join(f"{c:>9}" for c in CONFIG_NAMES)
    print(head)
    print(f"{'':>7} {'':>7} " + " ".join(f"{'L1a':>9}" for _ in CONFIG_NAMES))
    for seed in seeds:
        row = f"{seed:>7} {l0[seed][1]:>7.2f} " + " ".join(f"{l1a[(c, seed)][1]:>9.2f}" for c in CONFIG_NAMES)
        print(row)
        print(f"{'L2':>7} {'':>7} " + " ".join(f"{l2[c][seed]['test_acc'][-1]:>9.2f}" for c in CONFIG_NAMES))
    print(f"{'mean':>7} {np.mean([l0[s][1] for s in seeds]):>7.2f} "
          + " ".join(f"{np.mean([l1a[(c, s)][1] for s in seeds]):>9.2f}" for c in CONFIG_NAMES))
    print(f"{'L2':>7} {'':>7} " + " ".join(
        f"{np.mean([l2[c][s]['test_acc'][-1] for s in seeds]):>9.2f}" for c in CONFIG_NAMES))

    # ---- table 2: what each rung costs, and what training gives back ---------------------
    base = float(np.mean([l0[s][1] for s in seeds]))
    print("\ndrop against L0, in points. A smaller number is better.")
    print(f"{'config':>10} {'L1a drop':>9} {'L2 drop':>9} {'recovered by training':>22}")
    for c in CONFIG_NAMES:
        a = base - float(np.mean([l1a[(c, s)][1] for s in seeds]))
        b = base - float(np.mean([l2[c][s]['test_acc'][-1] for s in seeds]))
        print(f"{c:>10} {a:>9.2f} {b:>9.2f} {a - b:>22.2f}")

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

    counters = {k: 0 for k in qat.COUNTERS}
    for c in CONFIG_NAMES:
        for s in seeds:
            for epoch in l2[c][s]["counters"]:
                for k in qat.COUNTERS:
                    counters[k] += epoch[k]
    print("\nscale counters, summed over every epoch of every seed and every configuration")
    for k, v in counters.items():
        print(f"  {k:>18}: {v}")

    # ---- the pre-registered decision rule ----------------------------------------------
    print("\npaired comparisons. Two conditions, both required: the sign agrees in 8 of 10 "
          "seeds, and the mean beats twice its standard error.")
    print(f"{'a':>10} {'b':>10} {'mean a-b':>9} {'se':>6} {'agree':>6} {'cond1':>6} {'cond2':>6} {'winner':>7}")
    pairs = []
    for c in CONFIG_NAMES:
        pairs.append((f"L2 {c}", "L0", [l2[c][s]["test_acc"][-1] for s in seeds], [l0[s][1] for s in seeds]))
        pairs.append((f"L2 {c}", f"L1a {c}", [l2[c][s]["test_acc"][-1] for s in seeds],
                      [l1a[(c, s)][1] for s in seeds]))
    for rule, other in (("a6", "f6"), ("f6", "a6")):
        for fmt in ("e6m2", "e5m3"):
            a, b = f"{rule}-{fmt}", f"{other}-{fmt}"
            if a in CONFIG_NAMES and b in CONFIG_NAMES:
                pairs.append((f"L2 {a}", f"L2 {b}", [l2[a][s]["test_acc"][-1] for s in seeds],
                              [l2[b][s]["test_acc"][-1] for s in seeds]))
    for fmt, other in (("e6m2", "e5m3"), ("e5m3", "e6m2")):
        for rule in ("a6", "f6"):
            a, b = f"{rule}-{fmt}", f"{rule}-{other}"
            if a in CONFIG_NAMES and b in CONFIG_NAMES:
                pairs.append((f"L2 {a}", f"L2 {b}", [l2[a][s]["test_acc"][-1] for s in seeds],
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
