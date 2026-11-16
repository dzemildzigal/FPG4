"""Rung L3: the operands of the weight-gradient GEMM are rounded to FP4, everything else stays float.

Scope, fixed by docs/experiment_L3.md:

    forward pass   float. Nothing is quantized, so no weight sits on an FP4 grid
    weight grads   the two operands are rounded once, blocked along the BATCH axis
    gradient-down  float, because it needs the weights on a grid, and that is rung L5
    biases         float. The bias gradient is a sum of dZ, not an operand of a GEMM
    optimizer      float32, Adam

Two cells:

    L3-D    the delta operands only: dZ3, dZ2, dZ1
    L3-DA   the deltas and the activation operands: X, A1, A2

The block lies along the batch, because the weight-gradient GEMM sums over the batch:
dW = A.T @ dZ with A of shape (batch, features). So a block of 16 samples shares one scale, and every
feature column has its own scales. That is the transposed form of the forward blocking, and it is the
reason the block is a property of the GEMM, not of the tensor (ADR-024).

Run from python/:

    uv run python l3.py mnist            # both cells, 10 seeds, 4 configurations, 80 runs
    uv run python l3.py mnist 1          # the first seed only, a smoke run
    uv run python l3.py mnist1d

A finished run is written to out/l3/<dataset>/ and never repeated, so the sweep can be stopped and
restarted. The reported accuracy is the accuracy of the trained model. In L3 the trained model is the
float model, because nothing is stored in FP4.
"""
import json
import os
import sys

import numpy as np

import quant as q
from model import adam, evaluate, forward, init_params, onehot, relu_derivative
from train import DATASETS, DTYPE, load_dataset, load_weights

SEEDS = [42, 4242, 424242, 123, 456, 789, 1010, 2020, 3030, 4040]
EPOCHS = 10
BATCH = 128
BLOCK = 16                 # samples per block along the batch
WEIGHTS = ("W1", "W2", "W3")
CELLS = {"L3-D": False, "L3-DA": True}

CONFIGS = {
    "a6-e6m2": (q.e6m2, "L1b"),
    "f6-e6m2": (q.e6m2, "L1c"),
    "a6-e5m3": (q.e5m3, "L1b"),
    "f6-e5m3": (q.e5m3, "L1c"),
}
CONFIG_NAMES = list(CONFIGS)

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")

DATASET = sys.argv[1] if len(sys.argv) > 1 else "mnist"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else len(SEEDS)
assert DATASET in DATASETS, f"unknown dataset {DATASET}, choose from {list(DATASETS)}"
OUT = os.path.join(HERE, "..", "out", "l3", DATASET)


def result_path(cell, config, seed):
    return os.path.join(OUT, f"{cell}_{config}_seed{seed}.json")


def quantize_batch(T, table, rule):
    """Round a tensor of shape (batch, features) to FP4, with one 8-bit scale per 16 samples.

    The weight-gradient GEMM sums over the batch, so the block runs along the batch. T is cut into
    (batch / 16, 16, features) and then transposed to (batch / 16, features, 16), which is the
    (rows, blocks, 16) layout that quant.quantize_blocks expects: one "row" is one block of samples,
    one "block" is one feature column.

    Returns (Tq, idx, blocks). idx is (batch / 16, features, 16) and holds the payload code of every
    value. blocks is the same shape and holds the input, so the vanished share can be counted.
    """
    n, K = T.shape
    if n % BLOCK != 0:
        raise ValueError(f"batch must be a multiple of {BLOCK}, got {n}")
    blocks = np.ascontiguousarray(T.reshape(n // BLOCK, BLOCK, K).transpose(0, 2, 1))
    Tq, _, idx, _, _ = q.quantize_blocks(blocks, table, rule)
    return Tq.transpose(0, 2, 1).reshape(n, K), idx, blocks


def vanished(blocks, idx):
    """Share of non-zero values whose payload code is 0. Those values send no update in that step."""
    nz = blocks != 0
    return float(np.mean(idx[nz] == 0)) if nz.any() else 0.0


def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(np.sum(u * v) / (nu * nv)) if nu > 0 and nv > 0 else float("nan")


def backward_l3(p, y, cache, table, rule, round_acts, telemetry=False):
    """The backward pass of L3. Returns grads, and a record when telemetry is True.

    What is rounded, and what is not:

        dZ3          rounded -> dW3 = A2.T @ dZ3
        dA2 = dZ3 @ W3.T        float. This is the gradient-down GEMM, rung L5
        dZ2          rounded -> dW2 = A1.T @ dZ2
        dA1 = dZ2 @ W2.T        float
        dZ1          rounded -> dW1 = X.T @ dZ1
        A2, A1, X    rounded only in cell L3-DA
        bias grads   float. They are sums of dZ, not GEMM operands

    The float gradient is computed in the same call when telemetry is True, so the cosine against it
    measures the rounding of one step and nothing else.
    """
    Y = onehot(y)
    X, Z1, A1, Z2, A2, P = cache["X"], cache["Z1"], cache["A1"], cache["Z2"], cache["A2"], cache["P"]
    B = y.shape[0]
    rec = {"vanished": {}, "cos": {}, "sqnr": {}} if telemetry else None

    dZ3 = (P - Y) / B
    dA2 = dZ3 @ p["W3"].T                      # float, before any rounding
    dZ2 = dA2 * relu_derivative(Z2)
    dA1 = dZ2 @ p["W2"].T
    dZ1 = dA1 * relu_derivative(Z1)
    float_d = {"dZ3": dZ3, "dZ2": dZ2, "dZ1": dZ1}
    float_a = {"A2": A2, "A1": A1, "X": X}

    qd, qa = {}, {}
    for name, T in float_d.items():
        qd[name], idx, blocks = quantize_batch(T, table, rule)
        if telemetry:
            rec["vanished"][name] = vanished(blocks, idx)
            rec["sqnr"][name] = q.sqnr(T, qd[name])
    if round_acts:
        for name, T in float_a.items():
            qa[name], idx, blocks = quantize_batch(T, table, rule)
            if telemetry:
                rec["vanished"][name] = vanished(blocks, idx)
                rec["sqnr"][name] = q.sqnr(T, qa[name])

    lhs_a2 = qa["A2"] if round_acts else A2
    lhs_a1 = qa["A1"] if round_acts else A1
    lhs_x = qa["X"] if round_acts else X
    grads = {
        "W3": lhs_a2.T @ qd["dZ3"],
        "b3": np.sum(dZ3, axis=0),
        "W2": lhs_a1.T @ qd["dZ2"],
        "b2": np.sum(dZ2, axis=0),
        "W1": lhs_x.T @ qd["dZ1"],
        "b1": np.sum(dZ1, axis=0),
    }
    if telemetry:
        # The float gradient of the same step, from the same cache. Only the cosine is needed.
        fg = {"W3": A2.T @ dZ3, "W2": A1.T @ dZ2, "W1": X.T @ dZ1}
        for k in WEIGHTS:
            rec["cos"][k] = _cos(grads[k], fg[k])
    return grads, rec


def train_l3(seed, epochs, batch_size, X, y, X_test, y_test, config, cell, verbose=False):
    table, rule = CONFIGS[config]
    round_acts = CELLS[cell]
    rng = np.random.default_rng(seed)
    p = init_params(seed, dtype=DTYPE)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    t = 1
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
               "vanished": [], "cos": [], "sqnr": []}
    for epoch in range(epochs):
        ep = {"vanished": {k: [] for k in WEIGHTS}, "cos": {k: [] for k in WEIGHTS},
              "sqnr": {k: [] for k in WEIGHTS}}
        indices = np.arange(X.shape[0])
        rng.shuffle(indices)
        for start in range(0, X.shape[0], batch_size):
            batch = indices[start:start + batch_size]
            X_batch, y_batch = X[batch], y[batch]
            P, cache = forward(p, X_batch)          # float forward. Nothing is quantized here
            # Telemetry only on the first batch of the epoch, so the epochs are comparable and the
            # cost of the second float gradient stays at one batch per epoch.
            tel = start == 0
            grads, rec = backward_l3(p, y_batch, cache, table, rule, round_acts, telemetry=tel)
            if tel:
                for name in rec["vanished"]:
                    ep["vanished"].setdefault(name, []).append(rec["vanished"][name])
                    ep["sqnr"].setdefault(name, []).append(rec["sqnr"][name])
                for k in WEIGHTS:
                    ep["cos"][k].append(rec["cos"][k])
            p, m, v = adam(p, grads, m, v, t)
            t += 1
        train_loss, train_acc = evaluate(p, X, y)
        test_loss, test_acc = evaluate(p, X_test, y_test)
        # The accuracies are stored as percentages, the same unit as l2_sweep and block_sweep.
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(100.0 * float(train_acc))
        history["test_loss"].append(float(test_loss))
        history["test_acc"].append(100.0 * float(test_acc))
        history["vanished"].append({k: float(np.mean(v_)) for k, v_ in ep["vanished"].items() if v_})
        history["cos"].append({k: float(np.mean(v_)) for k, v_ in ep["cos"].items() if v_})
        history["sqnr"].append({k: float(np.mean(v_)) for k, v_ in ep["sqnr"].items() if v_})
        if verbose:
            print(f"{epoch:2d} loss {train_loss:.4f} acc {100 * train_acc:6.2f} "
                  f"test {test_loss:.4f} {100 * test_acc:6.2f} "
                  f"cos {history['cos'][-1]['W1']:.4f} {history['cos'][-1]['W2']:.4f} "
                  f"{history['cos'][-1]['W3']:.4f} "
                  f"vanished {100 * history['vanished'][-1]['W2']:.2f}%", flush=True)
    return p, history


def run_cell(cell, config, seed, data, X_test, y_test):
    path = result_path(cell, config, seed)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    print(f"  {cell} {config} seed {seed}: training", flush=True)
    p, history = train_l3(seed, EPOCHS, BATCH, data[0], data[1], X_test, y_test, config, cell)
    obj = {"cell": cell, "config": config, "seed": seed, **history}
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    return obj


def paired(a, b):
    """The pre-registered rule of docs/experiment_L3.md: 8 of 10 seeds agree, and |mean| > 2 se."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    agree = int(np.sum(np.sign(d) == np.sign(mean))) if mean != 0 else 0
    cond1 = (agree >= 8) if n >= 8 else None
    cond2 = bool(abs(mean) > 2 * se) if n > 1 else False
    winner = ("a" if mean > 0 else "b") if (cond1 and cond2) else None
    return {"mean": mean, "se": se, "agree": agree, "n": n, "winner": winner}


def draw(seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, EPOCHS + 1)
    colour = {"L3-D": "#1f77b4", "L3-DA": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for ax, key, ylabel in ((axes[0], "loss", "cross entropy loss"),
                            (axes[1], "acc", "accuracy, percent")):
        for cell in CELLS:
            for split, style in (("train", "--"), ("test", "-")):
                curves = []
                for config in CONFIG_NAMES:
                    for seed in seeds:
                        r = read(cell, config, seed)
                        if r is not None:
                            curves.append(r[f"{split}_{key}"])
                if not curves:
                    continue
                arr = np.array(curves, dtype=np.float64)
                mean = arr.mean(axis=0)
                se = arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros(EPOCHS)
                ax.plot(epochs, mean, style, color=colour[cell], linewidth=1.7 if split == "test" else 1.1,
                        marker="o" if split == "test" else None, markersize=3,
                        label=f"{cell} {split}")
                ax.fill_between(epochs, mean - se, mean + se, color=colour[cell], alpha=0.10, linewidth=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.set_xticks(epochs)
    axes[0].legend(fontsize=7.0, ncol=1, loc="upper right")
    fig.suptitle(f"{DATASET}: rung L3, FP4 operands of the weight-gradient GEMM. Dashed is train, solid is "
                 f"test. Each line is the mean over {len(CONFIG_NAMES)} configs and {len(seeds)} seeds, "
                 f"band is plus or minus one standard error", fontsize=8.6)
    fig.text(0.012, 0.045, f"docs/experiment_L3_{DATASET}_curves.png  |  runs: out/l3/{DATASET}, "
                          f"{len(seeds)} seeds x {len(CONFIG_NAMES)} configs x 2 cells  |  "
                          f"forward and optimizer float32, 10 epochs, batch {BATCH}", fontsize=7.0, color="#555555")
    fig.text(0.012, 0.020, "L3-D = deltas only. L3-DA = deltas and activation operands  |  "
                           "dashed = train, solid = test, band = plus or minus one standard error",
             fontsize=7.0, color="#555555")
    fig.tight_layout(rect=(0, 0.080, 1, 0.95))
    path = os.path.join(DOCS, f"experiment_L3_{DATASET}_curves.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)
    return path


def read(cell, config, seed):
    path = result_path(cell, config, seed)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    data = load_dataset(DATASET)
    X_test, y_test = data[2], data[3]
    seeds = SEEDS[:N_SEEDS]
    runs = len(CELLS) * len(CONFIG_NAMES) * len(seeds)
    print(f"dataset {DATASET}: {len(seeds)} seeds, {len(CONFIG_NAMES)} configurations, cells "
          f"{list(CELLS)}, {runs} runs, {EPOCHS} epochs, batch {BATCH}", flush=True)

    # The reference: the saved L0 float weights, evaluated here and not retrained.
    l0 = {}
    for seed in seeds:
        p0 = load_weights(seed, DATASET)
        l0[seed] = 100.0 * float(evaluate(p0, X_test, y_test)[1])
    print(f"  L0 reference: {np.mean(list(l0.values())):.2f} "
          f"+-{np.std(list(l0.values()), ddof=1):.2f} over {len(seeds)} seeds", flush=True)

    for cell in CELLS:
        for config in CONFIG_NAMES:
            for seed in seeds:
                run_cell(cell, config, seed, data, X_test, y_test)

    print(f"\nfinal test accuracy, percent, mean of {len(CONFIG_NAMES)} configs")
    print(f"{'cell':>7} " + " ".join(f"{c:>16}" for c in CONFIG_NAMES) + f" {'mean':>7}")
    for cell in CELLS:
        cells = []
        for config in CONFIG_NAMES:
            accs = [read(cell, config, s)["test_acc"][-1] for s in seeds]
            cells.append(f"{np.mean(accs):9.2f} +-{np.std(accs, ddof=1):4.2f}")
        flat = [read(cell, c, s)["test_acc"][-1] for c in CONFIG_NAMES for s in seeds]
        print(f"{cell:>7} " + " ".join(f"{c:>16}" for c in cells) + f" {np.mean(flat):7.2f}")
    print(f"{'L0':>7} {np.mean(list(l0.values())):9.2f} +-{np.std(list(l0.values()), ddof=1):4.2f}")

    print(f"\npaired against L0, same seed, per cell and configuration")
    print(f"{'cell':>7} {'config':>9} {'mean diff':>10} {'se':>7} {'agree':>6} {'decision':>12}")
    for cell in CELLS:
        for config in CONFIG_NAMES:
            accs = [read(cell, config, s)["test_acc"][-1] for s in seeds]
            r = paired(accs, [l0[s] for s in seeds])
            dec = "no decision" if r["winner"] is None else f"{'L0' if r['winner'] == 'b' else cell} better"
            print(f"{cell:>7} {config:>9} {r['mean']:10.2f} {r['se']:7.2f} {r['agree']:>3}/{r['n']:<2} {dec:>12}")

    print(f"\ncell L3-DA against L3-D, same seed")
    for config in CONFIG_NAMES:
        a = [read("L3-DA", config, s)["test_acc"][-1] for s in seeds]
        b = [read("L3-D", config, s)["test_acc"][-1] for s in seeds]
        r = paired(a, b)
        dec = "no decision" if r["winner"] is None else f"{'L3-DA' if r['winner'] == 'a' else 'L3-D'} better"
        print(f"{config:>9} {r['mean']:10.2f} {r['se']:7.2f} {r['agree']:>3}/{r['n']:<2} {dec:>12}")

    print(f"\nrounded operands, mean over the last epoch and over every seed and configuration")
    for cell in CELLS:
        names = ["dZ3", "dZ2", "dZ1"] + (["X", "A1", "A2"] if CELLS[cell] else [])
        print(f"  {cell}")
        print(f"  {'tensor':>8} {'SQNR dB':>9} {'vanished':>10} {'cosine, first batch':>20}")
        for name in names:
            sq = [read(cell, c, s)["sqnr"][-1].get(name) for c in CONFIG_NAMES for s in seeds]
            vn = [read(cell, c, s)["vanished"][-1].get(name) for c in CONFIG_NAMES for s in seeds]
            sq = [x for x in sq if x is not None]
            vn = [x for x in vn if x is not None]
            print(f"  {name:>8} {np.mean(sq):9.2f} {100 * np.mean(vn):9.2f}%")
        for name in WEIGHTS:
            cs = [read(cell, c, s)["cos"][-1].get(name) for c in CONFIG_NAMES for s in seeds]
            cs = [x for x in cs if x is not None]
            print(f"  {name:>8} {'':>9} {'':>10} {np.mean(cs):20.4f}")

    draw(seeds)


if __name__ == "__main__":
    main()
