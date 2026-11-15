"""The weight-blocking experiment: five arms against the L2 reference, on one dataset.

Run from python/:

    uv run python block_sweep.py mnist            # all five arms, all four configs, ten seeds
    uv run python block_sweep.py mnist 1 A        # one arm, one seed, a smoke run
    uv run python block_sweep.py mnist1d

What is measured, and against what
----------------------------------
The reference is rung L2, which is already on disk in out/w/<dataset>. Its scope: the forward
pass uses FP4 weights, the backward pass is float32, the optimizer is float32. Every arm below
keeps that scope and changes ONE thing, the block shape of the weight matrices:

    A    two 1D copies: 16 inputs per block for the forward pass, 16 outputs per block for the
         backward pass. W2 and W3 only, because W1 has no backward GEMM and so no second copy.
    B2   16 by 16 tiles, W2 and W3 only. One scale per 256 weights, same value in both passes.
    B3   16 by 16 tiles, all three matrices.
    C2   4 by 4 tiles, W2 and W3 only.
    C3   4 by 4 tiles, all three matrices.

Why B and C come in two forms: the W2/W3 form changes 416 of 13,002 weights, which is 3.2%.
A null result from that would not say whether the blocking is harmless or whether 96.8% of the
weights never saw the change. The all-matrix form is the same question with the whole store in
tiles. Both are reported.

The pre-registered decision rule is l2_sweep.paired, unchanged: the sign of the paired
difference agrees in at least 8 of 10 seeds AND |mean| is more than 2 standard errors. A run with
fewer than 8 seeds is a smoke run and decides nothing.

MNIST-1D is run only if a rule fires here. The trigger, fixed before the numbers exist: run
MNIST-1D when any arm reaches |mean| of at least one standard error on MNIST, decided or not.

What is written
---------------
    out/block/<dataset>/<arm>_<config>_seed<seed>.json   one file per run, written when it ends
    docs/block_<dataset>.png    four curves per arm: train and test loss, train and test accuracy

A finished run is never repeated, so the sweep restarts where it stopped. The figure is redrawn
after every arm, so a figure exists even if the sweep is stopped early.

To stop a running sweep:

    pkill -f "[b]lock_sweep"
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qat
from train import load_dataset

# l2_sweep reads sys.argv at import time. Give it the arguments it expects, then put ours back.
# The reason to import it at all is that its paired() is the pre-registered rule, and a second
# copy of a decision rule is a second rule.
_argv = sys.argv
sys.argv = ["l2_sweep.py", "mnist", "10", "w"]
import l2_sweep as sweep
sys.argv = _argv

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
DATASET = sys.argv[1] if len(sys.argv) > 1 else "mnist"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else len(sweep.SEEDS)
WANTED = sys.argv[3:] if len(sys.argv) > 3 else list(qat.TILES)
OUT = os.path.join(HERE, "..", "out", "block", DATASET)
REF = os.path.join(HERE, "..", "out", "w", DATASET)
CONFIG_NAMES = list(qat.CONFIGS)
ARM_LABEL = {"ref": "reference, L2", "A": "A  1D fwd + 1D bwd, W2 W3",
             "B2": "B2 16x16 tiles, W2 W3", "B3": "B3 16x16 tiles, all",
             "C2": "C2 4x4 tiles, W2 W3", "C3": "C3 4x4 tiles, all"}
ARM_COLOUR = {"A": "#c0392b", "B2": "#1f77b4", "B3": "#7fb3d5", "C2": "#2e7d32", "C3": "#8bc34a"}
COLOURS = ["#111111", "#c0392b", "#1f77b4", "#7fb3d5", "#2e7d32", "#8bc34a"]


def result_path(arm, config, seed):
    return os.path.join(OUT, f"{arm}_{config}_seed{seed}.json")


def ref_path(config, seed):
    return os.path.join(REF, f"w_{config}_seed{seed}.json")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_arm(arm, config, seed, data):
    """One training run, or the stored result of one that already finished."""
    path = result_path(arm, config, seed)
    stored = load_json(path)
    if stored is not None:
        return stored
    X_train, y_train, X_test, y_test = data
    print(f"  {arm} {config} seed {seed}: training", flush=True)
    result = qat.train_qat(seed, sweep.EPOCHS, sweep.BATCH, X_train, y_train, X_test, y_test,
                           config, acts=False, verbose=False, arm=arm)
    h = result["history"]
    obj = {
        "arm": arm,
        "config": config,
        "seed": seed,
        "test_loss": [float(x) for x in h["test_loss"]],
        "test_acc": [100.0 * float(a) for a in h["test_acc"]],
        "train_loss": [float(x) for x in h["train_loss"]],
        "train_acc": [100.0 * float(a) for a in h["train_acc"]],
        "sqnr": [{k: float(v) for k, v in d.items()} for d in h["sqnr"]],
        "counters": h["counters"],
        "frozen_steps": h["frozen_steps"],
        "stale_elements": [{k: float(v) for k, v in d.items()} for d in h["stale_elements"]],
        "drift": h["drift"],
    }
    save_json(path, obj)
    print(f"    final: train loss {obj['train_loss'][-1]:.4f} test loss {obj['test_loss'][-1]:.4f} "
          f"test acc {obj['test_acc'][-1]:.2f}", flush=True)
    return obj


def read_arm(arm, config, seed):
    """A run result, or the L2 reference for arm "ref"."""
    return load_json(ref_path(config, seed)) if arm == "ref" else load_json(result_path(arm, config, seed))


def stack(arm, key, seeds):
    """Per-seed curves of one arm under one key, averaged over the four configs.

    Returns (mean, standard error) arrays of shape (epochs,), and the number of runs behind them.
    The mean over configs is allowed because no config is selected by this experiment: the
    question is the arm, and each arm is run with all four configs so the arm effect is not
    confounded with a config choice. The band is the standard error over the runs.
    """
    rows = []
    for config in CONFIG_NAMES:
        for seed in seeds:
            r = read_arm(arm, config, seed)
            if r is not None and key in r:
                rows.append(r[key])
    if not rows:
        return None, None, 0
    a = np.asarray(rows, dtype=float)
    return a.mean(axis=0), a.std(axis=0, ddof=1) / np.sqrt(len(a)), len(a)


def draw(seeds):
    """Loss and accuracy, train and test, one line per arm. Redrawn after every arm."""
    arms = ["ref"] + [a for a in qat.TILES if read_arm(a, CONFIG_NAMES[0], seeds[0]) is not None]
    epochs = np.arange(1, sweep.EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for ax, key, ylabel in ((axes[0], "loss", "cross entropy loss"),
                            (axes[1], "acc", "accuracy, percent")):
        for i, arm in enumerate(arms):
            colour = "#777777" if arm == "ref" else ARM_COLOUR[arm]
            for split, style in (("train", "--"), ("test", "-")):
                name = f"{split}_{key}"
                mean, se, n = stack(arm, name, seeds)
                if mean is None:
                    continue
                ax.plot(epochs, mean, style, color=colour, linewidth=1.7 if split == "test" else 1.1,
                        marker="o" if split == "test" else None, markersize=3,
                        label=f"{ARM_LABEL[arm]} {split}")
                ax.fill_between(epochs, mean - se, mean + se, color=colour, alpha=0.10, linewidth=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.set_xticks(epochs)
    axes[0].legend(fontsize=6.4, ncol=1, loc="upper right")
    fig.suptitle(f"{DATASET}: weight blocking. Dashed is train, solid is test. "
                 f"Each line is the mean over {len(CONFIG_NAMES)} configs and {len(seeds)} seeds, "
                 f"band is plus or minus one standard error", fontsize=8.6)
    fig.text(0.01, 0.005, "reference is rung L2, out/w/. Its train loss was never stored, so that "
                          "one dashed line is absent.", fontsize=7.2, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    path = os.path.join(DOCS, f"block_{DATASET}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)
    return path


def table(seeds, arms):
    """Final values per arm and config, and the paired test of every arm against the reference."""
    print(f"\nfinal test accuracy, percent, mean of {len(seeds)} seeds")
    print(f"{'arm':>4} " + " ".join(f"{c:>9}" for c in CONFIG_NAMES) + "     mean")
    for arm in arms:
        vals = []
        for config in CONFIG_NAMES:
            a = [read_arm(arm, config, s) for s in seeds]
            acc = [r["test_acc"][-1] for r in a if r is not None]
            vals.append(np.mean(acc) if acc else float("nan"))
        print(f"{arm:>4} " + " ".join(f"{v:9.2f}" for v in vals) + f" {np.mean(vals):8.2f}")

    print("\nfinal test loss, mean of the same seeds")
    print(f"{'arm':>4} " + " ".join(f"{c:>9}" for c in CONFIG_NAMES) + "     mean")
    for arm in arms:
        vals = []
        for config in CONFIG_NAMES:
            a = [read_arm(arm, config, s) for s in seeds]
            loss = [r["test_loss"][-1] for r in a if r is not None]
            vals.append(np.mean(loss) if loss else float("nan"))
        print(f"{arm:>4} " + " ".join(f"{v:9.4f}" for v in vals) + f" {np.mean(vals):8.4f}")

    print("\nfinal train loss, mean of the same seeds")
    print(f"{'arm':>4} " + " ".join(f"{c:>9}" for c in CONFIG_NAMES) + "     mean")
    for arm in arms:
        vals = []
        for config in CONFIG_NAMES:
            a = [read_arm(arm, config, s) for s in seeds]
            loss = [r["train_loss"][-1] for r in a if r is not None and "train_loss" in r]
            vals.append(np.mean(loss) if loss else float("nan"))
        print(f"{arm:>4} " + " ".join(f"{'n/a' if np.isnan(v) else f'{v:9.4f}'}" for v in vals)
              + f" {np.mean(vals):8.4f}")

    print("\npaired test against the reference, per arm and config, on final test accuracy")
    print(f"{'arm':>4} {'config':>9} {'mean diff':>10} {'se':>8} {'agree':>6} {'cond1':>6} "
          f"{'cond2':>6} {'winner':>7}")
    for arm in arms:
        if arm == "ref":
            continue
        for config in CONFIG_NAMES:
            a = [read_arm(arm, config, s) for s in seeds]
            b = [read_arm("ref", config, s) for s in seeds]
            if any(r is None for r in a) or any(r is None for r in b):
                continue
            p = sweep.paired([r["test_acc"][-1] for r in a], [r["test_acc"][-1] for r in b])
            print(f"{arm:>4} {config:>9} {p['mean']:10.3f} {p['se']:8.3f} {p['agree']:6d} "
                  f"{str(p['cond1']):>6} {str(p['cond2']):>6} {str(p['winner']):>7}")

    print("\nthe drift telemetry of arm A, first and last epoch, mean of the seeds")
    print("  cos_dA2 and cos_dW2 come from the W3 copy. cos_dA1 and cos_dW1 come from the W2 copy.")
    print("  cos_dW3 is 1.0 by construction: dW3 = A2.T @ dZ3 never reads W3.")
    for config in CONFIG_NAMES:
        rows = [read_arm("A", config, s) for s in seeds]
        rows = [r for r in rows if r is not None and r["drift"]]
        if not rows:
            continue
        keys = ("cos_w_W2", "cos_w_W3", "cos_dA2", "cos_dA1", "cos_dW2", "cos_dW1", "cos_dW3")
        f = {k: np.mean([r["drift"][0][k] for r in rows]) for k in keys}
        l = {k: np.mean([r["drift"][-1][k] for r in rows]) for k in keys}
        print(f"  {config} epoch 1  " + " ".join(f"{k} {f[k]:.4f}" for k in keys))
        print(f"  {config} epoch {len(rows[0]['drift'])} " +
              " ".join(f"{k} {l[k]:.4f}" for k in keys))


def main():
    assert DATASET in ("mnist", "mnist1d"), f"unknown dataset {DATASET}"
    for arm in WANTED:
        assert arm in qat.TILES, f"unknown arm {arm}, choose from {list(qat.TILES)}"
    seeds = sweep.SEEDS[:N_SEEDS]
    runs = len(WANTED) * len(CONFIG_NAMES) * len(seeds)
    print(f"dataset {DATASET}: {len(WANTED)} arms {WANTED}, {len(CONFIG_NAMES)} configs, "
          f"{len(seeds)} seeds, {sweep.EPOCHS} epochs, {runs} runs", flush=True)
    print(f"  reference read from {REF}, arms written to {OUT}", flush=True)

    data = load_dataset(DATASET)
    for arm in WANTED:
        print(f"\narm {arm}: {ARM_LABEL.get(arm, arm)}", flush=True)
        for config in CONFIG_NAMES:
            for seed in seeds:
                run_arm(arm, config, seed, data)
        draw(seeds)

    table(seeds, WANTED)
    print("\nthe figure is", os.path.join(DOCS, f"block_{DATASET}.png"))


if __name__ == "__main__":
    main()
