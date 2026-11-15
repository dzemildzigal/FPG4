"""Loss and accuracy against training epoch, for L0, L2 and L2a.

Nothing is retrained here. Every number comes from a run that already happened.

Terms used here
---------------
epoch            one full pass over the training set. It is the x axis of both figures.
train loss       the mean cross entropy over the training batches of that epoch.
test loss        the mean cross entropy over the whole test set, measured at the end of that epoch.
train accuracy   the fraction of training samples classified correctly during that epoch.
test accuracy    the fraction of the whole test set classified correctly at the end of that epoch.
cross entropy    the loss this project trains with. Lower is better. It is not a probability,
                 so only its change against epoch, and its difference between rungs, are read here.

Where the data comes from
-------------------------
L0
    python/fp32_probe.py trained L0 in float32 for the seeds 42, 4242 and 424242 and its
    run log printed one line per epoch, because train() prints them. The log is the only
    place those curves exist: train() returns the history but the L0 runs did not save it.
    parse_probe_log() reads the log back into numbers, and the result is stored in
    python/l0_history.json so the curves survive the deletion of the log.
L2 and L2a
    out/w/<dataset> and out/wa/<dataset> hold one JSON per run, and every JSON stores the
    full per-epoch history. Nothing is recomputed.

The seeds used in the figures
-----------------------------
The L0 telemetry exists for 3 seeds, the L2 and L2a telemetry for 10. A mean over 3 seeds
and a mean over 10 seeds differ by seed noise, so the figures use the 3 seeds that both
have: 42, 4242 and 424242. The three L0 curves and the three L2 curves are then the same
draws of the same seeds and the gap between them is the rung, not the seed set.

Which configuration is drawn for L2 and L2a
-------------------------------------------
The best one, and the rule is mechanical: highest mean final test accuracy over the 10
available seeds, and the lower mean final test loss when two configs sit inside 0.01
accuracy points of each other. The rule is computed on all 10 seeds because that is all
the telemetry there is, then its winning config is drawn on the 3 shared seeds.

Usage
-----
    uv run python curves.py mnist
    uv run python curves.py mnist1d
"""

import json
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import l2_sweep as sweep

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
# The figures live in docs/ and only there: docs/experiment_L2.md shows them, and two copies of the
# same PNG in one repository is one copy too many. The JSON stays in out/ with the other results.
DOCS = os.path.join(HERE, "..", "docs")
KEYS = ("train_loss", "train_acc", "test_loss", "test_acc")
CONFIGS = ("a6-e6m2", "f6-e6m2", "a6-e5m3", "f6-e5m3")
MODES = {"w": "L2", "wa": "L2a"}
COLOUR = {"L0": "#222222", "L2": "#1f77b4", "L2a": "#d95f02"}
PROBE_SEEDS = (42, 4242, 424242)
# The jitter measurement below starts at this epoch and ignores the fast climb before it. Before
# epoch 4 every rung is still gaining more than a point per epoch, and that gain is the fit, not
# the step. From epoch 4 the curve is in its slow tail, where a movement is a movement.
FIRST_STABLE = 4
PROBE_LOG = "/tmp/fp32_probe.log"
L0_FILE = os.path.join(HERE, "l0_history.json")


def parse_probe_log(path):
    """Read the per-epoch lines that fp32_probe.py printed while it trained L0 in float32.

    The log format is fixed by train.py, which prints one line per epoch:
        <epoch> <train_loss> <train_acc> <test_loss> <test_acc>
    and fp32_probe.py prints the seed number after the ten lines of that seed:
        <seed> | <float64 accuracy> | <float32 accuracy> | <difference>

    The epoch line counter starts at 0, so epoch 0 of the log is epoch 1 of the figure.
    Returns {dataset: {seed: {key: [10 values]}}}.
    """
    found = {}
    dataset = None
    block = []
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if line.startswith("=== ") and line.endswith(" ==="):
            dataset = line.strip("= ").strip()
            found[dataset] = {}
            block = []
            continue
        if dataset is None:
            continue
        head = line.split("|")[0].strip()
        if "|" in line and head.isdigit():
            # the seed line closes the block of ten epoch lines above it
            if len(block) != 10:
                raise ValueError(f"seed {head} closed a block of {len(block)} lines, not 10")
            columns = np.array(block, dtype=float).T
            found[dataset][int(head)] = {k: v.tolist() for k, v in zip(KEYS, columns)}
            block = []
        elif re.match(r"^\d+ ", line) and len(line.split()) == 5:
            block.append([float(x) for x in line.split()[1:]])
    return found


def load_l0():
    """Return the L0 history per dataset, from python/l0_history.json or from the probe log."""
    if os.path.exists(L0_FILE):
        return json.load(open(L0_FILE))
    if not os.path.exists(PROBE_LOG):
        raise FileNotFoundError(
            f"neither {L0_FILE} nor {PROBE_LOG} exists. The L0 per-epoch history was never "
            "saved anywhere else, and this script does not retrain.")
    found = parse_probe_log(PROBE_LOG)
    json.dump(found, open(L0_FILE, "w"), indent=1)
    print(f"wrote {os.path.basename(L0_FILE)} from {PROBE_LOG}")
    # The file round trip turns the integer seed keys into strings. Do it here as well, so that the
    # first call of this script and every later one index the seeds the same way.
    return json.loads(json.dumps(found))


def read_run(mode, dataset, config, seed):
    path = os.path.join(OUT, mode, dataset, f"{mode}_{config}_seed{seed}.json")
    return json.load(open(path))


def choose_best(mode, dataset, seeds):
    """Pick the config by the rule in the module docstring, using every available seed."""
    table = {}
    for config in CONFIGS:
        rows = [read_run(mode, dataset, config, s) for s in seeds]
        # test_acc in these files is already a percentage. test_loss is the raw cross entropy.
        table[config] = (np.mean([r["test_acc"][-1] for r in rows]),
                         np.mean([r["test_loss"][-1] for r in rows]))
    ranked = sorted(table.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
    best = ranked[0][0]
    if abs(ranked[1][1][0] - table[best][0]) < 0.01:
        best = min([best, ranked[1][0]], key=lambda c: table[c][1])
    return best, table


def rung_curves(mode, dataset, config, seeds):
    """Mean over the seeds, per epoch, of every series the run files hold.

    l2_sweep.py stored test_loss, test_acc and train_acc per epoch. It did NOT store
    train_loss, so the training loss curve of L2 and L2a does not exist and is not drawn.
    The absent key is reported by draw() and never filled in.
    """
    rows = [read_run(mode, dataset, config, s) for s in seeds]
    out = {}
    for key in KEYS:
        if key not in rows[0]:
            continue
        a = np.array([r[key] for r in rows], dtype=float)
        out[key] = a.mean(axis=0)
        out[key + "_se"] = a.std(axis=0, ddof=1) / np.sqrt(len(rows))
    return out


def l0_curves(l0, dataset, seeds):
    """Same, from the probe log. There accuracy is a FRACTION, so it is scaled to percent here."""
    rows = [l0[dataset][str(s)] for s in seeds]
    out = {}
    for key in KEYS:
        a = np.array([r[key] for r in rows], dtype=float)
        if key.endswith("acc"):
            a = a * 100.0
        out[key] = a.mean(axis=0)
        out[key + "_se"] = a.std(axis=0, ddof=1) / np.sqrt(len(rows))
    return out


def draw(dataset):
    l0 = load_l0()
    missing = [s for s in PROBE_SEEDS if str(s) not in l0[dataset]]
    if missing:
        raise KeyError(f"{dataset}: no L0 telemetry for seeds {missing}")

    curves = {"L0": l0_curves(l0, dataset, PROBE_SEEDS)}
    chosen = {}
    for mode, rung in MODES.items():
        best, table = choose_best(mode, dataset, sweep.SEEDS)
        chosen[rung] = best
        print(f"{rung}: drawn config {best}, chosen on {len(sweep.SEEDS)} seeds")
        for config, (acc, loss) in sorted(table.items(), key=lambda kv: -kv[1][0]):
            print(f"    {config:8s} test accuracy {acc:6.2f}   test loss {loss:.4f}"
                  f"{'  <- drawn' if config == best else ''}")
        curves[rung] = rung_curves(mode, dataset, best, PROBE_SEEDS)

    # Per seed test accuracy, shape (epochs, seeds), for the jitter measurement. The mean curves
    # above cannot show the jitter: the seeds move in different directions and the mean cancels it.
    acc_per_seed = {"L0": np.array([np.array(l0[dataset][str(seed)]["test_acc"]) * 100.0
                                    for seed in PROBE_SEEDS]).T}
    for mode, rung in MODES.items():
        rows = [read_run(mode, dataset, chosen[rung], seed) for seed in PROBE_SEEDS]
        acc_per_seed[rung] = np.array([r["test_acc"] for r in rows], dtype=float).T
    # |change| of one epoch to the next, from epoch FIRST_STABLE on, per seed. Then the mean of those.
    jitter = {rung: np.abs(np.diff(a[FIRST_STABLE - 1:], axis=0)).mean(axis=0)
              for rung, a in acc_per_seed.items()}

    epochs = np.arange(1, sweep.EPOCHS + 1)
    export = {"dataset": dataset, "config": chosen, "seeds": list(PROBE_SEEDS),
              "epochs": sweep.EPOCHS, "note": "each entry is the mean over the seeds above",
              "jitter_test_acc_epochs": [FIRST_STABLE, sweep.EPOCHS],
              "jitter_test_acc": {r: {"per_seed": np.asarray(v).tolist(),
                                      "mean": float(np.mean(v))} for r, v in jitter.items()},
              "rungs": {r: {k: np.asarray(v).tolist() for k, v in h.items()}
                        for r, h in curves.items()}}
    path_json = os.path.join(OUT, f"curves_{dataset}.json")
    json.dump(export, open(path_json, "w"), indent=1)

    print(f"\nmean |change| of test accuracy between one epoch and the next, "
          f"epochs {FIRST_STABLE} to {sweep.EPOCHS}, per seed then averaged")
    for rung, v in jitter.items():
        print(f"    {rung:4s} " + " ".join(f"{x:.3f}" for x in v) + f"   mean {np.mean(v):.3f}")

    for key, ylabel, suffix, title in (
            ("loss", "cross entropy loss", "loss", "loss"),
            ("acc", "accuracy, percent", "acc", "accuracy")):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for rung, h in curves.items():
            for split, style in (("train", "--"), ("test", "-")):
                name = f"{split}_{key}"
                if name not in h:
                    print(f"{rung}: no {name} was ever stored, so it is not drawn")
                    continue
                y = h[name]
                se = h[name + "_se"]
                ax.plot(epochs, y, style, color=COLOUR[rung], linewidth=1.8,
                        marker="o", markersize=3, label=f"{rung} {split}")
                ax.fill_between(epochs, y - se, y + se, color=COLOUR[rung], alpha=0.13, linewidth=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{dataset}: {title} against training epoch\n"
                     f"mean of {len(PROBE_SEEDS)} seeds, band is plus or minus one standard error")
        ax.grid(alpha=0.25)
        ax.set_xticks(epochs)
        ax.legend(ncol=2, fontsize=9, loc="best")
        fig.text(0.01, 0.01, f"L0 float32 | L2 {chosen['L2']} | L2a {chosen['L2a']} | "
                             f"dashed = train, solid = test", fontsize=7.5, color="#555555")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = os.path.join(DOCS, f"curves_{dataset}_{suffix}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"wrote {path}")
    print(f"wrote {path_json}")


if __name__ == "__main__":
    draw(sys.argv[1] if len(sys.argv) > 1 else "mnist")
