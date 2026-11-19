"""Figures for rung L5. Every number is read from a stored run file or from a stored probe file.

    python l5_figure.py        # writes four figures into docs/

Four files, one file name per experiment and dataset, as the naming rule requires:

    experiment_L5_mnist1d_arms.png          the three arms on MNIST-1D, float master and deployed model
    experiment_L5_mnist_arms.png            the same on MNIST, where only two arms exist
    experiment_L5_mnist1d_forward_ptq.png   the forward pass alone, six settings, no training
    experiment_L5_mnist_forward_ptq.png     the same on MNIST

The arms and their one-line difference:

    L5a  forward 1D block, stochastic payload
    L5b  forward 16 by 16 tile, stochastic payload
    L5c  forward 1D block, round-to-nearest payload

All three have the same backward pass: 16 by 16 tile, stochastic payload, rounded-up scale. So the
only difference between the arms is a property of the forward pass.
"""
import glob
import json
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
OUT = os.path.join(HERE, "..", "out")

ARMS = [("l5a", "L5a\n1d / SR"), ("l5b", "L5b\n16x16 / SR"), ("l5c", "L5c\n1d / RTN")]
ARM_COLOUR = {"l5a": "#1f77b4", "l5b": "#d62728", "l5c": "#2ca02c"}

# Run directories, absolute so the footer can name them without ambiguity. The three arms of MNIST-1D
# ran on the Ryzen and were pulled here. The two arms of MNIST ran on this machine.
RUNS = {
    ("mnist1d", "l5a"): "/tmp/hunt/ryzen_l5/l5a",
    ("mnist1d", "l5b"): "/tmp/hunt/ryzen_l5/l5b",
    ("mnist1d", "l5c"): "/tmp/hunt/ryzen_l5/l5c",
    ("mnist", "l5a"): os.path.join(OUT, "l5_w64_t16_f1d", "mnist"),
    ("mnist", "l5b"): os.path.join(OUT, "l5_w64_t16_ftile16", "mnist"),
}
PROBE = {d: f"/tmp/hunt/probe_{d}.json" for d in ("mnist1d", "mnist")}
L0_LOG = {"mnist1d": os.path.join(OUT, "l0_wide_mnist1d.log"),
          "mnist": os.path.join(OUT, "l0_wide_mnist.log")}
DATASET_NAME = {"mnist1d": "MNIST-1D", "mnist": "MNIST"}


def mean_se(values):
    m = st.mean(values)
    se = st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return m, se


def load_float(dataset):
    """(mean, standard error) of the ten final accuracies of the width-64 float run.

    The log holds ten blocks of ten epochs. A line ends with the test accuracy as a fraction, so the
    tenth line of a block is the final accuracy of that seed.
    """
    rows = [line.split() for line in open(L0_LOG[dataset], encoding="utf-8") if line.strip()]
    final = [100.0 * float(rows[i * 10 + 9][4]) for i in range(10)]
    return mean_se(final)


def load_arm(dataset, arm):
    """(master mean, master se, deployed mean, deployed se, n) of one arm."""
    master, deployed = [], []
    for path in glob.glob(os.path.join(RUNS[(dataset, arm)], "*.json")):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        master.append(obj["test_acc"][-1])
        deployed.append(obj["test_acc_q"][-1])
    return (*mean_se(master), *mean_se(deployed), len(master))


def arms_figure(dataset):
    f0, f0se = load_float(dataset)
    rows = [(a, label) + load_arm(dataset, a) for a, label in ARMS if (dataset, a) in RUNS]
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    x = range(len(rows))
    w = 0.36
    for i, (arm, label, mm, ms, dm, ds, n) in enumerate(rows):
        ax.bar(i - w / 2, mm, w, yerr=ms, capsize=4, color=ARM_COLOUR[arm],
               label="float master" if i == 0 else None,
               error_kw={"elinewidth": 1.2})
        ax.bar(i + w / 2, dm, w, yerr=ds, capsize=4, color=ARM_COLOUR[arm], alpha=0.45,
               hatch="//", edgecolor=ARM_COLOUR[arm],
               label="deployed, FP4 store" if i == 0 else None,
               error_kw={"elinewidth": 1.2})
        ax.text(i - w / 2, mm + ms + 0.25, f"{mm:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, dm + ds + 0.25, f"{dm:.2f}", ha="center", fontsize=9)
    ax.axhline(f0, color="black", ls="--", lw=1.0)
    ax.text(-0.45, f0 + 0.15, f"float32 {f0:.2f}", ha="left", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label, *_ in rows])
    lo = min(min(r[2] - r[3], r[4] - r[5]) for r in rows) - 1.2
    ax.set_ylim(lo, 100)
    ax.set_ylabel("test accuracy, percent")
    ax.set_title(f"L5 on {DATASET_NAME[dataset]}, width 64: the forward pass is the only change\n"
                 "bar = mean of 10 seeds, whisker = standard error, dashed line = float32")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    name = f"experiment_L5_{dataset}_arms.png"
    foot = (f"file {name}   runs " + " ".join(RUNS[(dataset, a)][-28:] for a, _ in ARMS
                                             if (dataset, a) in RUNS)
            + f"   float {os.path.basename(L0_LOG[dataset])}   10 seeds, 42 to 4040")
    fig.text(0.5, 0.005, foot, ha="center", fontsize=6.0, color="#444")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(DOCS, name), dpi=150)
    plt.close(fig)
    return name


def probe_figure(dataset):
    p = json.load(open(PROBE[dataset], encoding="utf-8"))
    res = p["results"]
    order = ["float", "weights only", "1d rtn", "1d sr", "tile rtn", "tile sr"]
    labels = ["float\npass", "weights only\nfloat acts", "1d\nRTN", "1d\nSR", "16x16\nRTN", "16x16\nSR"]
    colours = ["#444444", "#7f7f7f", "#2ca02c", "#98df8a", "#d62728", "#ff9896"]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    vals = [res[k]["mean"] for k in order]
    errs = [res[k]["sd"] / math.sqrt(res[k]["n"]) for k in order]
    ax.bar(range(len(order)), vals, 0.62, yerr=errs, capsize=4, color=colours,
           error_kw={"elinewidth": 1.2})
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.15, f"{v:.2f}", ha="center", fontsize=9)
        if i:
            ax.text(i, vals[0] + 0.9, f"{v - vals[0]:+.2f}", ha="center", fontsize=8,
                    color="#b22222" if v - vals[0] < -3 else "#333")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    lo = min(vals) - 2.0
    ax.set_ylim(lo, max(vals) + 2.2)
    ax.set_ylabel("test accuracy, percent")
    ax.set_title(f"The forward pass alone on {DATASET_NAME[dataset]}, width 64, saved float weights\n"
                 f"bar = mean of {len(p['seeds'])} seeds, SR drawn {p['draws']} times, scale nearest\n"
                 "red number = delta against the float pass")
    ax.grid(axis="y", alpha=0.3)
    name = f"experiment_L5_{dataset}_forward_ptq.png"
    foot = (f"file {name}   probe {os.path.basename(PROBE[dataset])}   source python/l5_probe.py   "
            f"seeds {','.join(str(s) for s in p['seeds'])}   config {p['config']}   no training")
    fig.text(0.5, 0.005, foot, ha="center", fontsize=6.0, color="#444")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(DOCS, name), dpi=150)
    plt.close(fig)
    return name


if __name__ == "__main__":
    for d in ("mnist1d", "mnist"):
        print("wrote", arms_figure(d))
        print("wrote", probe_figure(d))
