"""Figure for the 16 by 16 tile arm of L4, width 64, MNIST and MNIST-1D.

Every number is read from a stored run file. Nothing is recomputed and nothing is typed in.

    python tile_figure.py            # writes both figures into docs/

The two arms are the block shape and nothing else:

    1d     one scale per 16 values along the contraction axis. Cell DAW needs a second copy of dZ,
           blocked along the other axis, so dZ is rounded twice in one step
    tile   one scale per 16 by 16 tile. The tile holds both axes, so dZ is rounded once

Both arms use stochastic rounding of the payload and the rounded-up block scale, and both run at width
64 with the same ten seeds, so the comparison is paired seed by seed.
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

# Run directories, absolute so the footer can name them without ambiguity.
RUNS = {
    ("mnist1d", "1d"): "/tmp/hunt/ryzen_w64",
    ("mnist1d", "tile"): "/tmp/hunt/ryzen_t16",
    ("mnist", "1d"): "/tmp/hunt/ryzen_mnist_1d",
    ("mnist", "tile"): "/tmp/hunt/ryzen_t16_mnist",
}

# The float reference at width 64, read from its own log. The log holds ten blocks of ten epochs and the
# last column of a line is the test accuracy as a fraction, so the tenth line of each block is the final
# accuracy of that seed.
L0_LOG = {"mnist1d": os.path.join(HERE, "..", "out", "l0_wide_mnist1d.log"),
          "mnist": os.path.join(HERE, "..", "out", "l0_wide_mnist.log")}


def load_float(dataset):
    """(mean, standard error) of the ten final accuracies of the width-64 float run."""
    rows = [line.split() for line in open(L0_LOG[dataset], encoding="utf-8") if line.strip()]
    final = [100.0 * float(rows[i * 10 + 9][4]) for i in range(10)]
    return mean_se(final)

CONFIGS = ("a6-e6m2", "f6-e5m3")


def load(directory):
    """test accuracy of every stored DAW run, keyed by (config, seed)."""
    out = {}
    for path in glob.glob(os.path.join(directory, "*.json")):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if "DAW" in obj["cell"]:
            out[(obj["config"], obj["seed"])] = obj["test_acc"][-1]
    return out


def mean_se(values):
    m = st.mean(values)
    se = st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return m, se


def figure(dataset, outfile):
    arms = {shape: load(RUNS[(dataset, shape)]) for shape in ("1d", "tile")}
    fm, fse = load_float(dataset)

    labels, means, errs, colours = [], [], [], []
    labels.append("float32")
    means.append(fm)
    errs.append(fse)
    colours.append("#444444")
    for cfg in CONFIGS:
        for shape in ("1d", "tile"):
            seeds = sorted(s for (c, s) in arms[shape] if c == cfg)
            m, se = mean_se([arms[shape][(cfg, s)] for s in seeds])
            labels.append(f"{cfg}\n{shape}")
            means.append(m)
            errs.append(se)
            colours.append("#1f77b4" if shape == "1d" else "#d62728")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = range(len(labels))
    ax.bar(x, means, yerr=errs, capsize=4, color=colours, width=0.62)
    ax.axhline(fm, color="#444444", linestyle="--", linewidth=1)
    for i, (m, e) in enumerate(zip(means, errs)):
        ax.text(i, m + e + 0.06, f"{m:.2f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(min(means) - 1.0, max(means) + 0.9)
    ax.set_ylabel("test accuracy, percent")
    ax.set_title(f"L4-DAW at width 64 on {dataset}: the block shape is the only change\n"
                 f"bar = mean of 10 seeds, whisker = standard error, dashed line = float32", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.012, 0.012, f"file {os.path.basename(outfile)}   runs "
             f"{RUNS[(dataset, '1d')]} and {RUNS[(dataset, 'tile')]}   seed list 42,4242,424242,123,"
             f"456,789,1010,2020,3030,4040", fontsize=7, color="#666666")
    fig.savefig(outfile, dpi=140)
    print("wrote", outfile)


if __name__ == "__main__":
    figure("mnist1d", os.path.join(DOCS, "experiment_L4_mnist1d_tile_vs_1d_block.png"))
    figure("mnist", os.path.join(DOCS, "experiment_L4_mnist_tile_vs_1d_block.png"))
