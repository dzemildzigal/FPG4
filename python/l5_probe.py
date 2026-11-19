"""The price of the forward pass at width 64, measured with no training at all.

Why this probe exists. Rung L5 puts the forward pass on the FP4 grid, and a trained model cannot say
which part of the loss belongs to which rounding, because training moves the weights to compensate. So
this probe holds the weights still: it loads the saved float weights of L0 and changes ONLY how the
forward pass rounds. Every accuracy below is the same weight set put through a different forward pass.

The five settings, in the order the report uses them:

    float                    no rounding. The reference, and it must reproduce L0
    weights only             W1, W2, W3 on the FP4 grid, activations in float32. This is rung L1
    1d, round to nearest     weights and activations on the grid, one scale per 16 values along the
                             axis the forward GEMM sums over, payload rounded to the nearest code
    1d, stochastic           the same, payload drawn. This is arm L5a before it trains
    16 by 16, round to nearest
    16 by 16, stochastic     the same with one scale per 16 by 16 tile. This is arm L5b before it trains

The scale rule is the forward rule, the nearest table entry (ADR-032), for every setting. Nothing here
touches the backward pass.

    python l5_probe.py [dataset] [seeds]      # writes out/l5_probe.json

Why three draws for the stochastic settings: stochastic rounding is a random variable, so one draw is
one sample. The mean of three draws is the quantity the report quotes, and the spread across draws is
reported so the reader can see how much of the difference is noise.
"""
import json
import os
import sys

import numpy as np

DATASET = sys.argv[1] if len(sys.argv) > 1 else "mnist1d"
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42, 789, 2020]
# l3 reads sys.argv at import time to pick its rung, its seeds and its output directory. This probe does
# not run l3's main, but the import still parses the arguments, so the list is replaced before the import
# and the real arguments are already saved above.
sys.argv = ["l5_probe.py", DATASET, str(len(SEEDS)), "sr", "64"]

import l3  # noqa: E402
import quant as q  # noqa: E402
from model import evaluate  # noqa: E402
from train import load_dataset, load_weights  # noqa: E402
WIDTH = 64
CONFIG = "a6-e6m2"
DRAWS = 3
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "l5_probe.json")

table, rule = l3.CONFIGS[CONFIG]          # UE6M2, target amax/6


def quant_weights(P, shape, rng):
    """The weight store on the FP4 grid. The block runs along axis 0, the axis the forward GEMM sums over."""
    out = dict(P)
    for k in ("W1", "W2", "W3"):
        if shape == "tile":
            out[k] = l3.quantize_tile(P[k], table, rule, rng, scale_rounding="nearest")[0]
        else:
            out[k] = l3.quantize_axis(P[k], 0, table, rule, rng, scale_rounding="nearest")[0]
    return out


def hook(shape, rng):
    """The activation quantizer that model.forward calls on X, then A1, then A2."""
    def f(A):
        if shape == "tile":
            return l3.quantize_tile(A, table, rule, rng, scale_rounding="nearest")[0], {}
        return l3.quantize_axis(A, 1, table, rule, rng, scale_rounding="nearest")[0], {}
    return f


def main():
    X, y, Xt, yt = load_dataset(DATASET)
    rows = {}

    def put(name, acc):
        rows.setdefault(name, []).append(float(acc))

    for seed in SEEDS:
        P = load_weights(seed, DATASET, WIDTH)
        put("float", 100 * evaluate(P, Xt, yt)[1])
        # Weights on the grid, activations untouched: the L1 setting, which isolates the store.
        put("weights only", 100 * evaluate(quant_weights(P, "1d", None), Xt, yt)[1])
        for shape in ("1d", "tile"):
            # Round to nearest is one deterministic answer, so it is not repeated for each draw.
            put(f"{shape} rtn", 100 * evaluate(quant_weights(P, shape, None), Xt, yt,
                                               quant=hook(shape, None))[1])
            for draw in range(DRAWS):
                rng = np.random.default_rng([seed, 7, draw])
                put(f"{shape} sr", 100 * evaluate(quant_weights(P, shape, rng), Xt, yt,
                                                 quant=hook(shape, rng))[1])

    summary = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                   "n": len(v), "values": v} for k, v in rows.items()}
    obj = {"dataset": DATASET, "width": WIDTH, "config": CONFIG, "seeds": SEEDS,
           "draws": DRAWS, "scale_rule": "nearest", "results": summary}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)

    print(f"{DATASET}, width {WIDTH}, config {CONFIG}, scale rule nearest, {len(SEEDS)} seeds, "
          f"payload drawn {DRAWS} times")
    base = summary["float"]["mean"]
    for k, v in summary.items():
        print(f"  {k:16s} {v['mean']:6.2f}  sd {v['sd']:4.2f}  n {v['n']:2d}  delta {v['mean'] - base:+6.2f}")


if __name__ == "__main__":
    main()
