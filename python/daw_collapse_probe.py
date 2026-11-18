"""Which operand of the gradient-down GEMM kills training on MNIST-1D?

The cell L4-DAW rounds the operands of both GEMMs of the backward pass. On MNIST every cell
stays inside 0.2 points. On MNIST-1D the DAW cell collapses, and the loss of some seeds reads
NaN. This probe finds the cause with three short runs instead of a theory.

TERMS

  dW GEMM      dW = A.T @ dZ. The contraction is over the batch. It gives the weight gradient.
  dA GEMM      dA = dZ @ W.T. The contraction is over the layer output width N. It gives the
               gradient that flows one layer down, so it sets every weight below it.
  rounded along N   blocked so that the 16 values that share a scale are 16 different outputs
               of the same sample. For a layer of width 16 that is the WHOLE layer width, so
               one scale covers every output of the layer.
  rounded along the batch   blocked so that the 16 values that share a scale are the same
               output of 16 different samples.
  cell L4-DAW  rounds A, A1, A2, dZ along the batch for the dW GEMM, and dZ and W along N for
               the dA GEMM. Stochastic rounding or round to nearest, by the rung.

THE FOUR VARIANTS, two epochs each, one seed, MNIST-1D

  1. both operands of the dA GEMM rounded, stochastic rounding. This is the shipped cell.
  2. both operands rounded, round to nearest. This is the L3-DAW cell, which was never run
     on MNIST-1D, because the protocol only extends a rung to the second dataset when a rule
     fires on the first. It does not fire, so this comparator did not exist. This adds it.
  3. only dZ rounded, the weight left float.
  4. only the weight rounded, dZ left float.

The discriminator between a weight and a batch tensor is the row count: a weight has at most 16
rows, a batch tensor has 128. The batches of a seed are the same in every variant, because the
shuffle generator is seeded by the seed alone.

THE WIDTH IS AN ARGUMENT

    uv run python daw_collapse_probe.py [width]

At width 16 the discriminator between the two operands below was the row count: a weight has at most
16 rows and a batch tensor has 128. That discriminator is FALSE at width 64, because W2 is (64, 64)
and W3 is (64, 10), and both have more than 16 rows. l3.quantize_axis_n now carries the name of the
operand, and this probe uses the name. A shape says nothing about what a tensor is.

Run it from python/ with PYTHONPATH=.:
    uv run python daw_collapse_probe.py 16 > ../out/daw_collapse_probe.txt
    uv run python daw_collapse_probe.py 64 > ../out/daw_collapse_probe_w64.txt
Cost: about 12 minutes at width 16, about 20 at width 64.
"""
import sys

# l3 reads its dataset, seed count, rounding rule, width and cells from sys.argv at import time.
# This probe fixes all of those except the width, so sys.argv is reduced to the dataset name BEFORE
# the import. Without this line l3 reads this probe's width argument as a dataset name and stops.
WIDTH = int(sys.argv[1]) if len(sys.argv) > 1 else 16
sys.argv = [sys.argv[0], "mnist1d"]

import l3
from train import load_dataset

EPOCHS = 2
SEED = 42
CONFIG = "a6-e6m2"


def patched_quantize_axis(round_delta, round_weight, orig):
    """quantize_axis_n with one operand of the dA GEMM allowed to pass through untouched.

    The operand is identified by its name, never by its shape. W2 and W3 are named W2 and W3, and the
    two deltas are named dZ2 and dZ3.
    """

    def inner(T, axis, table, rule, name, rng=None):
        is_weight = name.startswith("W")
        if (is_weight and not round_weight) or (not is_weight and not round_delta):
            return T, None, None
        return orig(T, axis, table, rule, name, rng)

    return inner


def main():
    l3.WIDTH = WIDTH
    X, y, Xt, yt = load_dataset("mnist1d")
    orig = l3.quantize_axis_n
    # A variant that leaves an operand float has no index to count, and the telemetry of l3
    # expects one. A probe-level guard is enough, because this probe reads accuracy and loss only.
    orig_vanished = l3.vanished
    l3.vanished = lambda blocks, idx: 0.0 if idx is None else orig_vanished(blocks, idx)
    variants = [
        ("the DA cell, neither operand of the dA GEMM rounded", "sr", False, False),
        ("both operands, stochastic rounding, the shipped cell", "sr", True, True),
        ("both operands, round to nearest, the L3 cell never run here", "rtn", True, True),
        ("delta only, weight float", "sr", True, False),
        ("weight only, delta float", "sr", False, True),
    ]
    print(f"dataset mnist1d, {WIDTH} wide layers, {EPOCHS} epochs, seed {SEED}, config {CONFIG}, "
          f"cell DAW")
    print("the first row rounds neither operand of the dA GEMM. It is the reference every other row")
    print("is read against, and it is measured here so that the table stands alone at any width")
    print()
    print(f"  {'variant':56s} {'epoch 1':>8s} {'epoch 2':>8s} {'loss 2':>9s}")
    print("  " + "-" * 84)
    for label, rule, rd, rw in variants:
        l3.ROUNDING = rule
        l3.RUNG = "L4" if rule == "sr" else "L3"
        l3.CELLS = {f"{l3.RUNG}-{k}": v for k, v in l3.CELL_KINDS.items()}
        l3.quantize_axis_n = patched_quantize_axis(rd, rw, orig)
        p, h = l3.train_l3(SEED, EPOCHS, 128, X, y, Xt, yt, CONFIG, f"{l3.RUNG}-DAW")
        print(f"  {label:56s} {h['test_acc'][0]:8.2f} {h['test_acc'][1]:8.2f} "
              f"{h['test_loss'][-1]:9.3f}", flush=True)
    l3.quantize_axis_n = orig
    l3.vanished = orig_vanished


if __name__ == "__main__":
    main()
