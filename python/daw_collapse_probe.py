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

Run it from python/ with PYTHONPATH=.:  uv run python daw_collapse_probe.py > ../out/daw_collapse_probe.txt
Cost: about 12 minutes.
"""
import numpy as np

import l3
from train import load_dataset

EPOCHS = 2
SEED = 42
CONFIG = "a6-e6m2"


def patched_quantize_axis(round_delta, round_weight, orig):
    """quantize_axis with one operand of the dA GEMM allowed to pass through untouched."""

    def inner(T, axis, table, rule, rng=None):
        is_weight = T.shape[0] <= 16
        if (is_weight and not round_weight) or (not is_weight and not round_delta):
            return T, None, None
        return orig(T, axis, table, rule, rng)

    return inner


def main():
    X, y, Xt, yt = load_dataset("mnist1d")
    orig = l3.quantize_axis
    # A variant that leaves an operand float has no index to count, and the telemetry of l3
    # expects one. A probe-level guard is enough, because this probe reads accuracy and loss only.
    orig_vanished = l3.vanished
    l3.vanished = lambda blocks, idx: 0.0 if idx is None else orig_vanished(blocks, idx)
    variants = [
        ("both operands, stochastic rounding, the shipped cell", "sr", True, True),
        ("both operands, round to nearest, the L3 cell never run here", "rtn", True, True),
        ("delta only, weight float", "sr", True, False),
        ("weight only, delta float", "sr", False, True),
    ]
    print(f"dataset mnist1d, {EPOCHS} epochs, seed {SEED}, config {CONFIG}, cell DAW")
    print("for scale: the DA cell, which rounds neither operand of the dA GEMM, reads 59.97 after")
    print("epoch 1 and 62.24 after epoch 2 on this seed")
    print()
    print(f"  {'variant':56s} {'epoch 1':>8s} {'epoch 2':>8s} {'loss 2':>9s}")
    print("  " + "-" * 84)
    for label, rule, rd, rw in variants:
        l3.ROUNDING = rule
        l3.RUNG = "L4" if rule == "sr" else "L3"
        l3.CELLS = {f"{l3.RUNG}-{k}": v for k, v in l3.CELL_KINDS.items()}
        l3.quantize_axis = patched_quantize_axis(rd, rw, orig)
        p, h = l3.train_l3(SEED, EPOCHS, 128, X, y, Xt, yt, CONFIG, f"{l3.RUNG}-DAW")
        print(f"  {label:56s} {h['test_acc'][0]:8.2f} {h['test_acc'][1]:8.2f} "
              f"{h['test_loss'][-1]:9.3f}", flush=True)
    l3.quantize_axis = orig
    l3.vanished = orig_vanished


if __name__ == "__main__":
    main()
