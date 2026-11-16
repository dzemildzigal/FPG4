"""Why does stochastic rounding not remove the vanished entries?

L4 predicted that stochastic rounding (SR) would sharply lower the share of delta
entries that round to payload code 0. It did not: the share fell by 1 to 2 points.

This probe measures the reason on real numbers, instead of arguing about it.

TERMS

  delta            dZ, the gradient of the loss with respect to a layer's pre-activation.
                   This is the tensor that L3 and L4 round before the weight-gradient GEMM.
  block scale      one number per block of 16 values. It is `amax / 6` here, so the largest
                   value in the block has r = 6, where r is the next term.
  r                the value divided by its block scale, in scale units. The payload codes are
                   {0, 0.5, 1, 1.5, 2, 3, 4, 6}, so r says how many code steps away from zero
                   a value sits.
  RTN              round to nearest. A value with r < 0.25 lands on code 0. Code 0 is the
                   midpoint of [0, 0.5], so every value below a quarter of a code step is lost.
  SR               stochastic rounding. It takes code 0.5 with probability t = r / 0.5, so it
                   lands on 0 with probability 1 - r / 0.5 for r below 0.5.
  mass             the sum of |dZ|. An entry that vanishes holds some of it. The share of the
                   mass that vanishes says how much of the number is lost, which is different
                   from the share of entries that vanish.

THE THREE ANSWERS THIS PROBE GIVES

  1. the median r. If it is far below 0.25, then most entries are many code steps from the
     first code, and no rounding rule can send them.
  2. the predicted SR share, computed from r alone. If it matches the measured share in the
     L4 runs, then the mechanism is explained and the prediction was simply wrong.
  3. the mass lost, and the error it puts into dW. If the mass is a few percent, then the
     vanishing is numerically small.

Run it from python/ with PYTHONPATH=.:  uv run python vanished_probe.py > ../out/vanished_probe.txt
"""
import numpy as np

import l3
import quant
from model import forward, onehot, relu_derivative

BLOCK = 16
BATCH = 128


def deltas(p, X, y):
    """The three deltas of one batch, exactly as model.backward and l3.backward_l3 build them."""
    P, cache = forward(p, X)
    Y = onehot(y)
    dZ3 = (P - Y) / X.shape[0]
    dZ2 = (dZ3 @ p["W3"].T) * relu_derivative(cache["Z2"])
    dZ1 = (dZ2 @ p["W2"].T) * relu_derivative(cache["Z1"])
    return {"dZ3": dZ3, "dZ2": dZ2, "dZ1": dZ1}, {"dZ3": cache["A2"], "dZ2": cache["A1"], "dZ1": cache["X"]}


def blocked(T):
    """Split along the batch into blocks of 16, padding the last one with zeros.

    The padding is safe: a zero does not change the block maximum, so it does not change the
    scale, and a zero payload adds nothing to the block sum.
    """
    nb = (T.shape[0] + BLOCK - 1) // BLOCK
    pad = nb * BLOCK - T.shape[0]
    Tp = np.pad(T, ((0, pad), (0, 0))) if pad else T
    return Tp.reshape(nb, BLOCK, T.shape[1]), pad


def main():
    p = l3.load_weights(seed=42)
    X, y, _, _ = l3.load_dataset("mnist")
    dz, acts = deltas(p, X[:BATCH], y[:BATCH])
    table, rule = l3.CONFIGS["a6-e6m2"]

    print("model: the trained float weights of seed 42, the master of the L0 row")
    print("batch: the first 128 samples of MNIST. Data: 8 block pairs per column")
    print("scale rule: amax/6. A block's largest value therefore has r = 6")
    print()
    print(f"  {'tensor':6s} {'median r':>9s} {'p90 r':>7s} {'RTN vanish':>11s} {'SR vanish':>10s} "
          f"{'mass lost':>10s}")
    print("  " + "-" * 60)
    for name in ("dZ3", "dZ2", "dZ1"):
        B, _ = blocked(dz[name])
        scale = quant.encode_scale(np.abs(B).max(axis=1, keepdims=True) / 6.0, table)[0]
        r = np.abs(B) / scale
        rtn = float((r < 0.25).mean() * 100)
        sr = float((1.0 - np.minimum(1.0, r / 0.5)).mean() * 100)
        mass = float(np.abs(B)[r < 0.25].sum() / np.abs(B).sum() * 100)
        print(f"  {name:6s} {np.median(r):9.4f} {np.percentile(r, 90):7.4f} {rtn:10.1f}% "
              f"{sr:9.1f}% {mass:9.2f}%")

    print()
    print("what the lost entries cost the weight gradient dW = Aq.T @ dZ on the same batch")
    print("  the relative L1 error compares the gradient built from the kept entries only")
    print("  against the gradient built from all of them")
    for name in ("dZ3", "dZ2", "dZ1"):
        T, A = dz[name], acts[name]
        B, pad = blocked(T)
        scale = quant.encode_scale(np.abs(B).max(axis=1, keepdims=True) / 6.0, table)[0]
        keep = (np.abs(B) / scale) >= 0.25
        Tkeep = (B * keep).reshape(B.shape[0] * BLOCK, B.shape[2])[: T.shape[0]]
        Aq = l3.quantize_batch(A[: len(T)], table, rule)[0]
        full = Aq.T @ T
        kept = Aq.T @ Tkeep
        print(f"  {name}: {np.abs(full - kept).sum() / np.abs(full).sum() * 100:5.2f}%")

    print()
    print("for comparison, the measured shares over 40 runs of each cell of L3 and L4 on MNIST,")
    print("last epoch, mean of every seed and configuration:")
    print("  L3-D round to nearest   dZ3 80.6%  dZ2 66.6%  dZ1 66.3%")
    print("  L4-D stochastic         dZ3 79.4%  dZ2 64.2%  dZ1 64.0%")


if __name__ == "__main__":
    main()
