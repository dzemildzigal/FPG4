"""
How many significand bits does the encoder need?

Two separate questions, and the answer to one is not the answer to the other.

  amax    the scale comes from the largest magnitude in the block. If the amax tree
          compares truncated values, it returns the truncated maximum, because
          truncation is monotone:  max(trunc(x)) == trunc(max(x)).  So the tree does
          not need a wider comparator, it needs a narrower one. The only question is
          how many bits before the chosen scale code starts to change.

  payload the 4-bit code picks one of 8 magnitudes. 3 bits separate them, so the
          payload path needs a few more than 3. How many is the second question.

Nothing here is rounded: truncation toward zero of the significand is exact, and the
rest is the same quantizer the sweep uses.

Run from python/ with:  uv run python amax_probe.py [mnist|mnist1d]
"""
import os
import sys

import numpy as np

import quant as q
from model import evaluate
from train import load_weights

FULL = 53                          # float64 carries 53 significand bits, so 53 is "no truncation"
BITS = [FULL, 24, 16, 12, 8]       # 24 is FP32
RULES = [("amax/6", "L1b"), ("four6", "L1c")]


def load_test(name):
    # The test set only. The training set of MNIST-1D is 3.1 GB and this probe never needs it.
    mod = __import__(name)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", name)
    return mod.get_x_y_test(path)


def trunc_significand(v, bits):
    # Floor the magnitude to `bits` significand bits, toward zero. Exact:
    # frexp gives v = m * 2^e with |m| in [0.5, 1), so m * 2^bits is an integer part
    # that np.trunc removes, and ldexp puts the power of two back.
    m, e = np.frexp(v)
    m = np.trunc(m * (1 << bits)) / (1 << bits)
    return np.ldexp(m, e)


def quantize(B, table, rule, bits_amax, bits_value):
    # Same shape and the same five steps as quant.quantize_blocks, with two truncations.
    # B is (rows, blocks, 16). Returns (decoded, scale codes).
    Bv = B if bits_value == FULL else trunc_significand(B, bits_value)
    src = B if bits_amax == FULL else trunc_significand(B, bits_amax)
    amax = np.max(np.abs(src), axis=2, keepdims=True)
    scale6, code6 = q.encode_scale(amax / 6.0, table)
    scale4, code4 = q.encode_scale(amax / 4.0, table)
    B_hat, codes, _, _ = q.four6(Bv, scale6, scale4, code6, code4, rule)
    return B_hat, codes


def build(P, table, rule, bits_amax, bits_value):
    # Quantize the weights and return the activation hook, so one call does one L1d pass.
    Q = dict(P)
    wcodes = {}
    for key in ("W1", "W2", "W3"):
        K, N = P[key].shape
        Wb = P[key].T.reshape(N, K // 16, 16)
        Wq, codes = quantize(Wb, table, rule, bits_amax, bits_value)
        Q[key] = Wq.reshape(N, K).T
        wcodes[key] = codes

    def hook(A):
        n, K = A.shape
        Aq, _ = quantize(A.reshape(n, K // 16, 16), table, rule, bits_amax, bits_value)
        return Aq.reshape(n, K), {}

    return Q, hook, wcodes


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    X, y = load_test(name)
    table = q.e5m3
    P = load_weights(42, name)

    print(f"{name}, seed 42, UE5M3, {X.shape[0]} test samples, L1d scope (weights and activations)")
    print("bits  means the significand bits of the value. 53 is float64, 24 is float32.")
    print()

    for label, rule in RULES:
        ref_codes = None
        ref_acc = None
        for mode in ("amax only", "amax and payload"):
            print(f"rule {label:7s} | {mode}")
            print("  bits | accuracy | scale codes that changed against 53 bits")
            print("  -----|----------|-----------------------------------------")
            for bits in BITS:
                ba = bits if bits != FULL else FULL
                bv = bits if mode == "amax and payload" else FULL
                Q, hook, wcodes = build(P, table, rule, ba, bv)
                _, acc = evaluate(Q, X, y, chunk=20_000, quant=hook)
                flat = np.concatenate([wcodes[k].ravel() for k in ("W1", "W2", "W3")])
                if ref_codes is None:
                    ref_codes, ref_acc = flat, acc
                    diff = "-"
                else:
                    diff = f"{np.mean(flat != ref_codes) * 100:5.2f}%"
                print(f"  {bits:4d} | {100*acc:8.2f} | {diff:>39s}")
                del Q, hook, wcodes
            print()

    print("The L1d reference numbers for seed 42 are the rows at 53 bits.")
    print("A changed scale code is the event that costs accuracy. Counting it is the")
    print("cheap measurement, because the code is an integer and needs no accuracy run.")


if __name__ == "__main__":
    main()
