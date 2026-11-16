"""How far a block scale is from one exponent per value. A format question, not an engine question.

Names. A block scale is one 8-bit number shared by a group of values. The payload is 4 bits per value.
Shrinking the group moves the format towards IEEE floating point, where every value has its own exponent.

Three things are measured, all on saved L0 weights, 10 seeds, 4 configurations, no training:

    scales       how many scale factors a matrix needs, and the bits per value that follows
    flush        the share of non-zero values that the payload rounds to 0. A value is flushed when it is
                 more than 12 times smaller than the largest value of its group. 12 is 6 / 0.5: the top
                 payload code over the smallest non-zero payload code.
    sqnr         the signal to quantization noise ratio of the weights

Block side 1 is one scale per value. That is the IEEE form: sign, exponent, payload, per value. The probe
measures what that form would give, so the distance to it can be read in dB and in bits per value.

Run from python/:  uv run python granularity_probe.py
"""
import math

import numpy as np

import quant as q
from tile_probe import CONFIGS, SEEDS, quantize_tiles
from train import load_weights

SIDES = (16, 8, 4, 2, 1)
BLOCK = 16          # one block holds 16 values along the reduction axis
PAYLOAD = 4         # bits per value
SCALE_BITS = 8      # bits per scale factor


def scales_1d(K, N):
    """One scale per 16 values along K, for each of the N outputs."""
    return N * math.ceil(K / BLOCK)


def scales_tile(K, N, a):
    """One scale per a by a tile."""
    return math.ceil(K / a) * math.ceil(N / a)


def bits_per_value(K, N, scales):
    return (PAYLOAD * K * N + SCALE_BITS * scales) / (K * N)


def geometry():
    """The two counts, and the bits that follow from them, for the shapes this project owns."""
    shapes = (("W1", 784, 16), ("W2", 16, 16), ("W3", 16, 10),
              ("a wider layer, 256 by 256", 256, 256), ("a layer of a 1B transformer, 1024 by 1024", 1024, 1024))
    for name, K, N in shapes:
        n1 = scales_1d(K, N)
        print(f"\n{name}, {K} by {N} = {K * N:,} values")
        print(f"{'layout':>8} {'scales':>9} {'against 1D':>11} {'bits per value':>15} {'against 1D':>11} {'values per scale':>17}")
        print(f"{'1D':>8} {n1:>9,} {'':>11} {bits_per_value(K, N, n1):15.3f} {'':>11} {K * N / n1:17.1f}")
        for a in SIDES:
            n = scales_tile(K, N, a)
            print(f"{a:>5}x{a:<2} {n:>9,} {n / n1:11.2f} {bits_per_value(K, N, n):15.3f} "
                  f"{bits_per_value(K, N, n) / bits_per_value(K, N, n1):11.2f} {K * N / n:17.1f}")


def flush_and_sqnr(dataset):
    """Per matrix, per block side: how many values the payload round to 0, and the SQNR."""
    rows = {m: {a: [] for a in ("1D",) + SIDES} for m in ("W1", "W2", "W3")}
    for seed in SEEDS:
        p = load_weights(seed, dataset=dataset)
        for m in ("W1", "W2", "W3"):
            W = np.asarray(p[m], dtype=np.float64)
            nz = W != 0
            for table, rule in CONFIGS.values():
                Wq1, _, _, _, _ = q.quantize_blocks_weights(W, table, rule)
                rows[m]["1D"].append((np.mean(Wq1[nz] == 0), q.sqnr(W, Wq1)))
                for a in SIDES:
                    Wq = quantize_tiles(W, (a, a), table, rule)[0]
                    rows[m][a].append((np.mean(Wq[nz] == 0), q.sqnr(W, Wq)))
    print(f"\n===== {dataset}, saved L0 weights, {len(SEEDS)} seeds x 4 configurations =====")
    print("flush = share of non-zero values whose payload code is 0. SQNR = signal to noise, dB.")
    for m in ("W1", "W2", "W3"):
        print(f"\n  {m}")
        print(f"  {'layout':>8} {'flush':>8} {'SQNR dB':>9}   (mean of {len(rows[m]['1D'])} runs)")
        for a in ("1D",) + SIDES:
            f = np.mean([x[0] for x in rows[m][a]])
            s = np.mean([x[1] for x in rows[m][a]])
            print(f"  {str(a):>8} {100 * f:7.2f}% {s:9.2f}")
    return rows


def marginal(rows):
    """What each step towards one scale per value buys, in dB, and what it costs, in bits per value.

    The payload error is zero when a value sits exactly on a payload code, and the block scale puts the
    largest value of the block on the top code. So the noise of a small group comes from the values that sit
    between codes, and from the values that fall below the smallest code. Both shrink with the group.
    """
    K, N = 16, 16
    print(f"\n{'layout':>8} {'bits per value':>15} {'SQNR dB':>9} {'dB bought by this step':>23} {'bits paid':>10} {'dB per bit':>11}")
    order = (16, 8, 4, "1D", 2, 1)
    prev = None
    for a in order:
        f = np.mean([x[0] for x in rows["W2"][a]])
        sq = np.mean([x[1] for x in rows["W2"][a]])
        bits = 4.5 if a == "1D" else bits_per_value(K, N, scales_tile(K, N, int(a)))
        if prev is None:
            print(f"{a:>8} {bits:15.2f} {sq:9.2f} {'':>23} {'':>10} {'':>11}")
        else:
            db, dbit = sq - prev[1], bits - prev[0]
            print(f"{a:>8} {bits:15.2f} {sq:9.2f} {db:23.2f} {dbit:10.2f} "
                  f"{(db / dbit if dbit else float('nan')):11.2f}")
        prev = (bits, sq)


if __name__ == "__main__":
    geometry()
    for ds in ("mnist", "mnist1d"):
        marginal(flush_and_sqnr(ds))
