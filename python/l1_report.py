import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # draw to a file, no screen needed
import matplotlib.pyplot as plt

import quant as q
from train import load_weights

SEEDS = [42, 4242, 424242]
BLOCK = 16
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")

# name, scale table, M (mantissa bits, so e = code >> M), quantize function, theoretical max relative scale error
# Worst case: x exactly between two neighbor scales. UE5M3 neighbors are 1/8 apart at 1.0, so the error is (1/16) / (17/16) = 1/17.
# UE6M2 neighbors are 1/4 apart at 1.0, so the error is (1/8) / (9/8) = 1/9.
FORMATS = [
    ("UE6M2", q.e6m2, 2, q.quantize_fp4_b_e6m2, 1 / 9),
    ("UE5M3", q.e5m3, 3, q.quantize_fp4_b_e5m3, 1 / 17),
]


def block_amax(W):
    # Same reshape as the quantize functions. One amax per block of 16 along K. Result shape (N, B).
    Wt = W.T
    Wb = Wt.reshape(Wt.shape[0], Wt.shape[1] // BLOCK, BLOCK)
    return np.max(np.abs(Wb), axis=2)


def collect(fn):
    # Pool the blocks of W1, W2, W3 over all seeds into flat lists.
    all_codes, all_x = [], []
    counters = {}
    for seed in SEEDS:
        P = load_weights(seed)
        for key in ("W1", "W2", "W3"):
            W = P[key]
            _, codes, _, cnt = fn(W, rule="L1b")      # L1b: the encoder input is amax / 6
            all_codes.append(codes.ravel())
            all_x.append(block_amax(W).ravel() / 6.0)
            for k, v in cnt.items():
                counters[k] = counters.get(k, 0) + v
    return np.concatenate(all_codes), np.concatenate(all_x), counters


def stats(name, table, M, codes, x, counters, bound):
    scales = table[codes]                              # the scale the format stores for each block
    rel_err = np.abs(scales - x) / x                   # distance from the ideal scale amax/6, relative
    counts = np.bincount(codes, minlength=256)         # blocks per code, index = code
    e = codes >> M
    print(f"--- {name} (rule L1b, pooled W1+W2+W3, {len(SEEDS)} seeds) ---")
    print(f"blocks                 : {len(codes)}")
    print(f"code min / max         : {codes.min()} / {codes.max()}")
    print(f"distinct codes         : {(counts > 0).sum()} of 255")
    print(f"e min / max            : {e.min()} / {e.max()}")
    print(f"amax/6 min / max       : {x.min():.3e} / {x.max():.3e}")
    print(f"max rel scale error    : {100 * rel_err.max():.2f} %  (bound {100 * bound:.2f} %)")
    print(f"headroom top          : {table[254] / x.max():.3e}  (table[254] / max amax/6)")
    print(f"headroom bottom       : {x.min() / table[0]:.3e}  (min amax/6 / table[0])")
    print(f"counters               : {counters}")
    assert rel_err.max() <= bound + 1e-12, f"{name}: scale error above the theory bound"
    assert all(v == 0 for v in counters.values()), f"{name}: a counter is not 0"
    return counts



def plot(results):
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 3.2 * len(results)), sharex=True)
    for ax, (name, counts) in zip(axes, results):
        ax.bar(np.arange(256), counts, width=1.0)
        ax.axvline(254, color="red", linestyle="--", linewidth=1, label="code 254 (top scale)")
        ax.set_title(f"{name}: scale codes of the trained weights (L1b, 3 seeds, 2430 blocks)")
        ax.set_ylabel("blocks")
        ax.legend()
    axes[-1].set_xlabel("scale code (0 to 255)")
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "l1_scale_hist.png")
    fig.savefig(path, dpi=150)
    print("saved", os.path.abspath(path))


if __name__ == "__main__":
    results = []
    for name, table, M, fn, bound in FORMATS:
        codes, x, counters = collect(fn)
        counts = stats(name, table, M, codes, x, counters, bound)
        results.append((name, counts))
    plot(results)
