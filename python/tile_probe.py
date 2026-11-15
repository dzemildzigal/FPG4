"""The SQNR price of a tile, against the 1D block, on saved L0 weights. No training.

Why this exists: the block experiment (docs/experiment_block.md) shows that a 16 by 16 tile costs no
accuracy after ten epochs of training. This probe measures what it costs before training, so the two
numbers can be put next to each other: the tile loses SQNR, and training absorbs the loss.

What is measured, per dataset, per configuration, per seed, per matrix:

    sqnr 1D      one scale per 16 inputs, the form rung L2 uses
    sqnr tile    one scale per a-by-b tile, a rows along K and b columns along N
    price        sqnr 1D minus sqnr tile, in dB. Positive means the 1D block is better.
    four6 shares how often rule four6 keeps amax/4 instead of amax/6 in the tiles of that shape

Run from python/:  uv run python tile_probe.py
"""
import os

import numpy as np

import quant as q
from train import load_weights

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 4242, 424242, 123, 456, 789, 1010, 2020, 3030, 4040]
CONFIGS = {"a6-e6m2": (q.e6m2, "L1b"), "f6-e6m2": (q.e6m2, "L1c"),
           "a6-e5m3": (q.e5m3, "L1b"), "f6-e5m3": (q.e5m3, "L1c")}
TILES = {"4x4": (4, 4), "16x16": (16, 16)}
MATRICES = ("W1", "W2", "W3")


def quantize_tiles(W, tile, table, rule):
    """One scale per a-by-b tile. Returns (Wq, share of tiles where four6 kept amax/4)."""
    a, b = tile
    K, N = W.shape
    Kp, Np = -(-K // a) * a, -(-N // b) * b
    Wp = np.zeros((Kp, Np))
    Wp[:K, :N] = W
    Tl = Wp.reshape(Kp // a, a, Np // b, b).transpose(0, 2, 1, 3).reshape(Kp // a, Np // b, a * b)
    Th, codes = q.quantize_blocks(Tl, table, rule)[:2]
    amax = np.max(np.abs(Tl), axis=2)   # (tile rows, tile columns)
    codes6 = q.encode_scale((amax / 6.0)[..., None], table)[1]
    keep = amax > 0
    share4 = float(np.mean(codes[keep] != codes6[keep])) if keep.any() else 0.0
    Wq = Th.reshape(Kp // a, Np // b, a, b).transpose(0, 2, 1, 3).reshape(Kp, Np)[:K, :N]
    return Wq, share4


def main():
    for dataset in ("mnist", "mnist1d"):
        print(f"\n===== {dataset}: {len(SEEDS)} seeds, saved L0 weights, nothing trained =====")
        for config, (table, rule) in CONFIGS.items():
            res = {m: {t: [] for t in TILES} for m in MATRICES}
            f4 = {m: {t: [] for t in TILES} for m in MATRICES}
            for seed in SEEDS:
                p = load_weights(seed, dataset=dataset)
                for m in MATRICES:
                    W = np.asarray(p[m], dtype=np.float64)
                    one, _, _, _, _ = q.quantize_blocks_weights(W, table, rule)
                    base = q.sqnr(W, one)
                    for name, tile in TILES.items():
                        Wq, share4 = quantize_tiles(W, tile, table, rule)
                        res[m][name].append(base - q.sqnr(W, Wq))
                        f4[m][name].append(share4)
            print(f"\n  config {config}: SQNR price of the tile against the 1D block, dB, mean of {len(SEEDS)} seeds")
            print("  matrix " + " ".join(f"{t:>18}" for t in TILES) + "   (range over the seeds)")
            for m in MATRICES:
                cells = []
                for t in TILES:
                    v = np.array(res[m][t])
                    cells.append(f"{v.mean():6.2f} [{v.min():5.2f},{v.max():5.2f}]")
                print(f"  {m:>6} " + " ".join(f"{c:>18}" for c in cells))
            print("  share of tiles where four6 keeps amax/4, mean")
            for m in MATRICES:
                print(f"  {m:>6} " + " ".join(f"{100*np.mean(f4[m][t]):17.1f}%" for t in TILES))
            drops = [np.array(res[m][t]) > 0 for m in MATRICES for t in TILES]
            print(f"  cells where the tile is worse in every one of the {len(SEEDS)} seeds: "
                  f"{sum(int(np.all(d)) for d in drops)} of {len(drops)}")


if __name__ == "__main__":
    main()
