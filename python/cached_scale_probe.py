"""Two of Dzemo's questions about the tile, answered on saved L0 weights. No training.

Names. W is (K, N) in this code: K inputs, N outputs. The forward GEMM sums over K, so a forward block is
16 values along K for one output n (Dzemo calls it a row). The backward GEMM sums over N, so a backward
block is 16 values along N for one input k (Dzemo calls it a column).

QUESTION 1, the cached scale.
    Quantize W in 1D forward blocks and keep their scales. For the backward pass, reuse those scales as the
    scales of the backward blocks: "one row's scalers become the column's scalers". The backward encoder then
    skips amax and snap. It runs only the payload quantizer, with a scale that was given to it.

    The pairing used here: backward block (k, nb) takes the scale of forward block (n = k, kb = nb).
    It exists only when K = N, because only then the forward scale grid (N, K/16) and the backward scale
    grid (K, N/16) have the same shape. W2 is 16 by 16, so it has a pairing. W3 is 16 by 10. It has 10
    forward scales and 16 backward blocks, so 6 backward blocks have no scale to take.

    The weights come from the float master, so each value is rounded once. That is the BEST case for the idea.

QUESTION 2, the return.
    A model trained with a tile store goes back to 1D blocks for inference. The tile store holds weights on
    the tile grid. The 1D encoder takes a new amax over 16 of those values, makes a new scale, and rounds
    again. This probe measures how much that second rounding changes.

    The stand-in for a tile-trained model is the float L0 weight put through the tile once. It is on the tile
    grid, like a tile-trained weight, but it never adapted to the tile. So this gives the noise of the second
    rounding. It does not give the accuracy of a model that trained in the tile. That needs training.

Run from python/:  uv run python cached_scale_probe.py
"""
import numpy as np

import quant as q
from tile_probe import CONFIGS, SEEDS, quantize_tiles
from train import load_weights

TILES = {"16x16": (16, 16), "8x8": (8, 8), "4x4": (4, 4)}


def backward_own(W, table, rule):
    """The backward copy of the rejected arm A: blocks of 16 along N for each input k, each with its own amax.

    quantize_blocks_weights cuts blocks along the FIRST axis of the matrix it gets. W.T has N first, so the
    blocks run along N. The result is transposed back to (K, N).
    """
    Wq, _, idx, scales, _ = q.quantize_blocks_weights(W.T, table, rule)
    return Wq.T, idx, scales


def backward_cached(W, scales_fwd):
    """Blocks of 16 along N for each input k, quantized with the scale of the forward block of the same index.

    scales_fwd is (N, K/16, 1). With K = N that is (K, N/16, 1): one scale per backward block, in the same
    order. No scale is computed here. This is the whole encoder of the idea: the payload pass, nothing else.
    """
    K, N = W.shape
    blocks = W.reshape(K, N // 16, 16)               # row k, block nb, 16 values along N
    idx = q.payload_index(blocks, scales_fwd)        # nearest payload code. A value above 6 x scale gets code 7
    return q.decode_payload(blocks, idx, scales_fwd).reshape(K, N), idx, blocks


def one_matrix(W, table, rule):
    """Every number of question 1 for one square matrix, one configuration."""
    Wf, _, _, scales_f, _ = q.quantize_blocks_weights(W, table, rule)      # forward copy, blocks along K
    Wown, idx_own, scales_own = backward_own(W, table, rule)               # backward copy of arm A
    Wcac, idx_cac, blocks = backward_cached(W, scales_f)                   # backward copy of the idea
    ratio = np.max(np.abs(blocks), axis=2, keepdims=True) / (6.0 * scales_f)   # 1.00 means the scale fits exactly
    nz = blocks != 0
    # "Scale too big" blocks: the cached scale is at least 25% above what the block needs (ratio below 0.8).
    # Those are the blocks where small values can round to 0. Counted as (flushed, total) so runs can be pooled.
    big = np.broadcast_to(ratio < 0.8, blocks.shape) & nz
    out = {
        "sqnr_own": q.sqnr(W, Wown), "sqnr_cached": q.sqnr(W, Wcac),
        "differs_own": float(np.mean(Wf != Wown)), "differs_cached": float(np.mean(Wf != Wcac)),
        "blocks_clipped": float(np.mean(ratio > 1.0)),
        # A value above 6.6 x scale is clipped by more than 10%. Snapping a scale to its nearest code moves it
        # by at most 4.4%, so the own-amax copy has none of these. Any such value is a real overflow.
        "clipped_cached": float(np.mean(np.abs(blocks) / scales_f > 6.6)),
        "clipped_own": float(np.mean(np.abs(blocks) / scales_own > 6.6)),
        "big_flushed_cached": int(np.sum(idx_cac[big] == 0)), "big_flushed_own": int(np.sum(idx_own[big] == 0)),
        "big_total": int(np.sum(big)),
        "zero_cached": float(np.mean(idx_cac[nz] == 0)), "zero_own": float(np.mean(idx_own[nz] == 0)),
    }
    for name, tile in TILES.items():
        out["sqnr_" + name] = q.sqnr(W, quantize_tiles(W, tile, table, rule)[0])
    return out, ratio.ravel()


def summary(rows, key):
    v = np.array([r[key] for r in rows])
    return v.mean(), v.min(), v.max()


def print_question_1(title, matrices):
    """matrices: list of (W, table, rule), one per run. Prints the SQNR table and the event table."""
    rows, ratios = [], []
    for W, table, rule in matrices:
        r, ratio = one_matrix(W, table, rule)
        rows.append(r)
        ratios.append(ratio)
    ratios = np.concatenate(ratios)
    print(f"\n===== {title}: {len(rows)} runs =====")
    print(f"{'backward copy':>36} {'SQNR dB, mean [min, max]':>28} {'values that differ from the forward copy':>42}")
    for label, key, diff in (("1D, own amax (rejected arm A)", "sqnr_own", "differs_own"),
                             ("1D, cached forward scale", "sqnr_cached", "differs_cached"),
                             ("16x16 tile", "sqnr_16x16", None), ("8x8 tile", "sqnr_8x8", None),
                             ("4x4 tile", "sqnr_4x4", None)):
        m, lo, hi = summary(rows, key)
        d = f"{100 * summary(rows, diff)[0]:.1f}%" if diff else "0.0% (one copy)"
        print(f"{label:>36} {m:9.2f} [{lo:6.2f}, {hi:6.2f}] {d:>42}")
    print(f"  price of the cached scale against own amax: {summary(rows, 'sqnr_own')[0] - summary(rows, 'sqnr_cached')[0]:.2f} dB")
    p5, p50, p95 = np.percentile(ratios, [5, 50, 95])
    print(f"  amax of the backward block / (6 x cached scale), 5th, 50th, 95th percentile: {p5:.2f}, {p50:.2f}, {p95:.2f}"
          "   (above 1.00 the largest value is clipped)")
    print(f"  backward blocks where the largest value is clipped: {100 * summary(rows, 'blocks_clipped')[0]:.1f}%")
    print(f"  {'share of values':>36} {'cached scale':>14} {'own amax':>10}")
    print(f"  {'values clipped by more than 10%':>36} {100 * summary(rows, 'clipped_cached')[0]:13.2f}% "
          f"{100 * summary(rows, 'clipped_own')[0]:9.2f}%")
    print(f"  {'non-zero values that round to 0':>36} {100 * summary(rows, 'zero_cached')[0]:13.2f}% "
          f"{100 * summary(rows, 'zero_own')[0]:9.2f}%")
    tot = sum(r["big_total"] for r in rows)
    if tot:
        print(f"  {'... in blocks whose scale is >25% too big':>36} "
              f"{100 * sum(r['big_flushed_cached'] for r in rows) / tot:13.2f}% "
              f"{100 * sum(r['big_flushed_own'] for r in rows) / tot:9.2f}%")


def question_1():
    for dataset in ("mnist", "mnist1d"):
        mats = [(np.asarray(load_weights(seed, dataset=dataset)["W2"], dtype=np.float64), table, rule)
                for seed in SEEDS for table, rule in CONFIGS.values()]
        print_question_1(f"{dataset}, W2 (16 by 16), saved L0 weights", mats)
    # Best case with no structure in the weights: every value is an independent draw from one normal
    # distribution, so every row and every column has the same typical amax. Any clipping seen here comes
    # from the random spread of amax alone.
    for n in (16, 64, 256):
        mats = [(np.random.default_rng(seed).standard_normal((n, n)), table, rule)
                for seed in SEEDS for table, rule in CONFIGS.values()]
        print_question_1(f"normal random {n} by {n}, no structure", mats)
    print("\nW3 is 16 by 10. Forward scales: 10. Backward blocks: 16. The pairing has no scale for 6 of 16.")


def question_2():
    """The return from a tile to 1D blocks along K, for W2 and W3 together."""
    for dataset in ("mnist", "mnist1d"):
        rows = {t: [] for t in TILES}
        for seed in SEEDS:
            p = load_weights(seed, dataset=dataset)
            for table, rule in CONFIGS.values():
                Ws = [np.asarray(p[m], dtype=np.float64) for m in ("W2", "W3")]
                flat = lambda ms: np.concatenate([m.ravel() for m in ms])
                direct = [q.quantize_blocks_weights(W, table, rule)[0] for W in Ws]
                for name, tile in TILES.items():
                    tiled = [quantize_tiles(W, tile, table, rule)[0] for W in Ws]
                    back = [q.quantize_blocks_weights(T, table, rule)[0] for T in tiled]
                    rows[name].append({
                        "changed": float(np.mean(flat(tiled) != flat(back))),
                        "sig": float(np.sum(flat(tiled) ** 2)),
                        "err": float(np.sum((flat(tiled) - flat(back)) ** 2)),
                        "via_tile": q.sqnr(flat(Ws), flat(back)),
                        "direct": q.sqnr(flat(Ws), flat(direct)),
                        "tile_only": q.sqnr(flat(Ws), flat(tiled)),
                    })
        print(f"\n===== the return, {dataset}, W2 and W3 together, {len(rows['16x16'])} runs =====")
        print(f"{'tile':>6} {'tile only':>10} {'1D direct':>10} {'tile, then 1D':>14} {'noise of the return':>20} "
              f"{'values that change':>19}")
        print(f"{'':>6} {'SQNR dB':>10} {'SQNR dB':>10} {'SQNR dB':>14} {'SQNR(tile, back) dB':>20} {'on the return':>19}")
        print("       tile only = float vs tile.  1D direct = float vs 1D.  tile, then 1D = float vs the returned weights.")
        print("       noise of the return = tile weights vs the returned weights, energy pooled over the runs.")
        for name in TILES:
            r = rows[name]
            print(f"{name:>6} {summary(r, 'tile_only')[0]:10.2f} {summary(r, 'direct')[0]:10.2f} "
                  f"{summary(r, 'via_tile')[0]:14.2f} "
                  f"{10 * np.log10(sum(x['sig'] for x in r) / sum(x['err'] for x in r)):20.2f} "
                  f"{100 * summary(r, 'changed')[0]:18.1f}%")


def accuracy_proxy():
    """Test accuracy of the float L0 model with W1 in 1D and W2, W3 in the variants below. Whole test set."""
    from model import evaluate
    from train import load_dataset

    def fwd(W, table, rule):
        return q.quantize_blocks_weights(W, table, rule)[0]

    def tile(side):
        return lambda W, table, rule: quantize_tiles(W, TILES[side], table, rule)[0]

    def via(side):
        return lambda W, table, rule: fwd(quantize_tiles(W, TILES[side], table, rule)[0], table, rule)

    variants = {"1D": fwd, "16x16": tile("16x16"), "16x16 then 1D": via("16x16"),
                "4x4": tile("4x4"), "4x4 then 1D": via("4x4")}
    for dataset, seeds in (("mnist", SEEDS), ("mnist1d", SEEDS[:3])):
        X_train, y_train, X_test, y_test = load_dataset(dataset)
        X_test = X_test.astype(np.float32)
        print(f"\n===== accuracy of the float L0 model, W1 in 1D, W2 and W3 as named, {dataset}, "
              f"{len(seeds)} seeds x 4 configs, whole test set =====")
        for name, make in variants.items():
            accs = []
            for seed in seeds:
                p = load_weights(seed, dataset=dataset)
                for table, rule in CONFIGS.values():
                    pq = dict(p)
                    pq["W1"] = fwd(np.asarray(p["W1"], dtype=np.float64), table, rule).astype(p["W1"].dtype)
                    for m in ("W2", "W3"):
                        pq[m] = make(np.asarray(p[m], dtype=np.float64), table, rule).astype(p[m].dtype)
                    accs.append(100.0 * evaluate(pq, X_test, y_test)[1])
            print(f"{name:>16}: {np.mean(accs):6.2f}  [{np.min(accs):6.2f}, {np.max(accs):6.2f}]")


if __name__ == "__main__":
    question_1()
    question_2()
    accuracy_proxy()
