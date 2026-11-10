import numpy as np
import quant as q
from train import load_weights, load_dataset

# Answers the width questions of docs/datapath.md section 4.5 with exact integer
# arithmetic on real trained weights and real activations.
#
# Three different questions are asked, and they have three different answers.
#
#   A. How wide is one term, as a number?
#      A term is term_b = T_b * P_b. T_b is an exact integer, the block sum of 16
#      payload products divided by 4. P_b is the exact product of two block scales.
#      Every block scale is a dyadic rational, so P_b = (nA * nB) * 2^E and the whole
#      term is one integer times one power of two. There is no rounding anywhere in
#      this file. The answer does not depend on nb.
#
#   B. How wide must the SUM of nb terms be if every term were full size?
#      A + ceil(log2(nb)). That is the growth of a sum, and it is what the rule
#      `width = 20 + ceil(log2(nb))` encodes.
#
#   C. How wide must ONE fixed-point register be to drop no bit of any term?
#      The register has one binary point, so its top must hold the largest term and
#      its bottom must sit at the finest low bit any term uses. The gap between those
#      two positions is the alignment span. It is driven by the SCALES, not by the
#      mantissas: a block with tiny scales gives a term whose low bit is far below
#      the largest term.
#
# B and C are not the same requirement, and C is the one that says whether 32 bits
# is exact. C grows with the data, and it is not bounded, because the scale window
# permits a ratio of 2^(2 * 9.58 decades) between the largest and the smallest P.
#
# Run from python/ with:  uv run python width_probe.py

NS = 20_000


def payload_ints(codes_signed, idx):
    # The signed payload integer of every element: sign times the table entry.
    return np.sign(codes_signed).reshape(idx.shape).astype(np.int64) * q.PAYLOAD_INTS[idx]


def dyadic(scales, M):
    # Every block scale is (1 + m/2^M) * 2^(e-B). That is an integer n = 2^M + m in
    # [2^M, 2^(M+1)-1] times a power of two. frexp gives s = mant * 2^exp with mant in
    # [0.5, 1), so n = mant * 2^(M+1) is exact and the power is exp - M - 1.
    mant, exp = np.frexp(scales)
    n = (mant * 2.0 ** (M + 1)).round().astype(np.int64)
    k = exp - (M + 1)
    assert np.array_equal(n * 2.0 ** k, scales), "a scale was not a dyadic rational"
    return n, k


def widths(kw, ka, sw, sa, M):
    # kw (m, nb, 16), ka (n, nb, 16), sw (m, nb, 1), sa (n, nb, 1).
    nb = kw.shape[1]
    S = np.einsum("nbk,mbk->nmb", ka, kw).astype(np.int64)   # exact
    T = S // 4                                             # exact, the payload unit
    nw, ew = dyadic(sw[:, :, 0], M)                        # (m, nb)
    na, ea = dyadic(sa[:, :, 0], M)                        # (n, nb)

    # term = T * nw * na * 2^(ew + ea). One integer and one power, still exact.
    mant = T * nw[None, :, :] * na[:, None, :]             # (n, m, nb)
    expo = ew[None, :, :] + ea[:, None, :]                 # (n, m, nb)

    nz = mant != 0                                         # a zero block has no width
    bits = np.zeros_like(mant)
    bits[nz] = np.floor(np.log2(np.abs(mant[nz]))).astype(np.int64) + 1
    msb = bits + expo                                      # term < 2^msb

    A_mant = int(bits.max()) + 1                           # one term, sign included
    B_sum = A_mant + int(np.ceil(np.log2(nb)))
    C_span = int(msb[nz].max()) - int(expo[nz].min()) + 1  # one binary point
    return {
        "A_mant": A_mant,
        "B_sum": B_sum,
        "C_span": C_span,
        "top": int(msb[nz].max()),
        "low": int(expo[nz].min()),
        "octaves": int(msb[nz].max()) - int(expo[nz].min()),
        "largest": float(np.ldexp(np.abs(mant).astype(np.float64), expo).max()),
        "smallest": float(np.ldexp(np.abs(mant).astype(np.float64), expo)[nz].min()),
    }


def main():
    P = load_weights(42)
    for dataset in ("mnist", "mnist1d"):
        Xtr, ytr, Xte, yte = load_dataset(dataset)
        # The test set only, chunked by slicing, to keep the memory down.
        X = Xte[:NS]
        print("### %s, %d samples of the test set" % (dataset, X.shape[0]))
        for fmt, M, name in ((q.e6m2, 2, "UE6M2"), (q.e5m3, 3, "UE5M3")):
            Wq1, _, idxW1, sW1, _ = q.quantize_blocks_weights(P["W1"], fmt, "L1c")
            Aq0, _, idxA0, sA0, _ = q.quantize_blocks_act(X, fmt, "L1c")
            r = widths(payload_ints(Wq1.T, idxW1), payload_ints(Aq0, idxA0), sW1, sA0, M)
            print("  %s  layer 1, nb = 49" % name)
            print("    largest term          %+.6e" % r["largest"])
            print("    smallest nonzero term %+.6e" % r["smallest"])
            print("    A  one term            %2d bits" % r["A_mant"])
            print("    B  sum of full terms   %2d bits  (A + 6)" % r["B_sum"])
            print("    C  lossless fixed point %2d bits  (top 2^%d, finest low bit 2^%d, %d octaves)"
                  % (r["C_span"], r["top"], r["low"], r["octaves"]))
        print()

    # The bound for any legal input. Enumerate the 255 scale codes, keep the largest
    # and the smallest, and take 576 for the largest block sum, which is 12 payload
    # units in all 16 positions for both operands. Two cases, and they are different:
    #   worst VALUE  - nb maximal terms, all at the largest scale
    #   worst SPREAD - the same, plus one block at the floor of the scale window
    # The width is driven by the SPREAD, because one binary point must hold both ends.
    print("### the width a fixed point needs for ANY legal input, nb = 49")
    NB = 49
    for name, table, M in (("UE5M3", q.e5m3, 3), ("UE6M2", q.e6m2, 2)):
        k, n = [], []
        for c in range(255):
            mant, e = np.frexp(table[c])
            n.append(int(round(mant * 2.0 ** (M + 1))))
            k.append(e - (M + 1))
        kmin, kmax, nmax = min(k), max(k), max(n)
        top = int(np.ceil(np.log2(NB * 576 * nmax * nmax * 2.0 ** (2 * kmax))))
        w_value = top - 2 * kmax + 1
        w_spread = top - 2 * kmin + 1
        print("  %s  scale codes %d, k in [%d, %d], n in [%d, %d]"
              % (name, 255, kmin, kmax, min(n), nmax))
        print("    worst VALUE  top 2^%d, low bit 2^%d -> %3d bits" % (top, 2 * kmax, w_value))
        print("    worst SPREAD top 2^%d, low bit 2^%d -> %3d bits, 32 is short by %d"
              % (top, 2 * kmin, w_spread, w_spread - 32))


main()


# ---------------------------------------------------------------------------
# Can a legal weight set and input need more than 32 bits? Yes. This section
# builds one and measures what a 32-bit register does to it.
#
# The condition is one line. A term has its own low bit at 2^E and an integer
# mantissa. The register has one low bit at 2^F, and F is forced by the largest
# term, because the largest term must not overflow. So a term loses bits if and
# only if E < F. Over a whole reduction that is:
#
#     bits dropped from the smallest term = max(0, span - width)
#
# with span = top - low + 1. Nothing else enters.
# ---------------------------------------------------------------------------

FLOOR = 2.0 ** -15          # the smallest block scale UE5M3 can express, code 0


def exceed(name, W, X):
    Wq, _, idxW, sW, _ = q.quantize_blocks_weights(W, q.e5m3, "L1c")
    Aq, _, idxA, sA, _ = q.quantize_blocks_act(X, q.e5m3, "L1c")
    S = np.einsum("nbk,mbk->nmb", payload_ints(Aq, idxA),
                  payload_ints(Wq.T, idxW)).astype(np.int64)
    nw, ew = dyadic(sW[:, :, 0], 3)
    na, ea = dyadic(sA[:, :, 0], 3)
    mant = (S // 4) * nw[None, :, :] * na[:, None, :]
    expo = ew[None, :, :] + ea[:, None, :]
    nz = mant != 0
    bits = np.zeros_like(mant)
    bits[nz] = np.floor(np.log2(np.abs(mant[nz]))).astype(np.int64) + 1
    top, low = int((bits + expo)[nz].max()), int(expo[nz].min())
    span = top - low + 1

    term = np.ldexp(mant.astype(np.float64), expo)
    exact = term.sum(axis=2)
    print("  %-46s span %2d  %s 32" % (name, span, "OVER" if span > 32 else "under"))
    for Wb in (32, 40):
        F = top - (Wb - 1)
        grid = 2.0 ** F
        acc = (np.round(term / grid) * grid).sum(axis=2)
        err = acc - exact
        rmse = np.sqrt((err ** 2).mean())
        den = np.sqrt((exact ** 2).mean())
        frac = 100.0 * ((expo < F) & nz).sum() / nz.sum()
        snr = "infinite" if rmse == 0 else "%.1f dB" % (20 * np.log10(den / rmse))
        print("  %2d bits, low bit at 2^%3d: %5.1f%% of terms below the grid, SQNR %s"
              % (Wb, F, frac, snr))

def exceed_main():
    from train import load_weights
    P = load_weights(42)
    rng = np.random.default_rng(7)
    X = rng.standard_normal((512, 784))
    print("### can legal tensors need more than 32 bits?")
    exceed("trained weights, Gaussian input", P["W1"], X)

    # One block of both operands pushed to the floor of the window. Legal: the scale
    # takes code 0, the payloads stay small, nothing saturates.
    W = P["W1"].copy()
    W[16:32, :] *= 2.0 ** -8
    Xb = X.copy()
    Xb[:, 16:32] = rng.uniform(-1e-5, 1e-5, size=(512, 16))
    exceed("one block at the floor of the window", W, Xb)

    # The same, plus one block pushed to the top, so the span grows from both ends.
    W2 = W.copy()
    Xc = Xb.copy()
    W2[400:416, :] *= 2.0 ** 6
    Xc[:, 400:416] *= 2.0 ** 6
    exceed("one block at the floor and one at the top", W2, Xc)


exceed_main()
