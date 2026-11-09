"""
L1e sweep: the datapath. Same quantized weights and activations as L1d, different arithmetic.

Run from python/:  uv run python l1e_sweep.py [mnist|mnist1d]
Needs the weights from:  uv run python train.py [mnist|mnist1d]

L1d decodes every operand to float64 and runs a float64 GEMM. L1e replaces that GEMM with 7 stated
arithmetic variants and measures what the datapath costs on top of the operand quantization.

The variants (see docs/experiment_L1.md, section L1e):

  E0  float64 GEMM on the decoded values. The reference. Must equal L1d.
  E1  integer block sum, scale product float64, accumulate float64.
  E2  integer block sum, scale product rounded to float32, accumulate float32.
  E3  integer block sum, fixed-point accumulator of 24, 20 or 16 bits.
  E4  float32 GEMM on the decoded values. The product is still exact (4-bit payload times 4-bit payload
      needs at most 8 bits), so E4 only rounds the accumulation.

Why the payload product is exact, and why E1 must equal E0:
  A payload is k/2 with k in {0,1,2,3,4,6,8,12}. The product of two payloads is (k1*k2)/4.
  The integer k1*k2 is at most 144, so the product is a multiple of 0.25 and it is exact.
  One block is 16 of those products, summed as integers. The largest block sum is 16*144 = 2304,
  which needs 12 bits. Then one multiply by scale_activation * scale_weight turns the block sum
  into a real number. That is the same number E0 computes, only with the sum done in integers first.
"""
import sys

import numpy as np

import quant as q
from train import load_weights, load_dataset

SEEDS = [42, 4242, 424242]
DATASET = sys.argv[1] if len(sys.argv) > 1 else "mnist"
MAX_SAMPLES = 20_000         # the first N test samples, so the run stays short
MAX_ROWS = 2_000             # rows per chunk, so the (rows, outputs, blocks) integer product stays small
BITS = (24, 20, 16)          # accumulator widths for E3

CONFIGS = [
    ("UE6M2 amax/6", q.e6m2, "L1b"),
    ("UE6M2 four6", q.e6m2, "L1c"),
    ("UE5M3 amax/6", q.e5m3, "L1b"),
    ("UE5M3 four6", q.e5m3, "L1c"),
]

VARIANTS = ["E0", "E1", "E2", "E3-24", "E3-20", "E3-16", "E4"]


def grid(P, table, rule):
    # Quantize the three weight matrices and keep the payload codes and the block scales.
    # Returns a list of 3 tuples (Wq, idx, scales, bias): Wq is (K, N), idx is (N, blocks, 16),
    # scales is (N, blocks, 1). The biases stay float64, like L1 and L1d.
    out = []
    for W, b in (("W1", "b1"), ("W2", "b2"), ("W3", "b3")):
        Wq, _, idx, scales, _ = q.quantize_blocks_weights(P[W], table, rule)
        out.append((Wq, idx, scales, P[b]))
    return out


def act_grid(A, table, rule):
    # Quantize one activation tensor: A is (n, K). Returns (Aq, idx, scales), same layout as a weight.
    Aq, _, idx, scales, _ = q.quantize_blocks_act(A, table, rule)
    return Aq, idx, scales


def payload_ints(Aq, idx):
    # The integer value that each element carries, in units of 0.5: sign * {0,1,2,3,4,6,8,12}.
    # The product of two of these is an integer, so the block sum can run in an integer accumulator.
    # Aq is (rows, K) and idx is (rows, blocks, 16), so the sign has to take the block layout first.
    sign = np.sign(Aq).reshape(idx.shape).astype(np.int32)
    return sign * q.PAYLOAD_INTS[idx]


def matmul_dp(Aq, idxA, sA, Wq, idxW, sW, variant):
    # One layer, one arithmetic variant.
    # Aq (n, K), idxA (n, B, 16), sA (n, B, 1)  |  Wq (K, M), idxW (M, B, 16), sW (M, B, 1)
    # B = K // 16. The output is (n, M) float64.
    if variant == "E0":
        return Aq @ Wq
    if variant == "E4":
        # Both operands decoded, product and sum in float32. The products are exact, so this only
        # measures the float32 accumulator.
        return (Aq.astype(np.float32) @ Wq.astype(np.float32)).astype(np.float64)

    bits = int(variant.split("-")[1]) if variant.startswith("E3") else None
    ka = payload_ints(Aq, idxA)                       # (n, B, 16) int32
    kw = payload_ints(Wq.T, idxW)                     # (M, B, 16) int32. Wq.T is (N, K) = the row layout of idxW
    out = np.empty((len(Aq), Wq.shape[1]), dtype=np.float64)

    for start in range(0, len(Aq), MAX_ROWS):
        stop = min(start + MAX_ROWS, len(Aq))
        # STEP 1: the integer product of every payload pair, summed over the 16 elements of one block.
        # k1*k2 is at most 144, so a block sum is at most 2304 and fits in 12 bits. Exact, no rounding.
        dots = np.einsum("nbk,mbk->nmb", ka[start:stop], kw, optimize=True)          # (rows, M, B) int64
        # STEP 2: the scale product. This is the first place where a real datapath must round.
        # The integers are in units of 0.5, so a product of two is in units of 0.25. The factor 1/4 is a
        # power of two, so it is exact and it can be folded into the scale product.
        sp = 0.25 * sA[start:stop, None, :, 0] * sW[None, :, :, 0]                    # (rows, M, B) float64

        if variant == "E1":
            # Scale product in float64, block contributions summed in float64. Must equal E0.
            out[start:stop] = np.einsum("nmb,nmb->nm", dots.astype(np.float64), sp)
        elif variant == "E2":
            # Scale product rounded to float32, block contributions summed in float32.
            out[start:stop] = np.einsum("nmb,nmb->nm", dots.astype(np.float32),
                                        sp.astype(np.float32)).astype(np.float64)
        else:
            # STEP 3: a fixed-point accumulator. The step of the grid is 2^-frac, chosen so the largest
            # output magnitude still fits in `bits` including the sign. Each block contribution is
            # snapped to that grid before the add, which is the alignment loss of a real accumulator.
            mag = np.max(np.abs(np.einsum("nmb,nmb->nm", dots.astype(np.float64), sp)))
            int_bits = int(np.ceil(np.log2(mag + 1.0))) + 1                          # sign + integer part
            frac = bits - int_bits
            if frac < 0:
                raise ValueError(f"{variant}: {bits} bits cannot hold a magnitude of {mag}")
            step = 2.0 ** (-frac)
            contrib_int = np.rint(dots.astype(np.float64) * sp / step)               # integer counts of `step`
            out[start:stop] = contrib_int.sum(axis=2) * step                         # exact integer add, one scale
    return out


def forward_dp(P, X, variant, table, rule):
    # The full chain with the chosen arithmetic. Every activation is quantized before it enters a matmul,
    # exactly like L1d. Returns (outputs of the 3 layers, list of the 3 quantized inputs (Aq, idx, scales)).
    inputs, outputs = [], []
    A = X
    for Wq, idxW, sW, bias in P:
        Aq, idxA, sA = act_grid(A, table, rule)
        inputs.append((Aq, idxA, sA))
        A = matmul_dp(Aq, idxA, sA, Wq, idxW, sW, variant) + bias                            # bias in float64
        if len(outputs) < 2:
            A = np.maximum(0.0, A)                                                    # ReLU, float64, same as model.relu
        outputs.append(A)
    return outputs, inputs


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def score(Z, y):
    P = softmax(Z)
    logp = np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None))    # clip: a broken variant can give log(0)
    return 100 * np.mean(P.argmax(axis=1) == y), float(-np.mean(logp))


def layer_error(ref, got):
    # Max absolute error, RMS error, and SQNR of one layer output against the E0 reference.
    # A variant that is bit-identical to E0 has zero noise, and its SQNR is infinite, not a division by zero.
    d = got - ref
    noise = float(np.sum(d ** 2))
    snr = np.inf if noise == 0.0 else float(10 * np.log10(np.sum(ref ** 2) / noise))
    return float(np.max(np.abs(d))), float(np.sqrt(np.mean(d ** 2))), snr


if __name__ == "__main__":
    _, _, X_all, y_all = load_dataset(DATASET)
    X, y = X_all[:MAX_SAMPLES], y_all[:MAX_SAMPLES]
    print(f"dataset {DATASET}: {len(X)} test samples, seeds {SEEDS}\n")

    acc = {v: {c: [] for c, _, _ in CONFIGS} for v in VARIANTS}
    err = {v: [] for v in VARIANTS}          # layer-1 error against E0, isolated from the chain

    for name, table, rule in CONFIGS:
        for seed in SEEDS:
            P = grid(load_weights(seed, DATASET), table, rule)
            ref, inputs = forward_dp(P, X, "E0", table, rule)
            for variant in VARIANTS:
                out, _ = forward_dp(P, X, variant, table, rule)
                a, _ = score(out[-1], y)
                acc[variant][name].append(a)
                if variant != "E0":
                    # Isolated: feed the E0 input into the layer, so the error is the datapath only.
                    A1 = np.maximum(0.0, matmul_dp(*inputs[0], *P[0][:3], variant) + P[0][3])
                    e1 = layer_error(np.maximum(0.0, ref[0]), A1)
                    # Layer 2 is not measured this way. It has K = 16, so one block, and its input would be
                    # A1 of the variant instead of A1 of E0. A tiny datapath difference then moves a few
                    # values across a payload boundary, and the resulting jump belongs to the quantizer,
                    # not to the datapath. Layer 1 is the honest measurement: 49 blocks, no re-quantization.
                    err[variant].append(e1)
                    if variant == "E1":
                        # H1: the integer block sum with the scale applied once per block is the same number
                        # as the float64 GEMM, up to float rounding. A larger difference is a bug.
                        rel = np.max(np.abs(out[0] - ref[0])) / np.max(np.abs(ref[0]))
                        assert rel < 1e-12, f"E1 differs from E0 by {rel:.3e} relative, the factorization is wrong"

    print("accuracy (mean over the 3 seeds)")
    print(f"{'config':<16}" + "".join(f"{v:>9}" for v in VARIANTS))
    for name, _, _ in CONFIGS:
        print(f"{name:<16}" + "".join(f"{np.mean(acc[v][name]):>9.2f}" for v in VARIANTS))

    print("\nisolated datapath error against E0, layer 1 (mean over 3 seeds x 4 configs)")
    print(f"{'variant':<9}{'max abs':>12}{'RMS':>12}{'SQNR dB':>10}")
    for variant in VARIANTS:
        if variant == "E0":
            continue
        m1 = np.mean([e[0] for e in err[variant]])
        r1 = np.mean([e[1] for e in err[variant]])
        s1 = np.mean([e[2] for e in err[variant]])
        print(f"{variant:<9}{m1:>12.3e}{r1:>12.3e}{s1:>10.1f}")
