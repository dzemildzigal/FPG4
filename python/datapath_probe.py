import numpy as np
import quant as q
from train import load_weights, load_dataset

# Reproduces every measured number in docs/datapath.md.
# Run from python/ with:  uv run python datapath_probe.py


def payload_ints(codes_signed, idx):
    # idx holds the payload code index (0..7) of every element, the sign is in the decoded value.
    # This turns (decoded value, index) into the signed payload integer of the format.
    return np.sign(codes_signed).reshape(idx.shape).astype(np.int64) * q.PAYLOAD_INTS[idx]


def main():
    Xtr, ytr, Xte, yte = load_dataset("mnist")
    P = load_weights(42)
    NS = 2000

    # ---- stage 1 and 2: encode both operands of layer 1 ------------------
    # W1 is (784, 16). quantize_blocks_weights transposes it, so idxW is (16, 49, 16).
    Wq, _, idxW, sW, _ = q.quantize_blocks_weights(P["W1"], q.e6m2, "L1c")
    # X is (n, 784). quantize_blocks_act keeps the row layout, so idxA is (n, 49, 16).
    Aq, _, idxA, sA, _ = q.quantize_blocks_act(Xtr[:NS], q.e6m2, "L1c")

    ka = payload_ints(Aq, idxA)   # (n, nb, 16) units of 0.5
    kw = payload_ints(Wq.T, idxW)  # (16, nb, 16) units of 0.5

    print("=== shapes ===")
    print("ka", ka.shape, "kw", kw.shape)
    print("payload code width        4 bits (1 sign + 3 index)")
    print("payload integer width     5 bits (-12..12)")

    # ---- stage 3: block sums, exact ------------------------------------
    S = np.einsum("nbk,mbk->nmb", ka, kw)  # (n, 16, nb): one integer sum per (sample, neuron, block)
    print("\n=== stage 3, block sum S ===")
    print("bound  16 * 144 =", 16 * 144, "-> 12 magnitude bits, 13 with sign")
    print("largest |S| observed:", np.abs(S).max())
    print("distinct S values   :", np.unique(S).size)

    # ---- stage 4: scale products ---------------------------------------
    # sA is (n, nb, 1) and sW is (16, nb, 1). The block index is the shared axis.
    Pp = sA[:, :, 0][:, None, :] * sW[None, :, :, 0]  # (n, 16, nb)
    print("\n=== stage 4, scale product P ===")
    print("P min %.6e  P max %.6e  ratio %.6e" % (Pp.min(), Pp.max(), Pp.max() / Pp.min()))

    # ---- stage 5: term and layer accumulator ---------------------------
    # term = (S / 4) * P, the 1/4 is the payload unit 0.5 applied twice.
    term = 0.25 * S * Pp
    C = term.sum(axis=2)  # (n, 16)
    print("\n=== stage 5, term and layer accumulator ===")
    nz = np.abs(term[term != 0])
    print("largest term         %.6e" % np.abs(term).max())
    print("smallest nonzero term %.6e" % nz.min())
    print("largest |C|          %.6e" % np.abs(C).max())
    print("term range           %.1f bits" % np.log2(np.abs(term).max() / nz.min()))

    # ---- worked example: one element of C, all 49 blocks ---------------
    print("\n=== one worked element, C[0, 0], sample 0 neuron 0 ===")
    run = 0.0
    for b in range(S.shape[2]):
        t = term[0, 0, b]
        run += t
        if 9 <= b <= 42:
            print("b=%2d S=%5d P=%.6e term=%+.6e acc=%+.6e"
                  % (b, S[0, 0, b], Pp[0, 0, b], t, run))
    print("zero blocks skipped, C[0,0] =", C[0, 0])

    # ---- layer 2: nb = 1, so the block structure disappears ------------
    print("\n=== layer 2, nb = 1 ===")
    Wq2, _, idxW2, sW2, _ = q.quantize_blocks_weights(P["W2"], q.e6m2, "L1c")
    A1 = np.maximum(0, Xtr[:NS] @ P["W1"] + P["b1"])
    Aq1, _, idxA1, sA1, _ = q.quantize_blocks_act(A1, q.e6m2, "L1c")
    S2 = np.einsum("nbk,mbk->nmb", payload_ints(Aq1, idxA1), payload_ints(Wq2.T, idxW2))
    P2 = sA1[:, :, 0][:, None, :] * sW2[None, :, :, 0]
    print("S2 shape", S2.shape, "largest |S2|", np.abs(S2).max())
    print("largest |C2|", np.abs(0.25 * (S2 * P2).sum(axis=2)).max())


if __name__ == "__main__":
    main()
