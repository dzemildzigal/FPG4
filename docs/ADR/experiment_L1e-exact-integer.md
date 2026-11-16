# ADR-013: The integer path is exact, and the datapath has one rounding

**Issue**: Every rounding in a training datapath is a place where a gradient can disappear. A block scaled
format offers several places to round: the payload product, the block sum, the scale product and the
accumulation. The project had to decide how many of them round.

**Decision**: The integer path is exact from the payload code to the term. One rounding exists in the whole
datapath, and it is the re-encode of the accumulated result back to FP4 at the end of the GEMM. Between
those two points nothing rounds.

**Status**: Decided, 2026-11-11, commit `b7b8001`.

**Group**: datapath.

**Assumptions**: An exact path is cheaper to verify than a path with several small roundings, and the
roundings would not be independent anyway.

**Constraints**: `S` must fit: the largest magnitude of a block sum is 16 times 144, which is 2304, so `S`
needs 13 bits and it is exact for every input. The term must fit: `S * N` is at most 2304 times 15, which
needs 19 magnitude bits plus a sign.

**Positions**:
1. Round the payload product. Rejected: the product of two payload integers is a multiple of 0.25 and it is
   exact for free.
2. Round the block sum. Rejected: 13 bits hold the worst case.
3. Round the term. Rejected: the product is exact in 21 bits.
4. Round only the accumulator, at the re-encode.

**Argument**: Each of the rejected roundings would cost bits and add a bias, and none of them is needed.
The exactness is what makes the L1e comparison meaningful: the rung measures the accumulator's alignment
loss and nothing else.

**Implications**: The only error of the datapath is the alignment loss inside the accumulator plus the final
re-encode. That is why the accumulator width decision (ADR-014) is the only remaining precision decision.

**Related decisions**: ADR-006, ADR-012, ADR-014.

**Related requirements**: A gradient must not be lost before the optimizer sees it.

**Related artifacts**: `docs/spec.md` sections 5.3 to 5.6, `docs/experiment_L1.md`, `python/l1e_sweep.py`,
`python/datapath_probe.py`.

**Related principles**: Exactness is a budget item, and it is cheaper than it looks.

**Notes**: The L1e variants E0 to E4 differ from each other by 0.03 accuracy points or less, except one
cell, which is consistent with a path that rounds once.
