# ADR-027: L3 keeps float weights and activations, and quantizes only the backward operands

**Issue**: L3 had to be defined so that it measures one thing: what rounding the gradient to FP4 costs. An
earlier plan had the block experiment as a prerequisite of L3. That plan was wrong, because L3 keeps float
weights, so no weight sits on a grid and no second blocking exists in it.

**Decision**: L3 runs with float32 weights, float32 activations, float32 optimizer and float32 loss. Only the
operands of the weight-gradient GEMM are rounded to FP4, once each, with one 8-bit scale per 16 samples. The
rounding rule is round to nearest. Two cells are run: `L3-D`, which rounds only the loss gradients, and
`L3-DA`, which also rounds the activation operands.

**Status**: Approved, 2026-11-16. Not run.

**Group**: experiment protocol.

**Assumptions**: The weight-gradient GEMM is the GEMM whose operand rounding matters, because it produces the
number the optimizer uses.

**Constraints**: The gradient-down GEMM cannot be quantized in L3, because its operands include `W.T` and
that needs the weights on a grid, which is L5. The comparison of one rounding against two roundings also
belongs to L5, because a second rounding needs a stored FP4 activation.

**Positions**:
1. Quantize only the loss gradient `dZ`.
2. Quantize `dZ` and the activation operands.
3. Quantize the whole backward pass, including `W.T`. That is L5.

**Argument**: Positions 1 and 2 separate the two operands for 15 minutes each on MNIST, which is cheap
enough to run both. Position 3 mixes the gradient question with the weight question, and the block
experiment already shows that mixing them hides the cause.

**Implications**: No new code in the forward pass and no change in the store. The rung adds one hook in the
backward pass, and it can be run before any RTL work. The expected effect size is 6 to 12% error in the
gradient norm and 5 to 11 degrees in direction, from `python/second_rounding_probe.py`.

**Related decisions**: ADR-005, ADR-019, ADR-021, ADR-024, ADR-028.

**Related requirements**: One variable per rung.

**Related artifacts**: `docs/experiment_L3.md`, `docs/experiment_plan.md`, `docs/experiment_block.md`.

**Related principles**: Correct a plan in writing before you run it.

**Notes**: The scope correction is written into `docs/experiment_L3.md` as its second section, with the
sentence that the earlier claim was wrong.
