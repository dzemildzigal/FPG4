# ADR-024: The transposed blocking is an encoder decision, not a datapath change

**Issue**: One training step has three GEMMs per layer and each sums over a different axis. A weight matrix
therefore needs one blocking for the forward pass and another for the gradient-down pass. Two primary
sources confirm that this forces either two stored copies or a square tile. The project had to decide whether
this changes the datapath.

**Decision**: The transposed blocking is a decision about the encoder and about the store. The lane, the
block sum, the scale product and the accumulator do not change. For activations and gradients the blocking
stays 1D, which both sources do.

**Status**: Decided, 2026-11-15, commit `0e58015`.

**Group**: datapath, storage.

**Assumptions**: A lane consumes 16 values with one scale along the reduction axis, and both a 1D block and
a 16 by 16 tile present exactly that.

**Constraints**: A tile smaller than 16 along the reduction axis needs more than one term per block pair,
which is a datapath change. 8 by 8 needs two, 4 by 4 needs four.

**Positions**:
1. Two 1D copies, one blocked along K and one along N. Measured and rejected, see ADR-025.
2. One 16 by 16 tile, one scale per 256 weights, the same value in both passes. No datapath change.
3. A 4 by 4 tile, one scale per 16 weights. Four terms per block pair, so four sums and four scale products.
4. Change the lane to accept a transposed operand. No such change is small.

**Argument**: Positions 1 and 2 cost the same datapath, so the choice between them is accuracy and memory,
which the block experiment answers. Position 3 is the only one that reaches finer than 16 values along K,
and it costs area or cycles. Position 4 has no bounded cost.

**Implications**: The block shape becomes a field of the weight store. The device needs no mode bit, because
1D and 16 by 16 present the same thing to a lane.

**Related decisions**: ADR-007, ADR-025, ADR-026.

**Related requirements**: One datapath for inference and training.

**Related artifacts**: `docs/experiment_block.md`, `docs/datapath.md`, `python/second_rounding_probe.py`.

**Related principles**: Change the feeder before you change the array.

**Notes**: The sources are NVIDIA arXiv 2509.25149 Appendix E.5 and the OCP MX paper arXiv 2310.10537
section 4.1. Both were quoted, not paraphrased.
