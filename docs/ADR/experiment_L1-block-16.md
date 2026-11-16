# ADR-007: A block holds 16 values along the reduction axis

**Issue**: The block size trades bits per value against accuracy. It also decides what one lane can do in
one cycle, because one lane applies one scale to one block.

**Decision**: A block holds 16 values, and the block lies along the axis that the GEMM sums over. For a
weight that is the input axis K. For an activation it is the feature axis, which is the last axis of the
tensor. The block size is semi-fixed at 16.

**Status**: Decided, 2026-10-29. Restated as semi-fixed on 2026-11-16.

**Group**: format.

**Assumptions**: A scale per 16 values holds enough range for a weight matrix and for an activation
tensor, and 8 bits of scale are enough to place it.

**Constraints**: A block must not cross a row of an activation tensor, because then the scale of one
sample would depend on another sample. That property is what lets the input be pre-encoded, see ADR-020.

**Positions**:
1. Block of 8: 5.0 bits per value.
2. Block of 16: 4.5 bits per value.
3. Block of 32: 4.25 bits per value.
4. Block of 256, one per tensor: 4.03 bits per value.

**Argument**: The scale costs 8 bits per block, so a block of 16 costs 0.5 bits per value and a block of
32 costs 0.25. The measured accuracy does not reward the larger blocks: a 16 by 16 tile, which is a block
of 256, costs 0.48 to 2.73 dB of SQNR against a block of 16 (ADR-026). A block of 8 costs twice the scale
bits and buys nothing measurable. Sixteen is also the number a lane wants, because it fills one payload
product per cycle.

**Implications**: The number of scales of a weight matrix is `N * ceil(K/16)`, which is 810 blocks for the
model's three matrices. An activation tensor needs one scale per 16 features, so one scale per sample for
a 16 wide layer.

**Related decisions**: ADR-006, ADR-017, ADR-020, ADR-024, ADR-026.

**Related requirements**: Fewest bits per value without a measured accuracy loss.

**Related artifacts**: `docs/spec.md`, `docs/datapath.md`, `learn-fpg4/lessons/0002-why-the-block-holds-16.html`.

**Related principles**: The block is a hardware unit, not a tuning knob.

**Notes**: "Semi-fixed" means the experiments may test a square tile, and the lane stays built for 16.
