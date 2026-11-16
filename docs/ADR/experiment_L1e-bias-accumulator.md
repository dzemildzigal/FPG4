# ADR-015: The bias lives at the accumulator's width and point

**Issue**: A bias is added to every output of a layer. It can be stored in the FP4 format like the weights,
or in the accumulator's own fixed-point format. The choice decides whether the bias add costs anything.

**Decision**: The bias is stored at the accumulator's width and binary point, 48 bits, and it is added in
the accumulator. It is never quantized to FP4 and it is not blocked along K.

**Status**: Decided, 2026-11-11, commit `c899d81`.

**Group**: datapath, storage.

**Assumptions**: A bias add at the accumulator's point needs no shift, so it costs one adder per lane.

**Constraints**: The bias store is one value per output, not one value per block, so it is 42 values for the
model, which is 2,016 bits, and it is 3.4% of the weight store.

**Positions**:
1. FP4 bias, blocked along K. Measured in L1d-bias: it costs accuracy, and it needs a shift to reach the
   accumulator's point.
2. FP4 bias with one scale per value. Measured, and it recovers part of the loss and costs more scales.
3. Bias at the accumulator width and point.

**Argument**: At the accumulator's point the add is free: no shift, no scale, no second rounding. The
measured cost of the FP4 bias in the model is larger than the 2,016 bits it would save. Per seed on
MNIST-1D in the worst row, float bias gave 55.78, a block of 16 gave 55.55, and one scale per value gave
55.68, so the FP4 store pays for itself in nothing.

**Implications**: The bias parameters are never quantized in any rung, so every rung's bias gradient is a
float gradient. The bias gradient is 42 values, so the encoder never touches it.

**Related decisions**: ADR-014, ADR-020.

**Related requirements**: A bias add that costs no rounding and no shift.

**Related artifacts**: `docs/spec.md`, `docs/experiment_L1.md` (L1d-bias), `python/model.py`.

**Related principles**: Put a value where it costs nothing to use.

**Notes**: Two bias cost numbers exist and were once added together: 1,344 bits for a block format and 840
bits for one scale per value. They are two alternatives, never a sum.
