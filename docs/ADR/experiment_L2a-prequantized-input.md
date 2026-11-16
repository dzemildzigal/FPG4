# ADR-020: The input tensor may arrive pre-encoded

**Issue**: The input of layer 1 is 784 wide, so the encoder sees 49 blocks per sample and spends most of
its time on data that never changes during an epoch. Quantizing it inside the loop is both slow and
pointless if the same value comes out every time.

**Decision**: The input tensor may be quantized once, outside the training loop, and stored. This is
allowed because a block never crosses a row, so the scale of a block depends on that row only. The
equivalence is proved, not assumed.

**Status**: Decided, 2026-11-12, commit `aea728b`.

**Group**: experiment protocol, storage.

**Assumptions**: The stored tensor is one FP4 payload plus one scale per 16 values, exactly what the encoder
would produce.

**Constraints**: The claim is only true if the block never crosses a row. If it did, the result would depend
on the batch size and on which rows share a batch.

**Positions**:
1. Encode the input inside the loop. 49 blocks per sample, 68 ms per batch of 128, and it dominates the
   step time.
2. Pre-encode the input once and store it.
3. Pre-encode the input and check nothing.

**Argument**: `check_prequant_equivalence` compares one call on 4,096 rows against 32 calls on 128 rows and
requires bit-identical output. It passes, so the stored tensor is the tensor the loop would have produced.
The cost of the check is one second and it protects every L2a number.

**Implications**: A pre-quantized copy of MNIST-1D takes 1.25 GB on disk or in memory, which is why the
sweep prepares one configuration at a time. In the engine, the same decision means the input arrives
encoded, and the encoder handles only A1 and A2 at inference.

**Related decisions**: ADR-007, ADR-030.

**Related requirements**: The encoder must not dominate the inference step.

**Related artifacts**: `python/l2_sweep.py` (`prequantize`, `check_prequant_equivalence`),
`docs/experiment_L2.md`.

**Related principles**: Prove an equivalence, do not assume it.

**Notes**: At inference the encoder is the limit unless the input is pre-encoded: 204 cycles per sample
against 202.5 cycles of multiply.
