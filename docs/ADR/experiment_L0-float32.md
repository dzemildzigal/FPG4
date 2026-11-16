# ADR-005: Float32 is the simulation width

**Issue**: The Python rungs need one float width. A wider width is closer to the algebra and slower. A
narrower width is faster and adds a rounding that no document counted. The rung ladder said FP64.

**Decision**: Float32 is the simulation width for the weights, the activations, the optimizer state and
the arithmetic of every rung from L0 up. The FP64 numbers stay as the record of the width experiment.

**Status**: Decided, 2026-11-12, commit `aea728b`. Ladder rows updated 2026-11-16, commit `237f74f`.

**Group**: experiment protocol.

**Assumptions**: A float32 run measures the FP4 effect and not the float width effect, because the
difference between the two widths is smaller than the seed spread.

**Constraints**: A row must never mix widths. One float64 seed and nine float32 seeds give a mean that
belongs to no single run.

**Positions**:
1. FP64 everywhere. Closest to the algebra, twice the memory, and slower on MNIST-1D.
2. FP32 everywhere. The numbers of the host, faster, and one rounding per operation.
3. FP32 for training and FP64 for the reference.

**Argument**: `python/fp32_probe.py` measured the same three seeds on the same code path in both widths.
L0 in float64 is 95.18 on MNIST and 65.11 on MNIST-1D. L0 in float32 is 95.01 and 64.98. The difference is
0.17 and 0.13 accuracy points, and the seed spread is larger. The width therefore does not hide an effect
of the size the FP4 rungs measure.

**Implications**: The reference row of every later rung is the float32 L0. A weight file of the wrong
width is retrained and overwritten, never mixed into a row. The rungs L1a to L1e were measured in float64
and keep that width in `docs/experiment_L1.md`.

**Related decisions**: ADR-021, ADR-022.

**Related requirements**: A reference that is comparable to the rung under test.

**Related artifacts**: `python/train.py` (`DTYPE`), `python/fp32_probe.py`, `docs/experiment_plan.md`,
`docs/experiment_L2.md`.

**Related principles**: One variable per rung. The reference and the rung share every other setting.

**Notes**: The ladder text said FP64 for L2 to L5 until 2026-11-16. The text now says FP32.
