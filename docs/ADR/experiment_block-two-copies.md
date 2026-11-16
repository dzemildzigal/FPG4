# ADR-025: Two 1D copies of a weight matrix are rejected

**Issue**: The simplest way to satisfy both blockings is to store a weight twice: one copy blocked along K
for the forward pass, one blocked along N for the gradient-down pass. It needs no new shape and it keeps the
lane at one term. NVIDIA records that this form hurts a 1.2B model, and the project had to measure it at its
own size.

**Decision**: A weight matrix is not stored as two 1D copies. The form is rejected for training.

**Status**: Decided, 2026-11-15, commit `0e58015`, and confirmed on the second dataset, commit `14db604`.

**Group**: format, storage.

**Assumptions**: The mechanism is the drift between the two copies, and the drift repeats because both copies
come from the same master every step.

**Constraints**: The two copies must both come from the float master, so they differ. On the model, 62 to 67%
of the elements of W2 and W3 differ in the `e6m2` configurations and 76 to 79% in the `e5m3` configurations.

**Positions**:
1. Two 1D copies.
2. One 16 by 16 tile with the same value in both passes.
3. A 4 by 4 tile.
4. Keep one 1D copy and use the same blocked matrix in both passes, which is wrong but legal.

**Argument**: Against the L2 reference, the two-copy form loses 0.398 accuracy points on MNIST, decided in 3
of 4 configurations, and 2.541 points on MNIST-1D, decided in every one of the 4 configurations. It also loses on test loss in every
configuration and on both datasets, and it fits the training set worse: 0.1666 against 0.1505 to 0.1518 for
the tile arms. In its worst configuration it also multiplies the seed spread by 3.8.

**Implications**: The second copy is not built, so the 32 extra blocks and 2,304 bits per step are not spent.
The store needs a square tile instead, which is a shape decision and not a second copy.

**Related decisions**: ADR-024, ADR-026.

**Related requirements**: No drift between the matrix the loss sees and the matrix the gradient uses.

**Related artifacts**: `docs/experiment_block.md`, `python/block_sweep.py`, `python/shape_probe2.py`
material in the run notes, `docs/experiment_L2.md`.

**Related principles**: A repeated bias does not average out.

**Notes**: On MNIST the copies do not diverge: their cosine stays at 0.990 to 0.993 for ten epochs, and the
cost is a constant bias. **On MNIST-1D the same telemetry falls from 0.982 at epoch 1 to 0.945 at epoch 10,
so on a harder dataset the drift compounds.** That is why the loss is 0.40 points on MNIST and 2.54 on
MNIST-1D, and it is the reason to reject the form on any dataset that is not as easy as MNIST.
