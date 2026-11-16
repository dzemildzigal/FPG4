# ADR-019: L2 is FP4 forward with a float32 backward pass

**Issue**: L2 had to answer one question: what does FP4 in the forward pass cost during training. Any other
change at the same time would confound it. The rung also needed a name that does not overstate what it
proves.

**Decision**: L2 quantizes the three weight matrices to FP4 for the forward pass only. The activations, the
biases, the backward pass, the master weights and the Adam state stay in float32. The gradient at the master
is the straight-through gradient. The honest name of the rung is "FP4 forward, float32 backward".

**Status**: Decided, 2026-11-12, commits `aea728b` and `490ba9a`.

**Group**: experiment protocol.

**Assumptions**: The FP4 effect on the forward pass is visible in the final accuracy and in the loss, and the
backward pass is exact enough that the difference is the forward pass.

**Constraints**: L2 must not report a number that belongs to the float master, and it must not claim native
FP4 training.

**Positions**:
1. Quantize the weights and the activations. That is L2a, a separate rung.
2. Quantize the gradients too. That is L3.
3. Quantize the weights of the forward pass only.

**Argument**: One rung changes one thing. L2 is the smallest FP4 change that touches training at all, and it
gives the reference that L2a, L3 and L5 compare against.

**Implications**: L2 cannot show a loss of gradient precision, a frozen weight or a drift. Those need L3 to
L7. The measured result is 94.61 to 94.80 on MNIST against 95.01 for the float32 L0, and 63.26 to 63.55 on
MNIST-1D against 64.98.

**Related decisions**: ADR-005, ADR-010, ADR-020, ADR-021, ADR-022, ADR-027.

**Related requirements**: One variable per rung.

**Related artifacts**: `python/qat.py`, `python/l2_sweep.py`, `docs/experiment_L2.md`.

**Related principles**: Name a rung for what it does, not for what it hopes to show.

**Notes**: The name "FP4 training" was used for L2 in early notes. It is wrong and it is not used.
