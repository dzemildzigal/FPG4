# ADR-022: The reported number is always the deployed quantized model

**Issue**: A training loop holds two weight sets: the float master that the optimizer updates, and the FP4
copy that the fabric uses. Evaluating the master gives a better number, and that number describes a model
that never reaches the device.

**Decision**: Every reported accuracy and loss is measured on the deployed quantized model, never on the
float master.

**Status**: Decided, 2026-11-14, commit `490ba9a`.

**Group**: experiment protocol.

**Assumptions**: The deployed model is the quantized one for every rung, including rungs where the master is
updated in float.

**Constraints**: The evaluation must therefore quantize the weights, and in L2a and L1d it must also
quantize the activations, with its own hook so that the training telemetry stays separate.

**Positions**:
1. Report the master. Better numbers, and they belong to no rung.
2. Report the quantized model. Lower numbers, and they are the numbers of the device.
3. Report both.

**Argument**: The first version of the L2 results reported the master by mistake, and the error was 0.5 to
0.9 accuracy points per row. A number that the fabric cannot reproduce is not a result of this project.

**Implications**: Every rung re-evaluates at the end of every epoch with a fresh quantizer state. The
telemetry of the steps keeps its own state, so the two do not mix.

**Related decisions**: ADR-019, ADR-021, ADR-023.

**Related requirements**: A number that the device can reproduce.

**Related artifacts**: `python/qat.py` (`train_qat`), `python/model.py` (`evaluate`),
`docs/experiment_L2.md` section "Correction".

**Related principles**: Measure what you ship.

**Notes**: The bug was a call to `evaluate(p, ...)` where the parameter should have been `evaluate(pq, ...)`.
The correction is recorded in the L2 document rather than hidden.
