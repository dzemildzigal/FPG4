# ADR-028: L6 has two forms, and both must be run

**Issue**: L6 is the first rung with no float parameter anywhere. The optimizer state is the largest part of
the training state and the least obviously compressible. Whether the state can be FP4 is not known, and a
wrong guess would remove a factor of seven from the state budget or add it back late in the project.

**Decision**: L6 has two forms and both are run:
- `L6a`: Adam `m` and `v` stay float32, and the weights, the activations, the gradients and the update are
  FP4.
- `L6b`: `m` and `v` are FP4 as well.

The experiment must be run. It must not be decided by argument.

**Status**: Decided, 2026-11-15, commit `863110b`.

**Group**: experiment protocol, format.

**Assumptions**: Adam's `v` is a running mean of squared gradients, and its values span a wide range across
layers, so a block format of 8 magnitudes may or may not hold it.

**Constraints**: `L6b` needs the same encoder and the same scale format as the weights, so it costs no new
hardware. It costs an experiment.

**Positions**:
1. Only `L6a`. Optimizer state stays float, and the state budget keeps 832,128 bits for 13,002 weights.
2. Only `L6b`. Assumes the state compresses, and nothing supports it yet.
3. Both forms, measured one at a time.

**Argument**: If `L6b` works, the whole training state of 13,002 weights falls from 1,306,701 bits to
175,527 bits, which is 7.44 times less. That is the largest single saving left in the project, and it is
larger than the removal of the master.

**Implications**: Two runs per configuration and per seed. A result of `L6b` also changes the storage plan,
because the block RAM is what limits the number of weights one board can train.

**Related decisions**: ADR-014, ADR-029.

**Related requirements**: No master weights, and the fabric owns the training state.

**Related artifacts**: `docs/experiment_plan.md` section "L6 has two forms", `docs/spec.md`,
`python/blocks_per_step.py`.

**Related principles**: Measure a compression before you trust it.

**Notes**: The bit counts are payload plus scale, at 4.5 bits per value, and they exclude the bias.
