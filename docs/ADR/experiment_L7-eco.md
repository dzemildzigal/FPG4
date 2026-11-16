# ADR-029: L7 recovers lost updates through the momentum, with no extra tensor

**Issue**: When a weight is stored in FP4, most optimizer updates are smaller than the grid. The update is
then lost, and the frozen-weight telemetry of the model shows how often: the mean Adam update is 74 times
smaller than one grid step near the middle of a block. A rung was needed that recovers those updates
without a second FP4 tensor.

**Decision**: L7 implements error feedback in the optimizer: the weights are rounded after every step, the
rounding error is added into the momentum, and no second tensor is stored. This is the method of the ECO
paper, arXiv 2601.22101.

**Status**: Decided, 2026-11-15, commit `863110b`. Ladder row updated.

**Group**: format, storage.

**Assumptions**: The error of a step is correlated across steps, so carrying it forward in the momentum
recovers what the grid would lose.

**Constraints**: The optimizer state still exists. The paper keeps `m` and `v`, so L7 does not remove them,
and the master is what L7 removes.

**Positions**:
1. A second FP4 tensor that holds the remainder of each weight.
2. Error feedback into the momentum, with no extra tensor.
3. Stochastic rounding instead of a remainder. That is L4.

**Argument**: Position 1 costs 4.5 bits per weight and doubles the weight store. Position 2 costs no memory
at all, and the paper measures it: no master weights with round to nearest diverges in FP8, and the same run
with error feedback reaches 3.3317 against 3.3238 for the baseline with master weights, on a 30M model.
Position 3 removes the bias but does not accumulate the lost amount, so it is a different tool and it is
already a rung of its own.

**Implications**: L7 becomes the rung that makes master-weight-free training possible, which is the project's
long-term aim. The frozen-weight problem, which L6 will show, is the thing L7 must recover.

**Related decisions**: ADR-028.

**Related requirements**: No master weights, and the fabric owns the weights.

**Related artifacts**: `docs/experiment_plan.md`, `docs/experiment_L2.md` (the frozen steps),
`python/qat.py` (`frozen_steps`, `stale_elements`).

**Related principles**: Recover an error where it is cheapest to store it.

**Notes**: No source found does this in FP4. The paper covers FP8 pretraining and INT4 fine-tuning, and it
lists FP8-LM, DeepSeek-V3 and Kimi-K2 as systems that keep a wide accumulator.
