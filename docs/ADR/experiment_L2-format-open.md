# ADR-023: The format and the rule are not decided, and the question moves to the gradients

**Issue**: Rungs L1b, L1c, L2 and L2a each ran four configurations, which are the two scale rules times the
two scale formats. The project needed to know whether one configuration wins, so that later rungs could run
one configuration instead of four.

**Decision**: No configuration is selected. The question moves from the forward pass to the gradients, that
is to L3 and L4.

**Status**: Pending. Recorded 2026-11-14, commit `b21d0b6`.

**Group**: experiment protocol.

**Assumptions**: A finer effect, such as a quantized gradient, separates the configurations where a
quantized forward pass does not.

**Constraints**: Every rung keeps four configurations until one wins, so every rung costs four times the
runs.

**Positions**:
1. Select `f6-e5m3`, the best cell on both datasets. Rejected: it is the best cell of a set where the
   differences are inside the seed spread.
2. Select `a6`, the simpler rule, because `four6` never shows a gain. Rejected for the same reason.
3. Keep all four.

**Argument**: On MNIST the ten seed means of the four configurations sit inside 0.19 accuracy points, and on
MNIST-1D inside 0.29. The paired rule fires in a minority of the comparisons and with both signs. A choice
would be a choice without evidence, and the project rule is that an option stays open until one option is
clearly winning.

**Implications**: The next rung that quantizes the gradient must run all four configurations. The reported
numbers always carry the configuration name.

**Related decisions**: ADR-011, ADR-021, ADR-027.

**Related requirements**: No defaults.

**Related artifacts**: `docs/experiment_L1.md`, `docs/experiment_L2.md`, `python/l2_sweep.py`.

**Related principles**: No defaults. A cell that wins by less than the seed spread is not a winner.

**Notes**: `four6` changes the scale code in a third of the blocks and still moves the accuracy by less than
0.03 points. That is itself a result about the model, not about the rule.
