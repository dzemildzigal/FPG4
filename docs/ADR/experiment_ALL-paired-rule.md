# ADR-021: Two runs are compared by a pre-registered paired rule

**Issue**: Two means over seeds can differ by an amount that comes from the seeds alone. Without a rule
written before the runs, every comparison becomes a judgement, and a judgement made after seeing the numbers
is not a measurement.

**Decision**: Two runs are compared on final test accuracy, seed by seed, and a difference counts only when
both conditions hold:
1. The sign of the paired difference agrees with the sign of its mean in at least 8 of 10 seeds.
2. The absolute mean is more than twice its standard error.

"No decision" is a legal result. It does not select either option.

**Status**: Decided, 2026-11-12, commit `aea728b`. Used unchanged since.

**Group**: experiment protocol.

**Assumptions**: The same seed gives the same start weights and the same batch order in both runs, so the
difference is paired and the seed noise cancels.

**Constraints**: The rule is written for ten seeds. Fewer than eight seeds gives condition 1 the value
"not applicable", so a smoke run cannot decide anything.

**Positions**:
1. Compare means only. Ignores the spread and the sign pattern.
2. Compare a confidence interval on the unpaired means. Wastes the pairing and needs more seeds.
3. The paired rule above.

**Argument**: Paired differences remove most of the seed noise, because both runs start from the same weights
and see the same batches. The sign condition catches a mean that comes from two large opposite seeds, which
is the failure mode a standard error does not reveal at ten samples.

**Implications**: Every sweep stores per-seed results and never a mean only. A mean without its seeds cannot
be checked against the rule.

**Related decisions**: ADR-022, ADR-023.

**Related requirements**: Evidence that a reader can recompute.

**Related artifacts**: `python/l2_sweep.py` (`paired`), used by `python/block_sweep.py`.

**Related principles**: Write the rule before the run.

**Notes**: A second copy of the rule would be a second rule, so `block_sweep.py` imports it instead of
copying it.
