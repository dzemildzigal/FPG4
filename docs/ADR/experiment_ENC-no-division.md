# ADR-016: The encoder divides by nothing

**Issue**: The scale rule of ADR-009 needs `amax/6`, and the rule of ADR-010 also needs `amax/4`. A divider
is expensive in area and in cycles, and it sits on the critical path of every block.

**Decision**: The encoder never divides. It compares `amax` against 254 thresholds. Threshold `c` is the
midpoint between scale code `c` and scale code `c+1`. The code of a block is the number of thresholds that
`amax/6` is above, which is the nearest scale code including the even-code tie rule.

**Status**: Decided, 2026-11-11, commit `f621008`.

**Group**: datapath.

**Assumptions**: A table of 254 thresholds is cheaper than a divider, and the comparison tree can be built
from the same constant table that the scale format already needs.

**Constraints**: The thresholds are constants of the format, so they are built once and shared by every
encoder in the engine.

**Positions**:
1. A divider. Accurate, large, and it needs several cycles per block.
2. A reciprocal multiply. Smaller than a divider, and it adds a rounding that no document counts.
3. Two hundred and fifty four midpoint thresholds, about 35 LUT6 for the comparison tree.

**Argument**: The threshold form is exact against the specification's own definition of the nearest code,
because the midpoints are where the choice changes. It also makes the tie rule explicit: a value equal to a
threshold is a tie, and the even code wins. A reciprocal multiply would put a second rounding into the
encoder, and ADR-013 keeps the datapath to one rounding.

**Implications**: The encoder needs the 254 thresholds for each format, and it needs `amax` before it can
compare. That is why the encoder holds the block's values in a buffer, see ADR-017.

**Related decisions**: ADR-008, ADR-009, ADR-010, ADR-017, ADR-018.

**Related requirements**: No division in the datapath.

**Related artifacts**: `docs/datapath.md` section 4.8, `python/quant.py` (`encode_scale`),
`python/amax_probe.py`.

**Related principles**: Compare, do not compute.

**Notes**: The code and the specification must agree on the tie. `python/tests/test_quant.py` checks the
even-code rule on constructed ties.
