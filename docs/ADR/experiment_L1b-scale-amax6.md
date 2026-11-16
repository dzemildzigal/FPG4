# ADR-009: The block scale is amax/6, snapped to the nearest 8-bit code

**Issue**: A block needs one rule that turns its 16 values into one 8-bit scale code. The rule decides how
much of the payload range a block uses.

**Decision**: The target scale is `amax/6`, where `amax` is the largest magnitude in the block. The code is
the nearest representable scale, with a tie going to the even code. The rule is named `a6`, or `L1b` in the
code.

**Status**: Decided, 2026-10-29, commit `0e0b13b`.

**Group**: format.

**Assumptions**: Putting the block maximum at the top payload code uses the whole payload range, and a
block whose maximum sits below code 6 wastes payload range without gaining accuracy.

**Constraints**: The encoder must not divide, so `amax/6` must be reachable by comparison against a table,
see ADR-016.

**Positions**:
1. `amax/6`: the block maximum lands on payload code 6.
2. `amax/4`: the block maximum lands on payload code 4, which reserves headroom above it.
3. The mean of the magnitudes, times a constant.

**Argument**: `amax/6` is the choice that uses the whole payload. `amax/4` was kept as the second candidate
of rule `four6` (ADR-010) because it wins in a minority of blocks: when the top value is rare, pushing it to
code 6 coarsens the grid for the other 15 values, and `amax/4` is better.

**Implications**: The largest scale code that `amax/6` can need depends on the format, and the smallest
scale code bounds the distance between blocks, which sets the accumulator width (ADR-014). A zero block
has `amax = 0` and takes code 0 by the rule of ADR-008.

**Related decisions**: ADR-006, ADR-008, ADR-010, ADR-016.

**Related requirements**: The encoder chooses a scale without a division.

**Related artifacts**: `docs/spec.md`, `python/quant.py` (`quantize_blocks`, `encode_scale`),
`python/tests/test_quant.py`.

**Related principles**: Use the range the format gives you.

**Notes**: `a6` is the base rule of every rung. `f6` is a superset of it, so a comparison of `a6` and `f6`
is a comparison of the rule with itself plus one more candidate.
