# ADR-012: The term is built from two significands, with no product table

**Issue**: A lane must turn 16 payload products and two scales into one term. The two scales each carry a
significand, and the term needs their product. The product can be stored, or built.

**Decision**: The lane builds the term in two stages: `T = (S * N_w) * N_x`. `S` is the exact block sum of
the 16 payload integers. `N_w` and `N_x` are the significands of the two scale codes. The exponents add
and become a shift. No table of products exists in the datapath.

**Status**: Decided, 2026-10-09. Recorded in `docs/spec.md` section 9.

**Group**: datapath.

**Assumptions**: A multiply by a 4-bit significand is a small number of conditional adds.

**Constraints**: `N` is at most 15 for `UE5M3` and at most 7 for `UE6M2`, because the encoder never emits a
zero scale and a valid stream has `N` at least `2^M`.

**Positions**:
1. One stage with a 256 by 8 memory that holds `P = N_w * N_x`.
2. Two conditional-add stages, no memory.
3. One DSP48 per lane.

**Argument**: The two stage form costs 4 conditional adders and no lookup table for `UE5M3`, against 3 to 4
adders and 32 lookup tables for the memory form. For `UE6M2` it is 2 adders and no lookup table. It is
never worse, it removes a memory from the lane, and it lets the binary point move as a shift.

**Implications**: The multiply path of one lane is one 13-bit conditional-add array, one 8-bit add for the
exponent, and one alignment shift. The design reports about 105 full adders for the first stage.

**Related decisions**: ADR-003, ADR-013, ADR-014.

**Related requirements**: Pure logic, and exactness.

**Related artifacts**: `docs/spec.md` sections 5.3, 5.4 and 9, `docs/datapath.md` section 4.5,
`python/datapath_probe.py`.

**Related principles**: Buy exactness with structure.

**Notes**: The spec's algebra still writes `P = N_w * N_x`. It is a bound and a name, and no hardware
builds it.
