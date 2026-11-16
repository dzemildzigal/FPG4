# ADR-003: The datapath holds no hard multiplier IP

**Issue**: The scale product can be built in one stage, from a product table, or in two stages, from the
two significands of the two 8-bit scales. The choice fixes the block RAM, the LUT count and the portability
of the engine.

**Decision**: The datapath is pure logic. It contains no DSP48 and no product read-only memory. The term
is built in two stages: `T = (S * N_w) * N_x`, where `N_w` and `N_x` are the significands of the weight
scale and the activation scale.

**Status**: Decided, 2026-10-09. Recorded in `docs/spec.md` section 9 and `learn-fpg4/REVIEW.md` item F1.

**Group**: datapath.

**Assumptions**: A small multiply by a 4-bit significand is cheaper as conditional adders than as a DSP,
and it does not cost a placement constraint.

**Constraints**: No DSP48 may be used, so the engine fits a device whose DSP blocks are wanted elsewhere.

**Positions**:
1. One stage with a 256 by 8 read-only memory that holds `P = N_w * N_x`, then `T = S * P`.
2. Two stages, `T = (S * N_w) * N_x`, with no memory.
3. One DSP48 per lane, then `T` from a hardware multiply.

**Argument**: The two stage form is never worse. For UE5M3 it is 4 conditional adders and no lookup
table, against 4 adders and 32 lookup tables for the one stage form, because `P` reaches 225. For UE6M2
it is 2 adders and no lookup table, against 3 adders and 32 lookup tables. The two stage form also keeps
the exponent as a shift, so the binary point moves without a multiplier.

**Implications**: The lane is a chain of conditional adders and one align shifter. The scale product is
exact in both stages. The algebra still names `P = N_w * N_x`, and it is only a bound: 64 to 225 for
UE5M3, 16 to 49 for UE6M2.

**Related decisions**: ADR-012, ADR-013, ADR-014.

**Related requirements**: Pure logic, and a fit inside the fabric of the board.

**Related artifacts**: `docs/spec.md` section 5.4 and section 9, `docs/datapath.md` section 4.5,
`docs/rtl-plan.md` section 3.1.

**Related principles**: Buy exactness with structure, not with memory.

**Notes**: An earlier revision of the spec described the read-only memory form in section 5.3 and the two
stage form in section 5.4. The ruling ended the contradiction.
