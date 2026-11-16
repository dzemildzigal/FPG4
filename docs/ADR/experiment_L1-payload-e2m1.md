# ADR-006: The payload is E2M1, four bits

**Issue**: The payload format fixes which numbers a block can hold, and every later stage inherits it. The
two candidates were a 4-bit E2M1 with magnitudes up to 6, and an E3M0 with magnitudes up to 8.

**Decision**: The payload is E2M1 with one sign bit, two exponent bits and one mantissa bit. The eight
magnitudes are `k/2` for `k` in {0, 1, 2, 3, 4, 6, 8, 12}, that is {0, 0.5, 1, 1.5, 2, 3, 4, 6}. The
largest magnitude is 6.

**Status**: Decided, 2026-10-29, commit `0e0b13b`.

**Group**: format.

**Assumptions**: A block scale removes the need for a wide payload range, so a payload with two exponent
bits is enough.

**Constraints**: The product of two payload magnitudes must be exact in a small integer. With `k1 * k2` at
most 144, the product is a multiple of 0.25 and it is exact.

**Positions**:
1. E2M1, maximum 6. Eight magnitudes, four codes per octave, one extra magnitude at the top.
2. E3M0, maximum 8. Sixteen magnitudes, all powers of two, no value between 1 and 2.
3. A 3-bit payload.

**Argument**: Four bits is the smallest payload that holds a sign, a scale and a value. E2M1 keeps two
values per octave across much of its range, which E3M0 does not. The exactness of the payload product is
what makes the whole integer path exact, and `k1 * k2 <= 144` is the reason it fits 8 bits.

**Implications**: The 8-bit scale must be able to put a block maximum anywhere on this grid. The block sum
`S` has the bound 16 times 144, which is 2304, and it needs 13 bits.

**Related decisions**: ADR-007, ADR-009, ADR-012, ADR-013.

**Related requirements**: An exact integer path, and a payload product that needs no rounding.

**Related artifacts**: `docs/spec.md`, `python/quant.py` (`PAYLOAD_CODES`, `PAYLOAD_INTS`),
`learn-fpg4/lessons/0003-what-the-8-scale-bits-buy.html`.

**Related principles**: Exactness before precision.

**Notes**: `E2M1` and `NVFP4` payloads are the same eight magnitudes. The scale is not the same, and
ADR-011 records that.
