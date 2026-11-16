# ADR-017: The encoder holds a 25-bit value and takes amax over the full magnitude

**Issue**: The encoder needs the block maximum before it can choose a scale. It can find the maximum from a
rounded value or from the full magnitude, and it must keep the block's values while it decides.

**Decision**: A value enters the encoder at 25 bits: one sign, 8 exponent bits, 16 significand bits. `amax`
is the maximum of the full magnitude, about 720 LUT6 for the weight path. The value is buffered at 25 bits,
which is 400 flip-flops at 4 values per cycle. The window clamp sits before the payload pass.

**Status**: Decided, 2026-11-11, commit `18bf3ab` (the probe) and `f621008` (the unit).

**Group**: datapath.

**Assumptions**: The accumulator's output is fixed point, so it is normalised once to 25 bits before the
encoder sees it.

**Constraints**: Only one rounding may exist in the path (ADR-013), so `amax` must not be taken from an
already rounded value.

**Positions**:
1. Find `amax` from a 16-bit significand. Smaller, and it can under-run the maximum.
2. Find `amax` from the full magnitude. Larger, and exact.
3. Find `amax` from the payload codes after a first pass. Two passes over the payload, and the scale would
   depend on the scale.

**Argument**: `python/amax_probe.py` measured how many significand bits the encoder needs to reproduce the
scale code of the full magnitude. The answer is that the code is stable at 16 bits, and the full magnitude
costs about 720 LUT6, which is affordable once per encoder and not once per lane.

**Implications**: The block buffer is 25 bits per value, so a 16-value block is 400 flip-flops. A square
tile of 256 values would need 6,400 flip-flops if it were buffered, which is why a tile encoder uses two
passes instead (ADR-026).

**Related decisions**: ADR-013, ADR-016, ADR-018, ADR-026.

**Related requirements**: One rounding in the whole path.

**Related artifacts**: `docs/datapath.md` section 4.8, `python/amax_probe.py`.

**Related principles**: Measure a width before you truncate it.

**Notes**: The probe is a measurement of the scale code, not of the accuracy of a rung.
