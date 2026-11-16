# ADR-008: Special codes and the tie rule

**Issue**: An 8-bit scale has 256 codes and the payload has 16. A format must say what happens on a zero
block, on a block whose maximum is not representable, on a non-finite input, and on an exact tie between
two codes. Without a rule, the encoder in Python and the encoder in RTL disagree, and every number in the
project becomes a measurement of a different function.

**Decision**:
- Code 255 of the scale is NaN, in both formats. It is never a scale.
- No subnormals, no zero scale, no infinity.
- A zero block encodes to scale code 0.
- A finite block whose scale overflows clamps to code 254, and a counter counts it.
- A block whose maximum is not finite encodes to code 255, and a counter counts it.
- A tie between two codes rounds to the even code.

**Status**: Decided, 2026-11-11, commits `a8f1e58` and `b7b8001`.

**Group**: format.

**Assumptions**: A zero block is common enough to deserve a path, and it is common: 41.5% of the blocks of
the L1d configuration were zero blocks.

**Constraints**: The Python encoder and the RTL encoder must make the same choice on the same input, bit
for bit.

**Positions**:
1. Saturate silently. Simple, and a saturated stream is invisible.
2. Saturate and count. One counter per cause, and a saturated stream is visible.
3. Raise an error.

**Argument**: A counter is 4 bits of state and it turns an invisible failure into a number. The tie rule
matters because "always round down" pushes every tie in one direction, and a drift of that kind repeats
every step. Round to even sends half the ties up and half down.

**Implications**: `python/quant.py` holds the four counters in `scale_counters`: `scale_saturated`,
`scale_underflow`, `zero_blocks`, `nonfinite_blocks`. Every rung reports them, and they are all zero on
the weights of the model, which is evidence that the 8-bit scale covers the range it must cover.

**Related decisions**: ADR-009, ADR-016.

**Related requirements**: The RTL and Python encoders agree bit for bit.

**Related artifacts**: `docs/spec.md`, `python/quant.py` (`scale_counters`, `encode_scale`),
`python/tests/test_quant.py`.

**Related principles**: A silent failure is worse than a counted one.

**Notes**: The zero block path costs about 2 LUT6 in the encoder and it was measured to cost no accuracy.
