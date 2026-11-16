# ADR-010: Rule four6 keeps amax/6 or amax/4 per block, as a configuration bit

**Issue**: Rule `amax/6` puts the block maximum on code 6. When one value dominates a block, that choice
coarsens the grid for the other 15 values. The question was whether a second candidate scale, chosen per
block, pays for its cost.

**Decision**: The encoder offers two candidate scales, `amax/6` and `amax/4`, computes the absolute error
of the block for both, and keeps the better one. The rule is named `four6`, or `L1c` in the code. It stays
in the RTL as a configuration bit, and it is not the default until a rung shows a gain.

**Status**: Decided, 2026-10-29. Kept as a configuration bit on 2026-11-11, commit `f621008`.

**Group**: format.

**Assumptions**: A block whose error falls with the second candidate is common enough to pay for the second
pass, or the bit can be turned off.

**Constraints**: The rule needs the snap twice in one encoder pass, which doubles part of the encoder, and
it needs the block maximum before it can choose.

**Positions**:
1. `a6` only. One candidate, the smallest encoder.
2. `four6`: two candidates, the smaller error wins.
3. Stochastic rounding: pick one of the two neighbouring payload values with a probability.
4. A per-block search over all 255 scale codes.

**Argument**: Measured, `four6` changes the scale code of 31 to 35% of the 1D blocks and 26 to 46% of the
4 by 4 tiles. It is not equivalent to stochastic rounding. Stochastic rounding is 3.1 to 3.2 dB worse in
SQNR than round to nearest, and it is unbiased, so the two are different tools and they combine: RTN is L3
and SR is L4. A search over 255 codes is exact and much larger than two candidates.

**Implications**: The rule is worth nothing on a 16 by 16 tile, where it keeps `amax/4` in 0.0 to 1.2% of
tiles, so `a6` and `f6` produce almost the same weights there. It is worth 26 to 46% on the 4 by 4 tile.

**Related decisions**: ADR-009, ADR-011, ADR-016, ADR-018.

**Related requirements**: A configuration that can be turned off after a measurement.

**Related artifacts**: `docs/spec.md`, `python/quant.py` (`four6`), `docs/experiment_L1.md`,
`python/l1e_sweep.py`.

**Related principles**: An option that never wins is turned off, and it is measured before it is turned off.

**Notes**: No rung has yet shown that `four6` beats `a6` on accuracy. The four configurations differ by
0.03 accuracy points or less in L1e.
