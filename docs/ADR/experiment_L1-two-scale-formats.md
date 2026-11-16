# ADR-011: Both 8-bit scale formats stay, and neither is chosen yet

**Issue**: Eight bits of scale can be split into exponent and mantissa in more than one way. The split
trades range against resolution. The project must either choose one split or carry both.

**Decision**: Carry `UE5M3` and `UE6M2` together and keep both in every rung. Do not choose between them
until a rung separates them. `UE5M3` is 5 exponent bits and 3 mantissa bits, code `(e << 3) | m`, bias 15,
which gives 8 codes per octave and 32 octaves. `UE6M2` is 6 exponent bits and 2 mantissa bits, code
`(e << 2) | m`, bias 31, which gives 4 codes per octave and 64 octaves.

**Status**: Pending. Decision deferred on 2026-11-14, commit `b21d0b6`.

**Group**: format.

**Assumptions**: A rung with a finer effect than L2 or L2a will separate the two formats, most likely a
rung that quantizes the gradient.

**Constraints**: Every rung must therefore run four configurations, which multiplies the runs by four.

**Positions**:
1. `UE5M3` only. Better resolution near the block maximum, half the range.
2. `UE6M2` only. Twice the range, coarser resolution.
3. Both, four configurations, and decide later.
4. `UE8M0`, one code per octave. Rejected permanently: no resolution at all, and it needs a normalisation
   of the tensor.

**Argument**: L1b, L1c, L2, L2a and the block experiment do not separate the formats. The best cell moves
with the dataset and with the configuration, and the paired rule fires in a minority of cells with both
signs. A choice now would be a choice without evidence.

**Implications**: Four configurations in every sweep, so every rung costs four times what one format would
cost. The `P` bound differs per format (64 to 225 for `UE5M3`, 16 to 49 for `UE6M2`), which the datapath
must hold as a parameter.

**Related decisions**: ADR-006, ADR-009, ADR-010, ADR-023.

**Related requirements**: A format choice that a measurement supports.

**Related artifacts**: `docs/spec.md` section 5, `python/quant.py` (`e6m2`, `e5m3`, `scale_table_builder`),
`docs/experiment_L2.md`.

**Related principles**: No defaults. An option stays open until one option is clearly winning.

**Notes**: `UE8M0` and the OCP MX format are rejected permanently, and the rejection is Dzemo's.
