# ADR-004: Four lanes, not sixteen

**Issue**: The lane count sets the area and the number of cycles per sample. It is the main scheduling
decision of the engine, and it must be taken before the RTL is written.

**Decision**: The engine holds four lanes. Sixteen lanes are rejected for now.

**Status**: Decided, 2026-11-11, commit `6240628`.

**Group**: datapath.

**Assumptions**: One lane finishes one block-MAC per cycle, so one lane consumes 16 payload products and
one scale product per cycle.

**Constraints**: The encoder must keep up with the lanes. At four cycles per encoded block it produces
one block every four cycles, which is four blocks per sixteen cycles.

**Positions**:
1. Four lanes in logic: about 1,800 LUT6 for the multiply path and about 2,900 LUT6 for the whole engine,
   and 203 cycles per sample of the model.
2. Sixteen lanes in logic: about 7,200 LUT6, which is 13.5% of a Z020, and 51 cycles per sample.
3. Sixteen lanes with one DSP48 each.

**Argument**: The model needs 810 block-MACs per sample, so four lanes take 203 cycles and sixteen lanes
take 51. The encoder cannot feed sixteen lanes at four cycles per block: layer 1 alone needs 49 blocks per
sample, which is 196 cycles, against 51 cycles of multiply. Sixteen lanes would idle for three cycles in
four. Four lanes also cost 5.5% of a Z020, which leaves room for the encoder, the accumulator and the
training state.

**Implications**: The step of the model takes 52,864 cycles on four lanes. Four lanes leave the lane count
open for a later revision, because the figure of merit is per lane and scales linearly.

**Related decisions**: ADR-002, ADR-018, ADR-026.

**Related requirements**: Smallest area first, then fewest cycles for one operation.

**Related artifacts**: `docs/datapath.md` section 9, `docs/engine.md`, `python/blocks_per_step.py`.

**Related principles**: Size the array to the feeder, not to the multiplier.

**Notes**: Sixteen lanes stays possible on a Z020. It is not rejected forever, only for the first build.
