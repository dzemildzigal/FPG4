# ADR-018: Cycles per encoded block are open, balanced at four

**Issue**: The encoder must produce one block per some number of cycles, and the lanes consume one block
per lane per cycle. A slow encoder starves the lanes and a wide encoder costs area. The number also decides
the achievable clock.

**Decision**: The cycles per encoded block are open, with the options 8, 4, 2 and 1. The design is built
and costed at 4. At 1 cycle per block the clock is limited to 100 MHz. The number is settled by synthesis
in rung E9.

**Status**: Pending. Commit `f621008` records the balance point and the open set.

**Group**: datapath, scheduling.

**Assumptions**: Four lanes consume four blocks per four cycles, so an encoder at 4 cycles per block keeps
them busy in layer 1 without being on the critical path of the accumulator.

**Constraints**: The area of the encoder is a function of the values it handles per cycle: at 4 values per
cycle the buffer is 400 flip-flops, at 16 values per cycle it is 1,600.

**Positions**:
1. 8 cycles per block. Smallest area, and it cannot feed four lanes from a 784 wide input.
2. 4 cycles per block. About 1,005 LUT6, and it matches four lanes in layer 1: 49 blocks need 196 cycles
   against 203 cycles of multiply.
3. 2 cycles per block. Twice the width.
4. 1 cycle per block. Fastest, and the clock drops to 100 MHz.

**Argument**: Layer 1 of the model is 784 wide, so the encoder sees 49 blocks per sample and the lanes need
203 cycles. At 4 cycles per block the encoder needs 196 cycles, which is a near balance. At 8 cycles it
needs 392 and it is the limit of the engine. At 1 cycle the multiply path cannot close timing at 125 MHz.

**Implications**: The area of the encoder is quoted at 4 cycles per block, about 1,005 LUT6 for the nine
pieces of the unit. Every later cost table uses that assumption.

**Related decisions**: ADR-004, ADR-016, ADR-017, ADR-026.

**Related requirements**: Fewest cycles per operation, and the smallest area that reaches it.

**Related artifacts**: `docs/datapath.md` section 4.8, `python/blocks_per_step.py`, `docs/engine.md`.

**Related principles**: Balance the feeder against the array.

**Notes**: The figure of 8 cycles per block was once written as a decision and then withdrawn, because it
was never decided.
