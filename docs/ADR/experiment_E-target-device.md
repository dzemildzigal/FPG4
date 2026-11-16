# ADR-002: The target device is the PYNQ-Z2, part XC7Z020

**Issue**: Area and clock figures only mean something against one device. Documents written before this
decision used an Artix-7 35T as the reference part, and the percentages were computed against it.

**Decision**: The target device is the PYNQ-Z2 board with the XC7Z020-1CLG400C part. Every area share,
every clock cycle count and every timing claim is against this part.

**Status**: Decided, 2026-11-16.

**Group**: scope, then area.

**Assumptions**: The board can clock the programmable logic at 125 MHz, and 150 to 200 MHz is the target
for the datapath.

**Constraints**: 53,200 LUT6, 106,400 flip-flops, 220 DSP48E1, 140 blocks of 36 Kb block RAM, which is
5,160,960 bits or 630 KiB. Two Cortex-A9 cores at 650 MHz on the processing system side.

**Positions**:
1. Artix-7 35T: 20,800 LUT6, 41,600 flip-flops, 1.8 Mb block RAM, 90 DSP48.
2. PYNQ-Z2, XC7Z020: the part on the board that this project owns.
3. Defer the part.

**Argument**: The project runs on a PYNQ-Z2. A share of a part that is not on the board is not a
measurement. The same LUT6 count is 14% of a 35T and 5.5% of a Z020, so the wrong denominator changes the
verdict on 16 lanes from "too large" to "possible".

**Implications**: The percentage columns of `docs/datapath.md` and `docs/engine.md` were computed against
a 35T. They are to be rebased on the Z020, and the absolute LUT6 counts stay. The block RAM budget also
changes by 2.8 times, which matters for the training state: master, `m` and `v` in float32 for 13,002
weights take 24.2% of the Z020 block RAM.

**Related decisions**: ADR-004, ADR-015, ADR-026, ADR-028.

**Related requirements**: Fit one engine, one encoder and one training state on one board.

**Related artifacts**: `docs/spec.md` section 9, `docs/datapath.md`, `docs/engine.md`,
`python/blocks_per_step.py`.

**Related principles**: A share is quoted against the device that runs it.

**Notes**: The mismatch was found on 2026-11-16 and is recorded here before any document is rebased.
