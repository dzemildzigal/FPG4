# ADR-032: The block scale must never clip the block maximum

**Status**: Accepted. 2026-11-18. It corrects the scale rule of ADR-009 and ADR-010, and it answers ADR-031.

**Date**: 2026-11-18

**Deciders**: Dzemo, with the measurements of `python/l3.py`, of the probe scripts in this record, and of the
L4 re-run with the corrected rule.

## Context and problem statement

Every block of 16 values is stored as two things: 4 bits per value (a payload code) and one 8-bit code
(a scale). Two rules choose those two things, and they are independent:

1. **The payload rule** chooses the code of one value, between the two codes that bracket it. "SR",
   stochastic rounding, draws the upper code with probability equal to how far the value sits between the
   two codes. "RTN", round to nearest, takes the nearer code and no random number.
2. **The scale rule** chooses one of the 255 table entries for the whole block.

The scale rule was "the nearest table entry to `amax/6`" (ADR-009, rule L1b). The nearest entry can sit
**below** `amax/6`. Then the largest value of the block needs a ratio `r = |x| / scale` above 6, and code 7
is not in the payload. The largest value clips to `6 * scale`.

Measured over 20,000 random blocks that span 14 decades of magnitude:

| Rule | Blocks with `r > 6` | Worst `r` |
|-|-:|-:|
| `a6-e6m2`, nearest | 10,413 of 20,000 (52%) | 6.750 |
| `f6-e6m2`, nearest | 7,222 of 20,000 | 6.749 |
| `a6-e5m3`, nearest | 8,340 of 20,000 | 6.375 |
| any rule, scale rounded up | **0** | **6.000** |

A second sample of 32,000 blocks, drawn with a different random stream, clips 52, 37, 51 and 35 percent of
the blocks for the four configurations under the nearest rule, and none under the rounded-up rule.

`r = 6.75` means the block maximum is stored 11% low.

**Why a 2% error in one entry is not small.** The largest entry of a block is not a random entry. In a
softmax gradient row `dZ3 = (P - Y) / B` the largest entry is the true-class entry. Clipping it shrinks it
by 2.4% on average, and the shrink has the same sign in every row. A row of `dZ3` must sum to zero. The
rounded row sums to **+2.4%** of the true-class entry instead. Shifting all ten logits of a row by the same
amount changes no softmax probability, so the loss has no gradient against that shift. Adam sees a
constant-sign push on a direction the loss cannot correct, and it walks.

The same defect is in the forward pass, where the payload rule is RTN. It was inside every L1, L2 and L2a
number in `docs/`, and it was never separated from the forward-path cost.

## Decision drivers

| Driver | Number |
|-|-|
| Accuracy at stake in the backward pass | 64.98 to 36.03 before the fix, 64.95 to 65.03 after it |
| Entries that clip per block | one, but it is the maximum |
| Cost of the change in the datapath | none. The 254 thresholds move from midpoints to boundaries |
| Price in SQNR of the rounded tensors | 1.68 to 2.86 dB |
| The engine must hold one rule for every tile and every axis | yes. The clip depends on the block, not on the shape |

## Considered options

1. **Keep the nearest scale and accept the clip.** The shipped state. Measured: the dA cells collapse.
2. **Round the scale up to the next table entry.** `r = amax / scale` then never exceeds 6, so no block
   clips. The scale is on average half a step larger, which lowers the SQNR of the block.
3. **Widen the payload by one code.** A 9th code above 6, for example 8. It removes the clip and keeps the
   nearest scale, but it needs 4 bits plus one escape, and it changes every code table.
4. **Clip the input before the block.** Throw away the excess on purpose, and record that the block is
   clipped. It makes the error visible but it does not remove it.
5. **Give each block an extra exponent bit.** More scale bits, not fewer. Rejected: it costs one bit per
   block for a defect that costs nothing to remove.

## Decision outcome

**Option 2. The scale rule is "the smallest table entry at or above the target", for every block, in every
pass.** For rule `amax/6` the target is `amax/6`. For the `four6` rule the two candidates `amax/6` and
`amax/4` are both rounded up before the error test compares them.

The rule keeps one property that the format needs: `amax / scale <= 6` for every block whose `amax` is
inside the table range. A block then never clips. The overflow case that remains is the one already
recorded: `amax` outside the table range saturates to code 254 and raises the counter.

### The encoder must change, and it costs nothing

The encoder in `docs/datapath.md` section 4.8 does not divide. It compares `amax` against 254 **midpoint**
thresholds and returns the code of the nearest entry. Under this decision it compares against the 255
**boundaries** themselves and returns the code of the first entry that is not below `amax`.

| Item | Before | After |
|-|-|-|
| Number of thresholds | 254 | 255 |
| Comparator count | unchanged | unchanged |
| Cycles per block | unchanged | unchanged |
| Threshold ROM contents | midpoints | boundaries |
| LUT6, FF | unchanged | unchanged |

So the answer to "must the encoder change" is **yes**, and the change is a different constant table, not a
different circuit. Section 4.8 of `docs/datapath.md` carries the note.

### Evidence

**The sign of the drift matches the sign of the bias.** Width 64, MNIST-1D, seed 42, 1,800 steps, one step per
batch, config `a6-e6m2`. The class mean is the mean over the ten logits of a row, signed, averaged over the
batch. The loss cannot see it.

| Cell | Row-sum bias of `dZ3` | Class mean of the logits at step 1,800 | Test accuracy at step 1,800 |
|-|-:|-:|-:|
| float, nothing rounded | 0 | -0.51 | 73.91 |
| SR, nearest scale (shipped) | +2.42% | **-88.60** | 71.06 |
| RTN, nearest scale | -2.96% | **+218.43** | 70.81 |
| SR, scale rounded up | -0.02% | +4.76 | 73.10 |

SR leaves a positive residual and the logits walk negative. RTN leaves a negative residual and the logits
walk positive. The float cell has neither.

**The bias of the shipped rule, per configuration, and the price of the fix.** 4,000 softmax gradient rows,
60 draws per row, entry 1 is the true class.

| Config | Residual, nearest scale | Residual, scale up | SQNR, nearest | SQNR, up | Price |
|-|-:|-:|-:|-:|-:|
| `a6-e6m2` | +2.42% | -0.02% | 18.37 dB | 15.51 dB | 2.86 dB |
| `f6-e6m2` | +1.89% | -0.01% | 18.35 dB | 16.00 dB | 2.35 dB |
| `a6-e5m3` | +1.05% | -0.02% | 19.13 dB | 17.11 dB | 2.02 dB |
| `f6-e5m3` | +0.87% | -0.04% | 18.97 dB | 17.29 dB | 1.68 dB |

The order of the four configurations by bias is the same order as their accuracy in the L4 MNIST-1D table.

**The cell recovers.** MNIST-1D, 10 epochs, cell L4-DAW, SR payload.

| Width | Evidence | Scale up | Float, same seed |
|-|-|-:|-:|
| 16 | 5 runs, 2 configs, 5 seeds | **65.03 mean** | 64.95 mean |
| 16 | seed 42, `a6-e6m2` | 64.44 | 64.98 |
| 64 | seed 42, `a6-e6m2` | 89.90 | 91.00 |
| 16 | 2 epochs, seed 42, `a6-e6m2` | 61.26 | 62.24 (cell DA) |
| 64 | 2 epochs, seed 42, `a6-e6m2` | 81.76 | 82.56 (cell DA) |

Before the fix the same cell at width 16 gave 36.03 over 40 runs.

## Consequences

**Positive**

- The last known large error in the FP4 backward pass is gone. Cell L4-DAW at width 16 sits 0.09 points from
  float, which is inside the seed spread of 0.57.
- The rule is one rule. It does not depend on the tile, on the axis, or on the pass.
- The datapath does not grow. The comparator tree is the same tree.

**Negative**

- The scale of a block is on average half a step larger, so the SQNR of a rounded tensor falls 1.68 to 2.86 dB.
  The accuracy does not follow that fall, measured above.
- Every L1, L2 and L2a number in `docs/` was measured with the nearest rule, in the forward pass. Those
  numbers include the clip. They must be re-measured before any of them is compared with a later rung.
- Two rules now exist in `python/quant.py`, and the difference is one argument. `quant.SCALE_ROUNDING` is
  `"nearest"` by default, so every stored run keeps its meaning, and `FPG4_SCALE_ROUNDING=up` re-runs any
  sweep under the new rule. `l2_sweep.py` writes the rule into every result file and into the output path.

**Done on 2026-11-18.** `docs/spec.md` went from 0.13 to 0.14, section 4.4 step 2 and the threshold
paragraph, on Dzemo's instruction. `python/quant.py` has the switch and the default is bit-identical to the
code before the change, checked against a saved reference of 20 arrays. `python/l2_sweep.py` records the rule
and separates the output paths, so no stored file can be overwritten by a re-measurement.

## Pros and cons of the options

| Option | Pro | Con |
|-|-|-|
| 1, keep the nearest scale | no change, no work | measured: the dA cells collapse, and the forward path carries the same bias |
| 2, round the scale up | removes the clip completely, costs no area, one rule for every block | 1.68 to 2.86 dB of SQNR, and every forward number must be re-measured |
| 3, add a 9th payload code | keeps the nearest scale and the SQNR | one escape bit or a wider payload, and every code table changes |
| 4, clip on purpose | the error is visible in a counter | the error is still there |
| 5, another scale bit | also removes the clip | one bit per block forever, for a defect that costs nothing to remove |

## Links

- ADR-009, the rule `amax/6`, which this record corrects
- ADR-010, the `four6` rule, which this record also rounds up
- ADR-016, the encoder without a division, whose threshold table changes
- ADR-031, which this record answers: the collapse is not a property of the axis of the contraction
- `docs/datapath.md` section 4.8, the encoder
- `docs/experiment_L4.md`, the correction section
