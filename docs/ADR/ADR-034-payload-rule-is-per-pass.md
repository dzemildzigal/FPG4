# ADR-034: The payload rule is per pass, round to nearest forward and stochastic backward

**Status**: Accepted. 2026-11-19. It corrects the payload row of ADR-033, which recorded one payload rule for
the whole pipeline. It does not touch the scale rule, which ADR-032 already made per use.

**Date**: 2026-11-19

**Deciders**: Dzemo, with the three arms of rung L5 and the forward-pass probe of `docs/experiment_L5.md`.

## Context and problem statement

Every payload value is one of eight codes, `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, multiplied by the block scale. Two
values of the block bracket a real value, and the rounding rule picks which one. There are exactly two rules
in use:

- **Round to nearest, RTN.** Take the nearer of the two codes. No random number.
- **Stochastic rounding, SR.** Take the upper code with probability equal to how far the value sits between
  the two. Consumes one uniform random number per value.

ADR-033 recorded SR for the whole pipeline. That came from L4, where removing SR broke the training: with the
scale rule fixed and RTN in the backward pass, cell DAW gave 22.23 at width 16 (against 65.03 for SR) and
42.76 at width 64 (against 90.03 for SR). The record took the backward result and applied it everywhere. **L5
shows that the forward pass needs the opposite rule.**

## Decision drivers

1. The accuracy of the trained model, paired by seed, against the rule fixed before L4: 8 of 10 seeds agree
   and the mean difference is above two standard errors.
2. The accuracy of the deployed model, meaning the FP4 store read back with the FP4 forward pass, because
   that is the only number the fabric can produce at inference.
3. Area and power, which Dzemo ranks first and second. Two payload rules must not mean two encoders.
4. The random source. If the forward pass does not need it, it can be gated off at inference, so its power
   cost does not apply to the deploying configuration.

## Considered options

| Option | Forward pass | Backward pass | Measured |
|-|-|-|-|
| 1, ADR-033's setting | SR | SR | L5a, master 84.26, deployed 74.88 |
| 2 | RTN | RTN | not run, and the backward comparison of L4 already refuses it: 22.23 and 42.76 |
| **3** | **RTN** | **SR** | **L5c, master 85.58, deployed 82.30** |
| 4, a rule that switches by magnitude | mixed | SR | not run, and nothing in L5 suggests a threshold would help |

Option 2 is not run because the backward result is already decisive and independent of the forward pass: the
collapse belongs to the softmax gradient row, not to the forward pass.

Option 4 is refused because it would need a per-value decision with its own comparator and would add area to
save accuracy that option 3 already does not lose.

## Decision outcome

**The forward pass rounds the payload to the nearest code. The backward pass rounds it stochastically. The
encoder carries both addends and one instruction bit selects between them.**

The evidence, MNIST-1D, width 64, cell DAW, ten seeds, `a6-e6m2`:

| Comparison | Delta | se | Agree | 2 se | Fires |
|-|-:|-:|-:|-:|-|
| L5c against L5a, float master | +1.32 | 0.33 | 9 of 10 | 0.65 | **YES** |
| L5c against L5a, deployed model | +7.42 | 0.32 | 10 of 10 | 0.64 | **YES** |

The same probe with no training at all agrees, and it separates the two effects that L5 mixes together.
Forward pass only, saved float weights, 3 seeds:

| Forward pass | MNIST-1D | MNIST |
|-|-:|-:|
| float | 90.48 | 97.40 |
| 1D block, round to nearest | 83.19 | 96.76 |
| 1D block, stochastic | 77.95 | 96.20 |
| 16 by 16 tile, round to nearest | 79.63 | 96.48 |
| 16 by 16 tile, stochastic | 70.68 | 95.71 |

### Why the two passes want opposite rules

The two passes do not round the same kind of number, and the difference is the summing constraint.

**Backward pass.** The first gradient is a row of the softmax loss: `dZ = (p − y) / B`. Every entry of that row
is a difference from the true class, the true-class entry is negative and larger in magnitude than the others,
and the row sums to exactly zero in exact arithmetic. That row is then rounded to FP4. Round to nearest pushes
the large negative entry toward zero when it sits between two codes, and pushes the small positive entries by
much less, so the rounded row no longer sums to zero and it errs in one direction. Every step, in the same
direction, on the logits. Adam does not remove a constant-direction error; it walks the logits. Stochastic
rounding is unbiased, so the error of the row averages to zero across steps and the sum constraint holds on
average.

**Forward pass.** An activation has no summing constraint. Its rounding error is a difference between the real
value and the nearer grid point, so it takes both signs roughly equally, and the bias of the whole tensor is
near zero even under round to nearest. What round to nearest removes is the *variance*, and the variance is
what L5 measures: 5.24 points on MNIST-1D.

That is the whole mechanism. The rule follows the presence of a summing constraint, not the pass.

## Consequences

- **The encoder has two addends.** One adder per payload value, one multiplexer on its addend input, one
  instruction bit. The comparator tree, the leading-one detect, the cycle count and the block width are
  unchanged. The LUT6 of the multiplexer is **not counted** and is an E9 item.
- **The random source is still required.** It cannot be removed to save area, because the backward pass
  collapses without it. It can be gated off at inference, which removes its dynamic power in the deploying
  configuration.
- **`FPG4_FWD_PAYLOAD` exists in the simulation** and defaults to `sr`, so every run made before this record
  reproduces bit for bit.
- **The deployed accuracy is now a first-class number.** `test_acc_q` is recorded per epoch by `l3.py`, and
  it is the number to quote for the fabric. `test_acc` stays the float master so the ladder keeps one
  comparable column.
- **ADR-033 is amended**: its payload row is replaced by this record, and its forward block row is closed to
  the 1D block by L5a against L5b (+2.29, 10 of 10 seeds).
- **L6 inherits a 3.28 point gap.** The deployed model is 3.28 points below the float master on MNIST-1D with
  both payload rules right. L6 removes the float master, so this gap must not grow.

## Pros and cons of the options

| Option | Pro | Con |
|-|-|-|
| SR everywhere | one payload rule, one addend, no multiplexer | 1.32 points of master accuracy and 7.42 of deployed accuracy on MNIST-1D |
| RTN everywhere | one payload rule, no random source at all | the backward pass collapses: 22.23 at width 16, 42.76 at width 64 |
| **per pass** | the best measured accuracy of the three, and the random source gates off at inference | two addends, one multiplexer per payload value, one more instruction bit |
| a magnitude threshold | in principle keeps the variance out of the large values and the bias out of the small ones | unmeasured, needs a comparator per value, and no measurement asks for it |

## Links

- `docs/experiment_L5.md`, the three arms, the probe, the paired comparisons and the figures
- `docs/experiment_L4.md`, the collapse with round to nearest and the block-scale correction
- `docs/datapath.md` section 4.8, the encoder and its cost
- ADR-032 the block scale must never clip the block maximum, and the scale rule is per use
- ADR-033 what goes into L5, amended by this record
- `python/l5_probe.py`, `python/l3.py` (`FPG4_FWD_PAYLOAD`), `python/runs_l5.py`
