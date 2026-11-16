# Experiment L4: the same backward pass, stochastic rounding instead of round to nearest

## Objective

L3 rounded the operands of the backward GEMMs to FP4 with round to nearest. On MNIST that moved the final
accuracy by less than 0.2 points, in both of its cells. L4 asks the same question with one change: the
payload is drawn with **stochastic rounding**, and everything else is identical.

Two questions:

1. does stochastic rounding change the accuracy of the rung?
2. does it remove the vanishing problem that L3 measured, where 66% to 81% of the delta entries sent no update?

## Terms

| Term | Meaning |
|-|-|
| round to nearest, RTN | take the closer of the two neighbouring payload codes. A tie goes to the even code |
| stochastic rounding, SR | take the upper code with probability equal to the value's position inside the step: `t = (r - lower) / (upper - lower)` |
| bias | the mean of the rounding error over many draws. RTN has a bias for a value that never crosses a code step. SR has none, by construction |
| variance | the spread of the rounding error. SR is worse than RTN on every single value, and better in expectation |
| vanished entry | a value that rounds to payload code 0 and sends no update in that step |
| frozen weight | a float master that moves while its FP4 value does not. L3 and L4 both have float weights, so neither can freeze a weight. The frozen weight stays L5 and L6 |

SR trades variance for the removal of bias. That is the whole of it:

```
RTN:  a value at 74% of a step always lands on the upper code. Error +26% of one step, every step.
SR:   the same value lands on the upper code 74 times in 100, and on the lower code 26 times. Mean error 0.
```

`python/quant.py` has `payload_index_sr` and `quantize_blocks_sr`. Five checks ran before this document:

| Check | Result |
|-|-|
| SR is unbiased. One value at 1.37 with scale 1.0, 200,000 draws | mean 1.3704, bias +0.0004. RTN bias +0.13 |
| the draw is a Bernoulli with probability `t` | 74.09% of the draws took the upper code, `t = 0.74` |
| the spread is real | SR standard deviation 0.219, RTN 0.000 for that value |
| the same seed gives the same draw | yes |
| a value above the top code clips and is not drawn | yes, code 7 every time |

## What L4 changes, exactly

| Item | L3 | L4 |
|-|-|-|
| payload rounding | RTN | **SR** |
| the scale rule | `amax/6` or `four6` | the same. `four6` compares its two candidates with the RTN error, because that comparison is a decision about the scale and not about the payload |
| the scale encoding | RTN, midpoints of the code table | the same |
| the shakedown of the batches | `np.random.default_rng(seed)` | the same call, so the batch order of a seed is identical to L3 |
| the rounding draws | none | `np.random.default_rng([seed, 1])`, a separate generator |
| cells | D, DA, DAW | D, DA, DAW |

**The separate generator is a method decision, not a convenience.** The batch order of every seed is the
same in L3 and L4, so L4 against L3 is a paired comparison on identical data in an identical order. If the
draws came from the same generator as the shuffle, every seed would see a different batch order and the
comparison would carry the shuffle noise on top.

## The three cells

The cells are the three of L3, unchanged:

| Cell | `dW` operands rounded | `dA` operands rounded | What it isolates |
|-|-|-|-|
| L4-D | `dZ3`, `dZ2`, `dZ1` | no | the loss gradient alone |
| L4-DA | the same, plus `X`, `A1`, `A2` | no | the activation operands |
| L4-DAW | the same as DA | `dZ` along N, and `W2`, `W3` along N | the full backward pass, with the forward pass still float32 |

In L4-DAW the delta is rounded twice per step, once along the batch for `dW` and once along N for `dA`. One
scale per 16 values cannot serve two axes, so a 1D format needs two copies. That is a property of the
format, and it was measured in L3-DAW.

## Metrics

| Metric | Definition | Why |
|-|-|-|
| final test accuracy, paired | against the L0 row of the same seed. The same pre-registered rule as L3 | the rung answer |
| **vanished share** | share of non-zero operand entries that round to payload code 0 | **the prediction: SR must reduce it** |
| gradient cosine | against the float gradient of the first batch of the epoch | the direction of the error |
| SQNR | of every rounded operand | the size of the error |
| per-epoch curves | train and test loss and accuracy, as a figure | the shape of the damage |

## Decision rule, written before the run

The paired rule of `l2_sweep.paired`, unchanged, for two comparisons:

1. **L4 against L0**, per cell and per configuration. This repeats L3 with one change.
2. **L4 against L3**, per cell and per configuration, same seeds.

L4 differs from L3 only when both hold:

1. the sign of the paired difference agrees with its mean in at least 8 of 10 seeds;
2. the absolute mean is more than twice its standard error.

No decision is a legal result.

## A prediction, on the record before the run

| Claim | Predicted outcome |
|-|-|
| the vanished share falls | 66% to 81% of the delta entries vanish under RTN. Under SR a value at 10% of a step still takes the upper code one time in ten, so the share must fall to near the share of exact zeros |
| the SQNR falls | SR adds variance. SQNR under SR should be 3 to 5 dB below RTN |
| the MNIST accuracy does not move | L3's gap was 0.09 points at most, so there is little to recover and little to lose. My prediction: no decision against L0 and no decision against L3 |
| MNIST-1D is where it can matter | it is the harsher dataset. L3 was never run there, so L4 is the first backward measurement on it |

## Run matrix and cost

| Item | Value |
|-|-|
| datasets | **MNIST and MNIST-1D**, both, by Dzemo's instruction of 2026-11-16 |
| cells | 3 |
| configurations | the four of `qat.CONFIGS` |
| seeds | the ten of `l2_sweep.SEEDS` |
| epochs, batch | 10, 128 |
| runs | 120 per dataset |
| time per run on MNIST | 11 s for D, 105 s for DA, 107 s for DAW, from the L3 file times. SR adds one `searchsorted` and one draw per value |
| time on MNIST | about 2 hours 15 minutes |
| time per run on MNIST-1D | **not measured. One smoke run of each cell is timed before the sweep** |
| the reference rows | the L0 rows that exist. No new runs |

## What L4 does not settle

| Limit | Where it belongs |
|-|-|
| the frozen weight | L5 and L6. Both need FP4 weights |
| the second rounding of a stored FP4 tensor | L5 |
| the optimizer state | L6 |
| whether SR is worth its area | the draw needs a random source and a comparator per value. That cost is not designed yet, and it is a question for the RTL |

## Settled before the run

| Item | Value | Where it comes from |
|-|-|-|
| datasets | MNIST and MNIST-1D | Dzemo, 2026-11-16 |
| the one change | payload rounding, RTN to SR | this document |
| the scale rule | unchanged, both settings, `four6` compared with RTN error | this document |
| the generators | separate, so the batch order matches L3 seed by seed | this document |
| the code | `python/l3.py <dataset> <nseeds> sr`. The same file as L3 on purpose, so the one change is checkable by diffing the rounding argument only | `python/l3.py` |
