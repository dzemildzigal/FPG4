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
| time on MNIST | **measured: 2 h 46 min** |
| time per run on MNIST-1D | **measured: 82 s for D, 794 s for DA.** DAW is quoted when it is measured |
| the reference rows | the L0 rows that exist. No new runs |

## Results, MNIST

120 runs, three cells, four configurations, ten epochs. Run 2026-11-16, finished 23:46. The output is
`out/l4_mnist.log`, the runs are `out/l4/mnist/`, and the figure is
`docs/experiment_L4_mnist_curves.png`.

### Final test accuracy, percent

| Cell | a6-e6m2 | f6-e6m2 | a6-e5m3 | f6-e5m3 | Mean of the four |
|-|-:|-:|-:|-:|-:|
| **L4-D**, deltas rounded, SR | 95.02 ±0.25 | 94.99 ±0.32 | 95.04 ±0.31 | 94.96 ±0.25 | **95.00** |
| **L4-DA**, and activations | 94.95 ±0.31 | 95.07 ±0.20 | 94.96 ±0.29 | 95.01 ±0.26 | **95.00** |
| **L4-DAW**, and the gradient-down operands | 94.93 ±0.26 | 95.05 ±0.21 | 95.10 ±0.26 | 95.03 ±0.23 | **95.03** |
| L0, float32, no configuration | | | | | **95.01 ±0.30** |

### Paired against L0, same seed

| Cell | Configuration | Mean difference | se | Seeds agreeing | Decision |
|-|-|-:|-:|-:|-|
| L4-D | a6-e6m2 | +0.01 | 0.08 | 4 of 10 | no decision |
| L4-D | f6-e6m2 | −0.03 | 0.05 | 5 of 10 | no decision |
| L4-D | a6-e5m3 | +0.03 | 0.06 | 5 of 10 | no decision |
| L4-D | f6-e5m3 | −0.05 | 0.05 | 8 of 10 | no decision |
| L4-DA | a6-e6m2 | −0.06 | 0.07 | 7 of 10 | no decision |
| L4-DA | f6-e6m2 | +0.05 | 0.07 | 6 of 10 | no decision |
| L4-DA | a6-e5m3 | −0.06 | 0.06 | 7 of 10 | no decision |
| L4-DA | f6-e5m3 | −0.00 | 0.06 | 8 of 10 | no decision |
| L4-DAW | a6-e6m2 | −0.08 | 0.08 | 5 of 10 | no decision |
| L4-DAW | f6-e6m2 | +0.04 | 0.07 | 5 of 10 | no decision |
| L4-DAW | a6-e5m3 | +0.08 | 0.08 | 4 of 10 | no decision |
| L4-DAW | f6-e5m3 | +0.02 | 0.05 | 4 of 10 | no decision |

### Paired against L3, same seed, same batches, only the rounding rule differs

| Cell | Configuration | Mean difference | se | Seeds agreeing | Decision |
|-|-|-:|-:|-:|-|
| D | a6-e6m2 | +0.008 | 0.056 | 5 of 10 | no decision |
| D | f6-e6m2 | +0.000 | 0.101 | 5 of 10 | no decision |
| D | a6-e5m3 | +0.012 | 0.036 | 7 of 10 | no decision |
| D | f6-e5m3 | −0.029 | 0.055 | 5 of 10 | no decision |
| DA | a6-e6m2 | −0.034 | 0.061 | 4 of 10 | no decision |
| DA | f6-e6m2 | −0.009 | 0.041 | 6 of 10 | no decision |
| DA | a6-e5m3 | −0.043 | 0.107 | 6 of 10 | no decision |
| DA | f6-e5m3 | +0.027 | 0.069 | 3 of 10 | no decision |
| DAW | a6-e6m2 | −0.114 | 0.067 | 6 of 10 | no decision |
| DAW | f6-e6m2 | +0.055 | 0.040 | 7 of 10 | no decision |
| DAW | a6-e5m3 | +0.170 | 0.095 | 7 of 10 | no decision |
| DAW | f6-e5m3 | −0.023 | 0.073 | 5 of 10 | no decision |

**No decision in twelve comparisons against L0 and no decision in twelve comparisons against L3.** The
largest difference in the second table is 0.17 points with a standard error of 0.095, so the size condition
of the rule (`|mean| > 2 se`) is not met.

### Telemetry, last epoch, mean over every seed and configuration

| Cell | Tensor | SQNR, L3 with RTN | SQNR, L4 with SR | Fall | Vanished L3 | Vanished L4 |
|-|-|-:|-:|-:|-:|-:|
| D | `dZ3` | 25.29 dB | 20.93 dB | −4.36 | 80.63% | 79.42% |
| D | `dZ2` | 23.58 | 18.90 | −4.68 | 66.64% | 64.24% |
| D | `dZ1` | 23.94 | 19.80 | −4.14 | 66.26% | 63.97% |
| DAW | `dZ3` | 25.45 | 20.72 | −4.73 | 80.87% | 78.90% |
| DAW | `dZ2` | 23.63 | 18.61 | −5.02 | 59.48% | 57.80% |
| DAW | `dZ1` | 23.67 | 19.02 | −4.65 | 59.39% | 57.68% |
| DAW | `dZ3` along N | 24.04 | 20.06 | −3.98 | 70.70% | 68.66% |
| DAW | `dZ2` along N | 21.12 | 17.46 | −3.66 | 11.86% | 13.00% |

| Gradient cosine, first batch | L3, RTN | L4, SR |
|-|-:|-:|
| `W1` | 0.9963 | 0.9900 |
| `W2` | 0.9935 | 0.9853 |
| `W3` | 0.9934 | 0.9861 |
| `dA2` | 0.9938 | 0.9868 |
| `dA1` | 0.9932 | 0.9863 |

### Two of the four predictions held and one failed

| Prediction made before the run | Outcome |
|-|-|
| SQNR falls 3 to 5 dB | **held.** 3.7 to 5.0 dB, in every tensor |
| the accuracy does not move on MNIST | **held.** 95.00 and 95.03 against 95.01, no decision in 24 comparisons |
| **the vanished share falls to near the share of exact zeros** | **failed.** it fell by 1 to 2 points, from 66.6% to 64.2% on `dZ2` |
| SR costs direction per step | **held, and it costs nothing in the end.** the cosine falls 0.006 to 0.008, and the final accuracy is unchanged |

**I was wrong about the vanished share, and the reason is measured.** `python/vanished_probe.py` reads the
block scale of every entry and reports `r`, the value in scale units. On the trained float weights, first
batch of MNIST:

| Tensor | Median `r` | The 90th percentile | Entries below the first code | Predicted SR share | Share of the `\|dZ\|` mass lost | Error put into `dW` |
|-|-:|-:|-:|-:|-:|-:|
| `dZ3` | 0.0083 | 4.03 | 74.9% | 72.3% | 2.50% | 2.77% |
| `dZ2` | 0.0110 | 1.62 | 78.7% | 75.9% | 4.88% | 4.53% |
| `dZ1` | 0.0167 | 1.61 | 77.6% | 74.5% | 4.78% | 3.54% |

The median delta entry sits at **one hundredth of a scale unit**. The first payload code is at 0.5, so that
entry is fifty code steps from the first code. Stochastic rounding gives it a 2% chance of being sent, not a
large one. The predicted fall is 2 to 3 points, and the measured fall is 1 to 2 points.

The three statements that follow from this table are the useful part of the rung:

1. **A shared block scale throws away 75% of the delta entries, and 97% of the delta is unaffected.** The
   entries that vanish hold 2.5% to 4.9% of the magnitude. The gradient keeps 95% of its L1 mass.
2. **Stochastic rounding cannot fix that**, because the entries are not near a code boundary. SR repairs a
   value that sits between two codes. These values sit near zero.
3. **The vanishing is not a defect of the rounding rule.** It is a property of the block scale: one scale
   for 16 values means the small values of a block share the scale of the largest one. Fixing it needs a
   finer scale, which costs the term chain that ADR-026 measures.

### Cost measured

| Cell | Time per run | 40 runs |
|-|-:|-:|
| L4-D | 12 s | 8 min |
| L4-DA | 116 s | 78 min |
| L4-DAW | 120 s | 80 min |
| total | | **2 h 46 min** |

SR adds 1 to 4 s per run over L3. The draw is one `rng.random` over the block array and one search of the
two neighbouring codes, so it does not change the shape of the cost.

### One defect, mine

The figure of the rung crashed after all 120 runs had finished, with `KeyError: 'L4-D'`. The cause was a
hardcoded colour table that held the three L3 cell names, so no L4 cell had a colour. The 120 run files were
already written and no run was lost. The fix derives the colour from the cell order. The figure was then
drawn from the stored run files, with no training.

## Results, MNIST-1D

120 runs, three cells, four configurations, ten epochs. Run 2026-11-16 and 2026-11-17, finished 20:52.
The output is `out/l4_mnist1d.log`, the runs are `out/l4/mnist1d/`, and the figure is
`docs/experiment_L4_mnist1d_curves.png`.

### Final test accuracy, percent

| Cell | a6-e6m2 | f6-e6m2 | a6-e5m3 | f6-e5m3 | Mean of the four |
|-|-:|-:|-:|-:|-:|
| **L4-D**, deltas rounded | 65.02 ±0.62 | 65.07 ±0.46 | 64.87 ±0.38 | 64.86 ±0.79 | **64.96** |
| **L4-DA**, and activations | 65.04 ±0.59 | 65.03 ±0.49 | 64.98 ±0.83 | 65.05 ±0.69 | **65.03** |
| **L4-DAW**, and the gradient-down operands | **22.72 ±4.03** | **32.24 ±5.81** | **40.94 ±4.10** | **48.23 ±4.24** | **36.03** |
| L0, float32, no configuration | | | | | **64.98 ±0.57** |

### Paired against L0, same seed

| Cell | Configuration | Mean difference | se | Seeds agreeing | Decision |
|-|-|-:|-:|-:|-|
| L4-D | a6-e6m2 | +0.04 | 0.18 | 5 of 10 | no decision |
| L4-D | f6-e6m2 | +0.10 | 0.22 | 5 of 10 | no decision |
| L4-D | a6-e5m3 | −0.11 | 0.17 | 6 of 10 | no decision |
| L4-D | f6-e5m3 | −0.11 | 0.24 | 4 of 10 | no decision |
| L4-DA | a6-e6m2 | +0.06 | 0.18 | 6 of 10 | no decision |
| L4-DA | f6-e6m2 | +0.05 | 0.19 | 5 of 10 | no decision |
| L4-DA | a6-e5m3 | +0.00 | 0.32 | 4 of 10 | no decision |
| L4-DA | f6-e5m3 | +0.07 | 0.28 | 5 of 10 | no decision |
| **L4-DAW** | a6-e6m2 | **−42.26** | 1.23 | **10 of 10** | **L0 better** |
| **L4-DAW** | f6-e6m2 | **−32.74** | 1.90 | **10 of 10** | **L0 better** |
| **L4-DAW** | a6-e5m3 | **−24.04** | 1.26 | **10 of 10** | **L0 better** |
| **L4-DAW** | f6-e5m3 | **−16.75** | 1.34 | **10 of 10** | **L0 better** |

DA against D, paired: +0.02, −0.04, +0.11, +0.18 points, no decision in four of four.

**The rule fires, and it fires against the cell.** The operands of the weight-gradient GEMM and of the
activations cost nothing on MNIST-1D. Add the operands of the gradient-down GEMM and the network loses 17 to
42 points. The seed spread is also 6 to 10 times wider than every other row, which is what a network that
has stopped training looks like.

### The failure is a blow-up, not a loss of precision

Seed 42, configuration a6-e6m2, the cell DAW:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
| test accuracy | 57.00 | 34.35 | 23.38 | 20.06 | 21.52 | 24.28 | 23.78 | 21.43 | 21.18 | 24.61 |
| test loss | 1.13 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |

The first epoch is normal, the second collapses, and the accuracy then wanders near chance, which is 10
percent. A second configuration, f6-e5m3, degrades more slowly and holds 45 percent at the last epoch. The
shape is the same in both: the first epoch is the equal of the DA cell and the run then destroys itself.

**The `NaN` is a defect of my loss, and it hid the size of the failure.** In float32 a softmax probability
underflows to exactly zero once a logit sits about 104 below the largest logit of its row. `log(0)` is minus
infinity, and `onehot * log(P)` is then `0 * -inf`, which is `NaN`, for class entries that are not the true
class. So one confident wrong sample turns the whole recorded loss into `NaN`, while every gradient stays
finite. `model.loss` now clips `P` away from zero. The clip changes no gradient, because `dZ3` is built from
`P` and never from the loss value. The accuracies of every rung are untouched.

### The cause, measured in four short runs

`python/daw_collapse_probe.py` runs two epochs of a single seed four ways on MNIST-1D. Round to nearest was
never run on this dataset, because the protocol extends a rung to the second dataset only when a rule fires
on the first. The probe supplies that missing comparator.

| Variant | After epoch 1 | After epoch 2 | Loss at epoch 2 |
|-|-:|-:|-:|
| the DA cell, neither operand of the dA GEMM rounded | 59.97 | **62.24** | 0.90 |
| both operands rounded, stochastic rounding, the shipped cell | 57.00 | **34.35** | 2.97 |
| both operands rounded, round to nearest | 53.44 | **44.85** | 1.86 |
| **only the delta rounded**, weight left float | 58.23 | **28.77** | 4.64 |
| **only the weight rounded**, delta left float | 59.34 | **60.93** | 0.99 |

Three statements, each from one row:

1. **It is not stochastic rounding.** Round to nearest collapses too, from 53 to 45.
2. **It is not the weight.** Round the weight along N and leave the delta float, and the run is healthy:
   59.34 and 60.93, against 59.97 and 62.24 with nothing rounded.
3. **It is the delta, rounded along the contraction axis N.** That alone takes 58.23 to 28.77.

### Why the delta along N fails, and why only on this dataset

`dA = dZ @ W.T` contracts over the output width N. So the operands must be blocked along N. For layer 2 the
width is 16, and for layer 3 it is 10, so **one block along N is the whole layer width**. One scale then
covers every output of a layer, and the smallest outputs of a row vanish. The measured telemetry of the cell:

| Tensor | SQNR | Entries that vanish |
|-|-:|-:|
| `dZ3` along N, 10 outputs in one block | 21.55 dB | **71.48%** |
| `dZ2` along N, 16 outputs in one block | 17.66 dB | 7.80% |
| `dA1`, the gradient one layer down | | cosine **0.8866** against float |
| `dA2` | | cosine 0.9738 against float |

`dZ3 = (P - Y) / B` has one entry near `1/128` and nine entries near `-1/1280`. The ratio is ten, and one
scale for the row with eight codes cannot hold a ratio of ten and a small value at the same time. 71 percent
of the entries go to zero, so the gradient that reaches layer 1 points 27 degrees away from the true
gradient, and it does that on 31,250 consecutive steps.

On MNIST the same structure was present and `dA1` kept a cosine of 0.9932. The difference is the dataset:
MNIST-1D needs 31,250 steps of fine gradient and MNIST reaches its ceiling early.

**The caveat, and it matters for the engine.** The test fixture has layers 16 wide and 10 wide. A block of 16
along N is therefore the full width of a layer. A layer 4,096 wide would give 256 blocks along N, and each
block would span 16 neighbouring outputs, not a whole layer. The collapse is measured at N at or below 16, and
it is not yet known where it stops. That is the open question, and it is recorded as ADR-031.

### Cost measured, both datasets

| Cell | MNIST, 40 runs | MNIST-1D, 40 runs |
|-|-:|-:|
| L4-D, 12 s and 86 s per run | 8 min | 57 min |
| L4-DA, 116 s and 877 s per run | 78 min | 9 h 45 min |
| L4-DAW, 120 s and 909 s per run | 80 min | 10 h 06 min |
| **total, 240 runs** | **2 h 46 min** | **20 h 48 min** |

The MNIST-1D training set is 400,000 samples of 784 inputs, so one epoch is 3,125 steps against 469 on
MNIST. Almost all of the cost of the DA and DAW cells is the 255-code scale search over the input `X`, which
the D cell does not do.

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
