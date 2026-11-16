# Experiment L3: FP4 backprop, everything else float

## Objective

Does training survive when the operands of the backward GEMMs are rounded to FP4, while the weights,
the activations, the optimizer and the loss stay float?

The ladder asks "how many updates vanish, and what is the discretization error". This document turns
that into a run with numbers in it.

## One correction, written down before the run

In the conversation before this document I said the weight-blocking experiment was a prerequisite of
L3. **That was wrong.**

| Rung | Weights | So the weight blocking | So the second rounding of A |
|-|-|-|-|
| L3 | float | does not exist: no weight is on a grid | does not exist: A is never stored as FP4 |
| L5 | FP4 | exists, and `experiment_block.md` settles it | exists, and it is the route question |

So this rung measures one rounding of each backward operand, on one grid, with no second rounding
anywhere. The route question and the weight-blocking question are both L5 scope.

## Terms

| Term | Meaning |
|-|-|
| gradient operand | a tensor that enters a GEMM of the backward pass as an input. For `dW = A.T @ dZ` they are `A.T` and `dZ`. |
| contraction axis | the axis a GEMM sums over. An FP4 block must lie along it. |
| RTN | round to nearest. On a tie, round to the even code. Rung L4 replaces this with stochastic rounding. |
| STE | straight-through estimator. The backward pass treats a quantizer as the identity for the gradient. L3 has no quantizer in the forward pass, so the STE is not used here. |
| vanished update | a gradient entry that rounds to payload code 0. The weight it belongs to gets no update from that entry in that step. |
| effective update | the update Adam applies to a float weight. In L3 the weight is float, so the update is never lost: it is only wrong. |

## What L3 changes, exactly

The backward pass of one layer has two GEMMs. L3 rounds the operands of the weight-gradient GEMM only:

```text
dW3 = A2.T @ dZ3        round A2 along the batch, round dZ3 along the batch
dW2 = A1.T @ dZ2        round A1 along the batch, round dZ2 along the batch
dW1 = X.T  @ dZ1        round X  along the batch, round dZ1 along the batch
dA2 = dZ3 @ W3.T        float. This is rung L5, because it needs W3 on a grid
dA1 = dZ2 @ W2.T        float. Same reason
```

Each operand is rounded once, from its float value, with one 8-bit scale per 16 samples. Nothing is
stored as FP4, so there is no second rounding.

Why the weight-gradient GEMM is the one to test: it is the GEMM that produces the number the
optimizer uses. The gradient-down GEMM only changes how the error is shared between layers, and it
cannot be tested without putting the weights on a grid.

## The two cells

| Cell | Operands rounded | Question |
|-|-|-|
| L3-D | `dZ3`, `dZ2`, `dZ1` only | is the loss gradient the part that must stay wide? |
| L3-DA | `dZ` and the activations: `X`, `A1`, `A2` | does the activation operand add damage? |

L3-DA is the full form of the rung. L3-D is the cheap form, and it separates the two operands. The
activations are the larger tensor: `X` is 49 blocks per sample, and the encoder cost of `X` is the
reason the input is stored pre-encoded (`experiment_L2.md`, cost section).

## Metrics

The gradient cosine is measured on the **first batch of each epoch**, not on every step. The second float
gradient that the cosine needs costs one extra backward pass, and one batch per epoch keeps the epochs
comparable and the cost small.

| Metric | Definition | Why |
|-|-|-|
| final test accuracy, paired | the deployed model against the L0 row of the same seed and width | the rung answer |
| vanished share | share of gradient entries that round to payload code 0, per matrix, mean over the steps | the precursor of the frozen-weight problem |
| gradient cosine | cosine between the FP4 gradient and the float gradient of the same step, per matrix | the size of the error, before any training effect |
| per-epoch curves | train and test loss and accuracy, four curves, drawn as a figure | the shape of the damage, not only the end point |

## Decision rule (pre-registered, written before the run)

The rule of `l2_sweep.paired`, unchanged. L3 differs from L0 only when both hold:

1. The sign of the paired difference agrees with the sign of its mean in at least 8 of 10 seeds.
2. The absolute mean is more than twice its standard error.

No decision is a legal result. A difference smaller than the seed spread does not select anything.

## Run matrix and cost

| Item | Value |
|-|-|
| reference | the L0 rows that already exist. No new runs. |
| cells | 2 |
| configurations | the four of `qat.CONFIGS` |
| seeds | the ten of `l2_sweep.SEEDS` |
| epochs, batch | 10, 128 |
| runs per cell | 40 |
| time per run, measured | **11 s for L3-D and 105 s for L3-DA on MNIST**, from the run file times of the seed 42 smoke run. The 23 s quoted before the run was the L2 figure and it was wrong for both cells |
| time per cell, measured | 7 minutes for L3-D, 70 minutes for L3-DA |
| time for both cells | about 77 minutes on MNIST |

**Why L3-DA costs ten times L3-D.** The input `X` is 128 by 784 per step, and its block lies along the
batch, so it cannot be quantized once before training: the block of 16 samples changes with every shuffle.
The quantizer builds a distance array of (8 blocks, 784 columns, 256 scales) for every step, which is
1.6 million values. L3-D never touches a tensor that wide.

MNIST-1D runs only if a rule fires on MNIST, the same trigger as the block experiment.

## Expected effect size, before the run

`python/second_rounding_probe.py` measured the operands of the weight-gradient GEMM rounded once along
the batch, on saved L0 weights, 5 seeds, 4 batches each. It ran with FP4 weights, because it belongs to
L5. L3 has float weights, so the numbers below are the expected size, not the result.

| Dataset | Layer | Gradient error against the float gradient | Cosine |
|-|-|-:|-:|
| MNIST | 1, the 784-wide input | 6.8 to 9.4% | 0.9937 to 0.9955 |
| MNIST | 2 and 3, 16 wide | 7.3 to 9.7% | 0.9891 to 0.9940 |
| MNIST-1D | 1 | 10.8 to 12.1% | 0.9815 to 0.9863 |
| MNIST-1D | 2 and 3 | 6.2 to 8.4% | 0.9893 to 0.9961 |

So the gradient is 6 to 12% wrong in norm, and 5 to 11 degrees off in direction. The prediction on
record: L3-DA loses less than 0.5 accuracy points against L0 on MNIST, and the loss is not separable
from the seed spread. If that holds, the gradient width is not the binding constraint of the project,
and L5's weight and activation quantization is.

**A units correction to the stored runs, written down instead of hidden.** The first eight runs of the
sweep, seed 42 of both cells and all four configurations, were written with the two accuracy lists stored as
fractions instead of percentages. The report printed them as percentages, so the seed 42 report line was
wrong by a factor of 100. The stored lists were multiplied by 100 once, and `l3.py` now stores percentages.
No other value changed and no run was repeated.

## Results, MNIST, 10 seeds

80 runs, both cells, four configurations, ten epochs. Finished 2026-11-16. The output is
`out/l3_mnist.log`, the runs are `out/l3/mnist/`, and the figure is
`docs/experiment_L3_mnist_curves.png`.

Final test accuracy, percent, mean of the four configurations:

| Row | a6-e6m2 | f6-e6m2 | a6-e5m3 | f6-e5m3 | mean |
|-|-:|-:|-:|-:|-:|
| L0, the reference, no rounding | 95.01 ±0.32 | 94.99 ±0.23 | 95.03 ±0.27 | 94.99 ±0.31 | 95.01 |
| **L3-D**, deltas rounded | 95.01 ±0.32 | 94.99 ±0.23 | 95.03 ±0.27 | 94.99 ±0.31 | **95.01** |
| **L3-DA**, deltas and activations rounded | 94.98 ±0.27 | 95.07 ±0.12 | 95.00 ±0.30 | 94.98 ±0.30 | **95.01** |

Paired against L0, same seed. The rule is the pre-registered one:

| Cell | Config | Mean difference | Standard error | Seeds that agree | Decision |
|-|-|-:|-:|-:|-|
| L3-D | a6-e6m2 | 0.00 | 0.07 | 4 of 10 | no decision |
| L3-D | f6-e6m2 | −0.03 | 0.07 | 6 of 10 | no decision |
| L3-D | a6-e5m3 | +0.02 | 0.05 | 6 of 10 | no decision |
| L3-D | f6-e5m3 | −0.02 | 0.06 | 6 of 10 | no decision |
| L3-DA | a6-e6m2 | −0.03 | 0.05 | 4 of 10 | no decision |
| L3-DA | f6-e6m2 | +0.06 | 0.08 | 6 of 10 | no decision |
| L3-DA | a6-e5m3 | −0.01 | 0.07 | 5 of 10 | no decision |
| L3-DA | f6-e5m3 | −0.03 | 0.06 | 6 of 10 | no decision |
| L3-DA against L3-D | a6-e6m2 | −0.03 | 0.04 | 6 of 10 | no decision |
| L3-DA against L3-D | f6-e6m2 | +0.09 | 0.05 | 6 of 10 | no decision |
| L3-DA against L3-D | a6-e5m3 | −0.03 | 0.06 | 4 of 10 | no decision |
| L3-DA against L3-D | f6-e5m3 | −0.01 | 0.02 | 5 of 10 | no decision |

**The rule did not select L3-DA over L0 and it did not reject it.** The largest mean difference in the whole
table is 0.09 points, and its standard error is 0.05. So the statement this rung supports is an
**equivalence bound**: on MNIST, rounding the operands of the weight-gradient GEMM to FP4 moves the final
accuracy by less than 0.2 points, at two standard errors, in either direction.

The rounded operands, last epoch, mean over every seed and configuration:

| Cell | Tensor | SQNR, dB | Vanished share | Gradient cosine, first batch |
|-|-|-:|-:|-:|
| L3-D | `dZ3` | 25.29 | 80.63% | W1 0.9981, W2 0.9980, W3 0.9978 |
| L3-D | `dZ2` | 23.58 | 66.64% | |
| L3-D | `dZ1` | 23.94 | 66.26% | |
| L3-DA | `dZ3` | 25.42 | 80.44% | W1 0.9965, W2 0.9940, W3 0.9940 |
| L3-DA | `dZ2` | 23.40 | 66.03% | |
| L3-DA | `dZ1` | 23.89 | 65.85% | |
| L3-DA | `X` | 23.98 | 4.00% | |
| L3-DA | `A1` | 20.70 | 4.03% | |
| L3-DA | `A2` | 20.68 | 3.84% | |

### What the data says

| Finding | Evidence |
|-|-|
| **the gradient width is not the binding constraint on MNIST** | both cells land on the float accuracy, 95.01 against 95.01, and the largest paired difference is 0.09 points |
| the activation operands add almost nothing | L3-DA against L3-D is −0.03 to +0.09, no decision in all four configurations |
| **most delta entries send no update, and it does not matter** | 66% to 81% of the delta entries round to payload code 0, and the model is unchanged. The vanished entries are the small ones relative to their block maximum, and Adam divides each coordinate by `sqrt(v)` |
| the vanished share does not follow the noise | `A1` has 20.70 dB of SQNR and 4% vanished. `dZ1` has 23.89 dB and 66% vanished. The share depends on the spread inside a block, the dB depends on the error norm |
| the cosine moves with the number of rounded operands | 0.998 with the deltas alone, 0.994 with the deltas and the activations |
| the prediction on record held | the expected effect was less than 0.5 points and not separable, from `second_rounding_probe.py`. It measured 0.09 and no decision |
| the curves are indistinguishable by eye | `docs/experiment_L3_mnist_curves.png`. The train loss curves of the two cells lie on each other for ten epochs |

### What this means for the ladder

| Consequence | Detail |
|-|-|
| the answer to the ladder's question | most updates vanish, and on MNIST that is free. The damage of FP4 training does not come from the backward operands |
| L4 has no gap to recover on MNIST | L3's difference from L0 is 0.09 points at most. Stochastic rounding can only recover what round to nearest loses. **This is an inference, not a measurement.** The L4 trigger should be MNIST-1D, or L5 |
| the next constraint is the forward pass | L5 puts the weights and the activations on a grid, and the frozen-weight problem starts there. That is where `experiment_L2.md` already showed a cost: L2 94.61 to 94.80 against 95.01 |
| MNIST-1D was not run | the protocol says it runs only if a rule fires on MNIST, and no rule fired |

## What L3 does not settle

| Limit | Where it belongs |
|-|-|
| the frozen weights, where a float master moves and the FP4 value does not | L5 and L6. Needs FP4 weights. |
| the second rounding of A and dZ | L5. Needs a stored FP4 forward tensor. |
| stochastic rounding against RTN | L4, which is this rung with one change |
| the error of the optimizer state | L6, in both forms, see `experiment_plan.md` |

## Settled before the run

| Item | Value | Where it comes from |
|-|-|-|
| float width | **float32**, for the weights, the activations, the optimizer and the arithmetic | decided by Dzemo. The ladder rows L2 to L5 now say float32 |
| reference row | the L0 row in float32, 95.01 on MNIST and 64.98 on MNIST-1D | `out/` L0 weight files are float32 |
| cells | both, L3-D and L3-DA | decided by Dzemo. They cost 15 minutes each on MNIST |
| order | MNIST first, then the trigger for MNIST-1D | decided by Dzemo |

The float64 L0 numbers, 95.18 on MNIST and 65.11 on MNIST-1D, stay as the record of the width
experiment in `experiment_L2.md`. They are not the reference of this rung, because a float32 run
against a float64 row would mix the width with the change under test.
