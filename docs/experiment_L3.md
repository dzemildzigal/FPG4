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
| time per run | 23 s on MNIST, 167 s on MNIST-1D, from the run file times |
| time per cell | 15 minutes on MNIST, 1 hour 51 minutes on MNIST-1D |
| time for both cells | 31 minutes on MNIST, 3 hours 42 minutes on MNIST-1D |

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

The vanished share is not measured yet. It needs the L3 code.

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
