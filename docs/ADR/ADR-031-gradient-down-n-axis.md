# ADR-031: A 1D block along the contraction axis N is rejected at small N

**Status**: Pending. Measured, and the ladder cannot pass L5 until a position is chosen. 2026-11-17.

**Date**: 2026-11-17

**Deciders**: Dzemo, with the measurements of rung L4 and of `python/daw_collapse_probe.py`.

## Context and problem statement

The backward pass has two GEMM families. The weight-gradient GEMM `dW = A.T @ dZ` contracts over the batch.
The gradient-down GEMM `dA = dZ @ W.T` contracts over the layer output width N. Both contract over a
different axis, so both need their operands blocked along a different axis, and one 1D block cannot serve
both from one copy of `dZ`. That was measured in L3 and it is why the delta needs two copies.

The engine has one legal block size, 16 values per scale. Along the batch that is 16 samples of the same
output. Along N that is 16 different outputs of one sample.

Rung L4 measured what the second blocking costs. On MNIST-1D, rounding the operands of both families takes
the network from 64.98 to 36.03 points.

## Decision drivers

| Driver | Number |
|-|-|
| Accuracy at stake | 17 to 42 points on MNIST-1D, 0.2 points on MNIST |
| The layer widths of the test fixture | 16 and 10, so one block along N is a whole layer |
| The engine must generalise to layers thousands of values wide | that is the reason FPG4 is not an MLP engine |
| Area of a finer scale | a 4 by 4 tile costs four term paths, 4,716 to 5,196 LUT6, and the clock falls to about 96 MHz |
| The store | a finer tile costs 0.36 percent of the store, so bits are not the constraint |

## Considered options

1. **Keep the gradient-down GEMM in float.** The FP4 backward pass then covers the weight-gradient GEMM
   only. This is the L4-DA cell, and it is clean on both datasets.
2. **Round the delta along N at N of 16 or less with four values per scale** instead of 16. The scale then
   covers four outputs, and the term chain costs what ADR-026 measures for a 4 by 4 tile.
3. **Keep two copies of the delta, one per axis, both with a block of 16** and accept the error. Measured
   and rejected by L4.
4. **Extend the fixture to a wider layer** before deciding, and find where the collapse stops. A 784-64-64-10
   network has 64 values per layer, so one block along N covers a quarter of a layer instead of all of it.
5. **Change what is rounded.** `dZ3 = (P - Y) / B` has one entry near `1/128` and nine near `-1/1280`, a ratio
   of ten, inside one row. An operation that equalises the row before the block, and its exact inverse after,
   would fit a row into eight codes. Not designed.

## Decision outcome

No decision. Rung L5 cannot choose a backward operand scheme until this record is decided.

The measurement stands on its own: **a 1D block along the contraction axis N is rejected at N of 16 or less,
on MNIST-1D, at 10 seeds, for both rounding rules.**

## Consequences

**Positive**

- The engine knows that the axis of the contraction sets the cost twice: once in the lane, through the term
  chain, and once in the accuracy, through what shares a scale. ADR-024 and ADR-026 hold the first. This
  record holds the second.
- The rejection is measured with round to nearest as well as stochastic rounding, so the cause is the
  blocking and not the rounding rule. `python/daw_collapse_probe.py` runs four variants of one seed and
  separates the two operands: the delta alone costs 58.23 to 28.77, the weight alone costs nothing.

**Negative**

- L5 waits. The three dials of L5 were already open. This adds a fourth.
- The measurement is at N of 16 and 10 only. A network with a wide layer has not been measured, so the
  record cannot state where the collapse stops. Option 4 is the way to find out, and it changes the test
  fixture, which is Dzemo's decision and not mine.

## Pros and cons of the options

| Option | Pro | Con |
|-|-|-|
| 1, keep it float | measured clean on both datasets. No new area | the backward pass is only half FP4, so the goal of no master weights and no float tensor is not reached |
| 2, four values per scale along N | keeps the whole backward pass in FP4 | four term paths in the lane, and the clock estimate falls to about 96 MHz |
| 3, two copies at 16 | no lane change | measured: loses 28 points on MNIST-1D |
| 4, widen the fixture | finds the real boundary, and a wide layer may need nothing at all | changes the ruler, which every number in `docs/` is measured on |
| 5, equalise the row | could be nearly free in area | not designed, and it adds an exact inverse to the datapath |

## Links

- `docs/experiment_L4.md`, the results of both datasets and the four-variant probe
- `docs/experiment_L3.md`, the first two blocks of the backward pass, and the second rounding of the delta
- ADR-024, the axis of the blocking
- ADR-026, the term chain, which prices every option that uses a finer scale
- ADR-027, the scope of the FP4 backward pass, now measured on both datasets
