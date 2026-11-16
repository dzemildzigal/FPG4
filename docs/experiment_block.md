# Experiment: weight blocking

## Objective

One question, one variable: **does the block shape of the weight matrices change what FP4-aware training learns?**

**Answer, in one line**: yes on MNIST-1D and no on MNIST. The two 1D copies, the 16 by 16 tile and the 8 by
8 tile are all decided losses on MNIST-1D, by -2.54, -2.81 and -1.51 accuracy points. The 4 by 4 tile is
the only form that is not separable from the reference on either dataset, and it costs four term paths per
lane, 4,716 to 5,196 LUT6.

Rung L2 already runs with one scale per 16 inputs and one copy of each matrix. This experiment keeps
everything else of L2 and changes only the block shape, and the number of copies, of the weight
matrices.

## Terms

| Term | Meaning |
|-|-|
| block | 16 values that share one scale. The scale sets the grid the values are rounded onto. |
| tile | a block of `a` by `b` values. A 16 by 16 tile is one block of 256 values. |
| copy | one quantized version of a weight matrix. Two copies of one matrix are built from the same float master, so they hold 62 to 79% different values. |
| drift | the error a step makes when the loss sees one copy and the gradient is computed with the other copy. It repeats every step, so it does not average to zero. |
| arm | one choice of block shape under test. |
| reference | rung L2, the arm that already existed. It uses one copy, blocked along K. |
| paired seeds | the same seed, the same start weights and the same batch order in two runs. Only the arm differs. Paired differences remove most of the seed noise. |
| SQNR | signal to quantization noise ratio, in dB. Higher is better. |

## Why this question exists

A block-scaled GEMM needs the 16 values of a block to lie along the axis the GEMM sums over. One
training step has three GEMMs per layer and they sum over three different axes:

| GEMM | formula | sums over | one block of an operand holds |
|-|-|-|-|
| forward | `Z = A @ W` | the inputs K | 16 inputs of one output |
| gradient down | `dA = dZ @ W.T` | the outputs N | 16 outputs of one input |
| weight gradient | `dW = A.T @ dZ` | the samples | 16 samples of one feature |

So one matrix needs one blocking for the forward pass and another for the gradient-down pass. Two
ways exist to satisfy both:

| way | shape | property |
|-|-|-|
| two copies | 16 by 1 for the forward pass, 1 by 16 for the backward pass | both copies are legal blocks. The values are not equal, so the step drifts. |
| one square tile | 16 by 16 | one scale per 256 weights serves both passes, and the values are equal by construction |

The published positions, quoted:

- NVIDIA, arXiv 2509.25149 Appendix E.5, on a 1.2B model: two 1D copies "negatively impacts the loss throughout training", and a 16 by 16 tile is better "despite having a larger block granularity".
- NVIDIA Transformer Engine, NVFP4 page: "NVFP4 uses 2D scaling for weights and 1D scaling for activations and gradients". The reason given: "to ensure that rowwise and columnwise quantized tensors are numerically equivalent".
- OCP MX, arXiv 2310.10537 section 4.1: with 1D blocks, `W` and `W.T` "must be stored as two separate tensors".

The question for this project is not what NVIDIA does. It is whether the cost shows up at 13,002
parameters, and how large it is.

## Run matrix

Six arms. All run with the four configurations and the ten seeds of L2. Six arms, four configurations,
ten seeds is 240 runs per dataset. Datasets run: MNIST, all six arms, 240 runs. MNIST-1D, four arms, 160
runs: A and B2 because the MNIST trigger fired, then C2 and D2 because B2 was rejected and the dial had to
be walked down.

| Arm | What changes | Matrices touched | Share of the 13,002 weights | MNIST | MNIST-1D |
|-|-|-|-:|-|-|
| reference | nothing. Rung L2 | none | 0% | 40 runs | 40 runs |
| A | a second copy blocked along N, used by the backward pass | W2, W3 | 3.2% | 40 runs | 40 runs |
| B2 | one 16 by 16 tile, the same value in both passes | W2, W3 | 3.2% | 40 runs | 40 runs |
| B3 | one 16 by 16 tile, the same value in both passes | W1, W2, W3 | 100% | 40 runs | not run |
| D2 | one 8 by 8 tile, the same value in both passes | W2, W3 | 3.2% | 40 runs | 40 runs |
| C2 | one 4 by 4 tile, the same value in both passes | W2, W3 | 3.2% | 40 runs | 40 runs |
| C3 | one 4 by 4 tile, the same value in both passes | W1, W2, W3 | 100% | 40 runs | not run |

**Why B3 and C3 were not run on MNIST-1D.** They put the tile on W1 as well, and W1 has no gradient-down
GEMM, so it has no second use for a square tile. On MNIST both forms were equal to their W2/W3 partner
(B3 94.70 against B2 94.67, C3 94.71 against C2 94.73), so the extra 40 runs would not have changed the
tile shape question. They test a different question: whether the whole store can go into tiles.

**Why W1 appears in no arm A.** W1 has no gradient-down GEMM: there is no layer below it, and the
input X is data. So it has no transposed use, so it has no second copy. Arm A cannot touch it.

**Why B and C come in two forms.** The W2/W3 form changes 3.2% of the weights. A null result from
it would not separate "the blocking is harmless" from "96.8% of the weights never saw the change".
The all-matrix form is the same question with the whole store in tiles.

## Scope

Everything not in the table above stays as it is in L2:

| thing | value |
|-|-|
| forward pass | FP4 weights, decoded to float32 |
| backward pass | float32, straight-through estimator |
| optimizer | Adam, float32 master, float32 `m` and `v` |
| activations | never quantized |
| biases | never quantized |
| epochs, batch | 10, 128 |
| seeds | 42, 4242, 424242, 123, 456, 789, 1010, 2020, 3030, 4040 |
| configurations | `a6-e6m2`, `f6-e6m2`, `a6-e5m3`, `f6-e5m3` |

The reported accuracy is the deployed quantized model in every arm.

## Decision rule (pre-registered, written before the run)

The rule of `l2_sweep.paired`, unchanged. An arm beats the reference only when both hold:

1. The sign of the paired difference agrees with the sign of its mean in at least 8 of 10 seeds.
2. The absolute mean is more than twice its standard error.

"No decision" is a legal result. It does not select an arm.

## What is logged

One JSON per run in `out/block/<dataset>/`, written when the run ends:

| key | content |
|-|-|
| `test_acc`, `test_loss` | per epoch, over the whole test set, the deployed model |
| `train_acc`, `train_loss` | per epoch, over the training batches |
| `sqnr` | per matrix, dB against the master, at the end of the epoch |
| `counters` | the four scale counters, summed over the epoch |
| `frozen_steps` | steps where the quantized matrix was bit-identical to the step before |
| `stale_elements` | share of elements whose FP4 value did not move, mean over the steps |
| `drift` | arm A only. See the drift section below |

## Cost

Per training step, taken from `python/blocks_per_step.py` and `docs/datapath.md`. The step is
211,456 block-MACs, which is 52,864 cycles on 4 lanes. The encoder is 4 cycles per block.

| Arm | Extra area | Extra cycles per step | Extra memory |
|-|-|-:|-:|
| A | none | 128, which is 0.24% | a second copy of W2 and W3, 2,304 bits, 0.045% of the Z020 BRAM |
| B2 | none | 104, which is 0.20% | none |
| B3 | none | 104 plus the W1 tiles | none |
| D2 | parallel: 1,572 to 1,732 LUT6, which is 3.0 to 3.3% of the Z020. Serial: none | parallel: none. Serial: 1,856, which is 3.5% | none |
| C2 | parallel: 4,716 to 5,196 LUT6, which is 8.9 to 9.8% of the Z020. Serial: none | parallel: none. Serial: up to 5,568, which is 10.5% | none |
| C3 | parallel: the C2 area. Serial: none | parallel: none. Serial: the C2 cycles plus 75,264 for the forward GEMM of W1, which is +142% | none |

The 104 cycles of B2 and B3 are a second pass over the tile for its maximum. A 16 by 16 tile holds
256 values, so the encoder reads 16 blocks of 16 values twice instead of once. No 256-value buffer
is needed, because the first pass keeps one running maximum.

**The D2 and C2 rows are different. They are term path costs, and a term path can be paid in two
ways.** The terms are defined in the section "The tile and the lane" below. One lane with one term path
takes 2 cycles per block pair at 8 by 8 and 4 cycles at 4 by 4 (serial, no area). A lane with 2 or 4
term paths takes 1 cycle (parallel, area). The two prices are alternatives, and they are never added.
**An earlier version of this table gave 2,784 cycles and 5.2% for D2. That was wrong.** It scaled the
C2 figure by 2 over 4. The extra cycles scale with the number of extra passes, which is 1 at 8 by 8 and 3
at 4 by 4, so the figure is 1,856 and 3.5%. `python/blocks_per_step.py` prints every number of this
table.

## Results, MNIST, 10 seeds

Reference, rung L2, from `out/w/mnist`:

| Config | Test accuracy, percent | Test loss | Train loss |
|-|-:|-:|-|
| a6-e6m2 | 94.72 | 0.1798 | not stored |
| f6-e6m2 | 94.77 | 0.1799 | not stored |
| a6-e5m3 | 94.61 | 0.1834 | not stored |
| f6-e5m3 | 94.80 | 0.1799 | not stored |
| **mean** | **94.73** | **0.1808** | |

The L2 runs did not store the training loss. So every train loss below has no reference partner.
The figure `docs/experiment_block_mnist_curves.png` shows the four curves of every arm, and the
reference appears in it with test curves only.

### Final test accuracy, percent, mean of 10 seeds

| Arm | a6-e6m2 | f6-e6m2 | a6-e5m3 | f6-e5m3 | Mean |
|-|-:|-:|-:|-:|-:|
| reference | 94.72 | 94.77 | 94.61 | 94.80 | 94.73 |
| A | 94.53 | 94.20 | 94.56 | 94.01 | **94.33** |
| B2 | 94.31 | 94.78 | 94.75 | 94.84 | 94.67 |
| B3 | 94.67 | 94.45 | 94.77 | 94.92 | 94.70 |
| D2 | 94.74 | 94.64 | 94.70 | 94.56 | 94.66 |
| C2 | 94.64 | 94.74 | 94.69 | 94.85 | 94.73 |
| C3 | 94.74 | 94.63 | 94.82 | 94.66 | 94.71 |

### Final test loss, mean of 10 seeds

| Arm | a6-e6m2 | f6-e6m2 | a6-e5m3 | f6-e5m3 | Mean |
|-|-:|-:|-:|-:|-:|
| reference | 0.1798 | 0.1799 | 0.1834 | 0.1799 | 0.1808 |
| A | 0.1882 | 0.1973 | 0.1895 | 0.2061 | **0.1953** |
| B2 | 0.1939 | 0.1818 | 0.1790 | 0.1763 | 0.1827 |
| B3 | 0.1817 | 0.1888 | 0.1790 | 0.1745 | 0.1810 |
| D2 | 0.1801 | 0.1835 | 0.1826 | 0.1880 | 0.1835 |
| C2 | 0.1804 | 0.1807 | 0.1812 | 0.1788 | 0.1803 |
| C3 | 0.1782 | 0.1832 | 0.1772 | 0.1812 | 0.1800 |

### Final train loss, mean of 10 seeds

| Arm | a6-e6m2 | f6-e6m2 | a6-e5m3 | f6-e5m3 | Mean |
|-|-:|-:|-:|-:|-:|
| A | 0.1620 | 0.1676 | 0.1609 | 0.1761 | **0.1666** |
| B2 | 0.1630 | 0.1493 | 0.1488 | 0.1441 | 0.1513 |
| B3 | 0.1521 | 0.1585 | 0.1499 | 0.1468 | 0.1518 |
| D2 | 0.1497 | 0.1530 | 0.1517 | 0.1578 | 0.1530 |
| C2 | 0.1524 | 0.1505 | 0.1526 | 0.1482 | 0.1509 |
| C3 | 0.1489 | 0.1531 | 0.1479 | 0.1521 | 0.1505 |

### The decision rule, applied to every pair

Final test accuracy against the reference, 10 paired seeds:

| Arm | Config | Mean difference | Standard error | Sign agrees | Rule 1 | Rule 2 | Winner |
|-|-|-:|-:|-:|-:|-:|-|
| A | a6-e6m2 | -0.195 | 0.080 | 8 of 10 | yes | yes | reference |
| A | f6-e6m2 | -0.566 | 0.130 | 10 of 10 | yes | yes | reference |
| A | a6-e5m3 | -0.046 | 0.186 | 2 of 10 | no | no | none |
| A | f6-e5m3 | -0.785 | 0.302 | 9 of 10 | yes | yes | reference |
| B2 | a6-e6m2 | -0.419 | 0.230 | 7 of 10 | no | no | none |
| B2 | f6-e6m2 | 0.010 | 0.134 | 5 of 10 | no | no | none |
| B2 | a6-e5m3 | 0.141 | 0.130 | 6 of 10 | no | no | none |
| B2 | f6-e5m3 | 0.043 | 0.072 | 7 of 10 | no | no | none |
| B3 | a6-e6m2 | -0.056 | 0.053 | 7 of 10 | no | no | none |
| B3 | f6-e6m2 | -0.317 | 0.343 | 6 of 10 | no | no | none |
| B3 | a6-e5m3 | 0.161 | 0.110 | 6 of 10 | no | no | none |
| B3 | f6-e5m3 | 0.119 | 0.105 | 6 of 10 | no | no | none |
| D2 | a6-e6m2 | 0.020 | 0.057 | 4 of 10 | no | no | none |
| D2 | f6-e6m2 | -0.128 | 0.125 | 6 of 10 | no | no | none |
| D2 | a6-e5m3 | 0.091 | 0.167 | 5 of 10 | no | no | none |
| D2 | f6-e5m3 | -0.243 | 0.142 | 6 of 10 | no | no | none |
| C2 | a6-e6m2 | -0.082 | 0.133 | 5 of 10 | no | no | none |
| C2 | f6-e6m2 | -0.033 | 0.040 | 6 of 10 | no | no | none |
| C2 | a6-e5m3 | 0.077 | 0.081 | 7 of 10 | no | no | none |
| C2 | f6-e5m3 | 0.046 | 0.070 | 7 of 10 | no | no | none |
| C3 | a6-e6m2 | 0.014 | 0.118 | 6 of 10 | no | no | none |
| C3 | f6-e6m2 | -0.142 | 0.094 | 9 of 10 | yes | no | none |
| C3 | a6-e5m3 | 0.212 | 0.055 | 9 of 10 | yes | yes | C3 |
| C3 | f6-e5m3 | -0.143 | 0.086 | 8 of 10 | yes | no | none |

24 comparisons. One fires: C3 in `a6-e5m3`, plus 0.212 accuracy points. One cell in 24 at the two
standard error threshold is what a false positive looks like. C3's mean over the four configs is
94.71, against 94.73 for C2, 94.70 for B3 and 94.66 for D2, so the cell does not repeat.

**No tile shape separates from the reference on MNIST.** 19 of the 20 tile cells are undecided, and the
one that fires is inside the false positive rate. The four tile means span 0.07 accuracy points:
C2 94.73, B3 94.70, B2 94.67, D2 94.66.

### The tile price before training

`python/tile_probe.py`, saved L0 weights, 10 seeds, no training. The number is the SQNR of the 1D block
minus the SQNR of the tile, so a positive number is a loss. The output of the run is in
`out/tile_probe_price.json`. Each cell is 40 values: 4 configurations times 10 seeds.

| Dataset | Matrix | 4 by 4 | 8 by 8 | 16 by 16 |
|-|-|-:|-:|-:|
| MNIST | W1 | -0.00 [-0.24, 0.11] | 1.15 [0.68, 1.69] | 1.81 [1.20, 2.48] |
| MNIST | W2 | 0.11 [-0.88, 1.13] | 1.20 [-0.29, 2.27] | 1.52 [0.30, 2.90] |
| MNIST | W3 | -0.18 [-1.01, 1.08] | 0.67 [-1.10, 2.41] | 1.02 [-0.93, 2.64] |
| MNIST-1D | W1 | -0.03 [-0.22, 0.21] | 0.90 [0.36, 1.46] | 1.32 [0.62, 2.12] |
| MNIST-1D | W2 | -0.31 [-1.20, 1.02] | 1.33 [0.48, 2.82] | 2.67 [0.67, 4.55] |
| MNIST-1D | W3 | -0.40 [-1.84, 1.41] | 1.03 [-1.21, 2.43] | 1.97 [0.22, 4.00] |

Each cell is the mean and the full range of 40 values, over the four configurations and the ten seeds.
**The first version of this table was wrong in two ways and both are corrected here.** It carried the
numbers of one configuration, `a6-e6m2`, and its caption said the range was over the four
configurations. It also had no 8 by 8 column, because arm D2 did not exist yet.

Collapsed to the two matrices that the tile touches, W2 and W3:

| Dataset | 4 by 4 | 8 by 8 | 16 by 16 |
|-|-:|-:|-:|
| MNIST, W2 and W3, mean | -0.03 | 0.94 | 1.27 |
| MNIST-1D, W2 and W3, mean | -0.36 | 1.18 | 2.32 |

- A 4 by 4 tile costs nothing, and in half the cells it gains a little. That is expected: 4 values per
  scale is finer than 16 along K, so the blocks are smaller along both axes.
- A 16 by 16 tile costs 1.02 to 2.67 dB, and it costs more where the matrix has fewer rows: W2 is 16 by
  16, so it is exactly one tile and the scale becomes a per-matrix scale.
- An 8 by 8 tile lands between the two, at 0.67 to 1.33 dB.
- `four6` is worth nothing on a 16 by 16 tile: it keeps `amax/4` in 0.0 to 1.2% of the tiles, against 26
  to 46% of the 4 by 4 tiles and 0 to 32% of the 8 by 8 tiles. So `a6` and `f6` produce almost the same
  weights on a 16 by 16 layout, and they differ on a 4 by 4 layout.

### The drift telemetry of arm A

Once per epoch, on a fixed probe batch: one forward pass with the forward copy, then the backward
pass twice on the same cache, once with each copy.

| Config | cos of the two W2 copies | cos of the two W3 copies | cos dA2, from the W3 copy | cos dA1, from the W2 copy | cos dW2 | cos dW1 | Epoch |
|-|-:|-:|-:|-:|-:|-:|-:|
| a6-e6m2 | 0.9920 | 0.9924 | 0.9915 | 0.9863 | 0.9917 | 0.9841 | 1 |
| a6-e6m2 | 0.9914 | 0.9923 | 0.9915 | 0.9857 | 0.9921 | 0.9849 | 10 |
| f6-e6m2 | 0.9927 | 0.9928 | 0.9923 | 0.9871 | 0.9940 | 0.9871 | 1 |
| f6-e6m2 | 0.9922 | 0.9924 | 0.9920 | 0.9884 | 0.9930 | 0.9868 | 10 |
| a6-e5m3 | 0.9913 | 0.9912 | 0.9907 | 0.9849 | 0.9906 | 0.9844 | 1 |
| a6-e5m3 | 0.9897 | 0.9908 | 0.9905 | 0.9833 | 0.9893 | 0.9802 | 10 |
| f6-e5m3 | 0.9920 | 0.9927 | 0.9920 | 0.9857 | 0.9936 | 0.9875 | 1 |
| f6-e5m3 | 0.9917 | 0.9920 | 0.9925 | 0.9869 | 0.9931 | 0.9860 | 10 |

`cos dW3` is 1.0000 in every run, and it must be: `dW3 = A2.T @ dZ3` reads A2 and dZ3, and never
reads W3. The telemetry was checked against that identity.

Share of elements where the two copies hold a different value, last epoch, mean of 10 seeds:

| Config | W2 | W3 |
|-|-:|-:|
| a6-e6m2 | 64.2% | 61.6% |
| f6-e6m2 | 66.6% | 64.6% |
| a6-e5m3 | 76.7% | 75.7% |
| f6-e5m3 | 78.4% | 78.9% |

## MNIST-1D

Four arms, four configurations, ten seeds each, 160 runs. Arms A and B2 ran on 2026-11-15. Arms C2 and
D2 ran on 2026-11-16, after arm B2 was rejected, because the dial had to be walked down. Arms B3 and C3
were not run: they also put a tile on W1, and W1 has no gradient-down GEMM, so its second blocking does
not exist. Run files: `out/block/mnist1d`. Figure: `docs/experiment_block_mnist1d_curves.png`.

Reference, rung L2, from `out/w/mnist1d`: 63.26, 63.35, 63.26, 63.55 test accuracy, mean 63.36. Test
loss 0.9288, mean over the four configurations.

| Arm | Config | Difference, accuracy | Difference, test loss | Standard error | Sign agrees | Winner |
|-|-|-:|-:|-:|-:|-|
| A | a6-e6m2 | -2.210 | +0.0857 | 0.515 | 10 of 10 | reference |
| A | f6-e6m2 | -2.991 | +0.0923 | 0.566 | 10 of 10 | reference |
| A | a6-e5m3 | -1.950 | +0.0786 | 0.326 | 10 of 10 | reference |
| A | f6-e5m3 | -3.014 | +0.0999 | 0.363 | 10 of 10 | reference |
| B2 | a6-e6m2 | -2.304 | +0.0596 | 0.302 | 10 of 10 | reference |
| B2 | f6-e6m2 | -3.239 | +0.0860 | 1.073 | 9 of 10 | reference |
| B2 | a6-e5m3 | -2.530 | +0.0668 | 0.475 | 10 of 10 | reference |
| B2 | f6-e5m3 | -3.160 | +0.0827 | 0.417 | 10 of 10 | reference |
| D2 | a6-e6m2 | -1.255 | +0.0304 | 0.309 | 9 of 10 | reference |
| D2 | f6-e6m2 | -1.575 | +0.0392 | 0.414 | 9 of 10 | reference |
| D2 | a6-e5m3 | -1.423 | +0.0366 | 0.467 | 8 of 10 | reference |
| D2 | f6-e5m3 | -1.776 | +0.0423 | 0.535 | 10 of 10 | reference |
| C2 | a6-e6m2 | -0.522 | +0.0146 | 0.438 | 6 of 10 | none |
| C2 | f6-e6m2 | -0.137 | +0.0009 | 0.469 | 5 of 10 | none |
| C2 | a6-e5m3 | -0.182 | +0.0024 | 0.262 | 4 of 10 | none |
| C2 | f6-e5m3 | -0.694 | +0.0108 | 0.377 | 8 of 10 | none |

Means: arm A -2.541 accuracy and +0.0891 loss. Arm B2 -2.808 and +0.0738. Arm D2 -1.507 and +0.0371.
Arm C2 -0.384 and +0.0072. **The two coarser forms are decided against in 4 of 4 cells. The 4 by 4 tile
is decided against in none.**

Final test accuracy, percent, mean of 10 seeds:

| Arm | a6-e6m2 | f6-e6m2 | a6-e5m3 | f6-e5m3 | Mean | Seed standard deviation |
|-|-:|-:|-:|-:|-:|-:|
| reference | 63.26 | 63.35 | 63.26 | 63.55 | 63.36 | |
| A | 61.05 | 60.36 | 61.31 | 60.54 | 60.82 | 1.43 |
| B2 | 60.96 | 60.11 | 60.73 | 60.39 | 60.55 | 1.52 |
| D2 | 62.01 | 61.78 | 61.84 | 61.78 | 61.85 | 1.10 |
| C2 | 62.74 | 63.21 | 63.08 | 62.86 | 62.97 | 0.82 |

The seed spread moves with the arm too. Arm C2 has the smallest spread of the four, 0.82 points, and the
reference is smaller still at 0.26 to 0.31 by configuration.

## Arm against arm, same seeds, MNIST-1D

The rule of the experiment compares an arm against the reference. Two arms can also be compared to each
other, paired on the same seeds, and that comparison needs no reference at all:

| Comparison | Mean difference | Standard error | Configurations decided | Sign agreement per config |
|-|-:|-:|-:|-|
| C2 minus A, 4 by 4 ahead | +2.157 | 0.256 | 4 of 4 | 9, 10, 10, 9 |
| C2 minus B2, 4 by 4 ahead | +2.425 | 0.302 | 4 of 4 | 10, 10, 9, 9 |
| C2 minus D2, 4 by 4 ahead | +1.123 | 0.222 | 3 of 4 | 8, 9, 7, 9 |
| D2 minus B2, 8 by 8 ahead | +1.301 | 0.317 | 2 of 4 | 9, 8, 8, 9 |
| D2 minus A, 8 by 8 ahead | +1.034 | 0.305 | 0 of 4 | 7, 6, 7, 8 |
| A minus B2 | +0.267 | 0.323 | 0 of 4 | 5, 7, 4, 5 |

Each mean is over the 40 pairs. The order on MNIST-1D is C2, then D2, then B2 and A together. The gap
between 4 by 4 and 8 by 8 is decided in 3 of 4 configurations, and the gap between 4 by 4 and 16 by 16 in
4 of 4.

The same comparisons on MNIST give nothing: every gap is between -0.40 and +0.01 accuracy points, and 0
to 2 of 4 configurations are decided. **MNIST cannot tell any two of these forms apart. MNIST-1D can.**

The drift telemetry of arm A on MNIST-1D is worse than on MNIST and it grows: `cos_dA1` falls from 0.982
at epoch 1 to 0.945 at epoch 10, and `cos_dW1` from 0.985 to 0.945, in `a6-e5m3`. Compare MNIST, where
the same numbers stay at 0.986 and 0.986.

## What the data says

1. **Two 1D copies make training worse.** It loses in 3 of the 4 configurations by the pre-registered
   rule on MNIST and in 4 of 4 on MNIST-1D, and it is worse on the test loss in all eight. Mean over the
   configurations: -0.398 accuracy points on MNIST and -2.541 on MNIST-1D.
2. **The damage is largest where the scale is finest.** `f6-e5m3` loses 0.785 accuracy points on MNIST,
   and it is the configuration where the copies disagree most, 78 to 79% of the elements. `e6m2`
   disagrees in 62 to 67% and loses 0.20 to 0.57 points. The two numbers move together.
3. **Arm A also fits the training set worse on MNIST.** Its training loss is 0.1666 against 0.1505 to
   0.1518 for every tile arm. The drift is not a generalisation effect. The optimizer is following a
   different function.
4. **Arm A adds seed variance on MNIST.** In `f6-e5m3` the seed-to-seed standard deviation is 0.98
   accuracy points against 0.26 for the reference. Same model, same seeds, same epochs.
5. **The gap does not close.** See `docs/experiment_block_mnist_curves.png`. The accuracy gap opens at
   epoch 2 and is still there at epoch 10. The test loss gap widens from epoch 3 to epoch 10.
6. **The copies drift, and MNIST-1D is where the drift compounds.** On MNIST the cosine of the two copies
   stays at 0.990 to 0.993 for ten epochs. On MNIST-1D the same telemetry falls from 0.982 at epoch 1 to
   0.945 at epoch 10. So the cost is a constant bias on an easy dataset and a growing drift on a harder
   one, which is why the loss is 0.40 points on MNIST and 2.54 on MNIST-1D.
7. **On MNIST no tile separates from the reference.** 19 of the 20 tile cells are undecided, and the one
   that fires is inside the false positive rate. A 16 by 16 tile costs 1.02 to 2.67 dB of SQNR against a
   1D block before any training (see the table above), and that cost does not reach the accuracy after
   training. On MNIST, training absorbs it.
8. **On MNIST a finer tile is not better.** The four tile means are 94.66, 94.67, 94.70 and 94.73, a
   spread of 0.07 points. A finer scale buys SQNR, and on MNIST SQNR does not buy accuracy.
9. **MNIST-1D reverses the tile result, and the reversal is ordered.** 16 by 16 loses 2.808 points, 8 by
   8 loses 1.507, 4 by 4 loses 0.384 and is not decided. Each halving of the tile side takes about half
   the loss away.
10. **On MNIST-1D the accuracy loss tracks the SQNR price of W2 and W3 almost exactly.** The three points
    are (-0.36 dB, -0.38 points), (1.18 dB, -1.51 points) and (2.32 dB, -2.81 points), which is about
    1.2 accuracy points per dB. On MNIST the same three points give about 0.05 points per dB, twenty
    times flatter. The slope is a property of the dataset, not of the format, so a new dataset must be
    measured and cannot be extrapolated from MNIST.
11. **The 4 by 4 tile is the only form that survives on both datasets.** It is undecided against the
    reference on MNIST (0 of 4 cells) and undecided on MNIST-1D (0 of 4 cells). It is ahead of the 8 by 8
    tile on MNIST-1D by 1.123 points, decided in 3 of 4 configurations, and ahead of the 16 by 16 tile by
    2.425 points, decided in 4 of 4.
12. **The two 1D copies and the 16 by 16 tile cannot be separated from each other on MNIST-1D.** Arm A
    is ahead of B2 by 0.267 points, inside its standard error. Both are 2 to 3 points behind.
13. **On MNIST, no two arms in this experiment can be separated at all.** Every arm-to-arm gap on MNIST
    sits between -0.40 and +0.01 points, and at most 2 of 4 configurations fire. **Ten seeds of MNIST
    have no power to choose a weight layout.** The choice therefore rests on MNIST-1D, which is the
    harsher dataset and the one with a continuous input.
14. **The SQNR probe ranks the shapes and the training runs agree with the ranking.** 4 by 4 costs -0.36
    dB on MNIST-1D, 8 by 8 costs 1.18 and 16 by 16 costs 2.32. The training loss is ordered the same
    way. A cheap probe of this kind is enough to reject a shape before a 40 run sweep, and it is not
    enough to accept one.

## Verdict

The first version of this section, written after MNIST only, said 16 by 16 was taken. The MNIST-1D run
overturned that, and the C2 and D2 runs then walked the dial down. The verdict rests on 400 block runs,
240 on MNIST and 160 on MNIST-1D, compared against 80 reference runs of rung L2, plus the SQNR probe on
10 seeds per dataset.

| Item | MNIST | MNIST-1D | Verdict |
|-|-:|-:|-|
| two 1D copies, arm A | -0.398, decided in 3 of 4 configs | -2.541, decided in 4 of 4 | **rejected** |
| 16 by 16 tiles, arm B2 | -0.056, no decision | -2.808, decided in 4 of 4 | **rejected** |
| 8 by 8 tiles, arm D2 | -0.065, no decision | -1.507, decided in 4 of 4 | **rejected** |
| 4 by 4 tiles, arm C2 | +0.002, no decision | -0.384, no decision | **the only survivor** |
| W1 blocked along K | no decision, and no second blocking exists | not run | **taken**, on structure |

Why the tile loses on MNIST-1D, in scales: a scale serves every weight whose row along the reduction axis
and column along the output axis sit in one tile. So for W2, which is 16 by 16:

| Layout | Scales of W2 | Scales of W3, 16 by 10 | Values under one scale along K | Values under one scale across N |
|-|-:|-:|-:|-:|
| 16 along K, one copy | 16 | 10 | 16 | 1 |
| 16 by 16 tile | 1 | 1 | 16 | 16 |
| 8 by 8 tile | 4 | 4 | 8 | 8 |
| 4 by 4 tile | 16 | 12 | 4 | 4 |

The 16 by 16 tile has the same granularity along K as the 1D block. It loses because it merges 16 outputs
across N under one scale, and on MNIST-1D that merge costs 2.8 points. The 4 by 4 tile is the only shape
that is finer along both axes at once.

### The dial this leaves

A square tile that keeps one value in both passes costs one term of the lane per 16 values along K. So
the accuracy of a tile is bought in term paths:

| Tile | Terms per lane | Parallel: extra LUT6, 4 lanes | Serial: extra cycles per step | Scales of W2 | MNIST | MNIST-1D |
|-|-:|-:|-:|-:|-:|-:|
| 16 by 16 | 1 | 0 | 0, and 0.20% for the second pass over the tile maximum | 1 | no decision, 94.67 | **-2.81, rejected** |
| 8 by 8 | 2 | 1,572 to 1,732, which is 3.0 to 3.3% of the Z020 | 1,856, which is 3.5% | 4 | no decision, 94.66 | **-1.51, rejected** |
| 4 by 4 | 4 | 4,716 to 5,196, which is 8.9 to 9.8% of the Z020 | up to 5,568, which is 10.5% | 16 | no decision, 94.73 | no decision, -0.38 |

Dzemo's cost ranking is smallest area first, then fewest cycles. On that ranking the 16 by 16 tile is the
cheapest and the 4 by 4 tile is the most expensive, and the two cheaper settings are decided losses.

The dial is now a single question with a single answer on the evidence: **pay for four terms, in area
(4,716 to 5,196 LUT6, which is 8.9 to 9.8% of the Z020, no extra cycles) or in cycles (up to 10.5% per
step on this ruler, no extra area), or accept a decided accuracy loss.** The 8 by 8 setting saves 3,144 to
3,464 LUT6 against the 4 by 4 and costs 1.12 accuracy points on MNIST-1D. No third option is measured.

**The cycle price belongs to this ruler.** W1 holds 96.5% of the weights and stays 1D, so only 3.5% of
the block-MACs read a tile. In a network where every layer except the first is tiled, a serial lane
trains 3.0 times slower at 4 by 4 and 1.67 times slower at 8 by 8. The area price does not depend on the
network.

### What must run next

| Run | Why | Cost |
|-|-|-|
| nothing for the tile shape | every setting of the dial is now measured on both datasets | 0 |
| the tile shape on a third dataset, when one exists | the slope of the loss against SQNR is a property of the dataset, and two datasets cannot fix a slope | 40 runs per shape |
| W2 tile 4 by 4 with W3 tile 8 by 8 | not measured. The two matrices need not share a shape, and W3 is 16 by 10 so it holds 12 tiles at 4 by 4 against 16 for W2 | 40 runs |

Until Dzemo accepts one shape, no tile shape is accepted for training, and the RTL is not written.

## The store this settles, inference against training

One table, every stored tensor, with the blocking it uses in each pass:

| Tensor | At inference | In training, forward pass | In training, backward pass |
|-|-|-|-|
| W1, 784 by 16 | one scale per 16 inputs, 784 scales | the same | not an operand. There is no layer below W1, and X is data |
| W2, 16 by 16 | one scale per 4 by 4 tile, 16 scales | the same tile, the same values | the same tile, read as W.T. One scale per 4 outputs |
| W3, 16 by 10 | one scale per 4 by 4 tile, 12 scales | the same | the same |
| X, A1, A2 | one scale per 16 features, so one scale per sample at 16 wide | the same | the weight-gradient GEMM needs 16 samples per block. **Not decided** |
| dZ3, dZ2, dZ1 | they do not exist | they do not exist | 16 neurons per block for the gradient-down GEMM. **Not decided** |
| the accumulator | 48 bits, one binary point per layer | the same | the same |
| the biases | the accumulator's 48-bit format, never FP4 | the same | the same |

**This table was written for the 16 by 16 tile and it now shows the 4 by 4 tile**, which is the only shape
that passed both datasets. The bit counts change by less than 1%, see the store table below.

Three statements that this table corrects:

1. **"Only the weights are quantized" is not right.** At inference both operands of every multiply are on
   the FP4 grid. A block of 16 activations has its own 8-bit scale, exactly like a block of 16 weights.
   What is true: an activation never has a float master. It leaves the 48-bit accumulator and the encoder
   rounds it once, in the same pass. The accumulator format is the output side of the GEMM. It is not an
   exemption from quantization.
2. **W1 stays blocked along K in training too.** It has no gradient-down GEMM, so no second blocking
   exists for it, and every tile shape costs it SQNR while buying nothing: 1.15 dB at 8 by 8 and 1.81 dB
   at 16 by 16, on MNIST.
3. **The block shape of the activations and the gradients is not settled here.** This experiment kept them
   in float, that is the L2 scope. Their second blocking is the L5 question, with three measured or
   named options: one rounding from the wide value, two roundings from the FP4 store, or 16 by 16 tiles
   for the activations.

**The mode does not change the store.** For W2 and W3 the same tile serves both passes and both modes,
so there is no conversion at a mode change. The bits, from `python/blocks_per_step.py`:

| Layout | W1 | W2 | W3 | Total |
|-|-:|-:|-:|-:|
| all three blocked along K | 56,448 | 1,152 | 720 | 58,320 bits |
| W2 and W3 as one 16 by 16 tile, W1 along K | 56,448 | 1,032 | 648 | 58,128 bits |
| W2 and W3 as 4 by 4 tiles, W1 along K | 56,448 | 1,152 | 736 | 58,336 bits |
| W2 and W3 as 8 by 8 tiles, W1 along K | 56,448 | 1,056 | 672 | 58,176 bits |

The tile shape changes the store by 208 bits at most, which is 0.36% of 58,320. **The store is not a
reason to choose a shape.** The reason to use a tile is the equal values in both passes, and the reason
to use a 4 by 4 tile instead of a 16 by 16 tile is 2.42 accuracy points on MNIST-1D. The reason not to
switch shape between modes is that a store with two shapes needs two encoders and a conversion step at
every mode change.

## In production: inference most of the time, training occasionally

### The two modes, per sample of a batch of 128, 4 lanes, encoder at 4 cycles per block

From `python/blocks_per_step.py`:

| Mode | Block-MACs per sample | MAC cycles | Encoder cycles | What sets the time |
|-|-:|-:|-:|-|
| inference, the input pre-encoded | 810 | 202.5 | 8 | the MAC |
| inference, the input encoded online | 810 | 202.5 | 204 | the encoder |
| training, forward and both backward GEMMs | 1,652 | 413 | 34.5 | the MAC |
| training, plus the weight re-encode | 1,652 | 413 | 59.8 | the MAC |

- The input `X` is 784 wide, so it is 49 blocks per sample. That is why the encoder overtakes the MAC in
  inference unless the input arrives pre-encoded. `experiment_L2.md` already pre-encodes it, and proves
  that pre-encoding gives the same values.
- Training does 4 times the block-MACs of inference, and it adds the weight re-encode. So the device is
  sized by the MAC, and inference has the encoder as its own limit.

### What the tile costs the datapath: nothing at 16 by 16, four terms at 4 by 4

A lane consumes 16 values that share one scale, along the axis the GEMM reduces over.

| Layout | What a lane sees in the forward pass | What a lane sees in the gradient-down pass | Datapath | Accuracy on MNIST-1D |
|-|-|-|-|-:|
| 16 values along K | one block of a column of W | not used | one `S` sum, one scale product | reference |
| one 16 by 16 tile | one column of the tile: 16 values, one scale | one row of the tile: 16 values, the same scale | one `S` sum, one scale product | -2.81 |
| one 8 by 8 tile | two columns, two scales | two rows, the same two scales | two `S` sums, two scale products | -1.51 |
| one 4 by 4 tile | four scales inside the 16 values | four scales | four `S` sums, four scale products | -0.38 |

So **1D and 16 by 16 are the same datapath with a different memory layout and a different encoder
grouping.** The 4 by 4 tile is not: it changes the lane, and it changes it into four copies of the same
thing. That is the price of the only tile shape that survives both datasets.

**An earlier version of this section said: "The block shape is a field of the weight store, not a
hardware mode. The device does not need a mode bit to choose it." That was wrong for every tile with a
side below 16.** The lane needs a term count per GEMM: 1 for W1 and for a 16 by 16 tile, 2 for an 8 by 8
tile, 4 for a 4 by 4 tile. That setting is a lane mode. It changes inside one forward pass of the ruler,
because W1 is 1D and W2 is a tile. The same earlier version also gave 10.5% as the cost "of the second
pass over the tile maximum and the re-encode". That was wrong too. The 10.5% is the serial term path.

### The tile and the lane: what changes for each side length

Terms, defined: the lane adds 16 payload products into one block sum `S` (the intra-block accumulator of
`datapath.md` section 4.3), and then makes one **term**, `S` times one scale product `P`. A tile of side
`a` puts `16 / a` different scales inside those 16 values along the contraction axis. Each group of `a`
values needs its own `S` and its own `P`. So the lane makes `t = 16 / a` terms per block pair.

| Tile | Values per scale along the contraction axis | Values per scale in the 2D tile | Terms `t` | Lane |
|-|-:|-:|-:|-|
| 1D block | 16 | 16 | 1 | the lane of `datapath.md` |
| 16 by 16 | 16 | 256 | 1 | the same lane. The scale is read 16 times |
| 8 by 8 | 8 | 64 | 2 | changes |
| 4 by 4 | 4 | 16 | 4 | changes |

**The lane sees one axis. Only the "values along the contraction axis" column matters to it.** The 4 by 4
tile has 16 values per scale in 2D, and the same count as the 1D block (16 scales for W2). That
equality explains why its accuracy matches. It does not make the lane see 16 values per scale. An
earlier version of the 4 by 4 argument mixed the two counts. NVIDIA repeats the scale for 16 by 16: "2D block scales are
replicated for each of the 1×16 blocks when being passed into Tensor Cores" (arXiv 2509.25149 section
4.3).

Two ways to pay for `t > 1`. Numbers from `python/blocks_per_step.py`, LUT6 are `datapath.md` estimates:

| Tile | Parallel: `t` term paths per lane | Serial: one term path used `t` times | Inference, serial, tile kept | Wide network, training, serial |
|-|-|-|-|-|
| 8 by 8 | +1,572 to 1,732 LUT6, 3.0 to 3.3% of the Z020 | +1,856 cycles, +3.5% | +3.2% | x 1.67 |
| 4 by 4 | +4,716 to 5,196 LUT6, 8.9 to 9.8% of the Z020 | +5,568 cycles, +10.5% (9.6% if the empty group of W3 is skipped) | +9.6% | x 3.00 |

In the serial lane the 16-wide product array is used once per `t` cycles. It is idle for `t - 1`
cycles out of `t`. The wide network figure is `(2t + 1) / 3`: three GEMMs of one layer have the same
size, two of them read the tile and cost `t` each, one does not.

### Three designs for the two modes

| Design | Store at inference | Store in training | Lane at inference | Lane in training | Conversion at the switch | Deployed model is the trained model |
|-|-|-|-|-|-|-|
| one store, the tile in both modes | W1 1D, W2 and W3 tiles | the same | 1 term on W1, `t` on W2 and W3 | the same | none | yes |
| two stores: 1D at inference, the tile in training | all three 1D | W1 1D, W2 and W3 tiles | 1 term | 1 term on W1, `t` on W2 and W3 | W2 and W3 only, 416 weights, 104 cycles, each way | **no. Not measured** |
| 16 by 16 only | W1 1D, W2 and W3 tiles | the same | 1 term | 1 term | none | yes. Costs -2.81 on MNIST-1D |

The two-store design is the one that keeps inference at one term. Two facts about it:

1. Only W2 and W3 change shape. W1 is 1D in both modes. In the ruler that is 416 weights, 3.2%, and 104
   cycles at 4 values per cycle.
2. Going back to 1D re-rounds W2 and W3. A 1D block of 16 values along K takes values from 4 tiles with 4
   different scales, finds a new maximum, and rounds again. So the model that runs at inference is a
   different function from the model that training optimized and tested. `docs/ADR/experiment_ALL-deployed-model.md`
   requires that the reported accuracy is the deployed model, so the number after the return conversion is
   the number that counts. **It has not been measured.** The nearest measured number is the float to
   tile conversion: 0.04 to 0.83 points on MNIST, 2.8 on MNIST-1D.

### The conversion price, measured

A model trained somewhere else, in float, converted to FP4. Weights only, activations float, whole test
set, from `python/tile_probe.py`:

MNIST, 10 seeds, test accuracy in percent:

| Layout | a6-e6m2 | f6-e6m2 | a6-e5m3 | f6-e5m3 |
|-|-:|-:|-:|-:|
| float, no quantization | 95.01 | 95.01 | 95.01 | 95.01 |
| 1D, all three matrices | 93.25 | 93.55 | 92.57 | 93.16 |
| 16 by 16, W2 and W3 | 92.72 | 92.72 | 92.53 | 92.53 |
| 16 by 16, all three | 92.57 | 92.57 | 92.34 | 92.34 |
| 4 by 4, all three | 93.08 | 93.07 | 93.50 | 93.82 |

MNIST-1D, 3 seeds, test accuracy in percent:

| Layout | a6-e6m2 | f6-e6m2 | a6-e5m3 | f6-e5m3 |
|-|-:|-:|-:|-:|
| float, no quantization | 64.94 | 64.94 | 64.94 | 64.94 |
| 1D, all three matrices | 56.77 | 58.62 | 55.51 | 55.09 |
| 16 by 16, W2 and W3 | 53.18 | 53.18 | 55.23 | 55.23 |
| 16 by 16, all three | 53.05 | 53.01 | 54.91 | 54.91 |
| 4 by 4, all three | 55.91 | 58.46 | 55.64 | 59.09 |

Three findings:

1. **Converting to the tile costs 0.04 to 0.83 points on MNIST.** That is the QAT gap: in training the
   weights move onto the coarse grid, and a converted model cannot. Training with the tile is free on
   MNIST, and it costs 2.3 to 3.2 points on MNIST-1D. The tile is not free anywhere that the weights need
   a fine scale.
2. **4 by 4 pays off in the conversion path.** It is better than 16 by 16 on MNIST in all four
   configurations and better than 1D in three of four, and on MNIST-1D it is better than 16 by 16 in all
   four. It is the store to convert into, and it is the store that needs the four term paths.
3. **The MNIST-1D conversion numbers are not resolved.** 3 seeds, and the spread is 2.8 to 5.7 points.
   Two of the four rows move by less than the spread.

The L1 rows in `experiment_L1.md` are the **1D store**. They are not the numbers of a 16 by 16 store. The
difference on MNIST is 0.04 to 0.83 points, and the row above gives it.

### What this means for the two uses

| Use | Block shape | Why |
|-|-|-|
| a model trained on FPG4, MNIST-like | any shape. 16 by 16 is cheapest | no drift, no second copy, and no accuracy cost on MNIST |
| a model trained on FPG4, MNIST-1D-like | 4 by 4 for W2 and W3, 16 along K for W1 | the 16 by 16 tile loses 2.81 points and the 8 by 8 loses 1.51. 4 by 4 costs 4,716 to 5,196 LUT6 |
| a float model converted for inference | 16 along K for W1, then 4 by 4 for W2 and W3 | a converted model cannot adapt to the coarser grid, and 4 by 4 is the best converted store in 3 of 4 configurations |
| both on one device | the shape is a per-matrix field in the store, and the term count is a per-GEMM setting of the lane | 1D and 16 by 16 use one term. 4 by 4 uses four, in area or in cycles |

## What this experiment does not settle

| Limit | Why |
|-|-|
| scale | 13,002 parameters and 10 epochs. NVIDIA's claim is for 1.2B parameters and 1T tokens. A null result here is not proof of safety there. |
| a loss generalises, a null result does not | arm A's result does transfer, because the mechanism is the same. The tile arms' null result on MNIST does not. |
| the drift with FP4 gradients | in this experiment the gradient is float32. NVIDIA suspects the gradient `dA` is where the 1D/1D form hurts most. That needs L5. |
| biases | the bias store is 48 bits at the accumulator's point, so it is not blocked and not tested here. |
| a third dataset | the slope of the accuracy loss against the SQNR price is 1.2 points per dB on MNIST-1D and 0.05 on MNIST. Two datasets cannot fix a slope, so a new dataset needs its own arm. |
| layers wider than one tile | W2 is one 16 by 16 tile and W3 is one tile, so the 16 by 16 arm gave each matrix one scale. NVIDIA used 16 by 16 weight tiles on a 1.2B model and found them better than two 1D copies. Here they tie, A minus B2 = +0.27 with a standard error of 0.32. A wider ruler would show if the -2.81 belongs to tiny layers. |
| the return from the tile to 1D | the two-store design re-rounds W2 and W3 at the end of training. Not measured, because the sweep did not save the final weights. |
| W2 and W3 with different shapes | not measured. W3 is 16 by 10, so a 4 by 4 grid gives it 12 scales against 16 for W2. |
| 20 epochs | every number here is 10 epochs. A longer run may narrow or widen a gap, and the protocol stays at 10 unless a rung is ambiguous. |

## Reproduction

```
cd python
uv run python block_sweep.py mnist                          # 200 runs, 66 minutes on 4 cores
uv run python block_sweep.py mnist1d 10 A B2                # 80 runs, 3 hours 1 minute from the first file to the last
uv run python block_sweep.py mnist1d 10 C2 D2              # 80 runs, 2 hours 55 minutes from the first file to the last
uv run python tile_probe.py                                # SQNR and converted accuracy of every layout, 7 seconds
uv run python blocks_per_step.py                           # the cost counts, from the shapes
uv run python second_rounding_probe.py                     # the second rounding of A and dZ, for L5
```

The store bits follow one formula, so any row of the store table can be checked by hand:
`bits = scales * 8 + K * N * 4`, where `scales` is `N * ceil(K/16)` for a 1D block and
`ceil(K/a) * ceil(N/b)` for an `a` by `b` tile.

The figures are `docs/experiment_block_mnist_curves.png` and
`docs/experiment_block_mnist1d_curves.png`. `block_sweep.py` writes them and redraws after every arm, so
a figure exists if a sweep is stopped early. Each figure states its own file name, the run directory and
the number of runs per curve in its footer.
