# Experiment: weight blocking

## Objective

One question, one variable: **does the block shape of the weight matrices change what FP4-aware training learns?**

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

Five arms. All run with the four configurations and the ten seeds of L2.

| Arm | What changes | Matrices touched | Share of the 13,002 weights |
|-|-|-|-:|
| reference | nothing. Rung L2 | none | 0% |
| A | a second copy blocked along N, used by the backward pass | W2, W3 | 3.2% |
| B2 | one 16 by 16 tile, the same value in both passes | W2, W3 | 3.2% |
| B3 | one 16 by 16 tile, the same value in both passes | W1, W2, W3 | 100% |
| C2 | one 4 by 4 tile, the same value in both passes | W2, W3 | 3.2% |
| C3 | one 4 by 4 tile, the same value in both passes | W1, W2, W3 | 100% |

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
| C2 | four term paths per lane, 4,716 to 5,196 LUT6, which is 8.9 to 9.8% of the Z020, or 4 times the cycles on the term path | 5,568, which is 10.5%, if one term path is kept | none |
| C3 | the C2 area, and W1 in tiles | the C2 cycles | none |

The 104 cycles of B2 and B3 are a second pass over the tile for its maximum. A 16 by 16 tile holds
256 values, so the encoder reads 16 blocks of 16 values twice instead of once. No 256-value buffer
is needed, because the first pass keeps one running maximum.

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
The figure shows the four curves of every arm, and the reference appears in it with test curves only.

### Final test accuracy, percent, mean of 10 seeds

| Arm | a6-e6m2 | f6-e6m2 | a6-e5m3 | f6-e5m3 | Mean |
|-|-:|-:|-:|-:|-:|
| reference | 94.72 | 94.77 | 94.61 | 94.80 | 94.73 |
| A | 94.53 | 94.20 | 94.56 | 94.01 | **94.33** |
| B2 | 94.31 | 94.78 | 94.75 | 94.84 | 94.67 |
| B3 | 94.67 | 94.45 | 94.77 | 94.92 | 94.70 |
| C2 | 94.64 | 94.74 | 94.69 | 94.85 | 94.73 |
| C3 | 94.74 | 94.63 | 94.82 | 94.66 | 94.71 |

### Final test loss, mean of 10 seeds

| Arm | a6-e6m2 | f6-e6m2 | a6-e5m3 | f6-e5m3 | Mean |
|-|-:|-:|-:|-:|-:|
| reference | 0.1798 | 0.1799 | 0.1834 | 0.1799 | 0.1808 |
| A | 0.1882 | 0.1973 | 0.1895 | 0.2061 | **0.1953** |
| B2 | 0.1939 | 0.1818 | 0.1790 | 0.1763 | 0.1827 |
| B3 | 0.1817 | 0.1888 | 0.1790 | 0.1745 | 0.1810 |
| C2 | 0.1804 | 0.1807 | 0.1812 | 0.1788 | 0.1803 |
| C3 | 0.1782 | 0.1832 | 0.1772 | 0.1812 | 0.1800 |

### Final train loss, mean of 10 seeds

| Arm | a6-e6m2 | f6-e6m2 | a6-e5m3 | f6-e5m3 | Mean |
|-|-:|-:|-:|-:|-:|
| A | 0.1620 | 0.1676 | 0.1609 | 0.1761 | **0.1666** |
| B2 | 0.1630 | 0.1493 | 0.1488 | 0.1441 | 0.1513 |
| B3 | 0.1521 | 0.1585 | 0.1499 | 0.1468 | 0.1518 |
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
| C2 | a6-e6m2 | -0.082 | 0.133 | 5 of 10 | no | no | none |
| C2 | f6-e6m2 | -0.033 | 0.040 | 6 of 10 | no | no | none |
| C2 | a6-e5m3 | 0.077 | 0.081 | 7 of 10 | no | no | none |
| C2 | f6-e5m3 | 0.046 | 0.070 | 7 of 10 | no | no | none |
| C3 | a6-e6m2 | 0.014 | 0.118 | 6 of 10 | no | no | none |
| C3 | f6-e6m2 | -0.142 | 0.094 | 9 of 10 | yes | no | none |
| C3 | a6-e5m3 | 0.212 | 0.055 | 9 of 10 | yes | yes | C3 |
| C3 | f6-e5m3 | -0.143 | 0.086 | 8 of 10 | yes | no | none |

20 comparisons. One fires: C3 in `a6-e5m3`, plus 0.212 accuracy points. One cell in 20 at the two
standard error threshold is what a false positive looks like. C3's mean over the four configs is
94.71, against 94.73 for C2 and 94.70 for B3, so the cell does not repeat.

### The tile price before training

`python/tile_probe.py`, saved L0 weights, 10 seeds, no training. The number is the SQNR of the 1D block
minus the SQNR of the tile, so a positive number is a loss:

| Dataset | Matrix | 4 by 4, mean and range | 16 by 16, mean and range |
|-|-|-:|-:|
| MNIST | W1 | -0.01 [-0.21, 0.06] | 1.33 [1.20, 1.43] |
| MNIST | W2 | -0.06 [-0.74, 0.83] | 0.93 [0.30, 1.44] |
| MNIST | W3 | -0.04 [-0.76, 1.08] | 0.48 [-0.93, 2.07] |
| MNIST-1D | W1 | -0.07 [-0.15, -0.01] | 0.74 [0.62, 0.87] |
| MNIST-1D | W2 | -0.28 [-0.88, 0.64] | 2.26 [0.67, 4.03] |
| MNIST-1D | W3 | -0.56 [-1.84, 0.23] | 1.44 [0.22, 2.86] |

Ranges are over the four configurations and the ten seeds.

- A 4 by 4 tile costs nothing, and in half the cells it gains a little. That is expected: 4 values per
  scale is finer than 16 along K, so the blocks are smaller along both axes.
- A 16 by 16 tile costs 0.48 to 2.73 dB, and it costs more where the matrix has fewer rows: W2 is 16 by
  16, so it is exactly one tile and the scale is a per-matrix scale.
- `four6` is worth nothing on a 16 by 16 tile: it keeps `amax/4` in 0.0 to 1.2% of the tiles, against 26
  to 46% of the 4 by 4 tiles. So `a6` and `f6` produce almost the same weights on a 16 by 16 layout.

So the picture is complete: the tile loses SQNR, and ten epochs of training give the accuracy back.

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

## What the data says

1. **Two 1D copies make training worse.** It loses in 3 of the 4 configurations by the pre-registered
   rule, and it is worse on the test loss in all four. Mean over the configurations: -0.398 accuracy
   points and +0.0145 test loss.
2. **The damage is largest where the scale is finest.** `f6-e5m3` loses 0.785 accuracy points, and it
   is the configuration where the copies disagree most, 78 to 79% of the elements. `e6m2` disagrees in
   62 to 67% and loses 0.20 to 0.57 points. The two numbers move together.
3. **Arm A also fits the training set worse.** Its training loss is 0.1666 against 0.1505 to 0.1518 for
   every tile arm. The drift is not a generalisation effect. The optimizer is following a different
   function.
4. **Arm A adds seed variance.** In `f6-e5m3` the seed-to-seed standard deviation is 0.98 accuracy
   points against 0.26 for the reference. Same model, same seeds, same epochs.
5. **The gap does not close.** See the figure. The accuracy gap opens at epoch 2 and is still there at
   epoch 10. The test loss gap widens from epoch 3 to epoch 10.
6. **The copies do not diverge.** The cosine of the two copies stays at 0.990 to 0.993 for ten epochs.
   The cost is a constant bias, not a runaway.
7. **No tile arm separates from the reference.** 15 of the 16 tile cells are undecided, and the one that
   fires is inside the false positive rate. A 16 by 16 tile costs 0.48 to 2.73 dB of SQNR against a 1D
   block before any training (see the table above), and that cost does not reach the accuracy after
   training. Training absorbs it.
8. **A 4 by 4 tile is not better than a 16 by 16 tile.** Their means are 94.73 and 94.67 to 94.70. A
   finer scale buys SQNR, and SQNR does not buy accuracy here.

## Verdict: what L5 takes

| Item | Decision | Reason |
|-|-|-|
| two 1D copies | **rejected** | decided by the rule in 3 of 4 configs, and worse on test loss in 4 of 4 |
| 4 by 4 tiles | **rejected** | no accuracy gain over 16 by 16, and 4,716 to 5,196 LUT6 or 10.5% of the cycles |
| 16 by 16 tiles for W2 and W3 | **taken** | the only tile that serves one scale to both passes at the lane width, no extra area, 0.20% extra cycles |
| W1 | **stays blocked along K** | it has no gradient-down GEMM, so no second blocking exists for it, and a tile only lowers its SQNR by 0.74 to 2.29 dB |

This is arm B2. The blockers on the choice were cost and structure, because the accuracy data does
not separate the tile shapes at this model size.

## The store this settles, inference against training

One table, every stored tensor, with the blocking it uses in each pass:

| Tensor | At inference | In training, forward pass | In training, backward pass |
|-|-|-|-|
| W1, 784 by 16 | one scale per 16 inputs, 784 scales | the same | not an operand. There is no layer below W1, and X is data |
| W2, 16 by 16 | one 16 by 16 tile, 1 scale | the same tile, the same values | the same tile, read as W.T. One scale per 16 outputs |
| W3, 16 by 10 | one 16 by 16 tile, 1 scale | the same | the same |
| X, A1, A2 | one scale per 16 features, so one scale per sample at 16 wide | the same | the weight-gradient GEMM needs 16 samples per block. **Not decided** |
| dZ3, dZ2, dZ1 | they do not exist | they do not exist | 16 neurons per block for the gradient-down GEMM. **Not decided** |
| the accumulator | 48 bits, one binary point per layer | the same | the same |
| the biases | the accumulator's 48-bit format, never FP4 | the same | the same |

Three statements that this table corrects:

1. **"Only the weights are quantized" is not right.** At inference both operands of every multiply are on
   the FP4 grid. A block of 16 activations has its own 8-bit scale, exactly like a block of 16 weights.
   What is true: an activation never has a float master. It leaves the 48-bit accumulator and the encoder
   rounds it once, in the same pass. The accumulator format is the output side of the GEMM. It is not an
   exemption from quantization.
2. **W1 stays blocked along K in training too.** It has no gradient-down GEMM, so no second blocking
   exists for it, and the 16 by 16 tile measured no accuracy gain while costing 0.74 to 2.29 dB of SQNR.
3. **The block shape of the activations and the gradients is not settled here.** This experiment kept them
   in float, that is the L2 scope. Their second blocking is the L5 question, with three measured or
   named options: one rounding from the wide value, two roundings from the FP4 store, or 16 by 16 tiles
   for the activations.

**The mode does not change the store.** For W2 and W3 the same 16 by 16 tile serves both passes and
both modes, so there is no conversion at a mode change. The bits, from `python/blocks_per_step.py`:

| Layout | W1 | W2 | W3 | Total |
|-|-:|-:|-:|-:|
| all three blocked along K | 56,448 | 1,152 | 720 | 58,320 bits |
| W2 and W3 as one tile, W1 along K | 56,448 | 1,032 | 648 | 58,128 bits |

The tile saves 192 bits, which is 0.33%. The reason to use it is the equal values in both passes, not the
bits. The reason not to switch shape between modes is that a store with two shapes needs two encoders and
a conversion step at every mode change.

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

### What the tile costs the datapath: nothing

A lane consumes 16 values that share one scale, along the axis the GEMM reduces over.

| Layout | What a lane sees in the forward pass | What a lane sees in the gradient-down pass | Datapath |
|-|-|-|-|
| 16 values along K | one block of a column of W | not used | one `S` sum, one scale product |
| one 16 by 16 tile | one column of the tile: 16 values, one scale | one row of the tile: 16 values, the same scale | one `S` sum, one scale product |
| one 4 by 4 tile | four scales inside the 16 values | four scales | four `S` sums, four scale products |

So **1D and 16 by 16 are the same datapath with a different memory layout and a different encoder
grouping.** 4 by 4 is not: it changes the lane.

That is the production answer. The block shape is a field of the weight store, not a hardware mode. The
device does not need a mode bit to choose it.

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

1. **Training with the tile costs nothing. Converting to the tile costs 0.04 to 0.83 points on MNIST.**
   That is the QAT gap: in training the weights move onto the coarse grid, and a converted model cannot.
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
| a model trained on FPG4 | 16 by 16 for W2 and W3, 16 along K for W1 | no drift, no second copy, and no accuracy cost |
| a float model converted for inference | 16 along K, or 4 by 4 if the four term paths are built | a converted model cannot adapt to the coarser grid |
| both on one device | the shape is a per-matrix field in the store | 1D and 16 by 16 cost the same datapath |

## What this experiment does not settle

| Limit | Why |
|-|-|
| scale | 13,002 parameters and 10 epochs. NVIDIA's claim is for 1.2B parameters and 1T tokens. A null result here is not proof of safety there. |
| a loss would generalise, a null result would not | arm A's result does transfer, because the mechanism is the same. The tile arms' null result does not. |
| the drift with FP4 gradients | in this experiment the gradient is float32. NVIDIA suspects the gradient `dA` is where the 1D/1D form hurts most. That needs L5. |
| biases | the bias store is 48 bits at the accumulator's point, so it is not blocked and not tested here. |

## MNIST-1D

Arm A and arm B2, four configurations, ten seeds, started because the trigger fired on MNIST: an arm
reached at least one standard error. The other three arms are not run, because two datasets are not
needed to confirm that the two-copy form is worse, and the tile arms already show nothing.

| Arm | Config | Mean difference, accuracy | Standard error | Sign agrees | Winner |
|-|-|-:|-:|-:|-|
| pending | | | | | |

## Reproduction

```
cd python
uv run python block_sweep.py mnist            # 200 runs, 66 minutes on 4 cores
uv run python block_sweep.py mnist1d 10 A B2  # 80 runs
uv run python curves.py mnist                 # not needed, nothing is retrained there
```

The figure is `docs/block_mnist.png`, written by `block_sweep.py` and redrawn after every arm.

The cost numbers above come from `python/blocks_per_step.py`, which counts the 16-value blocks of one
training step from the shapes alone. The measurement of the two copies before any training, on the
saved L0 weights, is `python/second_rounding_probe.py`. Neither script trains anything.
