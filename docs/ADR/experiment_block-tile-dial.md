# ADR-026: The tile is chosen on a dial of one, two or four terms per lane

**Issue**: A square tile keeps one value in both passes, and its cost is one term of the lane per 16 values
along the reduction axis. A 16 by 16 tile fits the lane at one term, an 8 by 8 tile needs two terms, and a
4 by 4 tile needs four. The project had to decide which tile the training datapath uses.

**Decision**: W1 stays blocked along K, because it has no gradient-down GEMM and no second blocking exists
for it. W2 and W3 take a square tile. **Two shapes are carried in every measurement that has a weight store:
4 by 4 in both passes, and 16 by 16 in training with 1D blocks at inference.** The 8 by 8 tile and the two
1D copies are dropped. The choice between the two carried shapes is deferred to rung L5.

**Status**: **Deferred to rung L5, by Dzemo on 2026-11-16.** The 16 by 16 tile was accepted on MNIST, commit
`0e58015`, and rejected on MNIST-1D, commit `14db604`. The 8 by 8 arm and the 4 by 4 arm on MNIST-1D
finished on 2026-11-16. The 8 by 8 tile is rejected, the two 1D copies are rejected, and the 4 by 4 tile is
the only shape that passes both datasets. Two shapes are carried from now on. **The choice is not made.**

**Group**: format, datapath.

**Assumptions**: The layer that needs the fine scale is W2 and W3, not W1. The layer-drop probe found the
damage in W2 and W3.

**Constraints**: A tile must be square to give one value in both passes, and it must cover at least 16
values along the reduction axis to fit one term. A tile smaller than that costs 2 or 4 terms.

**Positions**:
1. 16 by 16 tile: 1 term, no extra area, 0.20% extra cycles for the second pass over the maximum, 1 scale
   for W2. Measured: no decision on MNIST, 2.808 points lost on MNIST-1D, decided in 4 of 4. **Rejected.**
2. 8 by 8 tile: 2 terms. Parallel: 1,572 to 1,732 LUT6 for four lanes. Serial: 1,856 cycles, 3.5%. 4 scales
   for W2. Measured:
   no decision on MNIST, 1.507 points lost on MNIST-1D, decided in 4 of 4. **Rejected.**
3. 4 by 4 tile: 4 terms. Parallel: 4,716 to 5,196 LUT6. Serial: up to 5,568 cycles, 10.5%. 16 scales for W2.
   Measured: no decision
   on MNIST with a mean difference of +0.002, no decision on MNIST-1D with a mean difference of -0.384.
   **The only survivor.**
4. Keep W2 and W3 blocked along K and accept that the gradient-down pass is not FP4. This is the
   reference, and it is what every number above is measured against. Its accuracy is the best of the set
   by 0.38 points on MNIST-1D, and it is not separable from the 4 by 4 tile.
5. Two stores. Inference reads 1D blocks and uses one term. Training reads a tile in W2 and W3. A
   conversion runs at each switch: 416 weights, 104 cycles. Proposed by Dzemo, 2026-11-16. The return to 1D
   re-rounds W2 and W3, so the deployed model is not the trained model. Measured on float weights, no
   training: 54% of the values change after a 4 by 4 tile and 20% after a 16 by 16 tile, and the SQNR falls
   by 2.5 dB at 4 by 4. The accuracy of a tile-trained model after the return is not measured.
6. One store, with the tile in both modes. No conversion, and the deployed model is the trained model.
   Inference pays the terms: 3.2% more cycles at 8 by 8 and 9.6% at 4 by 4 on the ruler, serial lane.
7. Cached forward scales. Forward blocks are 1D along K. The backward blocks reuse the forward scales
   as enforced scales, and the backward encoder skips amax and snap. Proposed by Dzemo, 2026-11-16. The
   forward store is the inference store, so there is no return. Measured on W2, no training: the backward
   copy is 3.8 dB (MNIST) and 8.6 dB (MNIST-1D) worse than arm A's copy, 46% to 54% of the backward blocks
   clip their largest value, and 69% to 71% of the values still differ from the forward copy. The pairing
   exists only for square matrices. Not run as a training arm. **Expected worse than the two 1D copies of ADR-025 (arm A).**

**The three dials that L5 decides**: the scale rule (`amax/6` or `four6`), the scale format (UE6M2 or
UE5M3), and the tile (4 by 4 in both passes, or 16 by 16 in training with 1D at inference). L5 is the
first rung where the forward pass and the backward pass are both FP4, so it is the first rung where the
tile can be compared under the conditions that matter. Rungs L3 and L4 keep float weights and float
activations. No tile exists there and no store is read, so they are not doubled.

**Argument**: On MNIST no two of these forms can be separated: every arm-to-arm gap sits between -0.40 and
+0.01 accuracy points, and at most 2 of 4 configurations fire. Ten seeds of MNIST have no power to choose a
weight layout. So the choice rests on MNIST-1D, and there the order is clean and monotone: 16 by 16 loses
2.808 points, 8 by 8 loses 1.507, 4 by 4 loses 0.384 and is not decided. Each halving of the tile side
takes about half the loss away. The 4 by 4 tile is ahead of the 8 by 8 tile by 1.123 points, decided in 3
of 4 configurations, and ahead of the 16 by 16 tile by 2.425 points, decided in 4 of 4. So the question is
no longer "which shape is accurate": it is "is 1.12 accuracy points on the harsher dataset worth 3,144 to
3,464 LUT6, which is 5.9 to 6.5% of the Z020". The dial is ordered by area, which is the first criterion
of the project, and the area price of the surviving setting is the highest one.

**Implications**: The lane needs four sums and four scale products per block pair. The lane changes, and
the change is a term count per GEMM, which is a mode of the lane. It can be paid in area (parallel term
paths, 4,716 to 5,196 LUT6 for four lanes, against about 2,900 LUT6 for the whole engine today) or in
cycles (one term path used four times, up to 10.5% per step on the ruler, and x 3.0 on a network where every
layer but the first is tiled). The two prices are alternatives. The store does not change: 58,336 bits
against 58,320 for a pure 1D store, which is 0.03%. The RTL is not written until L5 picks one shape.

**Open items that can change the choice**:
- The ruler has layers one tile wide. W2 and W3 are one 16 by 16 tile each, so the 16 by 16 arm gave each
  matrix one scale. NVIDIA found 16 by 16 weight tiles better than two 1D copies on a 1.2B model (arXiv
  2509.25149, appendix E.5). Here they tie. A wider ruler would show if the -2.81 belongs to tiny layers.
- The accuracy after the return in position 5 is not measured. The sweep did not save the final weights.
  The float-weight probe gives the size of the change, not the accuracy of a tile-trained model.

**Related decisions**: ADR-004, ADR-007, ADR-017, ADR-022, ADR-024, ADR-025.

**Related requirements**: Smallest area first, then fewest cycles for one operation.

**Related artifacts**: `docs/experiment_block.md`, `python/block_sweep.py`, `python/tile_probe.py`,
`python/blocks_per_step.py`, `python/cached_scale_probe.py`, `out/tile_probe_price.json`, `docs/experiment_block_mnist_curves.png`,
`docs/experiment_block_mnist1d_curves.png`.

**Related principles**: Buy accuracy with structure only when the data asks for it.

**Motion against the 4 by 4 shape, and the measurement that answers it**: Dzemo raised it on 2026-11-16. A
fine tile could lose the benefit of block floating point and move the format towards IEEE floating point,
where one value has one exponent. The answer is in `python/granularity_probe.py`. For W2 the 4 by 4 tile
and the 1D block both need 16 scales and 4.50 bits per value, because `ceil(K/4) * ceil(N/4)` equals
`N * ceil(K/16)` for every aligned matrix. The two forms spend the same budget in a different shape. The
4 by 4 tile rounds 6.94% of the non-zero values to 0 on MNIST against 7.26% for the 1D block, and its SQNR
is 20.56 dB against 20.67 dB. The slide to one scale per value starts below 4: a 2 by 2 tile costs 1.50 more
bits per value for 2.60 dB, and one scale per value costs 7.50 more bits for 9.66 dB. The 16 by 16 tile is
the most efficient shape per bit, at 19.15 dB for 4.03 bits, and it is the shape that loses 2.81 points. So
the price of the 4 by 4 shape is four terms in the lane, and the price of the 16 by 16 shape is accuracy.

**Notes**: The 16 by 16 tile is free on MNIST for two reasons: the SQNR loss is small there and ten epochs of
training absorb it. A converted float model does not get that back, so the conversion path prefers 1D or
4 by 4. The accuracy loss on MNIST-1D is about 1.2 points per dB of SQNR price on W2 and W3, and the same
slope is 0.05 points per dB on MNIST. The slope belongs to the dataset, so it cannot be extrapolated from
MNIST to a new dataset.
