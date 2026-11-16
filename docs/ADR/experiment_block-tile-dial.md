# ADR-026: The tile is chosen on a dial of one, two or four terms per lane

**Issue**: A square tile keeps one value in both passes, and its cost is one term of the lane per 16 values
along the reduction axis. A 16 by 16 tile fits the lane at one term, an 8 by 8 tile needs two terms, and a
4 by 4 tile needs four. The project had to decide which tile the training datapath uses.

**Decision**: The dial is recorded and the choice is pending. W1 stays blocked along K, because it has no
gradient-down GEMM and no second blocking exists for it. W2 and W3 take a square tile, and the size of that
tile is not settled.

**Status**: Pending, and one candidate survives. The 16 by 16 tile was accepted on MNIST, commit
`0e58015`, and rejected on MNIST-1D, commit `14db604`. The 8 by 8 arm and the 4 by 4 arm on MNIST-1D
finished on 2026-11-16. The 8 by 8 tile is rejected and the 4 by 4 tile is the only shape that passes
both datasets. The choice is Dzemo's and it is not made.

**Group**: format, datapath.

**Assumptions**: The layer that needs the fine scale is W2 and W3, not W1. The layer-drop probe found the
damage in W2 and W3.

**Constraints**: A tile must be square to give one value in both passes, and it must cover at least 16
values along the reduction axis to fit one term. A tile smaller than that costs 2 or 4 terms.

**Positions**:
1. 16 by 16 tile: 1 term, no extra area, 0.20% extra cycles for the second pass over the maximum, 1 scale
   for W2. Measured: no decision on MNIST, 2.808 points lost on MNIST-1D, decided in 4 of 4. **Rejected.**
2. 8 by 8 tile: 2 terms, 1,572 to 1,732 LUT6 for four lanes, 5.2% extra cycles, 4 scales for W2. Measured:
   no decision on MNIST, 1.507 points lost on MNIST-1D, decided in 4 of 4. **Rejected.**
3. 4 by 4 tile: 4 terms, 4,716 to 5,196 LUT6, 10.5% extra cycles, 16 scales for W2. Measured: no decision
   on MNIST with a mean difference of +0.002, no decision on MNIST-1D with a mean difference of -0.384.
   **The only survivor.**
4. Keep W2 and W3 blocked along K and accept that the gradient-down pass is not FP4. This is the
   reference, and it is what every number above is measured against. Its accuracy is the best of the set
   by 0.38 points on MNIST-1D, and it is not separable from the 4 by 4 tile.

**Argument**: On MNIST no two of these forms can be separated: every arm-to-arm gap sits between -0.40 and
+0.01 accuracy points, and at most 2 of 4 configurations fire. Ten seeds of MNIST have no power to choose a
weight layout. So the choice rests on MNIST-1D, and there the order is clean and monotone: 16 by 16 loses
2.808 points, 8 by 8 loses 1.507, 4 by 4 loses 0.384 and is not decided. Each halving of the tile side
takes about half the loss away. The 4 by 4 tile is ahead of the 8 by 8 tile by 1.123 points, decided in 3
of 4 configurations, and ahead of the 16 by 16 tile by 2.425 points, decided in 4 of 4. So the question is
no longer "which shape is accurate": it is "is 1.12 accuracy points on the harsher dataset worth 3,144 to
3,464 LUT6, which is 5.9 to 6.5% of the Z020". The dial is ordered by area, which is the first criterion
of the project, and the area price of the surviving setting is the highest one.

**Implications**: The lane needs four sums and four scale products per block pair, so the term path becomes
the largest item of the datapath: 4,716 to 5,196 LUT6 for four lanes, against about 2,900 LUT6 for the
whole engine today. The cycle count rises by 10.5% per step. The store does not change: 58,336 bits
against 58,320 for a pure 1D store, which is 0.03%. The RTL is not written until Dzemo accepts a shape.

**Related decisions**: ADR-004, ADR-007, ADR-017, ADR-024, ADR-025.

**Related requirements**: Smallest area first, then fewest cycles for one operation.

**Related artifacts**: `docs/experiment_block.md`, `python/block_sweep.py`, `python/tile_probe.py`,
`python/blocks_per_step.py`, `out/tile_probe_price.json`, `docs/experiment_block_mnist_curves.png`,
`docs/experiment_block_mnist1d_curves.png`.

**Related principles**: Buy accuracy with structure only when the data asks for it.

**Notes**: The 16 by 16 tile is free on MNIST for two reasons: the SQNR loss is small there and ten epochs of
training absorb it. A converted float model does not get that back, so the conversion path prefers 1D or
4 by 4. The accuracy loss on MNIST-1D is about 1.2 points per dB of SQNR price on W2 and W3, and the same
slope is 0.05 points per dB on MNIST. The slope belongs to the dataset, so it cannot be extrapolated from
MNIST to a new dataset.
