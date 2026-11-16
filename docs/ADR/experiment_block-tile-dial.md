# ADR-026: The tile is chosen on a dial of one, two or four terms per lane

**Issue**: A square tile keeps one value in both passes, and its cost is one term of the lane per 16 values
along the reduction axis. A 16 by 16 tile fits the lane at one term, an 8 by 8 tile needs two terms, and a
4 by 4 tile needs four. The project had to decide which tile the training datapath uses.

**Decision**: The dial is recorded and the choice is pending. W1 stays blocked along K, because it has no
gradient-down GEMM and no second blocking exists for it. W2 and W3 take a square tile, and the size of that
tile is not settled.

**Status**: Pending. The 16 by 16 tile was accepted on MNIST, commit `0e58015`, and rejected on MNIST-1D,
commit `14db604`. The 4 by 4 and 8 by 8 arms are running.

**Group**: format, datapath.

**Assumptions**: The layer that needs the fine scale is W2 and W3, not W1. The layer-drop probe found the
damage in W2 and W3.

**Constraints**: A tile must be square to give one value in both passes, and it must cover at least 16
values along the reduction axis to fit one term. A tile smaller than that costs 2 or 4 terms.

**Positions**:
1. 16 by 16 tile: 1 term, no extra area, 0.20% extra cycles for the second pass over the maximum, 1 scale
   for W2. Measured: no decision on MNIST, 2.808 points lost on MNIST-1D, decided in 4 of 4.
2. 8 by 8 tile: 2 terms, 1,572 to 1,732 LUT6 for four lanes, 5.2% extra cycles, 4 scales for W2. Not
   measured yet.
3. 4 by 4 tile: 4 terms, 4,716 to 5,196 LUT6, 10.5% extra cycles, 16 scales for W2. Measured on MNIST: no
   decision. Not measured on MNIST-1D.
4. Keep W2 and W3 blocked along K and accept that the gradient-down pass is not FP4.

**Argument**: On MNIST the tile costs nothing and the 4 by 4 tile is equal, so cost decides. On MNIST-1D the
16 by 16 tile loses 2.808 points while SQNR predicts a loss of 0.5 to 2.7 dB, so the coarse tile is not
acceptable there and the finer tile has to be measured. The dial is ordered by area, which is the first
criterion of the project.

**Implications**: If the 4 by 4 tile is required, the lane needs four sums and four scale products per block
pair, and the term path becomes the largest item of the datapath. If an 8 by 8 tile is enough, the cost
halves. The RTL is not written until this is settled.

**Related decisions**: ADR-004, ADR-007, ADR-017, ADR-024, ADR-025.

**Related requirements**: Smallest area first, then fewest cycles for one operation.

**Related artifacts**: `docs/experiment_block.md`, `python/block_sweep.py`, `python/tile_probe.py`,
`python/blocks_per_step.py`, `docs/experiment_block_mnist_curves.png`,
`docs/experiment_block_mnist1d_curves.png`.

**Related principles**: Buy accuracy with structure only when the data asks for it.

**Notes**: The 16 by 16 tile is free on MNIST for two reasons: the SQNR loss is small there and ten epochs of
training absorb it. A converted float model does not get that back, so the conversion path prefers 1D or
4 by 4.
