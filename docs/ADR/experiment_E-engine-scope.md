# ADR-001: FPG4 is a general purpose FP4 GEMM engine

**Issue**: The project needed a stated scope before any format or datapath work could be judged. A first
draft treated the project as "FP4 for the MNIST network". That scope cannot answer any question about a
CNN, an RNN or a transformer, and it lets a design pass that only works for 784 wide inputs.

**Decision**: FPG4 is a general purpose FP4 block scaled GEMM engine for training and for inference. The
784-16-16-10 network is a test fixture and a ruler. Its numbers measure the engine. They are not the
target of the engine.

**Status**: Decided, 2026-11-09, commit `4ed9dce`.

**Group**: scope.

**Assumptions**: A GEMM engine with block scaled FP4 operands covers the layers of a CNN, an RNN and a
transformer, because every one of them is a matrix multiply with a bias and a non-linearity around it.

**Constraints**: No design choice may depend on 784, on 16, or on 10 as the layer width. A design choice
may depend on the block size 16, which is a property of the format.

**Positions**:
1. Target the MNIST network only. Smallest effort, and no result transfers.
2. Target a general GEMM engine. Larger effort, and every result transfers.
3. Defer the scope.

**Argument**: The cost per operation is the number that decides the project, and a cost per operation is
only meaningful for a reusable engine. `docs/engine.md` states the cost scoreboard and the figure of
merit, 28 LUT6 per payload-MAC per cycle. A model specific engine would have no figure of merit.

**Implications**: The lane, the encoder and the accumulator must be sized for one block pair and one
scale product, not for a specific layer. The model is used to count blocks per step and to measure
accuracy, and it is replaced when needed.

**Related decisions**: ADR-002, ADR-004, ADR-007, ADR-024, ADR-026.

**Related requirements**: A single datapath that serves inference and training.

**Related artifacts**: `docs/engine.md`, `docs/spec.md`, `docs/datapath.md`.

**Related principles**: One variable per experiment. A number without a script and a seed is not evidence.

**Notes**: The fixture is called "the model" in every later document, never "the target".
