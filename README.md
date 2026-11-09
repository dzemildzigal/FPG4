# FPG4 training

This repository holds the step-by-step implementation of native FPG4 training.
The work starts in Python, with numpy, and moves to RTL only after the numerical
experiments pass.

## Goal

Show that a block-FP4 datapath can train a neural network without a float32 or
bfloat16 master copy of the weights, and that every step of that datapath is
cheap enough in area and power to be worth building on an FPGA.

FPG4 is a **general purpose FP4 GEMM engine**, for inference and for training. It
is not an MLP engine. Any weight tensor that enters it is quantised to one 4-bit
E2M1 payload per value plus one unsigned 8-bit scale per block of 16 values along
the reduction axis. Every learned operation reduces to a GEMM, so the GEMM is the
unit of design and work.

The 784-16-16-10 network in this repository is a **test fixture**. It proves that
the arithmetic matches float64 and measures how many accuracy points the format
costs. It does not set the area budget and it does not shape any circuit. See
`docs/engine.md`.

The proof has two phases. Phase 1 is in software where we emulate the end goal in RTL.

1. A reference implementation in Python that emulates FPG4.
2. An RTL implementation that behaves the same as simulation.

## Structure

```
docs/     the engine, the format, the datapath, the experiment ladder, the results
python/   the simulation. One module per experiment
rtl/      the SystemVerilog sources. Empty until the simulation passes
data/     the datasets. Not tracked by git
```

## Status

Measured on both datasets, MNIST and MNIST-1D:

| rung | what it changes | state |
|---|---|---|
| L0 | float64 reference | done |
| L1a to L1c | FP4 weights, four scale rules | done |
| L1d | FP4 weights and activations | done |
| L1d-bias | the bias on the FP4 grid | done, the bias stays at 32 bits |
| L1e | the datapath variants E0 to E4 | done |
| L2 | quantisation aware training, weights only | written, not run |

Not done: the encoder circuit, the RTL, and **rung E9 synthesis**. Every LUT6
and DSP48 count in `docs/` is an estimate until E9 runs.

Figure of merit: **about 28 LUT6 per payload-MAC per cycle**, estimate. See
`docs/engine.md` section 4.

## Environment

```
uv sync
source .venv/bin/activate
```

The dependencies are numpy and matplotlib.

## Datasets

Two datasets. Both run through the same scripts, selected by name (`mnist` or `mnist1d`).

```
cd python
uv run python mnist1d.py              # generate MNIST-1D once (about 3 minutes, 1.6 GB in data/mnist1d)
uv run python train.py [dataset]      # L0: 3 seeds, writes out/weights_<dataset>_seed<seed>.npy.npz
uv run python l1_sweep.py [dataset]   # L1a to L1c: FP4 weights only
uv run python l1d_sweep.py [dataset]  # L1d: FP4 weights and activations
uv run python l1e_sweep.py [dataset]  # L1e: the datapath variants (accumulator, scale product)
uv run python l1_report.py [dataset]  # L1b scale statistics and the histogram plot
uv run python tests/test_quant.py
uv run python tests/test_mnist1d.py
```

MNIST comes from the four IDX files in `data/mnist`.
