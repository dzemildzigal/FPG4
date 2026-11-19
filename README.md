# FPG4 training

This repository holds the step-by-step implementation of native FPG4 training. The work starts in Python
with numpy, and it moves to RTL only after the numerical experiments pass.

## Goal

Show that a block-FP4 datapath can train a neural network without a float32 or bfloat16 master copy of the
weights, and that every step of that datapath is cheap enough in area and power to be worth building on an
FPGA.

FPG4 is a **general purpose FP4 GEMM engine**, for inference and for training. It is not an MLP engine.
Any weight tensor that enters it becomes one 4-bit E2M1 payload per value plus one unsigned 8-bit scale per
block of 16 values along the reduction axis. Every learned operation reduces to a GEMM, so the GEMM is the
unit of design.

The 784-16-16-10 network in this repository is a **test fixture**. It proves that the arithmetic matches
float and it measures how many accuracy points the format costs. It does not set the area budget and it does
not shape any circuit. See `docs/engine.md`.

## Status, 2026-11-16

The simulation runs on MNIST and MNIST-1D, 10 seeds per cell, 4 format configurations per cell.

| rung | what it changes | state |
|-|-|-|
| L0 | float32 reference. The network learns | done, 95.01 on MNIST and 64.98 on MNIST-1D |
| L1a to L1c | FP4 weights by post-training quantization, four scale rules | done |
| L1d | FP4 weights and activations | done |
| L1d-bias | the bias on the FP4 grid | done, and rejected: the bias is fixed point at the accumulator's width |
| L1e | the datapath variants: exact integer path, accumulator width, scale product | done, the accumulator is decided at 48 bits |
| L2 | quantization-aware training, weights only | done, 94.61 to 94.80 on MNIST |
| L2a | quantization-aware training, weights and activations | done, 93.88 to 94.10 on MNIST |
| block | the weight blocking for a backward pass that reads a transposed tile | done, 480 runs. Two 1D copies, 8 by 8 and 16 by 16 rejected. 4 by 4 carried with 16 by 16 to L5 |
| L3 | FP4 operands of the backward GEMMs, float forward | done, 3 cells, 120 runs. No decision anywhere, bound 0.2 points |
| L5 | FP4 forward and backprop, optimizer FP32 | **done, 50 runs.** Three arms at width 64. **The forward payload must be round to nearest and the backward payload must be stochastic**: +1.32 on the float master and +7.42 on the deployed model. The forward block stays 1D, the 16 by 16 tile costs 2.29. MNIST: 97.41 against float 97.45. MNIST-1D: 85.58 against float 90.50. ADR-034 | |
| L4 | L3 with stochastic rounding instead of round to nearest | **done, both datasets, narrow and width 64.** MNIST-1D rejected the cell that rounds the gradient-down operands, 64.98 to 36.03. **Cause: the block scale was rounded to the nearest table entry and clipped the block maximum. With the scale rounded up, at width 64, the same cell gives 90.03 to 90.40 against float 90.50.** The tile arm of 2026-11-19 adds the block shape: a 16 by 16 tile costs 0.01 points with `amax/6`, so L5 uses it. ADR-032, ADR-033 |
| L5 | FP4 forward and backward, optimizer float32 | not written. **The next rung, and the gate for three open dials** |
| L6 | FP4 forward, backward and optimizer, in two forms | planned. Both forms must run |
| L7 | error feedback into the momentum, no extra tensor | planned |
| L8 | Hadamard mixing | planned |
| E9 | synthesis, which turns every estimate into a measured number | **not run. Every LUT6 and DSP48 count is an estimate until it runs** |

Not done: the encoder circuit in RTL, and the RTL itself. `rtl/` is empty on purpose.

Figure of merit: **about 28 LUT6 per payload-MAC per cycle**, estimate. See `docs/engine.md` section 4.
The whole engine is about 2,900 LUT6 on a PYNQ-Z2, which has 53,200.

### Decided, and where

| Item | Decision | Record |
|-|-|-|
| the payload | E2M1, four bits, eight magnitudes | ADR-006 |
| the block | 16 values along the reduction axis | ADR-007 |
| the scale | unsigned 8 bits. **Both formats stay open**: UE6M2 `[7 sig][7 exp]` and UE5M3 `[8 sig][6 exp]` | ADR-011, ADR-023 |
| the accumulator | 48 bits, one binary point per layer at `min(y)`, `24.24` in the experiments | ADR-014 |
| the bias | fixed point at the accumulator's width and point, loaded as its initial value. Not FP4 | ADR-015 |
| the lanes | four | ADR-004 |
| the multiplier | logic only, no hard multiplier IP | ADR-003 |
| the simulation width | float32 from L0 up. L1a to L1e were measured before that, in float64 | ADR-005 |
| the reported model | always the deployed quantized model | ADR-022 |
| the device | PYNQ-Z2, XC7Z020 | ADR-002 |

The three dials that rung L5 decides: the scale rule (`amax/6` or `four6`), the scale format (UE6M2 or
UE5M3), and the tile (4 by 4 in both passes, or 16 by 16 in training with 1D blocks at inference). Both tile
shapes are carried in every measurement until L5. See `docs/experiment_plan.md`.

Every decision has a record in `docs/ADR/`, one file per decision, 30 files, indexed by `docs/ADR/README.md`.

## Structure

```
docs/          the engine, the format, the datapath, the ladder, the results, the ADRs
docs/ADR/      one file per architectural decision
python/        the simulation. One module per experiment, plus the probes
rtl/           the SystemVerilog sources. Empty until the simulation passes
data/          the datasets. Not tracked by git
out/           the run files and the logs. Not tracked by git
```

Start reading with `docs/README.md`, then `docs/engine.md`, then `docs/datapath.md`.

## Environment

```
uv sync
source .venv/bin/activate
```

The dependencies are numpy and matplotlib. Python 3.14, numpy 2.5, matplotlib 3.11.

## Running the rungs

Every script takes the dataset first: `mnist` or `mnist1d`. Run them from `python/`.

```
uv run python mnist1d.py                 # generate MNIST-1D once, about 3 minutes
uv run python train.py [dataset]         # L0. Writes the float weights
uv run python l1_sweep.py [dataset]      # L1a to L1c
uv run python l1d_sweep.py [dataset]     # L1d
uv run python l1e_sweep.py [dataset]     # L1e
uv run python l2_sweep.py [dataset] 10 w   # L2, weights only. "wa" is L2a
uv run python block_sweep.py [dataset]   # the weight blocking, six arms
uv run python l3.py [dataset] 10 rtn     # L3, round to nearest
uv run python l3.py [dataset] 10 sr      # L4, stochastic rounding. Same code, one argument
uv run python l1_report.py [dataset]     # the L1b scale statistics and the histogram
```

Every sweep writes one JSON file per run and never repeats a finished run, so any sweep can be stopped and
restarted. The reported accuracy is always the deployed quantized model (ADR-022).

## Probes, which train nothing

| script | what it measures | time |
|-|-|-|
| `blocks_per_step.py` | the block-MAC count, the cycles per step, the term paths of a tile, the alignment count toward one exponent per value | seconds |
| `tile_probe.py` | the SQNR price of every tile shape, and the converted accuracy | 7 s |
| `granularity_probe.py` | the scale count, the bits per value and the flush share of every block side down to one value | 5 s |
| `cached_scale_probe.py` | the return from a tile to 1D blocks, and the cached-scale idea | 14 s |
| `second_rounding_probe.py` | the second rounding of `A` and `dZ`, for L5 | seconds |
| `width_probe.py` | the accumulator headroom on the real tensors | seconds |
| `amax_probe.py` | the encoder width, and that 16 significand bits change no scale code | minutes |
| `curves.py` | the training curves and the exponential fits | read only |

## Datasets

Two datasets, through the same scripts, selected by name.

| dataset | inputs | classes | training set | test set | on disk |
|-|-:|-:|-:|-:|-:|
| `mnist` | 784 | 10 | 60,000 | 10,000 | 12 MB |
| `mnist1d` | 784 | 10 | 400,000 | 100,000 | 1.5 GB |

MNIST comes from the four IDX files in `data/mnist`. `python/mnist1d.py` generates MNIST-1D once. It is a
numpy rewrite of the public MNIST-1D generator: the 40-point signal is resampled to 784 points, so the input
width matches MNIST and the block count along K is the same. The samples are 500,000, not the 5,000 of the
published 40-point version, so a rung on MNIST-1D costs about seven times a rung on MNIST. See
`docs/experiment_L1.md` for the measurements behind those statements.

One rung needs care on a 16 GB machine: a pre-quantized copy of the MNIST-1D training set is 1.25 GB, and a
sweep should not hold more than one of them.

## The rule of the evidence

A number carries its source: the script, the command, and the run directory. A claim without a source does
not enter a document, and no figure is drawn by hand. Every figure is named
`experiment_<rung>_<dataset>_<content>.png` and its footer states its own file name and the run directory
behind its numbers (ADR-030).

A run that already exists is never repeated to make a figure. A file of run results is evidence, and
evidence is not regenerated.
