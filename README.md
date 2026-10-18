# FPG4 training

This repository holds the step-by-step implementation of native FPG4 training.
The work starts in Python, with numpy, and moves to RTL only after the numerical
experiments pass.

## Goal

Show that a block-FP4 datapath can train a neural network without a float32 or
bfloat16 master copy of the weights.

The proof has two phases. Phase 1 is in software where we emulate the end goal in RTL.

1. A reference implementation in Python that emulates FPG4.
2. An RTL implementation that behaves the same as simulation.

## Structure

```
docs/     the training contract, the experiment ladder, and the results
python/   the simulation. One module per experiment
rtl/      the SystemVerilog sources. Empty until the simulation passes
data/     the datasets. Not tracked by git
```

## Status

L0 done, scaffolding L1.

## Environment

```
uv sync
source .venv/bin/activate
```

The dependencies are numpy and matplotlib.
