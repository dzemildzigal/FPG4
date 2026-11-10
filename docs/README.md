# Documents

The documents are written in this order. Each one is short and each one gates
the work that follows it.

| Document | Content | Status |
| :--- | :--- | :--- |
| `engine.md` | What FPG4 is: a general purpose FP4 GEMM engine for inference and training. The model is a test fixture. The cost scoreboard, and the figure of merit | Active |
| `spec.md` | The format, the scale algebra, and the integer datapath. Authority for numbers | Active. Version 0.7. The amendment of 2026-10-18 is applied and the layer accumulator now follows `datapath.md` |
| `datapath.md` | One GEMM, stage by stage, from the payload codes to the encoded result. Widths, LUT counts, and what is decided | Active |
| `results-e8.md` | Experiment E8, delayed scaling against the block buffer. Restored from the archive because it agrees with the current architecture | Active. Measurement, not a plan |
| `experiment_plan.md` | The rungs and what each one is allowed to change | Active |
| `experiment_L1.md` | L1, L1d, L1d-bias and L1e measured on MNIST and MNIST-1D | Active |
| `experiment_L2.md` | L2, QAT on the weights only. Written, not yet run | Active |
| `contract.md` | The training datapath: model, optimizer, rounding points, metrics | Not written |
| `ladder.md` | The experiments L0 to L8, one factor per rung | Not written |
| `results.md` | The measured curve of every rung, with the seed spread | Not written |

## Rule

A number in `results.md` carries its source: the script, the command, and the
seed. A claim without a source does not enter the document.
