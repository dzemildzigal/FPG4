# Documents

The documents are written in this order. Each one is short and each one gates
the work that follows it.

| Document | Content | Status |
| :--- | :--- | :--- |
| `engine.md` | What FPG4 is: a general purpose FP4 GEMM engine for inference and training. The model is a test fixture. The cost scoreboard, and the figure of merit | Active |
| `spec.md` | The format, the scale algebra, and the integer datapath. Authority for numbers | Active. Version 0.13. The layer accumulator is decided: 48 bits, point at `min(y)` of the layer, and a fixed `24.24` for the experiments. Section 5.7 states the width rule with the distance term |
| `datapath.md` | One GEMM, stage by stage, from the payload codes to the encoded result. Widths, LUT counts, and what is decided | Active |
| `results-e8.md` | Experiment E8, delayed scaling against the block buffer. Restored from the archive because it agrees with the current architecture | Active. Measurement, not a plan |
| `experiment_plan.md` | The rungs and what each one is allowed to change | Active |
| `experiment_L1.md` | L1, L1d, L1d-bias and L1e measured on MNIST and MNIST-1D | Active |
| `experiment_L2.md` | L2, QAT on the weights only. Measured on MNIST and MNIST-1D | Active |
| `experiment_block.md` | The weight blocking: two 1D copies against a 16 by 16, an 8 by 8 and a 4 by 4 tile. Measured on MNIST and MNIST-1D, 480 runs. Verdict: both copies and both coarser tiles rejected, 4 by 4 is the only survivor | Active |
| `experiment_L3.md` | L3, FP4 backward pass with the weights, the activations and the optimizer in float. Measured on MNIST, three cells: no decision anywhere, and the whole backward pass stays within 0.2 points | Active |
| `experiment_L4.md` | L4, the same three cells with stochastic rounding instead of round to nearest. Measured on MNIST and MNIST-1D. MNIST-1D rejects the cell that rounds the gradient-down operands, and ADR-031 holds the case | Active |
| `ADR/` | Every architectural decision, one file per decision, in the Tyree and Akerman template. `ADR/README.md` is the index | Active |
| `contract.md` | The training datapath: model, optimizer, rounding points, metrics | Not written |
| `ladder.md` | The experiments L0 to L8, one factor per rung | Not written |
| `results.md` | The measured curve of every rung, with the seed spread | Not written |

## Rule

A number in `results.md` carries its source: the script, the command, and the
seed. A claim without a source does not enter the document.

A figure is named `experiment_<experiment id>_<dataset>_<content>.png`, so a file maps to one
experiment and one document. The footer of every figure states its own file name, the run directory
behind its numbers and the number of runs in each curve. A figure is written by a script, never by
hand.
