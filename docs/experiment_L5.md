# L5: the forward pass and the backward pass both on the FP4 grid, the optimizer in float32

**Date**: 2026-11-19.

**What L5 asks.** Can a model train with every operand of every GEMM on the FP4 grid, in both passes, and
still reach the accuracy of the float run? The optimizer stays in float32, so L5 is not yet the full native
loop. That is L6.

**Why it is the hard rung.** Below L5 each rung puts one part of the pipeline on the grid and leaves the rest
in float. L5 puts the forward weights, the forward activations and all four backward operands there at once,
so the error of every GEMM adds up inside one step. The forward pass is also the pass that feeds the loss
directly, so its error cannot be corrected by the backward pass.

## The three arms, and the one thing each changes

| Arm | Forward block | Forward payload | Forward scale |
|-|-|-|-|
| L5a | 1D, one scale per 16 values along the contraction axis | stochastic | nearest table entry |
| L5b | 16 by 16 tile | stochastic | nearest table entry |
| L5c | 1D, one scale per 16 values along the contraction axis | round to nearest | nearest table entry |

The backward pass is identical in the three arms, because L4 closed it: 16 by 16 tile, stochastic payload,
block scale rounded up to the next table entry, `dZ` rounded once. So a difference between two arms belongs
to the forward pass and to nothing else.

## Method

One switch, `FPG4_FWD`, and one more, `FPG4_FWD_PAYLOAD`. `FPG4_TILE=16` sets the backward block and
`FPG4_SCALE_ROUNDING=up` sets the backward scale rule, both unchanged from L4. The forward pass passes its own
scale rule explicitly, the nearest entry, so the two passes can hold different rules at the same time.

The fixture is width 64 (784-64-64-10), ten seeds, ten epochs, batch 128, cell DAW, configuration `a6-e6m2`.
That configuration is UE6M2 with the target `amax/6`, closed by ADR-033.

**Two numbers are recorded per run.**

- `test_acc` is the **float master** evaluated on the test set: the weights as Adam holds them, float
  activations, no rounding. It is comparable with L0, L3 and L4, and it is the number the ladder uses.
- `test_acc_q` is the **deployed model**: the FP4 weight store read back, the activations rounded, the whole
  forward pass on the grid. This is the only number the fabric can produce at inference. It is new in L5.

## Why a probe was needed before the arms could be read

The three arms moved three different things, and a trained model cannot say which rounding caused which part
of the loss, because training moves the weights to compensate. So `python/l5_probe.py` holds the weights
still. It loads the saved float weights of L0, changes only the forward pass, and evaluates. Nothing trains.

Forward pass only, width 64, 3 seeds, 100,000 test rows, configuration `a6-e6m2`, scale rule nearest.
Round-to-nearest settings are deterministic. Stochastic settings are the mean of three draws.

| Forward pass | MNIST-1D | Delta | MNIST | Delta |
|-|-:|-:|-:|-:|
| float | 90.48 | | 97.40 | |
| weights on the grid, activations float | 85.44 | −5.04 | 97.07 | −0.33 |
| 1D block, round to nearest | 83.19 | −7.29 | 96.76 | −0.65 |
| 1D block, stochastic | 77.95 | −12.53 | 96.20 | −1.20 |
| 16 by 16 tile, round to nearest | 79.63 | −10.85 | 96.48 | −0.92 |
| 16 by 16 tile, stochastic | 70.68 | −19.80 | 95.71 | −1.69 |

The probe answers three questions before a single arm trains:

1. **The weight store is the expensive operand.** Rounding only the weights costs 5.04 points on MNIST-1D.
   Rounding the activations as well costs another 2.25. At width 16 this was never visible.
2. **The 16 by 16 tile is the wrong shape for the forward pass.** It costs 3.56 points more than the 1D
   block on MNIST-1D with round to nearest, and 7.27 with stochastic rounding.
3. **Stochastic rounding is the wrong payload rule for the forward pass.** It costs 5.24 points against
   round to nearest on MNIST-1D and 0.56 on MNIST.

No third finding was expected. ADR-033 had recorded one payload rule for the whole pipeline.

## MNIST-1D, the dataset that separates

Ten seeds, width 64, cell DAW, `a6-e6m2`. The float reference is 90.50 with a standard error of 0.20, ten
seeds, from `out/l0_wide_mnist1d.log`.

| Arm | float master | sd | deployed | sd | gap | master against float |
|-|-:|-:|-:|-:|-:|-:|
| L5a | 84.26 | 0.33 | 74.88 | 1.23 | 9.38 | −6.24 |
| L5b | 81.97 | 0.72 | 69.61 | 1.00 | 12.36 | −8.53 |
| **L5c** | **85.58** | 0.86 | **82.30** | 0.73 | **3.28** | **−4.92** |

Figure: `experiment_L5_mnist1d_arms.png`.

## MNIST, where the forward pass is nearly free

Ten seeds, width 64, cell DAW, `a6-e6m2`. The float reference is 97.45 with a standard error of 0.05.

| Arm | float master | sd | deployed | sd | gap | master against float |
|-|-:|-:|-:|-:|-:|-:|
| L5a | 97.41 | 0.11 | 96.59 | 0.24 | 0.82 | −0.04 |
| L5b | 97.34 | 0.10 | 96.28 | 0.41 | 1.06 | −0.11 |

Figure: `experiment_L5_mnist_arms.png`.

L5c was not run on MNIST. The two dials it decides were already invisible there: L5a against L5b is 0.07
points, which is the whole size of the forward tile's effect on this dataset, and the probe shows the payload
rule is worth 0.56. A run that cannot separate the arms cannot decide the dial, so the MNIST arm of L5c would
have cost 20 runs and answered nothing.

## The paired comparisons, under the rule fixed before L4

The rule: a comparison fires only when 8 of 10 seeds agree in sign and the mean difference is above two
standard errors. Comparisons are paired seed by seed.

| Dataset | Comparison | Delta | se | Agree | 2 se | Fires |
|-|-|-:|-:|-:|-:|-|
| MNIST-1D | **L5c against L5a, float master** | **+1.32** | 0.33 | 9 of 10 | 0.65 | **YES** |
| MNIST-1D | **L5c against L5a, deployed** | **+7.42** | 0.32 | 10 of 10 | 0.64 | **YES** |
| MNIST-1D | **L5a against L5b, float master** | **+2.29** | 0.29 | 10 of 10 | 0.57 | **YES** |
| MNIST-1D | L5c against float | −4.92 | 0.34 | 10 of 10 | 0.67 | YES, worse |
| MNIST-1D | L5a against float | −6.24 | 0.26 | 10 of 10 | 0.52 | YES, worse |
| MNIST-1D | L5b against float | −8.53 | 0.28 | 10 of 10 | 0.56 | YES, worse |
| MNIST | L5a against L5b, float master | +0.07 | 0.03 | 6 of 10 | 0.07 | no |
| MNIST | L5a against float | −0.04 | 0.07 | 6 of 10 | 0.14 | no |
| MNIST | L5b against float | −0.11 | 0.07 | 7 of 10 | 0.13 | no |

**On MNIST nothing fires.** The forward pass on the FP4 grid costs 0.04 points and the model is at the float
level. MNIST keeps its role: it cannot separate anything at this width.

**On MNIST-1D three comparisons fire, and they close both forward-pass dials.**

## The three findings of L5

**1. The forward payload rule is round to nearest.** 1.32 points on the master and 7.42 on the deployed model.
This is the reverse of the backward pass, where round to nearest collapses the run: 22.23 at width 16 and
42.76 at width 64. The two passes need opposite rules, and one rule for the pipeline, which is what ADR-033
said, is wrong. ADR-034 records the correction.

Why the directions are opposite. In the backward pass the rounding error of a softmax gradient row has one
sign, because every entry in the row is a difference from the true class and the largest entry sits above the
top payload code. Stochastic rounding removes that bias and round to nearest keeps it, so the row stops
summing to zero and Adam walks the logits. In the forward pass there is no such summing constraint on the
activation, so the bias cancels between the two directions of rounding, and what is left is the variance,
which stochastic rounding adds and round to nearest does not.

**2. The forward block is the 1D block, not the tile.** 2.29 points, 10 of 10 seeds. The tile spans the
output axis as well as the contraction axis, so a forward weight loses a factor of 16 in scale count, and the
forward pass has no way to compensate because its error goes straight into the loss.

**Reading the trained rung against the probe.** The probe prices the forward pass at −7.29 points on a
float-trained model. L4-DAW, whose forward pass was float and whose backward pass is the same as L5's, scored
90.03. L5c scores 85.58, so the forward pass costs 4.45 points after training, and **training recovers 2.84 of
the 7.29 the probe measured.** QAT halves the price of the forward pass on this dataset and does not remove
it.

**3. The deployed model costs 3.28 points against the float master even with both dials right.** That gap is
the weight store read back as FP4 with no float copy in the forward pass. It is the price L6 inherits: L6
removes the float master, so it must keep this gap from growing.

## The datapath after L5, and every dial is now closed

| Element | Setting | Closed by |
|-|-|-|
| payload grid | E2M1, `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, 4 bits | Dzemo, 2026-11-18 |
| **forward payload rule** | **round to nearest** | L5c, +1.32 master, +7.42 deployed |
| **backward payload rule** | **stochastic** | L4 and L4c: round to nearest collapses, 22.23 and 42.76 |
| forward block | 1D block along the contraction axis | L5a against L5b, +2.29 |
| backward block | 16 by 16 tile | the L4 tile arm |
| forward scale rule | nearest table entry | L1d, ADR-032 |
| backward scale rule | smallest entry at or above the target | ADR-032 |
| scale format | UE6M2, one 8-bit code per block | ADR-033 |
| scale target | `amax/6` | ADR-033 |
| accumulator | 48 bits, point at `min(y)`, 24.24 | ADR-014 |
| lanes | 4 | `docs/datapath.md` |
| class layer | 16 by 16 tile padded to 16 columns in the backward pass, 1D block in the forward pass | Dzemo, 2026-11-19 |
| optimizer | Adam in float32 | L5 does not question it, L6 does |

## The silicon price of two payload rules, and what is not counted

The two rules share everything except one addend. Round to nearest adds the constant 0.5 to the position of
the value inside its grid step. Stochastic rounding adds a uniformly distributed number in `[0, 1)`. So the
price is a multiplexer on the addend input of one adder per payload value, selected by one instruction bit.

What that costs in LUT6 is **not counted**. It is an E9 item, and it joins the second scale-constant list of
ADR-032 as an estimate until synthesis.

The gain is on the other side: at inference the forward pass draws nothing, so the random source and its
comparators are clock-gated off in the inferring configuration. The random source stays in the area because
training needs it, and its power goes to leakage when the model infers.

## Run times, measured

| Work | Machine | Runs | Wall | Per run |
|-|-|-:|-:|-:|
| MNIST, L5a | this machine, 4 workers | 10 | 0.81 h | 1,160 s |
| MNIST, L5b | this machine, 4 workers | 10 | 0.62 h | 918 s first wave, 425 s tail |
| MNIST-1D, L5a | Ryzen, 8 workers | 10 | 2.42 h | 4,350 s under full load |
| MNIST-1D, L5b | Ryzen, 8 workers | 10 | 1.28 h | 2,300 s under full load |
| MNIST-1D, L5c | Ryzen, 8 workers | 6 | 1.12 h | 4,017 s |
| MNIST-1D, L5c | this machine, 4 workers | 4 | 2.16 h | 7,770 s |

The same code costs 1.9 times more per run on this machine than on the Ryzen for MNIST-1D. That is a measured
fact about the two boxes, not a guess.

**L5a was 1.9 times slower than L5b on the same machine, and the cause is not known.** Both run the same code
with the same shapes and the same eight workers, and the backward pass is identical. The only difference is the
shape of the forward block, and the 16 by 16 tile should be the slower of the two, not the faster. It does not
affect any accuracy above, but it is not explained and it is recorded here rather than smoothed over.

## Artifacts

| File | Content |
|-|-|
| `python/l3.py` | `FPG4_FWD` (float, 1d, tile16), `FPG4_FWD_PAYLOAD` (sr, rtn), `FPG4_TILE`, `FPG4_SCALE_ROUNDING`, `FPG4_RUNG` |
| `python/runs_l5.py` | the driver that maps an arm to those switches and runs one cell over a seed list |
| `python/l5_probe.py` | the forward-pass probe, writes `out/l5_probe.json` |
| `python/l5_figure.py` | the four figures, reads the run files and the probe file, types no number |
| `out/l5_w64_t16_f1d/mnist` | L5a on MNIST |
| `out/l5_w64_t16_ftile16/mnist` | L5b on MNIST |
| `/tmp/hunt/ryzen_l5/l5a` | L5a on MNIST-1D, pulled from the Ryzen |
| `/tmp/hunt/ryzen_l5/l5b` | L5b on MNIST-1D |
| `/tmp/hunt/ryzen_l5/l5c` | L5c on MNIST-1D, four from this machine and six from the Ryzen |
| `docs/experiment_L5_mnist1d_arms.png` | the three arms on MNIST-1D |
| `docs/experiment_L5_mnist_arms.png` | the two arms on MNIST |
| `docs/experiment_L5_mnist1d_forward_ptq.png` | the forward pass alone on MNIST-1D |
| `docs/experiment_L5_mnist_forward_ptq.png` | the forward pass alone on MNIST |

## What L5 does not measure

- **L5c on MNIST.** Not run, and the reason is above: 0.07 points of separation between the arms.
- **The deployed gap at width 16.** The 3.28 points are width 64 on MNIST-1D. Nothing here says what the gap
  is on the narrow fixture, and the narrow fixture is not the target.
- **The LUT6 of the addend multiplexer.** E9.
- **The random source's area.** Not designed yet. L5 proves it is required in the backward pass, so it cannot
  be removed to save area, but its cost is open.
- **The second scale-constant list** of ADR-032 in silicon. E9.
- **Anything about the optimizer on the grid.** L6a and L6b.

## Figures

| File | What it shows |
|-|-|
| `experiment_L5_mnist1d_arms.png` | L5a, L5b and L5c against the float line, float master and deployed model |
| `experiment_L5_mnist_arms.png` | L5a and L5b against the float line |
| `experiment_L5_mnist1d_forward_ptq.png` | the six forward settings, no training, deltas against the float pass |
| `experiment_L5_mnist_forward_ptq.png` | the same on MNIST |
