# ADR-033: What goes into L5

**Status**: Accepted. 2026-11-19. This record is the closed configuration of rung L5. It does not introduce a
new format. It names, for every open dial of the ladder, the setting L5 runs with, and it points at the run
that closed each dial.

**Date**: 2026-11-19

**Deciders**: Dzemo, on the measurements of `docs/experiment_L1.md`, `docs/experiment_L2.md`,
`docs/experiment_L4.md`, `docs/experiment_block.md` and the tile arm of 2026-11-19.

## Context and problem statement

L5 is the first rung where the forward pass is FP4 as well as the backward pass. Below L5 each rung rounds one
part of the pipeline and leaves the rest in float, so the ladder holds one difference per rung. L5 changes
three things at once: the forward weights, the forward activations and the backward operands all sit on the
FP4 grid, and the weights are stored on it, so the fabric owns them.

Three dials were left open for L5 by earlier records:

| Dial | Opened by | State before this record |
|-|-|-|
| the block shape, 1D block against a square tile | ADR-024, ADR-026 | two shapes carried: 4 by 4 for both passes, 16 by 16 for training with a 1D inference |
| the scale rule | ADR-009, ADR-010 | nearest, which clips the block maximum |
| the scale format, UE6M2 against UE5M3, and `amax/6` against `four6` | `docs/experiment_L1.md`, `experiment_L2.md` | four configurations measured, none separated at width 64 |

This record closes all three, and it states the two runs L5 makes first.

## Decision drivers

1. The accuracy of the trained model, judged by the pre-registered rule of `docs/experiment_L4.md`: a
   comparison fires only when 8 of 10 seeds agree in sign and the mean difference is above two standard
   errors. The reference is the float run of the same fixture and the same ten seeds. Where that reference
   itself sits 0.46 points away, the question is whether a setting costs anything against the setting it
   replaces, not whether it reaches the float number.
2. Area, because Dzemo ranks area first. The block shape sets the number of term paths in the lane, which is
   the largest single block of the engine.
3. Cycles for one term, the second of Dzemo's two criteria.
4. Dynamic range of the 8-bit scale, because a scale that cannot reach a tensor's magnitude clips it.
5. Bit-identical reproduction of every stored number, so that a change can always be checked against the run
   it replaced.

## Considered options

**The block shape.**

| Option | Term paths per block pair | Area | Accuracy at width 64 |
|-|-|-|-|
| 1D block, two copies of `dZ` | 1 | base | 90.04 and 97.36 |
| **16 by 16 tile** | **1** | **base** | **90.03 and 97.35** |
| 8 by 8 tile | 2 | 1,572 to 1,732 LUT6 | not measured at width 64 |
| 4 by 4 tile | 4 | 4,716 to 5,196 LUT6 | not measured at width 64 |

**The scale format and target.**

| Option | Range of the 8-bit scale | Tile arm, MNIST-1D | Tile arm, MNIST |
|-|-|-:|-:|
| **UE6M2, `amax/6`** | **4.66e-10 to 6.44e+09** | **90.03** | **97.35** |
| UE6M2, `four6` | the same | 89.81 | 97.32 |
| UE5M3, `amax/6` | 3.05e-05 to 1.15e+05 | not measured with the tile | not measured with the tile |
| UE5M3, `four6` | the same | not measured with the tile | not measured with the tile |

## Decision outcome

**The L5 configuration, closed.**

| Item | Setting | Closed by |
|-|-|-|
| payload grid | E2M1, `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, 4 bits | Dzemo, 2026-11-18. A wider top code was refused because it changes E2M1 |
| payload rounding | **stochastic**, drawn per value | L4. Round to nearest collapses at width 16 (22.23) and at width 64 (42.76), against 90.03 to 90.40 for stochastic |
| block scale | one 8-bit UE6M2 code per block, `amax/6` | this record. `four6` loses 0.59 points to the tile on MNIST-1D and 0.16 on MNIST, `amax/6` loses 0.01 and 0.00 |
| scale rule, forward pass | nearest table entry | L1d: the rounded-up rule costs 0.81 to 1.84 points on MNIST, and 0.68 to 1.32 in three of the four MNIST-1D configurations |
| scale rule, backward operands | smallest table entry at or above the target | L4: the nearest rule clips the block maximum, and the clip has one sign in every softmax gradient row |
| block shape, backward pass | **16 by 16 tile** | this record, 2026-11-19 |
| block shape, class layer | 16 by 16 tile, padded to 16 columns with zeros | Dzemo, 2026-11-19 |
| block shape, forward pass | **to be measured by L5a and L5b, see below** | the only property this record leaves open |
| second rounding of `dZ` | **does not exist** | the tile holds both axes, so one rounded copy serves `dW` and `dA` |
| accumulator | 48 bits, point at `min(y)`, format 24.24 | ADR-014 |
| lanes | 4 | `docs/datapath.md` |
| optimizer | Adam in float32, no other precision | L5 keeps the L6 question open |
| master weights | float32 for the forward pass of L5, FP4 in L6 | L5 does not store the forward weight in FP4 yet |
| fixture | 784-64-64-10, width 64 | ADR-031, refuted by the width-64 measurement |
| run | 10 epochs, batch 128, 10 seeds, MNIST and MNIST-1D | the ladder |

**Why the tile and not the 1D block.** Both shapes cost one term path, so the tile costs no area. The tile
costs 0.01 points on MNIST-1D and 0.00 on MNIST with `amax/6`. In exchange it removes the second rounding of
`dZ`, which means one encoder pass and one store per step fewer, and it is the shape NVIDIA measured as better
than two 1D copies on a 1.2B model (arXiv 2509.25149, Appendix E.5).

**Why not 4 by 4.** The coarsest shape costs 0.01 points against the 1D block on MNIST-1D and 0.00 on MNIST.
A finer tile can only improve a difference that already sits inside the noise of ten seeds, and it costs four
term paths, which is 8.9 to 9.8 percent of the Z020, against one. ADR-026 carried two shapes to this rung.
This record ends that, and 4 by 4 is dropped from the ladder.

**Why not `four6`.** Under a 16 by 16 tile it loses 0.59 points on MNIST-1D in 8 of 10 seeds. `amax/6` loses
0.01. `four6` also needs the payload error of two candidate scales for every block before it can choose, so
it is the more expensive of the two encodings, and under the tile it is now the less accurate one as well.

## The two runs L5 makes first, and the one property they close

The forward pass at width 64 with a 16 by 16 tile is **not measured**. The tile spans the output axis as well
as the contraction axis, so a forward weight loses a factor of 16 in the number of its scales, and the
granularity probe prices that at 1.5 dB (`experiment_block.md`, 19.15 dB against 20.67 dB). The forward pass
is also the pass with no headroom left: at width 16 it costs 2.3 points on MNIST-1D (L2a 62.30 against 64.98).

L5 therefore runs in two halves, in this order:

| Rung | Forward pass | Backward pass | Purpose |
|-|-|-|-|
| **L5a** | FP4 weights and activations, 1D block along the contraction axis | FP4, 16 by 16 tile | the conservative forward, so the first L5 number is not at risk from an unmeasured shape |
| **L5b** | FP4 weights and activations, **16 by 16 tile** | FP4, 16 by 16 tile | the shape the backward already uses |

L5b is compared against L5a with the same pre-registered rule that closed the tile in the backward pass: 8 of
10 seeds agree in sign and the mean difference is above two standard errors. If L5b matches L5a, the tile
wins in the forward pass too and the weight store shrinks by a factor of 16 in scales. If L5b is worse, the
forward pass keeps the 1D block along the contraction axis, and that is a per-matrix field of the same weight
store, so no format changes.

This is the only open property. Every other setting in the table above is closed, and no run of L5 may change
one of them without a new record.

## Consequences

- **The engine carries two block shapes**, the 1D block and the 16 by 16 tile, and the shape is a per-matrix
  field. The 4 by 4 shape is removed from the lane, which removes three of the four term paths.
- **The random source is in the datapath for good.** Round to nearest is not a fallback, and no area saving
  may be taken from the random generator.
- **The encoder carries both scale rules** and the instruction selects one per pass.
- **Every number in `docs/experiment_L1.md` and `docs/experiment_L2.md` was measured with the nearest scale
  rule**, which is now the forward rule, so those numbers remain the forward numbers and need no re-run. The
  L2 up-rule re-measurement that was queued on 2026-11-19 was cancelled for this reason.
- **The store shrinks.** A 16 by 16 tile holds one scale per 256 values, which is 4.03 bits per value against
  4.50 for the 1D block, once the class layer's padding is counted in.
- **L5 must not run with `four6`.** No arm of L5 exists for it.

## Pros and cons of the options

| Option | Pro | Con |
|-|-|-|
| 1D block in the backward pass | the shape every earlier run used | two copies of `dZ`, two encodings, two stores, and a second rounding |
| 16 by 16 tile in the backward pass | one copy, one encoding, no second rounding, free in accuracy with `amax/6` | costs 0.59 points with `four6`, so it forces the scale target to `amax/6` |
| 4 by 4 tile | 1.4 dB better signal to noise on the granularity probe | four term paths, 8.9 to 9.8 percent of the Z020, and no accuracy left to buy |
| `four6` scale target | 0.3 to 0.5 dB better at the 1D block granularity | loses 0.59 points under the tile, and needs a second payload pass |

## Links

- `docs/experiment_L4.md`, the tile arm of 2026-11-19 and the round-to-nearest table
- `docs/experiment_block.md`, the term count, the lane and the granularity probe
- `docs/experiment_L1.md`, the scale rule measured on the forward pass
- `docs/experiment_L2.md`, L2 and L2a, the forward pass in FP4
- `docs/datapath.md` section 4.8, the encoder, both scale rules and both block shapes
- `docs/spec.md` version 0.15, the per-use scale rule
- ADR-014 the accumulator, ADR-024 the block lies along the contraction axis, ADR-026 the tile,
  ADR-031 superseded, ADR-032 the block scale must never clip the block maximum
- `python/l3.py`, the switch `FPG4_TILE`, and `python/tile_figure.py`
