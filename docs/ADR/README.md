# Architecture decision records

Every architectural decision of this project, in the template of Jeff Tyree and Art Akerman:
[decision-record-template-by-jeff-tyree-and-art-akerman](https://github.com/architecture-decision-record/architecture-decision-record/blob/main/locales/en-001/templates/decision-record-template-by-jeff-tyree-and-art-akerman/index.md).

The template fields are used with these names: Issue, Decision, Status, Group, Assumptions, Constraints,
Positions, Argument, Implications, Related decisions, Related requirements, Related artifacts, Related
principles, Notes. Two fields are added: Date, inside Status, and the commit hash, because a decision in
this project must be traceable to a diff.

**Naming rule**: `experiment_<rung>-<title>.md`. The rung is the experiment the decision was made in, or
`ALL` for a rule that holds for every experiment. The title is short and says the decision, not the topic.

A decision that is not settled is still recorded, with status **Pending**. A pending record exists so that
the option set and the reason for waiting are not lost. It is not a decision.

## Index

| ADR | File | Rung | Group | Status | Date | Commit |
|-|-|-|-|-|-|-|
| 001 | `experiment_E-engine-scope.md` | E, engine | scope | Decided | 2026-11-09 | `4ed9dce` |
| 002 | `experiment_E-target-device.md` | E, engine | scope | Decided | 2026-11-16 | `237f74f` |
| 003 | `experiment_E-multiplier-in-logic.md` | E, engine | datapath | Decided | 2026-10-09 | spec section 9 |
| 004 | `experiment_E-four-lanes.md` | E, engine | datapath | Decided | 2026-11-11 | `6240628` |
| 005 | `experiment_L0-float32.md` | L0 | protocol | Decided | 2026-11-12 | `aea728b` |
| 006 | `experiment_L1-payload-e2m1.md` | L1 | format | Decided | 2026-10-29 | `0e0b13b` |
| 007 | `experiment_L1-block-16.md` | L1 | format | Decided | 2026-10-29 | `0e0b13b` |
| 008 | `experiment_L1-special-codes.md` | L1 | format | Decided | 2026-11-11 | `a8f1e58` |
| 009 | `experiment_L1b-scale-amax6.md` | L1b | format | Decided | 2026-10-29 | `0e0b13b` |
| 010 | `experiment_L1c-four6.md` | L1c | format | Decided | 2026-10-29 | `f621008` |
| 011 | `experiment_L1-two-scale-formats.md` | L1 | format | Pending | 2026-11-14 | `b21d0b6` |
| 012 | `experiment_L1e-two-stage-multiply.md` | L1e | datapath | Decided | 2026-10-09 | spec section 9 |
| 013 | `experiment_L1e-exact-integer.md` | L1e | datapath | Decided | 2026-11-11 | `b7b8001` |
| 014 | `experiment_L1e-accumulator-48.md` | L1e | datapath | Decided | 2026-11-11 | `6240628` |
| 015 | `experiment_L1e-bias-accumulator.md` | L1e | datapath | Decided | 2026-11-11 | `c899d81` |
| 016 | `experiment_ENC-no-division.md` | ENC | datapath | Decided | 2026-11-11 | `f621008` |
| 017 | `experiment_ENC-window-and-amax.md` | ENC | datapath | Decided | 2026-11-11 | `18bf3ab` |
| 018 | `experiment_ENC-cycles-open.md` | ENC | datapath | Pending | 2026-11-11 | `f621008` |
| 019 | `experiment_L2-scope.md` | L2 | protocol | Decided | 2026-11-12 | `aea728b` |
| 020 | `experiment_L2a-prequantized-input.md` | L2a | protocol | Decided | 2026-11-12 | `aea728b` |
| 021 | `experiment_ALL-paired-rule.md` | ALL | protocol | Decided | 2026-11-12 | `aea728b` |
| 022 | `experiment_ALL-deployed-model.md` | ALL | protocol | Decided | 2026-11-14 | `490ba9a` |
| 023 | `experiment_L2-format-open.md` | L2 | protocol | Pending | 2026-11-14 | `b21d0b6` |
| 024 | `experiment_ALL-transpose-is-encoder.md` | ALL | datapath | Decided, amended 2026-11-16 | 2026-11-15 | `0e58015` |
| 025 | `experiment_block-two-copies.md` | block | format | Decided | 2026-11-15 | `0e58015` |
| 026 | `experiment_block-tile-dial.md` | block | format | Deferred to L5, two shapes carried | 2026-11-16 | `14db604` |
| 027 | `experiment_L3-scope.md` | L3 | protocol | Approved | 2026-11-16 | `237f74f` |
| 028 | `experiment_L6-optimizer-forms.md` | L6 | protocol | Decided | 2026-11-15 | `863110b` |
| 029 | `experiment_L7-eco.md` | L7 | format | Decided | 2026-11-15 | `863110b` |
| 030 | `experiment_ALL-evidence.md` | ALL | tooling | Decided | 2026-11-16 | `be568b1` |

## What is not recorded here

A number. A number lives in the document of its experiment, next to the script and the seed that produced
it. A decision record says what was decided and why, and it points at the artifacts that hold the numbers.

## Reading order

If you read nothing else, read 001 (what the engine is for), 007 (the block), 014 (the accumulator), 026
(the tile, which is open) and 028 (the two forms of L6).
