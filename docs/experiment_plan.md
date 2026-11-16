## The experiment ladder

This is the general ladder to prove/disprove FPG4 viability.
All experiments will be in software, Python before RTL work starts.


|ladder|what changes|what does it answer|
|---|---|---|
|L0|nothing, all FP32|does the model learn at all?|
|L1|PTQ - FP4 after training|FP4 only for inference, what do we lose from FP32|
|L1a|FP4 w FP64 scale|what do we lose from just the payload quantization|
|L1b/c|FP4 w E6M2/E5M3 scale rules amax/6 or four6|what do we lose from the payload and scale quantization|
|L1d|PTQ - FP4 for weights **and activations**|what do we lose from FP32 when both **weights and activations** are quantized|
|L1e|L1d with a stated **datapath** (integer payload product, fixed-width accumulator, rounded scale product)|does the datapath rounding cost accuracy, and how wide must the accumulator be|
|L2|QAT - FP4 for **weights only** of the forward pass, master **weights and activations** in FP32, backprop FP32|how does FP4 influence the forward pass in training|
|L2a|QAT - FP4 for weights **and activations** of the forward pass, master weights in FP32, backprop FP32|how does weight+activation quantization affect the forward pass in training for full QAT|
|L3|FP32 weights and activations, FP4 backprop only with Round To Nearest|how many updates vanish? discretization error (gap)|
|L4|FP32 weights and activations, FP4 backprop with Stochastic Rounding| SR vs RTN, does it recover L3?|
|L5|FP4 forward and backprop (L2+L3+L4), optimizer FP32|does native weight and activations in FP4 train?|
|L6|FP4 forward, backprop AND optimizer|full native FP4 loop trains?|
|L7|add ECO, no second tensor: the rounding error goes into the momentum|does it recover updates below 1 grid step|
|L8|add Hadamard mixing|does outlier transformation help?|

The simulation width is **float32** from L0 up: the master, the activations, the optimizer state, the
arithmetic. The rungs L1a to L1e were measured when the width was float64, and `experiment_L1.md`
records their numbers with that width. In the L1a row, "FP64 scale" names an exact scale with no
8-bit code, not the simulation width.

Compare N to N-1. If delta is < than seed induced variance, the change was not substantial. Always test across seeds.

## The three dials that L5 decides

Decided by Dzemo, 2026-11-16. Three format questions are carried open until rung L5. L5 is the first rung
where the forward pass and the backward pass are both FP4, so it is the first rung where all three can be
compared under the conditions that matter.

| Dial | Settings carried | What decides it | Where the evidence stands |
|-|-|-|-|
| the scale rule | `amax/6` (L1b) or `four6` (L1c) | accuracy of a trained model, paired by seed | `f6-e5m3` is best on L2 and L2a. MNIST cannot decide. ADR-010 |
| the scale format | UE6M2 (`[7 sig][7 exp]`) or UE5M3 (`[8 sig][6 exp]`) | accuracy of a trained model, and the scale product width in the lane | UE5M3 is better on L2 and L2a. ADR-011 and ADR-023 stay Pending |
| the tile | 4 by 4 in the forward and the backward pass, or 16 by 16 in training with 1D blocks at inference | accuracy on MNIST-1D, then area and cycles | 4 by 4 is the only shape that passes both datasets in training. The 16 by 16 return to 1D is measured on float weights only. ADR-026 |

Two shapes are carried in every measurement that has a weight store. The 8 by 8 tile and the two 1D copies
are dropped. Rungs L3 and L4 are not affected, because both keep float weights and float activations, so no
tile exists and no store is read.

## L6 has two forms, and both MUST be run

L6 is the first rung with no float32 parameter anywhere. Two separate questions sit in it. They must be measured one at a time, the same way every other rung is measured.

|form|what stays float32|what is FP4|question|
|-|-|-|-|
|L6a, wide optimizer state|Adam `m` and `v`|weights, activations, gradients, the update|does a full FP4 loop train when the optimizer state stays float32?|
|L6b, FP4 optimizer state|nothing|the same as L6a, and `m` and `v`|can the optimizer state itself be FP4?|

`m` is Adam's running mean of the gradient. `v` is Adam's running mean of the squared gradient. Adam divides the update by `sqrt(v)`. The values of `v` are squares of gradients, so they fall in a wide range across the layers of one network. A 4-bit block format holds 8 magnitudes per block. Whether it holds `v` is not known.

This experiment MUST be run. It MUST NOT be decided by argument. If the answer is yes, the whole training state of 13,002 weights falls from 1,306,701 bits (master, `m`, `v` in float32) to 175,527 bits (weights, `m`, `v` in FP4 at 4.5 bits per value). That is 7.4 times less.

## The arithmetic axes (L1e)

The rungs above change which **values** sit on the FP4 grid. They do not change the **arithmetic**. In every Python rung the operands are decoded to FP32 and the product and the sum run in FP32. L1e is the first rung that changes the arithmetic.

Four separate questions, and they must be measured one at a time:

|axis|question|why it matters|
|-|-|-|
|payload product|does the product of two FP4 payloads round?|No. Payloads are `k/2` with k in {0,1,2,3,4,6,8,12}. The product is `(k1*k2)/4`. The integer `k1*k2` is at most 144, so the product is a multiple of 0.25 and it is exact.|
|scale product|does `scale_activation * scale_weight` round?|Yes in hardware. Two 8-bit codes give mantissas of 2 or 3 bits, so the product needs a width and a rounding step.|
|accumulator|how many bits, and where does the alignment loss happen?|Block sums have different scales. Adding them aligns mantissas, and that drops bits.|
|dequantize point|per block, or per output element?|Per block keeps the payload math integer and exact. Per element needs a real multiplier per product.|