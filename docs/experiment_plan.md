## The experiment ladder

This is the general ladder to prove/disprove FPG4 viability.
All experiments will be in software, Python before RTL work starts.


|ladder|what changes|what does it answer
|---|---|---|
|L0|nothing, all FP64|does the model learn at all?|
|L1|PTQ - FP4 after training|FP4 only for inference, what do we lose from FP64|
|L1a|FP4 w FP64 scale|what do we lose from just the payload quantization|
|L1b/c|FP4 w E6M2/E5M3 scale rules amax/6 or four6|what do we lose from the payload and scale quantization|
|L1d|PTQ - FP4 for weights **and activations**|what do we lose from FP64 when both **weights and activations** are quantized|
|L1e|L1d with a stated **datapath** (integer payload product, fixed-width accumulator, rounded scale product)|does the datapath rounding cost accuracy, and how wide must the accumulator be|
|L2|QAT - FP4 for **weights only** of the forward pass, master **weights and activations** in FP64, backprop FP64|how does FP4 influence the forward pass in training|
|L2a|QAT - FP4 for weights **and activations** of the forward pass, master weights in FP64, backprop FP64|how does weight+activation quantization affect the forward pass in training for full QAT|
|L3|FP64 weights and activations, FP4 backprop only with Round To Nearest|how many updates vanish? discretization error (gap)|
|L4|FP64 weights and activations, FP4 backprop with Stochastic Rounding| SR vs RTN, does it recover L3?|
|L5|FP4 forward and backprop (L2+L3+L4), optimizer FP64|does native weight and activations in FP4 train?|
|L6|FP4 forward, backprop AND optimizer|full native FP4 loop trains?|
|L7|add ECO, second FP4 tensor|does it recover updates below 1 grid step|
|L8|add Hadamard mixing|does outlier transformation help?|

Compare N to N-1. If delta is < than seed induced variance, the change was not substantial. Always test across seeds.

## The arithmetic axes (L1e)

The rungs above change which **values** sit on the FP4 grid. They do not change the **arithmetic**. In every Python rung the operands are decoded to FP64 and the product and the sum run in FP64. L1e is the first rung that changes the arithmetic.

Four separate questions, and they must be measured one at a time:

|axis|question|why it matters|
|-|-|-|
|payload product|does the product of two FP4 payloads round?|No. Payloads are `k/2` with k in {0,1,2,3,4,6,8,12}. The product is `(k1*k2)/4`. The integer `k1*k2` is at most 144, so the product is a multiple of 0.25 and it is exact.|
|scale product|does `scale_activation * scale_weight` round?|Yes in hardware. Two 8-bit codes give mantissas of 2 or 3 bits, so the product needs a width and a rounding step.|
|accumulator|how many bits, and where does the alignment loss happen?|Block sums have different scales. Adding them aligns mantissas, and that drops bits.|
|dequantize point|per block, or per output element?|Per block keeps the payload math integer and exact. Per element needs a real multiplier per product.|