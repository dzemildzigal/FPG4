## The experiment ladder

This is the general ladder to prove/disprove FPG4 viability.
All experiments will be in software, Python before RTL work starts.


|ladder|what changes|what does it answer
|---|---|---|
|L0|nothing, all fp32|does the model learn at all?|
|L1|FP4 after training|FP4 only for inference, what do we lose from FP32|
|L2|FP4 for forward pass, master weights fp32, backprop fp32|how does FP4 influence the forward pass in training|
|L3|FP4 backprop with Round To Nearest|how many updates vanish? discretization error (gap)|
|L4|FP4 backprop with Stochastic Rounding| SR vs RTN, does it recover L3?|
|L5|FP4 forward and backprop, optimizer fp32|does native weight FP4 train?|
|L6|FP4 forward, backprop AND optimizer|full native FP4 loop trains?|
|L7|add ECO, second FP4 tensor|does it recover updates below 1 grid step|
|L8|add Hadamard mixing|does outlier transformation help?|

Compare N to N-1. If delta is < than seed induced variance, the change was not substantial. Always test across seeds.