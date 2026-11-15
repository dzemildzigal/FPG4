"""
Counts the 16-value blocks that the encoder must produce in ONE training step of the 784-16-16-10 MLP, batch B = 128.
Every number comes from the shapes. Nothing is measured on hardware.
A block runs along the axis that its GEMM sums over (the contraction axis):
  forward   Z  = A  @ W      sums over K.      A blocks along K,     W blocks along K.
  dA        dA = dZ @ W.T    sums over N.      dZ blocks along N,    W.T blocks along N     (not needed for the first layer)
  dW        dW = A.T @ dZ    sums over batch.  A.T blocks along B,   dZ blocks along B
"""
import math
c = lambda n: math.ceil(n / 16)
B = 128
layers = [("L1", 784, 16), ("L2", 16, 16), ("L3", 16, 10)]
online, static = {}, {}
w_fwd = w_bwd = 0
for i, (name, K, N) in enumerate(layers):
    first = (i == 0)
    # the activation that enters this layer: the input X for L1 (static data), A1 / A2 otherwise
    tgt = static if first else online
    tgt[f"{name} A forward (along K)"] = B * c(K)
    tgt[f"{name} A for dW (along batch)"] = K * (B // 16)
    online[f"{name} dZ for dW (along batch)"] = N * (B // 16)
    if not first:
        online[f"{name} dZ for dA (along N)"] = B * c(N)
    w_fwd += N * c(K)                      # weight blocks along K, re-encoded after every optimizer step
    if not first:
        w_bwd += K * c(N)                  # the second 1D copy: blocks along N (the first layer needs none)
for k, v in online.items(): print(f"online  {k:34} {v:6d}")
for k, v in static.items(): print(f"static  {k:34} {v:6d}   (the dataset input, can be stored already encoded)")
print()
print(f"online blocks per step (activations + gradients) : {sum(online.values()):6d}")
print(f"static input blocks per step                     : {sum(static.values()):6d}")
print(f"weight blocks, forward copy (all 3 layers)       : {w_fwd:6d}")
print(f"weight blocks, second 1D copy (layers 2 and 3)   : {w_bwd:6d}   = {w_bwd*72} bits = {w_bwd*72//8} bytes")
tot = sum(online.values()) + w_fwd + w_bwd
print(f"second copy as a share of all blocks encoded     : {100*w_bwd/tot:6.2f} %   (static input not counted)")
print(f"second copy as a share of the online blocks      : {100*w_bwd/sum(online.values()):6.2f} %")
print(f"second copy if W1 were copied too (my old claim) : {w_fwd:6d} blocks = {w_fwd*72} bits = {w_fwd*72/8/1024:.1f} KiB")
# angles
import numpy as np
print()
for c_ in (0.9796, 0.9847, 0.9192, 0.9358, 0.9905, 0.9936, 0.9858, 0.9909, 0.3257, 0.4182, 0.1454, 0.1752, 0.4248, 0.5751, 0.6286, 0.6869):
    print(f"cos {c_:.4f} = {np.degrees(np.arccos(c_)):5.1f} degrees")
# MAC cycles per step for the "is the encoder the cost" question, 4 lanes, one block-pair per lane per cycle
fwd = sum(N * c(K) for _, K, N in layers) * B
dA = sum(K * c(N) for i, (_, K, N) in enumerate(layers) if i > 0) * B
dW = sum(K * N * (B // 16) for _, K, N in layers)
print()
print(f"block-MACs per step: forward {fwd}, dA {dA}, dW {dW}, total {fwd+dA+dW}")
print(f"4 lanes, 1 block-MAC per lane per cycle: {(fwd+dA+dW)/4:.0f} cycles per step")
print(f"encoder at 4 cycles per block, online only: {sum(online.values())*4} cycles; with the static input encoded online too: {(sum(online.values())+sum(static.values()))*4}")
