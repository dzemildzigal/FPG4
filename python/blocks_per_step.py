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

print("\n-- the weight store, in bits. A block of n values costs 4*n payload bits plus 8 scale bits.")
def store(layers, tiled, label):
    total = 0
    print(f"  {label}")
    for name, K, N in layers:
        blocks = 1 if name in tiled else N * (K // 16)
        bits = 4 * K * N + 8 * blocks
        total += bits
        print(f"    {name}: {K} by {N} = {K*N:5d} values, {blocks:4d} scales, {bits:6d} bits, "
              f"{bits/(K*N):.3f} bits per value")
    print(f"    total {total} bits = {total/8:.0f} bytes")
    return total
a = store(layers, set(), "all three matrices blocked along K, 16 values per block")
b = store(layers, {"L2", "L3"}, "W2 and W3 as one 16 by 16 tile, W1 blocked along K: the decision")
print(f"  the decision saves {a - b} bits, {100*(a-b)/a:.3f}%")

print("\n-- inference against training, per sample of a batch of 128, 4 lanes, encoder at 4 cycles per block")
LANES, ENC = 4, 4
fwd_blocks = B * (c(784) * 16 + c(16) * 16 + c(16) * 10)   # blocks along K times the outputs of the layer
act_online = B * (c(16) + c(16))                    # A1 and A2 only
per_sample = lambda blocks: blocks / B / LANES      # cycles, because one lane does one block-MAC per cycle
rows = [
    ("inference, input pre-encoded", fwd_blocks, act_online),
    ("inference, input encoded online", fwd_blocks, act_online + B * c(784)),
    ("training, forward + dA + dW", 211456, 1104),
]
for name, macs, enc_blocks in rows:
    print(f"  {name:34s} {macs/B:7.1f} block-MACs per sample, MAC {per_sample(macs):6.1f} cycles, "
          f"encoder {enc_blocks*ENC/B:6.1f} cycles, {enc_blocks*ENC/B > per_sample(macs) and 'ENCODER BOUND' or 'MAC bound'}")
print(f"  training also re-encodes the weights once per step: 810 blocks x {ENC} cycles = {810*ENC} cycles = "
      f"{810*ENC/B:.1f} cycles per sample")

print("\n-- the tile and the lane: t terms per block pair")
# A lane takes 16 payload products, adds them into one block sum S, and makes one term S * P with one scale product P.
# A tile of side a gives 16 / a groups inside those 16 values along the contraction axis. Each group has its own scale.
# So the lane makes t = 16 / a terms per block pair. 16 by 16: t = 1. 8 by 8: t = 2. 4 by 4: t = 4.
# Two ways to pay for the extra terms:
#   parallel  one term path per group. Costs LUT6. Costs no cycles.
#   serial    one term path, used once per group. Costs no LUT6 (a select on S and on the scale, about 21 bits per lane).
#             Costs cycles, because the 16 products wait for the term path.
# LUT6 of one term path of one lane, from datapath.md 4.5 and 9: multiply 135, barrel shifter 210 to 250, 48-bit add 48.
# They are estimates. Rung E9 measures them.
TERM_LO, TERM_HI = 135 + 210 + 48, 135 + 250 + 48
Z020_LUT = 53_200
groups = lambda n, a: math.ceil(n / a)          # groups of a values along a contraction axis of n values
# The GEMMs that read a tiled weight are the forward GEMM (contracts over K) and the dA GEMM (contracts over N).
# W1 is never tiled: it has no dA GEMM. So only layers 2 and 3 count here.
reads = []                                       # (outputs, contraction length) of every tiled GEMM
for i, (_, K, N) in enumerate(layers):
    if i > 0:
        reads.append((N, K))                     # forward: N outputs, contraction over K
        reads.append((K, N))                     # dA: K outputs, contraction over N
print(f"{'tile':>6} {'t':>2} | {'parallel: extra LUT6, 4 lanes':>32} {'Z020':>11} | "
      f"{'serial: extra cycles per step':>30} {'upper':>7} {'exact':>7} | {'inference':>9} | {'wide net, train':>15}")
for side in (16, 8, 4):
    t = 16 // side
    lut_lo, lut_hi = (t - 1) * 4 * TERM_LO, (t - 1) * 4 * TERM_HI
    upper = (t - 1) * sum(o * c(n) for o, n in reads) * B / 4                     # t groups in every block
    exact = sum(o * (groups(n, side) - c(n)) for o, n in reads) * B / 4           # real groups, the empty ones skipped
    infer = (t - 1) * sum(N * c(K) for i, (_, K, N) in enumerate(layers) if i > 0) / (fwd_blocks / B)
    wide = (2 * t + 1) / 3                      # three GEMMs of equal size, two of them read the tile, one does not
    print(f"{side:>2}x{side:<3} {t:>2} | {lut_lo:>13,} to {lut_hi:>6,} {100*lut_lo/Z020_LUT:>4.1f}-{100*lut_hi/Z020_LUT:>4.1f}% | "
          f"{upper:>16,.0f} {100*upper/(211456/4):>7.1f}% {100*exact/(211456/4):>6.1f}% | {100*infer:>8.1f}% | x{wide:>5.2f} cycles")
w23 = 16 * 16 + 16 * 10
print(f"\nweights that change shape when training starts and ends: W2 {16*16} + W3 {16*10} = {w23}, "
      f"which is {100*w23/13002:.1f}% of 13,002. At 4 values per cycle that is {w23/4:.0f} cycles.")

w1_slots = 16 * c(784) * B                       # forward GEMM of W1 per step
print(f"if W1 were tiled 4 by 4 and the lane were serial: +{3 * w1_slots / 4:,.0f} cycles per step, "
      f"which is +{100 * 3 * w1_slots / 4 / (211456 / 4):.0f}% (W1 forward is {w1_slots:,} of 211,456 block-MACs)")
