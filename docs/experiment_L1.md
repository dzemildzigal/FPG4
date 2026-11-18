## Our findings

We ran inference on three trained versions of the MLP. Seed 42, 4242 and 424242. ALL MAC and GEMM's are for L1 done in FP64. We simulate the effect of quantization on the weights while keeping the computation in full precision. This is, essentially, measuring the SQNR impact directly, and nothing else.

### L1a

For this experiment we built an E2M1 block quantization scheme with the block scaling factor of FP64.
Point of the experiment is to see how much E2M1 takes away from the unquantized weights just by its coarseness with ideal scaling factors (FP64)

Result table:
|seed|unquantized FP64 acc.|E2M1 block FP64 acc.|drop %|SQNR range dB|
|-|-|-|-|-|
|42|94.69%|92.65%|2.04|20.43 to 20.97|
|4242|95.42%|94.45%|0.97|20.41 to 20.50|
|424242|95.44%|94.29%|1.15|19.61 to 20.35|
|average|95.18%|93.80%|1.39|not viable for avg|


### L1b and L1c

For this experiment we build upon L1a but we make the block scaling factor E6M2 or E5M3 instead of FP64. Everything else remains the same. Now we test for E2M1 noise plus the quantized 8-bit scaling factor.

The only difference between L1b and L1c is that in L1b we force amax6 to be used as the rule, but in L1c we use four6 to choose between amax6 and amax4. This distinction allows us to evaluate the impact of using a more flexible scaling factor selection strategy in L1c compared to the fixed approach in L1b.

In both formats the scaling factor is clamped to the nearest code of the 8-bit table. Ties go to even code (last bit 0). Block of all 0's gets code 0. NaN or infinity is 255, satturation is 254.

#### L1b, UE6M2 (rule amax/6)

|seed|unquantized FP64 acc.|UE6M2 block acc.|drop %|SQNR range dB|
|-|-|-|-|-|
|42|94.69%|90.83%|3.86|20.18 to 20.51|
|4242|95.42%|94.09%|1.33|20.01 to 20.41|
|424242|95.44%|94.69%|0.75|19.56 to 20.07|
|average|95.18%|93.20%|1.98|not viable for avg|

#### L1c, UE6M2 (rule four6)

|seed|unquantized FP64 acc.|UE6M2 block acc.|drop %|SQNR range dB|
|-|-|-|-|-|
|42|94.69%|90.93%|3.76|20.73 to 21.40|
|4242|95.42%|94.55%|0.87|20.65 to 21.10|
|424242|95.44%|94.83%|0.61|20.44 to 20.97|
|average|95.18%|93.44%|1.75|not viable for avg|

#### L1b, UE5M3 (rule amax/6)

|seed|unquantized FP64 acc.|UE5M3 block acc.|drop %|SQNR range dB|
|-|-|-|-|-|
|42|94.69%|90.78%|3.91|20.22 to 20.63|
|4242|95.42%|94.22%|1.20|20.40 to 20.81|
|424242|95.44%|94.58%|0.86|19.51 to 20.26|
|average|95.18%|93.19%|1.99|not viable for avg|

#### L1c, UE5M3 (rule four6)

|seed|unquantized FP64 acc.|UE5M3 block acc.|drop %|SQNR range dB|
|-|-|-|-|-|
|42|94.69%|90.36%|4.33|21.03 to 21.33|
|4242|95.42%|94.66%|0.76|21.12 to 21.49|
|424242|95.44%|94.72%|0.72|20.61 to 21.04|
|average|95.18%|93.25%|1.94|not viable for avg|

#### Summary of all rungs

Averages over the three seeds. Pooled SQNR is computed over all weights of W1, W2 and W3 together, per seed, then averaged.

|experiment|mean acc.|mean test loss|mean pooled SQNR dB|
|-|-|-|-|
|L0 unquantized|95.18%|0.1686|none|
|L1a FP64 scale|93.80%|0.2136|20.41|
|L1b UE6M2 amax/6|93.20%|0.2313|20.12|
|L1c UE6M2 four6|93.44%|0.2296|20.90|
|L1b UE5M3 amax/6|93.19%|0.2277|20.34|
|L1c UE5M3 four6|93.25%|0.2325|21.11|

#### What the numbers show

1. The 8-bit scaling factor costs little. L1a to L1b is 0.60 points (UE6M2) and 0.61 points (UE5M3) of mean accuracy. The L0 spread over the three seeds is 0.75 points, so this step is below the noise.
2. four6 raises SQNR in all 6 seed and format pairs (both ends of the SQNR range go up). The mean pooled gain is 0.78 dB (UE6M2) and 0.77 dB (UE5M3). The mean accuracy changes by +0.24 (UE6M2) and +0.06 (UE5M3) points. This is below the 0.75 noise level. Loss does not follow SQNR: UE5M3 four6 has the best SQNR and the worst mean loss.
3. UE5M3 and UE6M2 give the same result on MNIST (93.19 against 93.20 for L1b).
4. Seed 42 is the outlier in every quantized rung (90.4% to 92.7%). The other two seeds are within 1 point of each other.
5. The test set has 10,000 images. The binomial standard error of one accuracy number is about 0.2 to 0.25 points (computed, not measured). One seed on its own cannot show a difference smaller than that.

Hypothesis (not tested): SQNR counts the error of every weight equally. Rows of W1 that belong to MNIST border pixels, which are always zero, carry error that changes no output. A test is to weight the error of each W1 row by the mean squared value of its input pixel.

#### Scale statistics (L1b, W1 + W2 + W3, three seeds, 2,430 blocks)

Script: `python/l1_report.py`. Plot: `out/experiment_L1_scale_hist_mnist.png`. The input of the encoder is amax/6.

|item|UE6M2|UE5M3|
|-|-|-|
|code min / max|97 / 116|66 / 104|
|distinct codes used|20 of 255|39 of 255|
|exponent e min / max|24 / 29|8 / 13|
|amax/6 min / max|1.02e-2 / 2.49e-1|same|
|max relative scale error|11.10%|5.84%|
|theory bound of that error|11.11% (1/9)|5.88% (1/17)|
|headroom to the top code (table[254] / max amax/6)|2.6e10|4.6e5|
|headroom to the bottom code (min amax/6 / table[0])|2.2e7|334|
|scale saturated / underflow / zero blocks / non-finite|0 / 0 / 0 / 0|0 / 0 / 0 / 0|

The measured scale error is just below the bound in both formats. This is the worst case of the theory, so the encoder is correct on real data. The used codes sit in a narrow cluster far from both ends of the code range. MNIST weights do not stress either format. The format choice needs a stress case (gradients, optimizer state) and cannot be made from L1.

#### What L1 does not test

1. The matrix multiply runs in float64 on the quantized weights (simulated quantization). A float64 product of two such numbers loses nothing due to being in FP64, and the accumulations are the same unless the exponents spread over about 40 bits. The accumulator width of a real FP4 MAC is not tested here and is for later experiments.
   L1d adds the activation quantization and L1e adds the datapath (accumulator and multiplier). Both are measured below.
2. Activations stay in float64 like GEMM. L2 decides whether the forward pass quantizes them.
3. Biases are not quantized.
4. The FP4 payload is not stored in packed form (they are not even stored, so far). Only the decoded weights and the scale codes exist. Packed storage starts at L5 and for the RTL test vectors.

#### Open decision ? 

MNIST is not a good stress test for FP4 so far. No obvious decisions can be made based on these results alone. 
Use amax/6 or four6 as the default from L2 on, with the other as an additional test. Same is true for E6M2 vs E5M3. No real difference (on MNIST test) between them. four6 costs a second payload quantizer and an error sum per block. It gave no reliable accuracy gain on MNIST. 


### L1 on MNIST-1D

Second test set. Same network 784-16-16-10, same Adam, lr 1e-3, batch 128, 10 epochs, seeds 42 / 4242 / 424242. Nothing tuned. Rerun after every change to `quant.py`: `uv run python l1_sweep.py mnist` and `uv run python l1_sweep.py mnist1d`.

Dataset: `python/mnist1d.py` (numpy rewrite of https://github.com/greydanus/mnist1d, Apache-2.0). 500,000 samples, 400,000 train and 100,000 test, sequence length 784, dataset seed 42. Test standard error of one accuracy: about 0.15 points (computed). The generator makes a 72-point signal and resamples it to 784 points. The inputs carry the same information as the 40-point original: 31 principal components hold 99% of the variance, 63 hold 99.9% (measured), and neighbor inputs correlate at 0.999. Length 784 gives the same K as MNIST (49 blocks of 16), it does not make the task harder. Results do not compare with published MNIST-1D numbers (length 40, MLP 68%).

Commands:

```
uv run python mnist1d.py                   # once, about 3 minutes, writes data/mnist1d (1.6 GB, not tracked)
uv run python train.py mnist1d             # about 2.3 minutes per seed
uv run python l1_sweep.py mnist1d
uv run python l1_report.py mnist1d
uv run python tests/test_mnist1d.py
```

#### Summary of all rungs, MNIST-1D

Pooled SQNR here sums over all seeds before the division (`l1_sweep.py`). The MNIST table above averages per seed. The two definitions differ by 0.01 dB (MNIST L1a: 20.40 pooled, 20.41 averaged).

|Rung|42|4242|424242|Mean acc|Mean loss|Drop vs L0|Pooled SQNR (dB)|
|-|-:|-:|-:|-:|-:|-:|-:|
|L0 float|65.15|64.87|65.31|65.11|0.8834|0.00|none|
|L1a fp64 scale|59.38|53.62|50.45|54.48|1.2087|10.63|20.09|
|L1b UE6M2 amax/6|59.24|56.53|55.74|57.17|1.1385|7.94|19.87|
|L1c UE6M2 four6|57.03|55.85|52.60|55.16|1.1800|9.95|20.87|
|L1b UE5M3 amax/6|57.46|51.76|52.21|53.81|1.1936|11.30|20.04|
|L1c UE5M3 four6|58.16|56.82|54.52|56.50|1.1138|8.61|21.01|

#### What the numbers show, MNIST-1D

1. The weight drop is 8 to 11 points. On MNIST it is 1.4 to 2.0. The test is much more sensitive to quantization.
2. The drop is large and differs a lot between seeds. For L1a the drop is 5.8, 11.3, 14.9 points. The rungs do not keep one order across seeds. With 3 seeds, the differences between rungs (up to 3.4 points in mean accuracy) are not significant. Quantizing the same weights is deterministic and the test set is large, so the spread comes from the 3 trained networks, not from test noise.
3. SQNR does not predict accuracy here. L1a has a float scale and no 8-bit scale error, and it has the lowest accuracy of the L1a/L1b UE6M2 pair (54.48 against 57.17). `four6` raises SQNR by 1.0 dB in both formats. It lowers accuracy for UE6M2 (-2.0) and raises it for UE5M3 (+2.7). This is noise, not a result.
4. The drop comes from W2 and W3, not W1. Measured on the first 20,000 test samples with L1a, one layer quantized at a time (scratch script, not in the repo):

|Seed|W1 only|W2 only|W3 only|all three|
|-|-:|-:|-:|-:|
|42|0.35|2.87|4.95|6.08|
|4242|0.21|2.16|6.35|10.93|
|424242|-0.03|4.66|7.66|14.84|

On MNIST the same measurement gives 0.03 to 0.21 (W1), 0.46 to 0.75 (W2), 0.29 to 1.52 (W3). W1 holds 784 of the 810 blocks and does not matter. The 16-wide layers matter. Hypothesis (not tested): L0 accuracy is 65%, so many test samples sit near a decision boundary, and a small shift of the logits flips them. At 95% most samples have a large margin.
5. A first guess was wrong: the larger drop does not come from a larger error in the first layer. The layer-1 output SNR is 32 to 33 dB on MNIST-1D and 26 to 27 dB on MNIST (L1a, first 10,000 test samples).

#### Scale statistics, MNIST-1D (L1b, W1 + W2 + W3, three seeds, 2,430 blocks)

Plot: `out/experiment_L1_scale_hist_mnist1d.png`.

|item|UE6M2|UE5M3|
|-|-|-|
|code min / max|96 / 117|65 / 105|
|distinct codes used|21 of 255|39 of 255|
|exponent e min / max|24 / 29|8 / 13|
|amax/6 min / max|8.4e-3 / 2.85e-1|same|
|max relative scale error|11.05%|5.88%|
|headroom to the top code|2.3e10|4.0e5|
|headroom to the bottom code|1.8e7|277|
|scale saturated / underflow / zero blocks / non-finite|0 / 0 / 0 / 0|0 / 0 / 0 / 0|

The scale range is as narrow as on MNIST. MNIST-1D weights do not stress either format. The format decision still needs gradients and optimizer state.

#### What MNIST-1D adds

It shows damage that MNIST hides, mainly in the two small layers. It cannot decide amax/6 against four6 or UE6M2 against UE5M3 with 3 seeds. A decision needs many paired seeds (10 networks cost about 23 minutes).

#### L1d (measured): weights and activations, no datapath model

L1a to L1c quantize the weights only. The input image X, the hidden activations A1 and A2, the logits and the biases stay float64. L1d quantizes the activations too, with the same format and the same rule as the weights, so **both operands of every MAC are on the FP4 grid**. That is real full-FP4 inference, except that the arithmetic stays float64. The arithmetic is L1e.

One format and one rule for both operands, as a real MAC would use. That gives 4 configs, not 16.

Where the blocks go:

|Tensor|Shape|Blocks along K|Scale codes per sample|
|-|-|-:|-:|
|X|(N, 784)|49|49|
|A1|(N, 16)|1|1|
|A2|(N, 16)|1|1|
|Z3|(N, 10)|not quantized|0|

A1 and A2 have K = 16, so one block per sample. The scale of a hidden activation is per sample, not per feature. The logits stay float64 because the softmax and the loss run in float. The biases stay float64.

Script: `python/l1d_sweep.py`. Commands: `uv run python l1d_sweep.py mnist` and `uv run python l1d_sweep.py mnist1d`. Three seeds, same weights as L1.

|Dataset|Config|L1: weights only|L1d: weights + activations|Extra drop|
|-|-|-:|-:|-:|
|MNIST|UE6M2 amax/6|93.20|90.85|2.35|
|MNIST|UE6M2 four6|93.44|91.50|1.94|
|MNIST|UE5M3 amax/6|93.19|91.37|1.82|
|MNIST|UE5M3 four6|93.25|91.52|1.73|
|MNIST-1D|UE6M2 amax/6|57.17|53.61|3.56|
|MNIST-1D|UE6M2 four6|55.16|52.50|2.66|
|MNIST-1D|UE5M3 amax/6|53.81|51.34|2.47|
|MNIST-1D|UE5M3 four6|56.50|53.87|2.63|

Per seed, MNIST, UE6M2 amax/6: 87.47 / 91.95 / 93.13. Against L0 (94.69 / 95.42 / 95.44) the full-FP4 loss is 7.2 / 3.5 / 2.3 points.

Activation SQNR, mean over the 3 seeds, one chunk of 10,000 test samples:

|Dataset|Config|X|A1|A2|
|-|-|-:|-:|-:|
|MNIST|UE6M2 amax/6|22.86|20.53|20.51|
|MNIST|UE6M2 four6|25.88|21.32|21.42|
|MNIST|UE5M3 amax/6|22.73|20.88|20.81|
|MNIST|UE5M3 four6|25.43|21.64|21.75|
|MNIST-1D|UE6M2 amax/6|20.5|21.0|20.3|
|MNIST-1D|UE5M3 four6|23.2|22.7|21.5|

What the numbers show:

1. The activations cost 1.7 to 3.6 points on top of the weights. The L1 numbers were an upper bound for full FP4, and this is the size of the bound.
2. `four6` helps the activations more than it helps the weights: +3.0 dB on X for MNIST and +2.7 dB for MNIST-1D. The input image is the tensor with the wide histogram (80.7% of the pixels are 0), so it is where the second candidate scale earns its place.
3. The scale counters stay at 0 for every tensor and every config, with one exception: the all-zero blocks of the MNIST input image (203,545 of 490,000 blocks, 41.5%, code 0 and no scale). No scale saturates, nothing underflows. **The activations do not stress the scale range either.** The formats still cannot be separated by their range.
4. The rank of the four configs changes between MNIST and MNIST-1D. UE5M3 four6 is worst on MNIST by 0.02 points and best on MNIST-1D by 0.26 points. Three seeds do not resolve that.

#### L1d-bias (measured): the biases on the FP4 grid

L1d above quantizes the weights and the activations. The biases stay float64. This measures what happens when the biases go onto the FP4 grid too, so that **every stored value in the model** is a 4-bit payload plus an 8-bit scale.

Why the bias is a separate decision: the bias is the only tensor in the model that enters **no multiply**. So its storage format and its add width are independent. The storage format decides the "full FP4" claim. The add width decides the accuracy. The datapath consequence is in `docs/datapath.md` section 4.6.

**Outcome: the bias stays in 32 bits, in the accumulator's fixed-point format. The FP4 bias was measured and rejected.** The reason is in item 5 below: every way to put 42 values on the FP4 grid either loses accuracy or adds a second encoder rule for a tensor that has no multiply to save. It costs 840 bits to stop arguing.

Method. The bias vectors are 16, 16 and 10 values. Two placements are tested:

| placement | what it means | storage |
|-|-|-|
| one block of 16 | the `k` biases of a layer share one scale, like a row of a weight matrix | 4 + 8 bits per layer |
| one scale per value | every bias gets its own scale, so nothing is crushed by a neighbour | 12 bits per value |

The payload and the scale come from the same encoder as everything else (`quantize_blocks`). The bias is **not** given its own rule and it is not tuned. Everything else is the L1d `four6` configuration, full test sets, 3 seeds. The `float64` column reproduces the L1d rows above exactly (91.50 MNIST UE6M2, 52.50 MNIST-1D UE6M2, 53.87 MNIST-1D UE5M3), so the harness is the same.

First a fact about the data. The `k` bias values of one layer have a wide dynamic range:

|vector|n|amax|amin|within-vector range|
|-|-:|-:|-:|-:|
|`b1`|16|0.243 to 0.364|0.0044 to 0.0276|8.8x to 75x|
|`b2`|16|0.319 to 0.413|0.0003 to 0.0057|56x to **1195x**|
|`b3`|10|0.296 to 0.359|0.0065 to 0.0284|12.7x to 45.6x|

So a shared scale has a problem that the weight matrices do not have: 16 values that differ by up to three orders of magnitude. Measured quantization error, as a fraction of `amax`:

|format|one block of 16|one scale per value|
|-|-:|-:|
|UE6M2|6.2% to 15.4%|1.9% to 9.1%|
|UE5M3|5.8% to 15.4%|1.9% to 7.3%|

Against that, the weight error bound is 11.1% (UE6M2) and 5.88% (UE5M3). A shared bias block is worse than the weights it is added to.

Accuracy, full test sets, 3 seeds, L1d `four6` configuration:

|Dataset|Format|bias float64|bias block of 16|bias, scale per value|
|-|-|-:|-:|-:|
|MNIST|UE6M2|91.50|91.34|**91.50**|
|MNIST|UE5M3|91.52|91.64|**91.53**|
|MNIST-1D|UE6M2|52.50|52.61|**53.06**|
|MNIST-1D|UE5M3|53.87|**52.73**|**53.77**|

Per seed, MNIST-1D UE5M3, which is the worst row: float64 55.78 / 53.99 / 51.85, block of 16 55.55 / 51.31 / 51.32, scale per value 55.68 / 54.19 / 51.44.

What the numbers show:

1. **The bias cost is at most 1.14 points, and only with a shared block scale.** UE5M3 on MNIST-1D loses 1.14 points with one block of 16 and 0.10 points with one scale per value. On MNIST the same config *gains* 0.12 points. The two datasets disagree in sign, so no bias result is a real gain, and the honest statement is that a shared block scale moves the accuracy by up to a point and a scale per value does not move it.
2. **The cause is the dynamic range, and it is measured.** The bias range reaches 1195x inside one vector. One scale for 16 such values throws away the small ones. One scale per value removes the problem, and then buys at most 0.10 points for 42 extra scale constants. The 32-bit store costs 1344 bits against 504 and needs no second rule. `docs/datapath.md` section 4.6 takes the 32 bits.
3. **This settles the bias, not the format.** The bias is 42 values in a 13,002 parameter model, 0.32%. It cannot separate UE6M2 from UE5M3, and it cannot decide the scale rule. It answers one narrower question: is it worth putting the bias on the FP4 grid at all.
4. **The answer to that is no, and this is why.** Three options were on the table. One shared scale per layer costs up to 1.14 points. One scale per value costs 42 extra scale constants and a block size of 1, for a movement of 0.00 to 0.10 points that is inside the seed spread. Keeping the bias at 32 bits, in the accumulator's own format, costs 840 bits more than the per-value FP4 store, which is 105 bytes and 1.4% of the weight store. **So the bias stays at 32 bits.** A block encoder that has to be told "except for these 42 values, use a block size of 1" is the convolution, and 105 bytes is not worth it.
5. **The claim survives.** Every value that enters a multiply is 4 bits plus an 8-bit scale per block of 16. The bias enters no multiply. The claim is written in `docs/datapath.md` section 2 as a sentence so that it can be checked.

What this does not test: the bias during training. Here the bias is quantized after training. L2 must update it. The update size is the problem already recorded for the weights (median optimizer update 1.431e-3 of the weight, 349 times below one 4-bit step) and the bias is smaller than the weights, so the bias does not make that problem easier. This section therefore leaves the question touched by 32-bit storage and settled by it: no quantized bias enters training in L2.

Not in the repo: the probe is a scratch script. It reuses `quantize_blocks` and `model.relu`, and it chunks the test set at 20,000 samples so it fits in memory.

#### L1e (measured): the datapath, accumulation and multiplier

L1a to L1d change which **values** sit on the FP4 grid. They do not change the **arithmetic**. In all of them the operands are decoded to float64 and the product and the sum run in float64, inside numpy. L1e is the first rung that changes the arithmetic. It runs on the trained L0 weights, with no retraining, so it is still PTQ (post-training quantization).

The question: does the rounding of a real datapath cost accuracy, and how wide must each register be?

### 1. Definitions

Read this first. The variants in section 4 are unreadable without these terms.

| Term | Meaning |
|-|-|
| MAC | Multiply-accumulate. One multiply plus one add. It is the basic operation of a matmul. |
| matmul layer | `Z = A @ W + b`. `A` is `(N, K)`, `W` is `(K, M)`, `Z` is `(N, M)`. `K` is the accumulation axis. |
| operand | One of the two numbers that a MAC multiplies. Here one operand is an activation, the other is a weight. |
| payload | The 4-bit code of an FP4 number. It decodes to a magnitude `k/2` with `k` in `{0,1,2,3,4,6,8,12}`, that is `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. One sign bit is stored separately. |
| block | 16 consecutive elements along K. The 16 elements share one scale. |
| scale | An 8-bit code. It decodes through the scale table to one real number. |
| decoded value | `sign * payload * scale`. |
| block sum | The sum of the 16 integer products `k_act * k_w` of one block. Integer, no rounding. |
| scale product | `scale_act * scale_w` of one block. |
| block contribution | `block sum * scale product / 4`. The factor 1/4 is exact, because `(k/2) * (k/2) = k*k/4`. |
| accumulator | The register that holds the running sum of the block contributions. |
| fixed point | A number stored as an integer times one fixed power of two. The chosen width in bits sets the resolution. |
| alignment | Making two numbers share an exponent before an add. The smaller number loses its low bits. |
| float32 | Real number format with a 24-bit significand. About 7 decimal digits. |
| float64 | Real number format with a 53-bit significand. About 16 decimal digits. |
| SQNR | Signal to quantization noise ratio, in dB. `10 * log10(sum(reference^2) / sum(error^2))`. Higher is better. |
| point | One percentage point of test accuracy. |
| isolated measurement | Both variants get the same input tensor, so the difference is the arithmetic only, with no chain effect. |

### 2. The datapath in 5 steps

Block FP4 wants a datapath with no real multiplier in front of every product. The steps are:

1. The payload decoder turns a 4-bit code into a small integer `k`. This is a lookup table or a mux, not arithmetic.
2. Multiply two of those integers: `k_act * k_w`. The largest value is `12 * 12 = 144`, which is 8 bits. The product is exact. **There is no FP4 multiplier that rounds.**
3. Sum the 16 integer products of one block. The largest value is `16 * 144 = 2304`, which is 12 bits with a sign. The sum is exact.
4. Multiply the block sum once by the scale product. This is the first place that must round.
5. Add the block contributions in the accumulator. Layer 1 has 49 of them, one per block of K = 784. This is the second place that must round.

L1e puts a number on steps 4 and 5. It does not touch steps 1 to 3, because those are exact.

### 3. Facts that need no measurement

| Item | Value | Bits |
|-|-|-|
| payload integer `k` | 0 to 12 | 4 |
| integer product `k_act * k_w` | 0 to 144 | 8 |
| block sum of 16 products | 0 to 2304 | 12 signed |
| scale mantissa | 2 or 3 bits, plus the implicit 1 | 4 |
| scale product mantissa | up to 4 or 6 bits | 6 |
| block contributions to add, layer 1 | 49, each with its own scale product | open |

### 4. The variants

The matmul has three places where a number can be rounded. Each variant holds two of them and changes one.

| id | Place 1: the 16 products of a block | Place 2: the block sum | Place 3: the 49 block contributions | The question it answers |
|-|-|-|-|-|
| E0 | decode both operands to float64 | float64 GEMM, no block structure | float64 | Reference. Must equal the L1d number. |
| E1 | integer `k*k` | integer, exact | float64, one scale multiply per block | Does the integer factorization give the same answer as float64? |
| E2 | integer `k*k` | integer, exact | scale product rounded to **float32**, sum in float32 | Is a float32 scale product good enough? |
| E3-24 | integer `k*k` | integer, exact | **fixed-point accumulator, 24 bits** | Is a 24-bit accumulator good enough? |
| E3-20 | integer `k*k` | integer, exact | **fixed-point accumulator, 20 bits** | Is a 20-bit accumulator good enough? |
| E3-16 | integer `k*k` | integer, exact | **fixed-point accumulator, 16 bits** | Is a 16-bit accumulator good enough? |
| E4 | decoded to **float32** | float32 | float32 | What if the integer path is dropped and a plain float32 GEMM runs? |

E4 is the control. The product of two 4-bit payloads needs at most 8 bits, so it is exact even in float32. E4 therefore measures the float32 accumulator only, not a lossy multiplier.

### 5. One block, with real numbers

Seed 42, config UE6M2 `four6`, MNIST test sample 0, layer 1, output 0, block 22. That block covers input rows 352 to 367, part of the digit.

```
raw pixels        [0, 0.086, 0.914, 1.0, 0.325, 0, 0, ..., 0]
decoded act       [0, 0.0781, 0.9375, 0.9375, 0.3125, 0, 0, ..., 0]
act scale         0.15625                        (decoded from its 8-bit code)
act payload k     [0, 1, 12, 12, 4, 0, 0, ..., 0]      in units of 0.5
weight payload k  [-8, -1, 2, -1, -1, 0, 4, 3, 8, -4, -12, 0, -3, 1, 4, 2]
integer products  [0, -1, 24, -12, -4, 0, 0, ..., 0]   sum = 7      <- the block sum
weight scale      0.0625
scale product / 4 0.00244140625
block contribution 7 * 0.00244140625 = 0.01708984375
```

Repeat for the other 48 blocks and add the 49 contributions:

```
sum of the 49 block contributions = 2.318084716796875
E0 float64 GEMM out[0,0]          = 2.318084716796875
```

The two numbers are identical. This is the result that E1 tests on all 20 outputs, all 10,000 samples, both datasets and all seeds.

### 6. How the measurement was made

Script: `python/l1e_sweep.py`, 205 lines. The one function that changes between variants is `matmul_dp` at line 74. The chain is `forward_dp` at line 122. The error measure is `layer_error` at line 148.

1. Load the three trained weight matrices W1, W2, W3 of one seed. They are float64.
2. Quantize them with one config, which is a format plus a rule. Keep four things per matrix: the decoded values, the payload index, the block scales, and the 8-bit scale codes.
3. For each layer in order: quantize the input activation with the same config, call `matmul_dp`, apply ReLU on layers 1 and 2, add the bias in float64.
4. After layer 3: softmax, `argmax`, compare with the label. That is the accuracy.
5. Repeat for 4 configs, 3 seeds and 7 variants. That is 84 full chains per dataset.
6. Separately, for the isolated number: give layer 1 the same input under E0 and under one variant, and compare the two output tensors. No chain effect, no re-quantization.

Measurements: MNIST uses all 10,000 test samples. MNIST-1D uses the first 20,000 of 100,000, to keep the run short. Layer 1 is processed in chunks of 2,000 rows, because the integer product tensor is `(rows, 16 outputs, 49 blocks)`.

Commands:

```
cd python
uv run python l1e_sweep.py mnist
uv run python l1e_sweep.py mnist1d
```

### 7. Results: accuracy

Mean over the 3 seeds. Each cell is the mean test accuracy in percent.

MNIST, 10,000 test samples:

| Config | E0 | E1 | E2 | E3-24 | E3-20 | E3-16 | E4 |
|-|-:|-:|-:|-:|-:|-:|-:|
| UE6M2 amax/6 | 90.85 | 90.85 | 90.85 | 90.85 | 90.85 | 90.88 | 90.85 |
| UE6M2 four6 | 91.50 | 91.50 | 91.50 | 91.50 | 91.50 | 91.49 | 91.50 |
| UE5M3 amax/6 | 91.37 | 91.37 | 91.37 | 91.37 | 91.36 | 91.36 | 91.37 |
| UE5M3 four6 | 91.52 | 91.52 | 91.52 | 91.52 | 91.52 | 91.53 | 91.52 |

MNIST-1D, first 20,000 test samples:

| Config | E0 | E1 | E2 | E3-24 | E3-20 | E3-16 | E4 |
|-|-:|-:|-:|-:|-:|-:|-:|
| UE6M2 amax/6 | 53.71 | 53.71 | 53.71 | 53.71 | 53.71 | 53.70 | 53.71 |
| UE6M2 four6 | 52.79 | 52.79 | 52.79 | 52.79 | 52.79 | 52.77 | 52.79 |
| UE5M3 amax/6 | 51.69 | 51.69 | 51.69 | 51.69 | 51.69 | 51.66 | 51.69 |
| UE5M3 four6 | 54.06 | 54.06 | 54.06 | 54.06 | 54.06 | 54.06 | 54.06 |

The E0 column of MNIST-1D gives 53.71 here and 53.61 in the L1d table. The reason is the sample count, not a code change. L1d uses all 100,000 samples, this table uses the first 20,000.

### 8. Results: the datapath error alone

Isolated measurement. Layer 1 gets the E0 input, so the number is the arithmetic only. Mean over 3 seeds and 4 configs. `max abs` is the largest absolute difference in the layer output. The layer output is `Z1`, before the ReLU.

| Variant | MNIST max abs | MNIST SQNR dB | MNIST-1D max abs | MNIST-1D SQNR dB |
|-|-:|-:|-:|-:|
| E1 | 0 | exact | 0 | exact |
| E2 | 3.8e-7 | above 200 | 8.9e-6 | 157.4 |
| E3-24 | 3.8e-6 | 148.9 | 6.5e-5 | 124.0 |
| E3-20 | 2.1e-4 | 103.1 | 1.9e-3 | 91.1 |
| E3-16 | 6.3e-3 | 70.0 | 3.7e-2 | 65.1 |
| E4 | 3.8e-7 | above 200 | 8.5e-6 | 158.2 |

For scale: the operand quantization gives the layer-1 activation an SQNR of 20.5 to 21.8 dB (section L1d). Every variant here is above 65 dB. The arithmetic is between 44 and 130 dB better than the quantization of the operands.

### 9. What the numbers show

1. **E1 equals E0 bit for bit.** Not "close". Zero difference on both datasets, all 4 configs, all 3 seeds, all 3 layers. The integer block sum with one scale multiply per block is exactly the float64 GEMM. The factorization costs nothing.
2. **The accumulator width does not matter above 16 bits.** E3-24 and E3-20 are identical to E0 in accuracy. E3-16 moves a mean by at most 0.03 points, and it moves them in both directions: up on MNIST (90.88 against 90.85 for E0) and down on MNIST-1D (51.66 against 51.69). That is inside the seed spread, not a trend.
3. **The error falls by about 30 dB for every 4 bits added to the accumulator.** 70.0 dB at 16 bits, 103.1 dB at 20 bits, 148.9 dB at 24 bits, MNIST. The accumulator width is a free design choice, as far as accuracy is concerned.
4. **The scale product at float32 is the smallest effect of the three.** 3.8e-7 on MNIST, 157 dB on MNIST-1D. A fixed-point multiplier of 6 or 8 bits will be far above that limit, because the operand quantization is already at 20 dB.
5. **The datapath is not where the accuracy goes.** Operand quantization costs 1.73 to 3.56 points (section L1d). The datapath costs at most 0.03 points. The ratio is over 50 to 1. A block-FP4 datapath with a 12-bit block sum, one scale multiply per block and a 20-bit accumulator loses nothing measurable.
6. **The four configs stay indistinguishable in L1e.** Every variant keeps the same ranking and the same gaps. So L1e gives no reason to choose UE6M2 or UE5M3, and no reason to choose `amax/6` or `four6`. That decision stays open and belongs to L2.

### 10. What you can decide from this

| Question | Answer | What it fixes in the design |
|-|-|-|
| Does the integer block sum equal float64? | Yes, bit-identical (E1) | The datapath is: payload decoder, integer multiply, integer add tree, one scale multiply per block. No real multiplier in front of the products. |
| How wide must the accumulator be? | 20 bits costs nothing, 16 bits costs 0.03 points | Choose 20 bits. Stop treating it as an accuracy question. |
| Does the scale product need float32? | No. float32 error is 3.8e-7, the operand error is at 20 dB | Use a small fixed-point multiplier. The width is an area question, not an accuracy question. |
| Does the datapath decide the format? | No | UE6M2 vs UE5M3 and amax/6 vs four6 stay open. L2 decides them. |
| What is left in L1e? | The width of the 49-block add tree, and a fixed-point scale multiplier instead of float32 | These are hardware cost questions. Accuracy cannot separate them, so E9 synthesis answers them. |

### 11. Limits of this measurement

1. **The E3 model is not bit-accurate to hardware.** It rounds each block contribution once to the accumulator grid, then adds the 49 contributions exactly. A real adder tree rounds after each add. The error source is the same and the size of the effect is small, but the model is slightly optimistic.
2. **The scale product is a float32 multiply in E2, not a fixed-point multiply.** E2 shows the size of the effect, not the cost of a real multiplier.
3. **Layer 2 and layer 3 are not measured in isolation.** They have `K = 16`, so one block. Their input would be the variant's own A1 instead of the E0 A1, and a difference of 1e-6 then moves a few values across a payload boundary. The resulting jump belongs to the quantizer, not to the datapath. Layer 1 is the honest measurement: 49 blocks and no re-quantization.
4. **Three seeds.** A difference of 0.03 points is not resolved by three seeds.

### 12. What L1e does not test

The update path. Nothing trains in L1e. The accumulator never has to hold a weight update that is 349 times smaller than one grid step. That problem is L3 and L7, and it is the hard one.

## Re-measurement under the rounded-up scale, 2026-11-18

`quant.SCALE_ROUNDING` has two settings, ADR-032. This section measures the second one on the forward pass,
paired against the first, both on today's float32 code. Nothing else changed: same seeds, same stored weights,
same script.

**Read this first.** The tables earlier in this document were measured in float64, before ADR-005 moved the
simulation to float32. A float64 row and a float32 row are not comparable, so the numbers below cannot be
subtracted from the tables above. Both arms below were produced today, so the difference between them is the
only thing this section measures.

### L1, weights only

| Dataset | Config | Nearest | Up | Delta |
|-|-|-:|-:|-:|
| MNIST | E6M2 `amax/6` | 93.41 | 93.22 | -0.19 |
| MNIST | E6M2 `four6` | 93.52 | 93.30 | -0.22 |
| MNIST | E5M3 `amax/6` | 93.54 | 92.47 | -1.07 |
| MNIST | E5M3 `four6` | 93.11 | 92.76 | -0.35 |
| MNIST-1D | E6M2 `amax/6` | 56.77 | 55.01 | -1.76 |
| MNIST-1D | E6M2 `four6` | 58.62 | 57.23 | -1.39 |
| MNIST-1D | E5M3 `amax/6` | 55.51 | 53.38 | -2.13 |
| MNIST-1D | E5M3 `four6` | 55.09 | 55.23 | +0.14 |
| both | L0 float, no scale | 95.18 and 64.94 | same | 0 |

### L1d, weights and activations

| Dataset | Config | Nearest | Up | Delta |
|-|-|-:|-:|-:|
| MNIST | E6M2 `amax/6` | 91.10 | 90.29 | -0.81 |
| MNIST | E6M2 `four6` | 91.64 | 90.80 | -0.84 |
| MNIST | E5M3 `amax/6` | 91.73 | 89.89 | -1.84 |
| MNIST | E5M3 `four6` | 91.40 | 90.58 | -0.82 |
| MNIST-1D | E6M2 `amax/6` | 53.53 | 52.36 | -1.17 |
| MNIST-1D | E6M2 `four6` | 55.64 | 54.32 | -1.32 |
| MNIST-1D | E5M3 `amax/6` | 52.71 | 52.03 | -0.68 |
| MNIST-1D | E5M3 `four6` | 52.91 | 53.76 | +0.85 |

Mean of the eight L1d deltas: **-0.58 points on MNIST-1D and -1.08 points on MNIST.** Seven of eight negative.

### The control

The rung `fp64 scale` uses no 8-bit table, so the scale rule cannot reach it. It is identical in both arms to
the last printed digit: MNIST 93.88 with 20.38 dB, MNIST-1D 54.36 with 20.11 dB. That is the proof that the
switch touches the table path and nothing else.

### What it means

The rounded-up scale keeps `amax / s` below the top payload code, so no block clips its maximum. In the
backward pass that is worth 29 points, because the clipped element is the true-class element of a softmax row
and the error has one sign in every row.

In the forward pass the error has no direction. The rule then only pays for itself: it loses 0.5 to 1.1 dB of
signal to noise ratio on the quantized tensors, and the accuracy follows. So the forward pass keeps the
nearest scale and the backward operands take the rounded-up scale. The datapath carries both forms, ADR-032.

**Caveat on the dataset.** These L1 and L1d numbers are on the 16-wide network, which is too small for
MNIST-1D: its float ceiling is 64.94 here against 90.50 at 64 wide. The MNIST-1D deltas are therefore a
statement about a strangled network, not about the format.
