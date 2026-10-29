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

Script: `python/l1_report.py`. Plot: `out/l1_scale_hist.png`. The input of the encoder is amax/6.

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
2. Activations stay in float64 like GEMM. L2 decides whether the forward pass quantizes them.
3. Biases are not quantized.
4. The FP4 payload is not stored in packed form (they are not even stored, so far). Only the decoded weights and the scale codes exist. Packed storage starts at L5 and for the RTL test vectors.

#### Open decision ? 

MNIST is not a good stress test for FP4 so far. No obvious decisions can be made based on these results alone. 
Use amax/6 or four6 as the default from L2 on, with the other as an additional test. Same is true for E6M2 vs E5M3. No real difference (on MNIST test) between them. four6 costs a second payload quantizer and an error sum per block. It gave no reliable accuracy gain on MNIST. 

