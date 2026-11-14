# Experiment L2

## Objective

This experiment aims to find the impact of FP4 E2M1 w. scaling factor E6M2 or E5M3 with amax/6 or four6 on the training by quantizing only the forward pass of the training process.

Every forward pass during the training will have it's weights quantized according to the specified FP4 format with the chosen scaling factor.

Then we will run the forward pass with those quantized weights. And compute the loss based on the output of that forward pass. Save the loss in FP64.

The backward pass will remain in full FP64 precision, ensuring the experiment can be compared to L1's 0th run of full FP64 training so we can see the impact of quantization on the forward pass.

L2 is the first phase of the experiment where the quantization acts during training. The weights are quantized again at every step (we don't touch the activations). L1 quantized one fixed set of weights (also didn't touch the activations) after training (**post-training quantization, PTQ**). L2 is, essentially, **quantization-aware training (QAT)**.

**We don't touch the activations during L2. We defer it to L2a, where we will specifically study the effect of quantizing activations in addition to the forward pass weights, isolating the impact of activation quantization from weight quantization.**

## Terms

|Term|Meaning|
|-|-|
|master weight|The FP64 copy of W that the optimizer updates. It is not on the FP4 grid.|
|Wq|The quantized copy of the master weight, decoded back to FP64. The forward pass uses Wq.|
|STE, straight-through estimator|A rule for the backward pass. The gradient flows through the quantizer as if the quantizer were the identity function. Formally: `dWq/dW := 1`.|
|paired seeds|The same seed, the same start weights and the same batch order in two runs. Only the option under test differs. Paired differences remove most of the seed noise.|

## Methodology

We will introduce multiple seeds to ensure our results aren't biased by a random initialization of the model weights. So far we used 42, 4242, and 424242. We will expand this to 10 seeds:

Our seeds will be 42, 4242, 424242, 123, 456, 789, 1010, 2020, 3030, and 4040.

1. Initialize the model with full FP64 precision weights.
2. For each forward pass during training, quantize the weights according to the specified FP4 format and chosen scaling factor.
3. Perform the forward pass using the quantized weights.
4. Compute the loss based on the forward pass output.
5. Perform the backward pass using full FP64 precision weights to update the model parameters.
6. Repeat steps 2-5 for the entire training process.
7. Evaluate the model's performance on a test set to assess the impact of forward pass quantization.
8. Compare the results with the baseline full FP64 training to quantify the effect of FP4 forward pass quantization.
9. Document any observed differences in training dynamics, convergence speed, and final model accuracy to provide a comprehensive analysis of the impact of forward pass quantization.

### Exact gradient rule (definition, do not vary it between runs)

The forward pass uses `Wq`. The backward pass computes `dL/dW` at the float master `W` (STE). It uses the activations that the quantized forward produced. The biases and the Adam state stay in FP64. The quantizer is treated as the identity in the backward pass.

We will create a run matrix for every experiment to systematically record the results for each combination of FP4 format, scaling factor, and seed, ensuring a comprehensive analysis of the impact of forward pass quantization.

This means the gradient never sees the quantization error of the weight. The gradient only sees its effect on the activations. This is the standard weight-quantization STE. L3 and L4 change exactly this rule, so it must be written down here.

### What L2 does not quantize

Activations stay in FP64. Biases stay in FP64. This is the same scope as L1, so L2 and L1 are comparable. The consequence: the L1 and L2 accuracies are **upper bounds** for real FP4 inference and training. Real hardware rounds the activations too.

**Decision needed (Dzemo):** the hardware MAC needs two FP4 operands. Choose one:
- **Option A (recommended):** L2 quantizes weights only. Activation quantization becomes its own step "L2a", with the same 4 configs, run after L2. Reason: one factor changes per rung, so the damage can be attributed.
- **Option B:** L2 crosses both factors, 8 configs (2 formats x 2 rules x weights-only/weights-and-activations). One longer run, attribution is weaker.

## Starting baseline

We pull the L0 baseline on MNIST and MNIST1D datasets

|seed|unquantized FP64 acc. MNIST|unquantized FP64 acc. MNIST1D|
|-|-|-|
|42|94.69%|65.15|
|4242|95.42%|64.87|
|424242|95.44%|65.31|
|average|95.18%|65.11|

|seed|unquantized FP64 test loss MNIST|unquantized FP64 test loss MNIST1D|
|-|-|-|
|average|0.1686|0.8834|

### Float width

The baselines above are float64. `python/fp32_probe.py` measured the same three seeds on the same code path in float32, changing nothing else:

|dataset|float64 mean|float32 mean|shift|seed spread of the float64 row|
|-|-:|-:|-:|-:|
|MNIST|95.18|95.18|+0.00|0.75|
|MNIST-1D|65.11|64.94|-0.17|0.44|

The shift is inside the seed spread on both datasets, so **float32 is the width of every run from here on**, and `train.DTYPE` is the one place that sets it. The three float64 rows above stay as history: a float64 row and a float32 row are not comparable, so the `L0` row of the run matrix is retrained in float32 for all ten seeds, and `l2_sweep.py` retrains any weight file whose width does not match instead of averaging the two.

From here we see that MNIST1D is a lot more sensitive than MNIST. That's why we have it included :)

## Run matrix

Every run uses: 784-16-16-10, Adam, lr 1e-3, batch 128, 10 epochs, dataset seed 42. Nothing is tuned. Each config runs on every seed below.

|Run name|Format|Rule|Seed count|
|-|-|-|-|
|`L0`|none|none|10|
|`L1a`|FP4 payload, FP64 scale|n/a|10 (reference row: the PTQ loss)|
|`L2-a6-e6m2`|E6M2|amax/6|10|
|`L2-f6-e6m2`|E6M2|four6|10|
|`L2-a6-e5m3`|E5M3|amax/6|10|
|`L2-f6-e5m3`|E5M3|four6|10|

Seeds: 42, 4242, 424242, 123, 456, 789, 1010, 2020, 3030, and 4040. Each dataset runs the whole matrix. Total: 6 rows x 10 seeds x 2 datasets = 120 runs, of which 100 need training.

`L1a` is the reference point that shows how much training recovers. `L0` is the reference point that shows what quantization costs.

## Decision rule (pre-registered, written before the run)

An option wins only if both conditions hold on **both** datasets:
1. The paired difference (same seed, L2 against L2) has the same sign in at least 8 of the 10 seeds.
2. The absolute mean paired difference is greater than 2 standard errors of that mean.

If the conditions do not hold, the result is **no decision**. The option stays open and moves to the next rung. Do not pick a default.

## What is logged

Per epoch, per seed, per run:
- train loss, train accuracy, test loss, test accuracy (from `model.evaluate`).
- SQNR of W1, W2, W3, measured on the quantized copy against the master, at the end of the epoch.
- The 4 counters of `scale_counters`: scale saturated, scale underflow, zero blocks, non-finite blocks.
- **New: number of steps where `Wq` did not change**, while the master weight did change. This is the mild form of the update problem, and it is a direct input for L3 and L7.

Expected under L2: the counters stay at 0. The L1 counters did (2,430 blocks, all zero). Weight bytes do not stress an 8-bit scale. Write the zeros down anyway, they are evidence.

## Expected outcomes (hypotheses, to be confirmed or rejected)

1. L2 is better than L1a on the same dataset. The weights adapt to the grid, so QAT recovers part of the PTQ loss. If L2 is worse than L1a, suspect a bug before suspecting a finding.
2. `four6` keeps its SQNR advantage (+0.8 to +1.0 dB, measured in L1). Whether it converts into accuracy is the open question.
3. E6M2 and E5M3 stay within noise of each other. The scale range is narrow (21 and 39 codes used of 255 in L1).
4. The counters stay at 0, so L2 cannot decide the format. The format decision needs gradients (L3, L4) and optimizer state (L6).

## Cost

Per training run: 625 steps per epoch x 10 epochs = 6,250 forward passes, so 6,250 quantizer calls per matrix. The quantizer cost is negligible against the GEMM.

|Dataset|One run, 10 epochs|100 runs|
|-|-|-|
|MNIST|about 20 s (estimate, from 30 s for the L0 run)|about 35 min|
|MNIST-1D|about 2.4 min (measured, L0)|about 4 h|

## Reproduction

```
cd FPG4/python
uv run python l2_sweep.py [mnist|mnist1d]     # script not written yet
uv run python tests/test_quant.py
```

Results go in `docs/experiment_L2.md`, section "Results", with the same table shape as `experiment_L1.md`.

## Open decisions

1. **Activations.** Option A (weights only in L2, activation quantization as a separate step) or Option B (8 crossed configs). See "What L2 does not quantize".
2. **Test set or validation split.** Step 7 uses "a test set", like L0 and L1. That keeps the numbers comparable, but the winner is then selected on the test set. A validation split out of train (for example 10% of the 400,000 MNIST-1D samples) would remove the selection bias but breaks comparability with L0 and L1. Recommendation: keep the test set for L2 and state the bias in the results.
3. **The cheap activation check.** Before any retraining, L1 can be rerun with quantized activations (no training, a few minutes). That gives the weight-only against weight+activation gap at inference, for the 4 configs, at the cost of one script. Recommendation: do it, it is the cheapest data we can get on the format question.

## Results

### Correction, and it voids the first version of this section

The first version of these results reported `evaluate(p, ...)`, where `p` is the float master weight. The master never reaches the fabric. The number that belongs here is `evaluate(pq, ...)`, the quantized model, and for L2a it is `evaluate(pq, X, y, quant=activation_hook)`.

| what was reported | what it was | what it should be |
|-|-|-|
| `L2` accuracy | the float master weight | the FP4 weight |
| `L2a` accuracy | the float master weight, on float activations | the FP4 weight on FP4 activations |

So the first L2 table measured a model the fabric never runs. It is withdrawn in full, together with the MNIST-1D L2 table and the `four6` verdict that came from it.

**Why the error moved the answer, and not only the number.** `four6` keeps the FP4 weights closer to the master than `amax/6` does, by 0.74 dB of SQNR (20.64 against 19.90 on W1). A float master with a smaller distance to its own quantization scores better in float. That is exactly what the wrong table showed, and it is why the wrong table promoted `four6` by a decided 0.42 points. **The advantage was the master's SQNR leaking into the reported accuracy. It is not an advantage of the quantized model.** The corrected numbers below do not reproduce it.

`python/qat.py` now evaluates the deployed model on both rungs, and `self_check()` checks two things that would have caught this: the quantizer is idempotent on its own output, and every activation tensor the forward pass hands the backward pass is already on the grid.

### MNIST, L2 and L2a, 10 seeds

Conditions: float32 (`train.DTYPE`), full test set, 10 epochs, batch 128, dataset seed 42, 784-16-16-10. Script `python/l2_sweep.py mnist 10 w` and `python/l2_sweep.py mnist 10 wa`. Every run is resumable in `out/w/mnist/` and `out/wa/mnist/`.

The rungs:

| rung | weights | activations | training |
|-|-|-|-|
| `L0` | float | float | no quantization |
| `L1a` | FP4, float64 scale | float | no, shares the L0 weights |
| `ptq` | FP4, 8-bit scale | float for L2, FP4 for L2a | no, shares the L0 weights |
| `L2` | FP4, quantized every step | float | quantization in the forward pass |
| `L2a` | FP4, quantized every step | FP4, quantized every step | quantization in the forward pass |

Mean over 10 seeds:

| rung | MNIST accuracy | drop against L0 |
|-|-:|-:|
| L0 | **95.01** | – |
| L1a | 92.65 | 2.37 |
| ptq for L2, a6-e6m2 / f6-e6m2 / a6-e5m3 / f6-e5m3 | 93.25 / 93.55 / 92.57 / 93.16 | 1.77 / 1.46 / 2.44 / 1.86 |
| **L2**, same order | **94.72 / 94.77 / 94.61 / 94.80** | **0.29 / 0.24 / 0.41 / 0.21** |
| ptq for L2a, same order | 90.70 / 91.63 / 90.28 / 91.28 | 4.31 / 3.39 / 4.74 / 3.74 |
| **L2a**, same order | **93.88 / 94.10 / 93.92 / 93.97** | **1.13 / 0.91 / 1.10 / 1.04** |

What training recovers:

|configuration|L2 ptq drop|L2 drop|recovered|L2a ptq drop|L2a drop|recovered|
|-|-:|-:|-:|-:|-:|-:|
|a6-e6m2|1.77|0.29|1.48|4.31|1.13|3.19|
|f6-e6m2|1.46|0.24|1.22|3.39|0.91|2.47|
|a6-e5m3|2.44|0.41|2.03|4.74|1.10|3.64|
|f6-e5m3|1.86|0.21|1.64|3.74|1.04|2.69|

**Per-seed accuracy, L2:**

|seed|L0|a6-e6m2|f6-e6m2|a6-e5m3|f6-e5m3|
|-|-:|-:|-:|-:|-:|
|42|94.71|94.56|94.34|95.06|94.42|
|4242|95.41|94.85|94.66|94.45|94.77|
|424242|95.42|94.91|94.98|94.94|94.95|
|123|94.89|94.50|94.60|93.96|94.25|
|456|94.79|94.74|94.66|94.27|95.03|
|789|94.66|94.81|94.90|94.89|94.87|
|1010|95.18|95.04|94.79|94.32|94.85|
|2020|95.37|94.40|94.88|94.90|95.04|
|3030|94.75|94.53|94.77|94.49|94.81|
|4040|94.95|94.90|95.13|94.80|95.00|
|**mean**|**95.01**|**94.72**|**94.77**|**94.61**|**94.80**|

**Per-seed accuracy, L2a:**

|seed|L0|a6-e6m2|f6-e6m2|a6-e5m3|f6-e5m3|
|-|-:|-:|-:|-:|-:|
|42|94.71|94.09|93.78|93.66|93.83|
|4242|95.41|94.00|94.03|94.37|94.28|
|424242|95.42|94.70|94.27|93.98|94.19|
|123|94.89|93.12|94.04|93.66|93.63|
|456|94.79|93.35|93.92|93.50|93.96|
|789|94.66|94.22|93.96|93.33|94.56|
|1010|95.18|94.10|94.09|94.00|93.75|
|2020|95.37|93.95|94.53|94.11|93.90|
|3030|94.75|93.78|94.09|94.00|93.80|
|4040|94.95|93.54|94.29|94.55|93.79|
|**mean**|**95.01**|**93.88**|**94.10**|**93.92**|**93.97**|

### The decision rule, applied to every pair

L2:

|a|b|mean a-b|standard error|sign agrees|condition 1|condition 2|winner|
|-|-|-:|-:|-:|-:|-:|-|
|L2 a6-e6m2|L0|-0.29|0.10|9/10|yes|yes|**L0**|
|L2 f6-e6m2|L0|-0.24|0.10|7/10|no|yes|no decision|
|L2 a6-e5m3|L0|-0.41|0.14|8/10|yes|yes|**L0**|
|L2 f6-e5m3|L0|-0.21|0.10|6/10|no|yes|no decision|
|L2 a6-e6m2|ptq a6-e6m2|+1.48|0.32|10/10|yes|yes|**L2**|
|L2 f6-e6m2|ptq f6-e6m2|+1.22|0.37|9/10|yes|yes|**L2**|
|L2 a6-e5m3|ptq a6-e5m3|+2.03|0.45|9/10|yes|yes|**L2**|
|L2 f6-e5m3|ptq f6-e5m3|+1.64|0.41|10/10|yes|yes|**L2**|
|L2 a6-e6m2|L2 f6-e6m2|-0.05|0.07|6/10|no|no|no decision|
|L2 a6-e5m3|L2 f6-e5m3|-0.19|0.12|8/10|yes|no|no decision|
|L2 a6-e6m2|L2 a6-e5m3|+0.12|0.13|6/10|no|no|no decision|
|L2 f6-e6m2|L2 f6-e5m3|-0.03|0.06|6/10|no|no|no decision|

L2a:

|a|b|mean a-b|standard error|sign agrees|condition 1|condition 2|winner|
|-|-|-:|-:|-:|-:|-:|-|
|L2a a6-e6m2|L0|-1.13|0.14|10/10|yes|yes|**L0**|
|L2a f6-e6m2|L0|-0.91|0.07|10/10|yes|yes|**L0**|
|L2a a6-e5m3|L0|-1.10|0.10|10/10|yes|yes|**L0**|
|L2a f6-e5m3|L0|-1.04|0.12|10/10|yes|yes|**L0**|
|L2a a6-e6m2|ptq a6-e6m2|+3.19|0.46|10/10|yes|yes|**L2a**|
|L2a f6-e6m2|ptq f6-e6m2|+2.47|0.47|10/10|yes|yes|**L2a**|
|L2a a6-e5m3|ptq a6-e5m3|+3.64|0.58|10/10|yes|yes|**L2a**|
|L2a f6-e5m3|ptq f6-e5m3|+2.69|0.51|10/10|yes|yes|**L2a**|
|L2a a6-e6m2|L2a f6-e6m2|-0.21|0.15|6/10|no|no|no decision|
|L2a a6-e5m3|L2a f6-e5m3|-0.05|0.17|4/10|no|no|no decision|
|L2a a6-e6m2|L2a a6-e5m3|-0.03|0.18|6/10|no|no|no decision|
|L2a f6-e6m2|L2a f6-e5m3|+0.13|0.12|6/10|no|no|no decision|

**Every comparison between two configurations is a `no decision` in L2a.** In L2 one pair reaches condition 1 and fails condition 2. The four configurations are inside each other's noise on MNIST at both rungs.

### What training does to the grid

End of training, mean over the 10 seeds. `stale` is the fraction of weight elements whose FP4 value did not move from one step to the next while the master weight was free to move. `frozen steps` counts the steps, of 469 in an epoch, in which the whole quantized matrix was bit-identical to the step before.

|rung|configuration|SQNR W1|stale W1|frozen steps|
|-|-|-:|-:|-:|
|L2|a6-e6m2|19.90|99.25%|300.6|
|L2|f6-e6m2|20.64|98.61%|353.3|
|L2|a6-e5m3|20.11|98.99%|302.7|
|L2|f6-e5m3|20.83|98.29%|337.0|
|L2a|a6-e6m2|19.89|99.33%|255.6|
|L2a|f6-e6m2|20.63|98.76%|308.1|
|L2a|a6-e5m3|20.09|99.09%|271.5|
|L2a|f6-e5m3|20.84|98.44%|309.1|

**98 to 99 percent of the weight elements never move their FP4 value. `four6` remains the configuration with the most frozen weights.**

### Activation SQNR, L2a

Mean over the 10 seeds at the end of training. The input is 784 wide and is quantized once, outside the loop, because a block lives inside one row and rows never mix. That claim is proved at run time: `check_prequant_equivalence()` quantizes 4,096 rows in one call and again in 32 batches of 128, and the two results must be bit-identical. Measured worst difference: 0.

|configuration|input X|A1|A2|
|-|-:|-:|-:|
|a6-e6m2|22.83|20.45|20.52|
|f6-e6m2|25.82|21.41|21.53|
|a6-e5m3|22.68|20.76|20.88|
|f6-e5m3|25.36|21.73|21.84|

**The input is the best-preserved tensor, not the worst.** That is the opposite of what a first guess would say, and it is why the four tensors are reported separately.

### Scale counters

On the weights, summed over every epoch of every seed and configuration: `scale_saturated` 0, `scale_underflow` 0, `zero_blocks` 0, `nonfinite_blocks` 0. The same as the L1 measurement.

On the activations, in L2a, same sum: `scale_saturated` 0, `scale_underflow` 0, **`zero_blocks` 6**, `nonfinite_blocks` 0.

**This is the first non-zero counter in the project.** ReLU produces all-zero blocks, and a weight matrix never does. **The zero-block path of the encoder is exercised by activations and by nothing else**, which is a fact for the RTL and not a hypothesis.

### Findings on MNIST

1. **Weight-only FP4 QAT costs 0.21 to 0.41 points.** Two of the four configurations are statistically indistinguishable from the float baseline; the other two are decided, in favour of L0, by 0.29 and 0.41 points.
2. **Weights and activations together cost 0.91 to 1.13 points, and all four configurations are decided.** So the activation half of the MAC costs about 3 times the weight half on MNIST: 1.13 against 0.29 for the same rule.
3. **Training recovers 1.2 to 2.0 points in L2 and 2.5 to 3.6 in L2a.** The more the rung quantizes, the more training has to give back, and every one of those comparisons is decided in favour of training.
4. **The two scale formats are not separated at either rung.** Under either rule, on either rung, the answer is no decision.
5. **`four6` against `amax/6` is not separated at either rung.** The decided 0.42 points that the first version of this section reported was the float master's SQNR, and it is withdrawn.
6. **The best configuration is `f6-e5m3` on both rungs** (94.80 and 93.97), and the worst is `a6-e5m3` on L2 (94.61) and `a6-e6m2` on L2a (93.88). The whole spread is 0.19 and 0.22 points.
7. **The staleness problem survives every correction.** It is measured on the optimizer and not on the evaluated model, so the fix did not touch it.

### MNIST-1D

**Withdrawn, pending a re-run with the corrected evaluation.** The numbers in the first version came from the float master and are not evidence for anything.

- size: `python/l2_sweep.py mnist1d 10 w` needs about 2.5 minutes for each of 40 runs, so 1.7 hours.
- `python/l2_sweep.py mnist1d 10 wa` needs about 2.5 minutes for each of 40 runs as well, because the input is quantized once outside the loop. So another 1.7 hours.

MNIST-1D is 2.5 times more sensitive to the same weight quantization (ptq drop 7.97 to 10.02 against 1.46 to 2.44), so it is where the four configurations separate, if they separate at all on this project's evidence so far.
