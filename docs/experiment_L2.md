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

### MNIST

Conditions: 10 seeds, float32 (`train.DTYPE`), full test set, 10 epochs, batch 128, dataset seed 42, 784-16-16-10. Script `python/l2_sweep.py mnist`. Every run is resumable in `out/l2/mnist/`.

Three rungs sit under L0 and they are different, so the column means are labelled:

| rung | what it is | needs training |
|-|-|-|
| `L0` | no quantization | yes, and it is the same run as rung L0 |
| `L1a` | FP4 payload with a float64 scale. The scale is exact, so this rung has no 8-bit scale at all | no, it shares the L0 weights |
| `ptq` | FP4 payload with the configuration's own 8-bit scale. This is rung L1b or L1c | no, it shares the L0 weights |
| `L2` | the same quantization, but the forward pass sees it during training. Option A: weights only | yes, a new run per configuration and seed |

`ptq` is the like-for-like reference for `L2`. `L1a` is the rung above it and it is shown so the ladder has no gap.

Accuracy, percent, full test set:

|seed|L0|L1a|a6-e6m2 ptq|f6-e6m2 ptq|a6-e5m3 ptq|f6-e5m3 ptq|
|-|-:|-:|-:|-:|-:|-:|
|42|94.71|92.42|91.21|90.82|91.51|90.69|
|4242|95.41|94.73|94.30|94.88|94.33|93.76|
|424242|95.42|94.48|94.73|94.86|94.78|94.87|
|123|94.89|89.01|93.12|92.56|91.21|94.07|
|456|94.79|91.78|92.26|92.96|92.06|91.42|
|789|94.66|92.28|92.60|92.83|91.49|92.51|
|1010|95.18|92.04|93.38|94.23|91.56|92.94|
|2020|95.37|94.67|94.08|94.83|94.90|94.18|
|3030|94.75|92.90|93.28|93.14|92.49|94.02|
|4040|94.95|92.15|93.52|94.38|91.40|93.11|
|**mean**|**95.01**|**92.65**|**93.25**|**93.55**|**92.57**|**93.16**|

L2, same seeds and same columns:

|seed|a6-e6m2|f6-e6m2|a6-e5m3|f6-e5m3|
|-|-:|-:|-:|-:|
|42|94.80|94.62|94.58|94.35|
|4242|94.42|94.71|92.89|94.49|
|424242|94.93|94.05|94.69|94.94|
|123|94.32|94.28|93.82|94.51|
|456|93.76|94.21|93.25|93.78|
|789|94.71|94.63|94.25|94.46|
|1010|94.27|95.08|94.78|94.57|
|2020|94.75|94.54|94.33|95.24|
|3030|94.04|94.97|94.44|94.85|
|4040|94.56|94.54|94.74|94.75|
|**mean**|**94.46**|**94.56**|**94.18**|**94.59**|

Drop against L0, in points. A smaller number is better:

|configuration|L1a drop|ptq drop|L2 drop|training recovers|
|-|-:|-:|-:|-:|
|a6-e6m2|2.37|1.77|0.56|1.21|
|f6-e6m2|2.37|1.46|0.45|1.01|
|a6-e5m3|2.37|2.44|0.84|1.60|
|f6-e5m3|2.37|1.86|0.42|1.44|

### What training does to the grid

End of training, mean over the 10 seeds. `stale` is the fraction of weight elements whose FP4 value did not move from one training step to the next, while the master weight was free to move. `frozen steps` counts the steps, of 469 in an epoch, in which the whole quantized matrix was bit-identical to the step before.

|configuration|SQNR W1|SQNR W2|SQNR W3|stale W1|stale W2|stale W3|frozen steps|
|-|-:|-:|-:|-:|-:|-:|-:|
|a6-e6m2|19.90|19.00|18.80|99.25%|98.97%|98.96%|300.6|
|f6-e6m2|20.64|19.92|19.95|98.61%|98.00%|97.74%|353.3|
|a6-e5m3|20.11|19.47|19.13|98.99%|99.07%|98.88%|302.7|
|f6-e5m3|20.83|20.08|19.94|98.29%|97.95%|97.83%|337.0|

The four scale counters, summed over every epoch of every seed and every configuration: `scale_saturated` 0, `scale_underflow` 0, `zero_blocks` 0, `nonfinite_blocks` 0. Weight bytes do not stress an 8-bit scale, exactly as the L1 counters predicted.

### The decision rule, applied to every pair

|a|b|mean a-b|standard error|sign agrees|condition 1|condition 2|winner|
|-|-|-:|-:|-:|-:|-:|-|
|L2 a6-e6m2|L0|-0.56|0.12|8/10|yes|yes|**L0**|
|L2 f6-e6m2|L0|-0.45|0.15|9/10|yes|yes|**L0**|
|L2 a6-e5m3|L0|-0.84|0.24|10/10|yes|yes|**L0**|
|L2 f6-e5m3|L0|-0.42|0.11|9/10|yes|yes|**L0**|
|L2 a6-e6m2|ptq a6-e6m2|+1.21|0.32|10/10|yes|yes|**L2**|
|L2 f6-e5m3|ptq f6-e5m3|+1.44|0.33|10/10|yes|yes|**L2**|
|L2 f6-e6m2|ptq f6-e6m2|+1.01|0.43|7/10|no|yes|no decision|
|L2 a6-e5m3|ptq a6-e5m3|+1.60|0.55|7/10|no|yes|no decision|
|L2 a6-e6m2|L2 f6-e6m2|-0.11|0.17|4/10|no|no|no decision|
|L2 a6-e5m3|L2 f6-e5m3|-0.42|0.18|8/10|yes|yes|**four6**|
|L2 a6-e6m2|L2 a6-e5m3|+0.28|0.18|7/10|no|no|no decision|
|L2 f6-e6m2|L2 f6-e5m3|-0.03|0.15|4/10|no|no|no decision|

### Findings

1. **Weight-only FP4 QAT costs 0.42 to 0.84 points on MNIST, and that cost is real.** All four configurations are below L0 with both conditions of the rule satisfied. Training recovers 1.01 to 1.60 points of the 1.46 to 2.44 that PTQ lost.
2. **`four6` beats `amax/6` in UE5M3 by 0.42 points**, decided, 8 of 10 seeds, 94.59 against 94.18. In UE6M2 the sign is the same, 94.56 against 94.46, but the rule gives no decision.
3. **MNIST does not separate the two formats.** e6m2 against e5m3 under `amax/6` gives +0.28 with 7 of 10 signs and under `four6` gives -0.03 with 4 of 10. Both are no decision.
4. **The best configuration is `f6-e5m3` at 94.59**, then `f6-e6m2` 94.56, `a6-e6m2` 94.46, `a6-e5m3` 94.18. The spread across the four is 0.41 points, which is smaller than the seed spread of one configuration.
5. **The training problem is visible for the first time.** 98 to 99 percent of the weight elements do not move their FP4 value from one step to the next, and 300 to 353 of 469 steps have the whole quantized matrix frozen while the master keeps moving. The optimizer spends most of its work on values the fabric cannot see. This is the number L3 and L7 have to move.
6. **The counters are all zero**, so L2 cannot separate the formats on range either. The separation has to come from gradients at L3 and L4, and from optimizer state at L6.
7. **The pre-registered hypotheses.** 1, L2 beats ptq: confirmed for two configurations and undecided for two. 2, `four6`'s SQNR advantage converts into accuracy: confirmed in UE5M3, undecided in UE6M2. 3, the formats stay within noise of each other: confirmed on MNIST.

### One anomaly, recorded and not explained

Seed 123 is the reason the `L1a` mean is below every `ptq` column. Measured on that seed:

|rung|accuracy|pooled SQNR|SQNR W1|
|-|-:|-:|-:|
|`L1a`, float64 scale|89.01|20.300|20.455|
|`a6-e5m3`|91.21|20.200|20.321|
|`a6-e6m2`|93.12|20.042|20.099|

`L1a` and the two `ptq` rows differ only in whether the scale is snapped to the 8-bit grid. On this seed the accuracy moves by 4.11 points while the pooled SQNR moves by 0.26 dB, and the accuracy runs **opposite** to the SQNR: the best SQNR gives the worst accuracy. So on this seed the accuracy follows the direction of the weight error, not its size. Not investigated. It is a warning that a single seed's PTQ number is not a measurement of a rule.

### Not yet run

MNIST-1D. It is 2.5 times more sensitive to the same quantization (L1: 1.7 to 2.4 points against 1.5 to 1.9), so it is where the four configurations separate, if they separate at all.

### MNIST-1D

Same conditions: 10 seeds, float32, full 100,000-sample test set, 10 epochs, batch 128, dataset seed 42. Script `python/l2_sweep.py mnist1d`. One epoch is 3,125 steps here, not 469.

Accuracy, percent, full test set:

|seed|L0|L1a|a6-e6m2 ptq|f6-e6m2 ptq|a6-e5m3 ptq|f6-e5m3 ptq|
|-|-:|-:|-:|-:|-:|-:|
|42|64.89|55.21|58.75|60.59|57.89|57.50|
|4242|64.75|57.46|57.98|60.30|57.10|56.14|
|424242|65.17|50.42|53.57|54.98|51.54|51.65|
|123|64.44|58.42|58.14|59.21|57.02|58.70|
|456|64.86|58.64|51.73|52.73|58.85|56.09|
|789|64.43|58.52|48.00|49.60|55.99|54.59|
|1010|65.33|47.94|55.34|55.21|52.81|54.09|
|2020|65.12|61.62|48.06|51.46|60.40|59.87|
|3030|66.33|60.58|59.17|58.89|51.81|61.94|
|4040|64.45|58.86|58.82|58.89|59.36|59.54|
|**mean**|**64.98**|**56.77**|**54.96**|**56.19**|**56.28**|**57.01**|

L2, same seeds and same columns:

|seed|a6-e6m2|f6-e6m2|a6-e5m3|f6-e5m3|
|-|-:|-:|-:|-:|
|42|64.40|63.69|63.87|63.89|
|4242|63.63|64.07|63.60|64.02|
|424242|63.83|65.14|65.03|65.17|
|123|63.92|64.53|62.12|62.62|
|456|63.70|62.31|62.96|63.07|
|789|63.89|64.41|63.31|63.30|
|1010|62.77|62.30|62.81|63.97|
|2020|64.52|64.43|65.09|65.20|
|3030|63.38|64.42|63.80|64.93|
|4040|63.69|64.65|63.70|64.26|
|**mean**|**63.77**|**64.00**|**63.63**|**64.05**|

Drop against L0, in points:

|configuration|L1a drop|ptq drop|L2 drop|training recovers|
|-|-:|-:|-:|-:|
|a6-e6m2|8.21|10.02|1.20|8.82|
|f6-e6m2|8.21|8.79|0.98|7.81|
|a6-e5m3|8.21|8.70|1.35|7.35|
|f6-e5m3|8.21|7.97|0.93|7.03|

End of training, mean over the 10 seeds. `frozen steps` counts an epoch of 3,125 steps:

|configuration|SQNR W1|SQNR W2|SQNR W3|stale W1|stale W2|stale W3|frozen steps|
|-|-:|-:|-:|-:|-:|-:|-:|
|a6-e6m2|19.79|18.05|18.55|98.17%|97.65%|97.90%|552.2|
|f6-e6m2|20.75|19.05|19.21|97.11%|95.70%|95.27%|797.1|
|a6-e5m3|20.00|18.06|18.56|97.47%|97.80%|97.81%|513.9|
|f6-e5m3|20.90|19.23|19.39|96.34%|95.92%|95.81%|869.7|

The four scale counters, summed over every epoch of every seed and every configuration: `scale_saturated` 0, `scale_underflow` 0, `zero_blocks` 0, `nonfinite_blocks` 0. Same as MNIST.

The decision rule, applied to every pair:

|a|b|mean a-b|standard error|sign agrees|condition 1|condition 2|winner|
|-|-|-:|-:|-:|-:|-:|-|
|L2 a6-e6m2|L0|-1.20|0.28|10/10|yes|yes|**L0**|
|L2 f6-e6m2|L0|-0.98|0.37|8/10|yes|yes|**L0**|
|L2 a6-e5m3|L0|-1.35|0.29|10/10|yes|yes|**L0**|
|L2 f6-e5m3|L0|-0.93|0.22|8/10|yes|yes|**L0**|
|L2 a6-e6m2|ptq a6-e6m2|+8.82|1.45|10/10|yes|yes|**L2**|
|L2 f6-e6m2|ptq f6-e6m2|+7.81|1.24|10/10|yes|yes|**L2**|
|L2 a6-e5m3|ptq a6-e5m3|+7.35|1.06|10/10|yes|yes|**L2**|
|L2 f6-e5m3|ptq f6-e5m3|+7.03|0.99|10/10|yes|yes|**L2**|
|L2 a6-e6m2|L2 f6-e6m2|-0.22|0.27|6/10|no|no|no decision|
|L2 a6-e5m3|L2 f6-e5m3|-0.42|0.14|9/10|yes|yes|**four6**|
|L2 a6-e6m2|L2 a6-e5m3|+0.15|0.26|5/10|no|no|no decision|
|L2 f6-e6m2|L2 f6-e5m3|-0.05|0.32|6/10|no|no|no decision|

### What the two datasets say together

The decision rule requires both conditions on **both** datasets, so a row is decided only when MNIST and MNIST-1D agree.

|question|MNIST|MNIST-1D|verdict|
|-|-|-|-|
|does weight-only FP4 QAT cost accuracy|0.42 to 0.84 points, all four decided|0.93 to 1.35 points, all four decided|**yes, it costs accuracy. Small on both**|
|does training recover what PTQ lost|1.01 to 1.60 of 1.46 to 2.44|7.03 to 8.82 of 7.97 to 10.02|**yes, and most of it**|
|is `four6` better than `amax/6` in UE5M3|**+0.42, decided**|**+0.42, decided**|**yes. The same size on both datasets**|
|is `four6` better than `amax/6` in UE6M2|+0.11, no decision|+0.22, no decision|**no decision**|
|does the format matter under `amax/6`|+0.28, no decision|+0.15, no decision|**no decision**|
|does the format matter under `four6`|-0.03, no decision|-0.05, no decision|**no decision**|

**L2's verdict is one line: `four6` wins in UE5M3, nothing else is decided, and weight-only FP4 QAT costs under 1.4 points on both datasets.**

### The two results that were not in the plan

1. **The `four6` advantage is the same size on both datasets, 0.42 points, and it is decided on both.** MNIST-1D is 2.5 times more sensitive to weight quantization than MNIST (ptq drop 7.97 to 10.02 against 1.46 to 2.44), and yet the gap between the two rules does not grow with the sensitivity. That is the strongest single result of L2, and it is the first result in this project that survives a change of dataset.
2. **`four6` is the configuration with the most frozen weights, not the fewest.** 869.7 of 3,125 steps frozen on MNIST-1D against 513.9 for `a6-e5m3`, and 337.0 of 469 on MNIST against 302.7. So the rule that quantizes better also tracks the master less. **Accuracy and staleness point in opposite directions.** That is a direct argument for rung L7 and it was not predicted.

### Consequence for the RTL

`four6` stays. Section 4.8 of `docs/datapath.md` costs it as a second scale snap and a second payload pass, close to twice the encoder. The simulation did not drop it, it promoted it, and it did so in the format that was already cheapest to compute: UE5M3 has 8 codes per octave and a 6-bit exponent, so the second snap is 35 LUT6 and not a second divider.
