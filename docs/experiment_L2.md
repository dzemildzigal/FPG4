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
