# FPG4 specification

Version 0.2. Status: draft. Owner: Dzemo.

## 1. Purpose

This document defines FPG4. FPG4 is an FPGA microarchitecture for a published 4-bit format. It is not a new datatype. The document defines four things:

1. The numeric format, called FPG4-FMT.
2. The exact integer datapath that implements FPG4-FMT in FPGA fabric.
3. The error budget of that datapath.
4. The reference model and the experiments that gate the hardware work.

Any implementation of FPG4 must make sure that its output bits agree with the reference model in section 8. A disagreement is a defect in the implementation, not a rounding difference.

### 1.1 Decision status

The scale configuration is not decided. Two configurations are registered, UE5M3 and UE6M2, and section 4.3 defines both. The decision stays open until experiment E7 measures end-to-end training.

The reason for the open decision is one failure mode. A tensor with a very wide dynamic range can produce a block maximum that is above the top of the UE5M3 window. A single very large activation value is enough. Only the UE6M2 configuration reaches such a block. The measurement is in `docs/results-e1-e3.md`, section 4.

## 2. Terms

A payload is the 4-bit element value. A block is a group of 16 consecutive payload values along the contraction dimension K. A block scale is the single 8-bit value that applies to one block. A contraction dimension is the dimension that the dot product sums over. A code is the integer form of one payload value, as defined in section 5. A guard bit is an extra low-order bit that keeps the datapath exact. A flush is the replacement of a value by zero when the value is too small to align.

## 3. Prior art and the position of this work

The format is published. FPG4 cites it and does not claim it.

| Work | Contribution | Relationship to FPG4 |
| :--- | :--- | :--- |
| Hu et al., arXiv 2601.19026 | Proposes UE5M3 as a hardware-friendly FP4 block scale. Shows that it removes global scaling for weights and activations. | FPG4 uses this scale format. |
| arXiv 2609.02846 | Block-16 UE5M3 FP4 recipe. Pretrains an 8B model for 188.7 billion tokens. Beats the Transformer Engine NVFP4 recipe on held-out negative log-likelihood. | FPG4 uses this recipe as the accuracy reference. |
| OCP MX specification v1.0 | MXFP4: E2M1 payload, block 32, E8M0 scale. | FPG4 keeps the E2M1 payload. |
| NVIDIA NVFP4 | E2M1 payload, block 16, E4M3 block scale, FP32 global scale. | FPG4 is compared against this format throughout. |
| Fahmy group, TRETS 2025 | MX minifloat systolic arrays on FPGA with exact accumulation. | Hardware baseline. |
| "Jack of All Scales", FPL 2026 | Versatile FPGA tensor block for MXFP precisions. | Hardware baseline. |
| arXiv 2310.10537, the MX paper | States that the principal axis of the shared scale is the reduction dimension, and that transposing an MX tensor changes that axis | FPG4 uses the same rule, and section 6 of `docs/motivation.md` records it |
| RPTQ, arXiv 2304.01089. TAGQuant. PermuQuant, arXiv 2605.09503. OCGQuant, arXiv 2609.00066. | A family that groups elements by value. It is a different model of a block. | Section 6.2 of `docs/motivation.md` states the difference. Section 3 of the MX paper states the rule that FPG4 follows. |

The claim of FPG4 is the datapath. The claims are:

1. No DSP slice is used anywhere in the arithmetic path.
2. The arithmetic is exact in integers, with a provable and bounded accumulator width.
3. No tensor-level reference and no second quantization pass are needed.
4. The layout is bit-identical to the published UE5M3 block-level recipe when the UE5M3 configuration is selected.

## 4. FPG4-FMT

### 4.1 Payload

The payload is FP4 E2M1, as defined by the OCP MX specification.

| Field | Bits | Position |
| :--- | :--- | :--- |
| Sign | 1 | bit 3 |
| Magnitude index | 3 | bits 2 down to 0 |

The magnitude index is the value `(exponent << 1) | mantissa`, with a 2-bit exponent field and a 1-bit mantissa field.

| Magnitude index | E2M1 value | Code c | Sign field |
| :--- | :--- | :--- | :--- |
| 0 | 0.0 | 0 | 0 or 1 |
| 1 | 0.5 | 1 | 0 or 1 |
| 2 | 1.0 | 2 | 0 or 1 |
| 3 | 1.5 | 3 | 0 or 1 |
| 4 | 2.0 | 4 | 0 or 1 |
| 5 | 3.0 | 6 | 0 or 1 |
| 6 | 4.0 | 8 | 0 or 1 |
| 7 | 6.0 | 12 | 0 or 1 |

The format has no infinity and no NaN. A value above 6.0 saturates to 6.0. A negative zero and a positive zero represent the same real value.

The magnitude index needs no decoding step in hardware. The index is the table address.

### 4.2 Block

The block size B is 16 payload values. The block runs along the contraction dimension K. Both operands of a dot product use the same block boundaries along K.

### 4.3 Block scale

The block scale is an unsigned 8-bit value. The split of the eight bits is a parameter of the implementation. The parameter is called the scale configuration. Two configurations are registered.

| Configuration | Exponent bits | Mantissa bits | Bias | Top exponent code |
| :--- | ---: | ---: | ---: | :--- |
| UE5M3 | 5 | 3 | 15 | Reserved. **Amended 2026-10-18: only code 255 is NaN. See the amendment below.** |
| UE6M2 | 6 | 2 | 31 | Used. **Amended 2026-10-18: code 255 is NaN.** |

Let E be the number of exponent bits, M the number of mantissa bits, and B the bias. The code splits into an exponent field e in the top E bits and a mantissa field m in the low M bits. Let T be 2^E - 1, the top exponent code.

The scale value s is:

- If the configuration has a subnormal band and e = 0: s = m * 2^(1 - B - M). **Superseded on 2026-10-18. No configuration has a subnormal band.**
- Otherwise: s = (1 + m / 2^M) * 2^(e - B).
- If the top exponent code is reserved and e = T: the encoder must not emit it. The decoder reads it as the largest finite value. **Superseded on 2026-10-18. See the amendment below.**

Only the UE5M3 configuration has a subnormal band. The UE6M2 configuration uses every code for a finite value. **Superseded on 2026-10-18. See the amendment below.**

#### Amendment 2026-10-18: special codes of the block scale

Three decisions replace the old rules. Dzemo approved all three on 2026-10-18.

**Why the old rule is overridden.** The table above reserved the whole exponent
code 31 for UE5M3. That is 8 codes. The project cannot find the origin of that
rule. `ref/fpg4.py` says "as in IEEE-754", and `ref/dump_vectors.py` only tests
that the code never appears. No document gives a functional reason, and UE6M2
never had the rule. A NaN needs 1 code and not 8. The old rule may have copied
the IEEE habit of a non-finite top exponent. That is a hypothesis, not a record.

| # | Decision | Rule |
| ---: | :--- | :--- |
| D1 | The scale has a NaN code | A NaN block scale means every value in the block is NaN. The payload bits are ignored. The FP4 payload has no NaN, so a block can carry NaN only through its scale. The mark stays inside the data when blocks move between chips |
| D2 | The NaN code is 255 | Both formats. 255 is MXFP4 (E8M0) compatible. A block of zero bits decodes to zero (payload 0 times a finite scale), so zero-filled memory is a valid zero tensor. Finite codes 0 to 254 keep the same order as their values |
| D3 | No subnormal band | Both formats use `s = (1 + m / 2^M) * 2^(e - B)` for every e. UE5M3 code 0 is `2^-15`. UE6M2 code 0 is `2^-31`. No configuration has a zero scale |

The measured E1 audit (synthetic multilayer perceptron, 400 steps, bias 15) shows
0.00 percent of blocks in the subnormal band, and a lowest exponent field of 1.
That is a hint and not a proof. The MNIST scale-code histogram of L1b checks it.
If blocks reach the bottom of the window, the band returns.

A zero scale is not needed to represent a zero block. A zero block has payload
index 0 in every element, and `0 * s = 0` for any finite `s`. A zero scale is a
hazard, because the encoder divides by the scale. So the zero-scale repair rule
of section 4.4 step 6 is replaced. A block with `amax = 0` takes code 0.

Encoder rules for special input:

| Input | Scale code | Count |
| :--- | ---: | :--- |
| `amax = 0` | 0 | none |
| `amax` is finite, `amax / 6` above the top finite scale | 254 | scale saturation +1 |
| `amax` is NaN or infinite | 255 | non-finite block +1 |
| `amax / 6` below the lowest scale | 0 | none |

Values after the amendment:

| Item | UE5M3 | UE6M2 |
| :--- | ---: | ---: |
| Finite codes | 0 to 254 | 0 to 254 |
| NaN code | 255 | 255 |
| Smallest scale | 2^-15 = 3.05e-5 | 2^-31 = 4.66e-10 |
| Largest finite scale (code 254) | 1.75 * 2^16 = 114,688 | 1.5 * 2^32 = 6.44e9 |
| Window in block-maximum decades (`6 * s`) | 9.58, from 1.83e-4 to 6.88e5 | 19.14, from 2.79e-9 to 3.87e10 |
| Largest relative scale error | 5.88 percent (unchanged) | 11.1 percent (unchanged) |
| Exponent offset of section 5.3 | 38 (unchanged) | 68 (unchanged) |

The offset is `2 * (bias + M + 1)`. It does not depend on the number of codes or
on the subnormal band, because every N field is now `2^M + m`.

Name rule: the U in UE5M3 and UE6M2 means unsigned, so the code has no sign bit.
Both formats have a bias. A signed E5M3 has 9 bits and does not fit in 8.

Sections that this amendment affects, to update when the text is next revised:

| Place | Old text | New rule |
| :--- | :--- | :--- |
| Section 4.3, last paragraph before 4.4 | No infinity and no NaN | NaN at code 255, no infinity |
| Section 4.4 step 6 | Zero scale repaired to code 1 | `amax = 0` takes code 0 |
| Section 5.3 | N is `2 * m` at e = 0 | N is `2^M + m` for every e |
| Section 5.3, bound on N | N at least 2 | N at least `2^M` |
| Section 5.4, N table | 16 entries for UE5M3 | 8 entries for both formats |
| Section 8, test 3 | Repaired scale code 1 | Scale code 0 |
| `ref/fpg4.py`, `ref/dump_vectors.py` | Old rules | Archived, not repaired. The new `python/quant.py` is the live implementation |

Two numbers follow from the split. The precision follows from M alone. The width of the window follows from E alone.

| Configuration | Window in block-maximum decades | Largest relative scale error |
| :--- | ---: | ---: |
| UE5M3, bias 15 | 9.91, from 4.58e-5 to 3.69e5 | 5.88 percent |
| UE6M2, bias 31 | 19.21, from 2.79e-9 to 4.51e10 | 11.1 percent |

The bias moves the window. It does not change the width of the window. Section 11 records the open decision.

Both configurations have no infinity and no NaN. A scale value above the largest finite value saturates. **Amended 2026-10-18: both configurations have a NaN at code 255 and no infinity. A finite scale value above the largest finite value saturates to code 254.**

### 4.4 Quantizer

The encoder reads 16 elements, called a block, and it produces one 8-bit scale code and sixteen 4-bit payload codes. The encoder sees one block only. It never reads the whole tensor, and it never sees a second block.

The selected rule is `four6`. The rule tests two candidate scale codes and keeps the better one.

1. Compute amax, the largest absolute value in the block.
2. Candidate `c6` is the scale code nearest to `amax / 6`. Candidate `c4` is the scale code nearest to `amax / 4`.
3. Encode the block with `c6`, and then with `c4`. Step 7 defines the encoding of one candidate.
4. Compute the absolute error of each candidate. The error is the sum over the 16 elements of `|x - x_hat|`.
5. Keep the candidate with the smaller error. A tie goes to `c6`.
6. If the chosen scale value is zero, replace the choice with scale code 1. This is the zero-scale repair rule. **Superseded on 2026-10-18. No scale code is zero. A block with `amax = 0` takes code 0.**
7. Encode each element against the chosen scale. Compare the absolute value against the seven thresholds of the payload grid, each one multiplied by s. Take the nearest value. A tie goes to the even mantissa bit.
8. Count the elements with a magnitude above `6 * s` as saturated.

Step 2 uses the thresholds of the scale grid in the amax domain. The threshold between two adjacent scale values `s_lo` and `s_hi` is `d * (s_lo + s_hi) / 2`, where `d` is 6 for `c6` and 4 for `c4`. Both forms are dyadic. The form for `c6` is `3 * (s_lo + s_hi)`, and the form for `c4` is `2 * (s_lo + s_hi)`.

The encoder divides nowhere. Every threshold is the product of a small dyadic constant and a scale value, so every comparison is exact in binary. The RTL uses this form. Test 4 of the conformance suite proves that the fast path agrees with an encoder that uses exact rational arithmetic. Test 10 proves the scale algebra.

Decoding is `x_hat = c / 2 * s`.

An implementation may ship the rule `amax6`, which tests `c6` only and skips steps 3 to 5. The rule is cheaper and it costs 0.70 dB on the real tensors. Experiment E9 measures whether the encoder for `four6` closes timing inside the block time.

### 4.5 Rounding modes

| Use | Mode | Reason |
| :--- | :--- | :--- |
| Weights | Round-to-nearest, ties-to-even | Deterministic and cheap. |
| Forward activations | Round-to-nearest, ties-to-even | Deterministic and cheap. |
| Saved activation for the weight-gradient GEMM | Round-to-nearest, ties-to-even | Deterministic and cheap. |
| Upstream gradient dY | Stochastic rounding, optional | Removes the bias in the expected value. |

Stochastic rounding is a design option, not a requirement. The hardware cost is one random-number source per lane. Experiment E6 in section 10 decides if the option is necessary.

### 4.6 No global scale

FPG4 has no tensor-level scale and no tensor-level amax. The encoder never reads the whole tensor. This is the main difference from the published recipe in arXiv 2609.02846, which keeps a tensor reference with a 50-step refresh.

Experiment E1 in section 10 must show that the removal is safe. If E1 fails, the specification adds a periodic tensor reference and the claim changes.

## 5. Exact integer datapath

This section states the integer rules. Sections 1 to 9 of `docs/motivation.md`
derive each part from the compression and give the purpose of each block.

### 5.1 Element product

The code of an element is c = 2 * value. The code set is {0, 1, 2, 3, 4, 6, 8, 12}. The product of two codes is an exact integer. The largest product is 144, so the product needs 8 bits.

The product table has 8 * 8 = 64 valid input pairs. The product is therefore a read-only memory of 64 entries with 8 bits each. In a 7-series FPGA one 6-input look-up table holds a 64 by 1 ROM. The multiplier costs 8 look-up tables plus 1 look-up table for the sign.

### 5.2 Block sum

For one block b and one output element, the block sum is

    S_b = sum over i in b of (c_w,i * c_x,i)

S_b is an integer. The bound is |S_b| <= 16 * 12 * 12 = 2304. The block accumulator is therefore 13 bits wide, signed.

### 5.3 Block scale application

Every scale value is an integer multiple of a power of two. Define the field N by

    N = 2 * m     if e = 0        the subnormal band, UE5M3 only
    N = 2^M + m   if e >= 1       the normal band

The scale value is then s = N * 2^(e - bias - M). The definition holds in the subnormal range and in the normal range.

| Format | M | bias | s | N range, normal | N, subnormal | exponent offset for E |
| :--- | ---: | ---: | :--- | ---: | :--- | ---: |
| UE5M3 | 3 | 15 | N * 2^(e - 18) | 8 to 15 | 2 * m | 38 |
| UE6M2 | 2 | 31 | N * 2^(e - 33) | 4 to 7 | none | 68 |

The exponent offset is 2 * (bias + M + 1).

Let the two block scales be s_w and s_x. The contribution of block b to the dot product is

    D_b = (s_w * s_x / 4) * S_b
        = S_b * (N_w * N_x) * 2^(e_w + e_x - offset)

The factor 4 comes from the payload code, which holds two times the value divided by the scale.

Define the product P = N_w * N_x and the exponent offset E = e_w + e_x - offset. Then

    D_b = S_b * P * 2^E

Conformance test 10 checks the identity D_b = (s_w * s_x / 4) * S_b over all 65536 pairs of scale codes in both formats. The identity is exact, and it is the whole scale algebra of the datapath.

The product P = N_w * N_x is a quantity of the algebra. It is in the range 0 to 225, so it needs 8 bits for UE5M3 and 7 bits for UE6M2. The datapath does not build P. It applies the two factors in two stages, so the only constant of each stage is one N field, and each N field is at most 15. Section 5.4 gives the structure. P appears in this specification only as a bound and as a name for the exact product. The encoder never emits a zero scale code, so N is at least 2 in a valid stream.

The project ruled on the structure of the term unit on 2026-10-09, and it selected the two-stage form. Section 11 records the ruling and its cost.

E is in the range -38 to 22 for UE5M3, which is 7 bits with a sign, and -68 to 58 for UE6M2, which is 8 bits with a sign.

### 5.4 Exactness

The product S_b * P is exact. The tight bound on T_b depends on the format, because N is at most 15 for UE5M3 and at most 7 for UE6M2.

| Format | Largest N | Bound on |T_b| | Width, signed |
| :--- | ---: | ---: | ---: |
| UE5M3 | 15 | 2304 * 225 = 518,400 | 20 bits |
| UE6M2 | 7 | 2304 * 49 = 112,896 | 18 bits |

The result is exact at either width. UE6M2 needs two bits fewer, and section 5.7 carries that saving into the group accumulator.

The term T_b is built from conditional adds of shifted copies of S_b, and not from a multiplier resource. The unit multiplies by N_w and then by N_x, so it holds two chained constants instead of one product table. The number of conditional adders is the canonical signed-digit count of each constant. N is at most 15, so the count is at most 2 for each stage. UE5M3 needs 4 adders and UE6M2 needs 2. A single-stage form with P as the constant would need 4 adders and 32 look-up tables for UE5M3, and 3 adders and 32 look-up tables for UE6M2, because P reaches 225. Section 9 gives the structure and the size.

The shift by E is exact when E is zero or positive. When E is negative, the shift discards low-order bits. This is the only rounding point in the datapath.

### 5.5 Bit widths

| Signal | Width | Bound |
| :--- | :--- | :--- |
| Payload code | 4 bits | 12 plus sign |
| Element product | 8 bits | 144 |
| Block sum S_b | 13 bits signed | 2304 |
| Mantissa product P, a bound and not a register | 8 bits | 225 |
| Scaled block value T_b | 18 bits signed (UE6M2), 20 bits signed (UE5M3) | 112,896 or 518,400 |
| Scale exponent offset E | 7 bits signed (UE5M3), 8 bits signed (UE6M2) | -38 to 22, or -68 to 58 |
| Group accumulator | T_bits + G + ceil(log2 n) bits | section 5.7 |
| Guard-bit budget G | 12 standard, 16 insurance | decided by E5 |

### 5.6 Alignment

The group accumulator sums the block contributions. The block with the largest E sets the reference exponent. Every other block aligns to that reference.

The reference exponent of a group comes from a pre-scan of the block scale codes of that group. The pre-scan makes the alignment static, so the datapath holds no variable shifter in the accumulator path. Section 3 of `docs/rtl-plan.md` states the reason, the cost, and the alternative.

When a block needs a right shift of more than G bits, the datapath flushes the block to zero and sets a sticky bit. The guard-bit budget G sets the smallest term that survives.

The guard-bit budget is G = 12 for the standard configuration and G = 16 for the insurance configuration. Experiment E5 decides this. The measured spread of the block exponents inside one group is one octave at the median and five octaves in the worst real case, so G = 12 is far above the requirement. The hard synthetic case with a 12-decade range needs G = 10 for an aggregate cost below 0.05 dB, and its error tail falls below 0.03 percent of the RMS at G = 16. Section 5.7 gives the accumulator width.

Guard bits are not an optimization. Measurement E5 shows that G = 0 cuts the signal to noise ratio of a real matrix product from 18.02 dB to 3.71 dB.

### 5.7 Group accumulator width

The aligned term is below 2 ** (T_bits + G - 1), and the sum of n blocks is below n times that value.

    width = T_bits + G + ceil(log2(n))

T_bits is 20 for UE5M3 and 18 for UE6M2. The width of T_b comes from section 5.4.

At G = 12 the UE5M3 width is 35 bits for a contraction of 128, which is 8 blocks, 38 bits for 1024, which is 64 blocks, and 40 bits for 4096, which is 256 blocks. UE6M2 needs two bits fewer in every case. The count is the number of blocks and not the contraction length, so a larger block size reduces the accumulator.

The bound is a proof and not a tight fit. The measured use of the accumulator is below 1 percent on the real matrix products, and below 0.2 percent on the long synthetic contractions. The designer can size to the bound and hold the proof, or size to the measurement and count the overflow events.

## 6. Error budget

The datapath has exactly three error sources.

| Source | Maximum relative error | Applies to | Notes |
| :--- | :--- | :--- | :--- |
| Payload rounding | 20 percent above one scale unit, 33.3 percent between 0.5 and 1.0, and 100 percent below 0.25 | Every element | The dominant term. The measured whole-tensor error is 9.4 percent RMS |
| Scale grid rounding | 5.88 percent | Every block | One error per block of 16. |
| Alignment and flush | Controlled by G | Every block pair | Below 0.05 dB at G = 12 on every case, and zero on the real data at G = 6 |
| Top-end scale saturation | Not bounded | Every block | A fault term, not a rounding term. |

The fourth line is not a rounding effect. A block whose ideal scale is above the largest scale of the format saturates, and the payload of that block saturates with it. The cost depends on the energy of the block. Experiment E3 shows a case where 2 blocks of 8192 carried 99.99 percent of the error energy.

The bottom of the range fails in the other direction and the damage is small. A block whose ideal scale is below the smallest scale of the format floors out. Values that small carry almost no energy, so the damage stays local.

The consequence is that the bias choice is not symmetric. The bias must come from the range audit in experiment E1.

The encoder must count every saturated block and report the count. A saturated block is then a visible event in a training run, and not a silent loss of accuracy.

The payload term dominates. The scale term is second order, because one scale error covers 16 elements.

The mantissa bits of the scale are the cause of the only multiplication in the datapath. Measurement E3 shows that they buy 3.5 to 6.3 dB against a scale with a power of two, which is the E8M0 format of MXFP4. Section 9 states the size of that multiplication, and section 5.4 gives the bound on the number of shift-and-add steps.

This table gives the reason for experiment E3. An E8M0 scale has a maximum relative error of 20 percent, which is equal to the payload term. A UE6M2 scale has a maximum relative error of 11.1 percent. Experiment E3 measures the cost of these two options.

The table also shows that the extra fraction bit in UE5M3 is cheap. The scale multiply happens once per block, so one extra adder in the boundary unit is the whole cost.

## 7. Storage and traffic

| Item | Bits per element | Notes |
| :--- | :--- | :--- |
| Payload | 4 | |
| Block scale | 8 / 16 = 0.5 | |
| Total | 4.5 | The same as NVFP4, without the FP32 tensor scalar. |

## 8. Reference model

The reference model is the authority. It is a small, readable Python module. It is bit-exact and it uses no floating-point arithmetic in the quantization path.

Requirements:

1. The quantizer accepts a float32 array and returns the payload codes, the scale codes, and a statistics record.
2. The decoder accepts payload codes and scale codes and returns float32 values.
3. The integer dot product accepts payload codes and scale codes for both operands and returns the exact integer result before the final flush.
4. The model emits and reads a test vector file, so an RTL testbench can compare bit for bit.
5. The model runs on the Zynq processor of the PYNQ-Z2 board.

Conformance tests:

1. All 256 scale codes decode to the values in section 4.3.
2. All 16 payload codes decode to the values in section 4.1.
3. A block of all zeros produces a repaired scale code of 1 and a zero block sum.
4. A block with one outlier equals the case where the outlier is the maximum.
5. A block that saturates reports a saturation count of at least one.
6. A dot product of maximum codes equals 2304 for B = 16.
7. The integer path agrees with the float path for a tolerance of zero, for a random test set of 10^6 elements, when E is positive.

## 9. Hardware plan

The build plan is `docs/rtl-plan.md`. It holds the architecture, the module list, the interfaces, the time budget, the test strategy, and the acceptance criteria. This section holds the target device and the three variants that the paper compares.

| Item | Value | Source |
| :--- | :--- | :--- |
| Device | XC7Z020-1CLG400C | PYNQ-Z2 |
| Look-up tables | 53,200 | Device |
| Flip-flops | 106,400 | Device |
| DSP48E1 slices | 220 | Device |
| Block RAM | 140 blocks of 36 Kb, or 630 KB | Device |
| PL clock | 125 MHz on board, target 150 to 200 MHz | Board and speed grade |
| DDR3 | 16-bit at 1050 Mbps, about 2.1 GB/s peak | Board |
| PS | 2 Cortex-A9 at 650 MHz | Board |

The processing element contains one 64 by 8 product read-only memory, one 13-bit block accumulator, and one boundary unit. The boundary unit contains the table that decodes the N field of each operand, the term unit of section 5.4, the alignment unit, and the group accumulator of section 5.7. The boundary unit holds no product table.

The table that decodes the N field is a function of the mantissa field and the subnormal flag, so it holds 16 entries of 4 bits for UE5M3 and 8 entries of 4 bits for UE6M2. One 6-input look-up table holds a 64 by 1 read-only memory, so the whole table fits in one look-up table. Section 5 defines the table, and the count comes from the width of the mantissa field and not from synthesis.

The product of two payload elements is not a multiplication in the arithmetic sense. It is a table read of 8 look-up tables, and it stays that way.

The datum has exactly one multiplication, and it is the term unit. The term unit holds two chained scale multiplications of the kind that the format defines. It holds no multiplier resource of any kind.

Every scale code decodes to a value of the form N * 2 ** (e - bias - M). A multiplication of two scales is therefore an addition of the exponents and a multiplication of the two N fields. The exact dequantized value of one block is S_b * s_w * s_x / 4, and the datapath computes S_b * N_w * N_x * 2 ** E with E = e_w + e_x - offset. The two agree for every pair of scale codes, and conformance test 10 checks that fact.

The addition of the exponents is one 7-bit or 8-bit adder for each block. The product N_w * N_x is a small integer product of two 3-bit or 4-bit fields, and the term unit applies it to S in two stages.

    T = (S * N_w) * N_x

A multiplication by N is a group of conditional adds of shifted copies of S. The number of adders is the number of nonzero digits of the canonical signed-digit form of N, less one. The count is bound by the mantissa width, and it is not estimated.

| Format | N field values | Adders for S * N, one stage | Stages | Adders in the term unit |
| :--- | :--- | ---: | ---: | ---: |
| UE6M2 | 4 to 7 | 0 or 1 | 2 | 2 at most, 1.5 on average |
| UE5M3 | 8 to 15 | 0, 1, or 2 | 2 | 4 at most, 2.25 on average |

UE6M2 is the cheaper format in this unit, because its N field is smaller. The shift of each partial product is a fixed amount, so it costs no logic.

A DSP48E1 slice holds a hardened network of the same kind. The AMD user guide UG479 describes the slice as a 25 by 18 two's-complement multiplier followed by an accumulator. UG369 describes the multiply operation as the generation of partial products that a second stage adds. The shift-and-add network and the DSP slice are the same arithmetic, and different hardware. FPG4 uses the look-up-table form, because the unit is two adders and the look-up tables are not scarce.

| Form | Look-up tables | Cycles | Note |
| :--- | ---: | ---: | :--- |
| One adder, shared by the two stages | about 25 | 2 | Fits the 16-cycle block period many times over |
| One adder for each stage, in parallel | about 45 | 1 | The faster form, and not necessary |

The unit runs once for each block, which is once for 16 multiply operations. The datapath therefore needs no DSP48E1 slice for the term unit, and the DSP count of the design stays at zero.

Experiment E9 measures the two forms. The counts of the table above come from the format, and they do not depend on synthesis.

A boundary unit is busy for one cycle in 16. Several lanes can therefore share one boundary unit in a round-robin. That is the lever for the area of an array, and section 3.1 of `docs/rtl-plan.md` records it.

A variant with the term multiply on a DSP48E1 exists for one purpose only. It measures the area that the look-up-table form costs. It is not a fallback for the design and it is not a plan for the hardware.

The scale configuration is three parameters: the exponent bits, the mantissa bits, and the bias. A configuration change touches the decode and the constants only. The payload quantizer, the integer multiply, the shifter, the accumulator, and the two conditional-add stages are common to every configuration, because each stage takes one N field as its constant. The size of the change is a synthesis question, and it belongs to experiment E9. The change costs no extra element storage.

The block scale encoder is a separate item of the area and timing budget. It contains a block buffer of 16 payload codes for each input lane, a maximum tree, the scale encoder, the payload quantizer, and the error accumulator that the rule `four6` needs. Experiment E9 measures its area and its effect on the clock.

The count of encoders is not one, and the count does not follow from the number of blocks. The boundary unit of the datapath runs once for each block, so several lanes share it by round-robin. The payload quantizer of the encoder runs once for each element, so it cannot use that lever. One encoder supplies 8 elements in a cycle with the rule `four6`, and 16 with the rule `amax6`. An array of R rows and C columns needs R/8 encoders for one operand and C/8 for the other, and the rule `amax6` halves both counts. For an array that fills the device, the encoders take about 10 percent of the look-up tables. Section 7.1 of `docs/rtl-plan.md` holds the table, the two bounds, and the case that fails. A stream from DDR supplies about 5 elements in a cycle for both operands together, so a design with a DDR feed needs one encoder for each operand. A stream that already arrives in FPG4 needs no encoder at all.

The buffer is a property of block scaling. Experiment E8 measured the cost of removing it. A delayed encoder uses the statistics of the previous block, and it costs 7.48 dB on the real tensors and 30.39 dB on a 12-decade stress case. The buffer costs about 4 look-up tables for each lane, because a 16-deep shift register fits in one LUT6 as an SRL. The buffer stays in the design. Section 8 of `docs/results-e8.md` records the measurement.

The rule `four6` doubles the arithmetic of the encoder, so it is a separate configuration with its own cost. The second pass of the rule stays as well. Measurement E8 shows that the choice of the rule changes on 38 to 55 percent of neighbouring blocks, so the choice cannot be made one block ahead.

The three variants to compare are:

1. FPG4 with a look-up-table accumulator.
2. FPG4 with a DSP48E1 accumulator.
3. Pure DSP-packed INT4, as the baseline.

## 10. Experiments

Each experiment has a deliverable and a gate. A gate is the hardware decision that depends on the result.

| ID | Experiment | Deliverable | Gate |
| :--- | :--- | :--- | :--- |
| E1 | Range audit over weights, activations, dY, dW and dX for a full training run. | Saturation rate, subnormal rate, and the required exponent window per tensor class. | The bias choice for the scale, the alignment width, and the validity of section 4.6. |
| E2 | Scale-rule sweep at B = 16. Rules: four6, the second largest value, the 75th percentile, the lowest block error, and the oracle. | Error per rule, the dead zone, and the worst-block error. | The specification of the scale unit. Done on 2026-10-07. The rule is `four6`. |
| E3 | Scale-format threat test at B = 16. Formats: E8M0, UE6M2, UE5M3, E4M3 with and without a tensor reference. | One error number and one table per format. | Whether the paper can defend the fraction bits in the scale. |
| E4 | Reference-model conformance. | A pass or fail record against section 8. | The authority for all RTL comparison. |
| E5 | Error attribution between payload, scale, and alignment. | The split of the total error. | The guard-bit budget G and the accumulator width. Done 2026-10-09. G = 12, and the accumulator is T_bits + G + ceil(log2 n) bits, with T_bits of 20 for UE5M3 and 18 for UE6M2. |
| E6 | Backward signal under stochastic rounding. | The cosine between the quantized update and the float32 update, and the rate of zero updates. | Whether the hardware needs a random-number source. |
| E7 | End-to-end training of a small model against FP32, BF16, FP16, FP8, MXFP4, and NVFP4. | Loss, and tokens to reach a loss target. | The acceptance criteria for the board demonstration. |
| E8 | Delayed scaling. The encoder uses the statistics of the previous block. | Error and the area saved by the removal of the buffer. | Whether the array needs a block buffer at all. Done 2026-10-09. The delay costs 7.48 dB on the real tensors and 30.39 dB on a stress case. The buffer stays. |
| E9 | Synthesis of the block scale encoder. | Area, and the change of the clock, for the rules `amax6` and `four6`. | Whether `four6` fits the timing budget. It is the fallback gate for the rule. |

E1, E3 and E4 come first. Each one costs less than one day of work and each one can stop the project.

## 11. Open decisions

0. The structure of the term unit. Decided 2026-10-09. The term unit multiplies by N_w and then by N_x, so `T = (S * N_w) * N_x`. It holds no product table. A ruling was needed because section 5.3 and section 9 of an earlier revision described a 256 by 8 read-only memory for `P = N_w * N_x` and a single stage `T = S * P`, while section 5.4 and `rtl-plan.md` section 3.1 described the two-stage form. The two-stage form is never worse and it is better for UE6M2, and it needs no read-only memory. The computed costs are 4 conditional adders and 0 look-up tables for UE5M3, and 2 adders and 0 look-up tables for UE6M2. The single-stage form is 4 adders and 32 look-up tables for UE5M3, because P reaches 225, and 3 adders and 32 look-up tables for UE6M2. The ruling is recorded in `learn-fpg4/REVIEW.md` item F1.
1. The bias choice for the scale. The default is 15, which matches the published work. E1 decides. E3 shows that the top of the range is the dangerous end, because the largest values carry the energy.
2. The scale-selection rule. Decided by E2. The rule is `four6`, which tests the two candidate scales amax/6 and amax/4 and keeps the better one. It gains 0.17 to 2.64 dB against the largest-value rule. Clipping the top of the block is rejected. The test that chooses between the two candidates is the absolute error of the block, not the squared error, and not a saturation test. The absolute error needs no multiplier, and it loses only 0.14 dB against the squared error. The rule doubles the work of the encoder. The area and the clock effect of that doubling are open, and they come from synthesis in experiment E9. If the encoder does not close timing, the fallback is the rule `amax6`, which costs 0.70 dB on the real tensors. The measured values on the real tensors are 21.27 dB for `four6` and 20.57 dB for `amax6`, both from measurement E2 in this project.
3. The guard-bit budget G. Decided by E5. The value is 12 for the standard configuration and 16 for the insurance configuration. The measured spread of the block exponents inside one group is one octave at the median and five octaves in the worst real case, so the measured requirement is 4 to 6. The accumulator width is T_bits + G + ceil(log2 n) bits, with T_bits of 20 for UE5M3 and 18 for UE6M2.
4. Stochastic rounding for dY. The default is off. E6 decides.
5. The tensor reference. The default is none. E1 decides.
6. The scale configuration. This decision is open until E7. Two positions are defensible, and the hardware supports both. The hardware cost is not the same for the two, and the measurements of this revision change the balance.

   Position A, UE5M3 with bias 15. It is the more accurate configuration. It costs 0.17 dB against the payload floor on the real tensors, and it matches the NVFP4 surrogate. The measured spread of block maxima in the audited run is 4.5 decades, and the window holds 9.91 decades. Its hardware is the larger of the two: the term unit holds 4 adders, T_b is 20 bits, and the group accumulator is 2 bits wider.

   Position B, UE6M2 with bias 31. It is the safer configuration. It costs 0.38 dB against the payload floor on the real tensors, which is 0.21 dB more than position A. In exchange it holds 19.21 decades, so a tensor with a very wide dynamic range cannot push a block maximum above the window. Measurement E3 shows that such a tensor costs UE5M3 a factor of 100 in error power, and costs UE6M2 nothing. Its hardware is the smaller of the two: the term unit holds 2 adders, T_b is 18 bits, and the group accumulator is 2 bits narrower. It needs one bit more for the exponent offset E, which the alignment shifter uses and the accumulator does not.

   The stated reason for the open decision is the risk of a single very large activation value. That risk is not present in the audited run, and no small model will produce it. Section 1.1 records the status. The counts above come from the format, and experiment E9 measures the two builds.
## 12. Out of scope

Inference kernels for a deployed model. Weight-only quantization. Sparsity. Non-16 block sizes, except in experiment E2. Attention and normalization kernels. Multi-FPGA systems.

A randomized Hadamard transform is out of scope. It is the answer to one case that no scale rule can repair. A block that holds one value 1000 times larger than its neighbours loses 92 percent of those neighbours to zero. A Hadamard transform spreads such an outlier across many blocks, and it uses additions only. Experiment E2 measured the case. Section 4.2 of `docs/results-e1-e3.md` records the result.

## 13. Revision history

| Version | Date | Change |
| :--- | :--- | :--- |
| 0.1 | 2026-10-07 | The format, the exact integer datapath, the error budget, and the experiments. |
| 0.2 | 2026-10-09 | Experiment E5 sets the guard budget at 12 and the accumulator at T_bits + G + ceil(log2 NB) bits. Experiment E2 sets the rule at `four6` with the absolute-error test. Experiment E8 rejects delayed scaling and keeps the block buffer. The alignment reference is a pre-scan of the group scale codes. The RTL plan is added. |
| 0.3 | 2026-10-09 | The term unit is stated as two chained scale multiplications, and not as a multiplier. The count of adders comes from the format, and it is 2 for UE6M2 and 4 for UE5M3. The width of T_b is 18 bits for UE6M2 and 20 bits for UE5M3. Section 5.4 gains the bound table, and the width table of section 5.7 is corrected to a count of blocks and not a contraction length. |
| 0.4 | 2026-10-09 | The MX paper `arXiv 2310.10537` is added to the prior art of section 3. It states the rule that sets the block axis, and the rule that a transposed tensor changes that axis. Section 6.1 and section 6.2 of `docs/motivation.md` record the two models of a block. |
| 0.5 | 2026-10-18 | The reserved exponent 31 of UE5M3 is overridden, because its origin is not recorded. Both scales get one NaN at code 255. Neither has a subnormal band or a zero scale. The zero-scale repair rule is replaced. The UE5M3 window changes from 9.91 to 9.58 decades. The UE6M2 window changes from 19.21 to 19.14 decades. Section 4.3 carries the amendment, and the table of affected sections. |
