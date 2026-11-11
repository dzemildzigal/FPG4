# The GEMM datapath

One matrix multiply, from the payload codes in memory to the encoded result.
Every number in this document is either proven from the format or measured with
`python/datapath_probe.py`. Nothing is estimated unless it says estimate.

**The model in this repository, 784-16-16-10, is a test fixture.** No circuit in
this document may depend on those numbers. The block size of 16 is the only fixed
number. `nb`, `k` and `K` are inputs to the circuit. See `engine.md` section 2 for the 
reasoning behind this decision.

This document defines a single GEMM. `engine.md` section 4 
holds the scoreboard and the figure of merit for the whole engine, 
and `engine.md` section 3 states the metrics needed for every experiment we do. 

---

## 1. What a GEMM is

**GEMM** = general matrix multiply. It is the operation `C = A @ B`.

| symbol | shape | name |
|---|---|---|
| `A` | `(i, j)` | left operand. `i` = rows, `j` = the reduction axis |
| `B` | `(j, k)` | right operand. `j` = the reduction axis, `k` = columns |
| `C` | `(i, k)` | result |

`C[i, k] = sum over j of A[i, j] * B[j, k]`.

**Reduction axis** = the axis that is summed away. Here it is `j`. After the GEMM,
`j` no longer exists in the result.

The GEMM is legal only if `j` is a multiple of the block size, 16. If it is not,
the operand is padded.

---

## 2. Definitions

Read this table once. Every later section uses these words.

| term | meaning |
|---|---|
| **payload** | the 4-bit FP4 code of one element, E2M1 |
| **payload integer `k`** | the signed integer that the payload code stands for, in units of 0.5. The set is `{0, 1, 2, 3, 4, 6, 8, 12}` with a sign |
| **block** | 16 consecutive elements along the reduction axis, for one owner |
| **owner** | for `A` it is one row `i`. For `B` it is one column `k`. For `C` it is one row `i` |
| **scale** | one unsigned 8-bit FP8 value per block, UE6M2 or UE5M3 |
| **decoded value** | `payload_integer * 0.5 * scale`. This is what the element stands for in "float value" (after compression). |
| **encode** | turn real numbers into payload codes plus one scale code per block |
| **decode** | turn payload codes plus a scale back into real numbers |
| **block sum `S`** | the sum of the 16 integer payload products of one pair of blocks. An integer because payloads are represented with `k` |
| **scale product `P`** | `scale of the A block * scale of the B block`. One value per pair of blocks. |
| **term** | the contribution of one pair of blocks to one element of `C`. `term = S * P / 4` |
| **intra-block accumulator** | `S`. Adds the 16 products inside one block |
| **layer accumulator** | the register that adds the terms of all block pairs into one element of `C`. It is the sum over `S * P`'s for all blocks in the current partial sum. |
| **two's complement** | the signed integer encoding where the top bit has weight `-2^(n-1)` |
| **magnitude bits** | the number of bits needed to store the absolute value, ignoring the sign |
| **binary point** | the fixed position inside a fixed-point number where the integer part ends and the fraction starts. It does not move |
| **precision** | how many bits of a value survive |
| **range** | how large and how small a value can be |
| **alignment** | shifting a term so its binary point matches the accumulator's binary point, before the add |
| **guard bit** | one extra bit kept at the bottom of a value so a right shift can round instead of truncate |
| **round to nearest, ties to even** | the rounding rule of this project |
| **bound** | the largest value the format can possibly produce. A bound is not a measurement |
| **the FP4 claim** | every value that **enters a multiply** is stored as a 4-bit E2M1 payload plus an 8-bit scale. The scale product, the intra-block sum and the layer accumulator are wider, because they are sums and not operands. This sentence is the claim. Nothing in this project says "every bit is 4 bits". **The bias is the one stored tensor with no multiply, and it is stored at the accumulator width, 48 bits, section 4.6** |

---

## 3. How many blocks

| object | shape | blocks | reason |
|---|---|---|---|
| `A` | `(i, j)` | `i * j / 16` | each row `i` is cut into `j / 16` blocks |
| `B` | `(j, k)` | `k * j / 16` | each column `k` is cut into `j / 16` blocks |
| `C` | `(i, k)` | `i * k * j / 16` **pairs** | every pair of blocks meets once |

Let `nb = j / 16`. Then one element `C[i, k]` is the sum of `nb` terms.

**We have *One* scale multiply *per the whole block pair, not per element(s) of that pair*.** Without blocks you would need `j`
multiplies per element, making the FP4 compression useless. 
You'd be remaking floating point, badly, and inefficiently, and with **a lot of rounding errors** all for nothing. 
With scale multiply, one block pair costs **`1` scale multiply and `15` integer adds**
(`16 - 1`, the adder tree over the 16 payload products). Nothing more.
For one element of `C` that is `nb` scale multiplies and `nb * 15` adds, so 49 and
735 for layer 1. `nb - 1` is not 15, `16 - 1` is 15.

---

## 4. The pipeline, stage by stage

```
encode A -> payload codes + scale codes
encode B -> payload codes + scale codes
   |
   v
for each (i, k, b):
    For A@B, here we are making a partial sum over the contraction axis j.
    In order to do that partial sum, we need to make many more partial sums over the blocks of A and B.
    Each of these block partial sums needs to be summed to add to the final partial sum of the contraction over j.
    
    1) do 16 LUT products                                                               -> integer, exact
    2) sum with adder tree the 16 products from 1)                                      -> makes S for that block pair, 13 bits, exact
    3a) take the scale code from the block from A, and scale code from the block from B
    3b) scale multiply those scales to make P.                                          -> makes P for that block pair, 14 bits, rounded once at 3b)
    4) calculate partial sum for this block S * P                                       -> makes the term, rounded once at 4)
    5) add partial sum from 4) (term) into layer accumulator (partial sum acc of AxB)   -> 48-bit fixed point accumulator, point at min(y) of the layer, gives C_i_k (without bias and ReLU)
   |
   v
add bias (as the accumulator's initial value), ReLU                                      -> activation(partial sum + bias)
   |
   v
encode C -> payload codes + scale codes
```

### 4.1 Stage 1, encode A

| step | operation | output |
|---|---|---|
| 1 | `amax` = largest magnitude in the block | one real number per block |
| 2 | `amax / 6` (or the `four6` rule) | one real number per block |
| 3 | pick the closest 8-bit scale code from the table | 8 bits per block |
| 4 | divide the block by the scale, round each element to a payload code | 4 bits per element |

Widths: **4 bits per element**, **8 bits per block of 16**.
That is `4 * 16 + 8 = 72 bits` per block, against `16 * 32 = 512 bits` for float32.
Compression ratio 7.1.

### 4.2 Stage 2, encode B

Identical. The only difference is the transpose, because `B` blocks along its
column direction. `quantize_blocks_weights` does that transpose once.

### 4.3 Stage 3, the block sum `S`

The 16 payload integers of the A block and the 16 payload integers of the B block
meet element by element.

| quantity | value | width | proof |
|---|---|---|---|
| payload integer `k` | `{0, 1, 2, 3, 4, 6, 8, 12}` | 4 magnitude bits | 12 needs 4 bits |
| payload product `k_a * k_w` | at most `12 * 12 = 144` | 8 magnitude bits | 144 needs 8 bits |
| block sum `S` | at most `16 * 144 = 2304` | 12 magnitude bits | 2304 needs 12 bits |
| `S` with sign | `[-2304, +2304]` | **13 bits** | 12 magnitude + 1 sign |

**The payload product is a lookup table, split into sign and magnitude.** Two
4-bit codes would address `16 * 16 = 256` entries of 9 bits. The 256 code pairs
produce only 37 distinct signed products (think, 2x4 is the same as 1x8...), so
the table is redundant. The cheap form takes the sign out and keeps 3 magnitude
bits on each side: `8 * 8 = 64` entries of 8 bits, 512 bits, and the sign is one
XOR. Section 9 builds both circuits and compares them.
Question by Dzemo: Why ROM? This looks like a good job for a LUT to take 16x16 and port it to the 37 distinct signed products? Like an encoder logic circuit? What is more considerate to our FPGA fabric and in what way?

**Answer: you are right, and the word "ROM" was wrong.** On an FPGA there is no
separate ROM. A **LUT6** is the fabric primitive and it holds 64 bits addressed
by 6 inputs, so a LUT6 *is* a 64-word ROM of one bit. "ROM or logic" is not a
choice. The real constraint is the **number of parallel ports**: the 16 products
of one block happen in the same cycle, and a memory has one or two read ports.
That rules out block RAM and distributed RAM. What is left is 16 combinational
copies of a table. The sign-magnitude split of your question is the cheapest
copy, 8 LUT6 of magnitude table plus a sign XOR, against 36 LUT6 for the full
256-entry signed table. Section 9 has the numbers.

The sum of the 16 products is an exact, concrete integer number (albeit with more bits to represent it). There is no rounding in `S`.

`S` is the first accumulator, the **intra-block accumulator**.
Its worst case, for any single `S` is `2304` and it holds that exactly. For that integer it needs 13 bits, no more.

### 4.4 Stage 4, the scale product `P`

`P = scale_A * scale_B`. Both scales are FP8. The multiply is done in three parts.

| part | operation | width |
|---|---|---|
| sign | always positive, both scales are unsigned | 0 bits |
| *multiply significands | `(1 + mantissa_A/2^M) * (1 + mantissa_B/2^M)`, each, result lands in `[1, 4)`. `M` is the mantissa bits of the format, so `/4` for UE6M2 and `/8` for UE5M3 | 2 integer bits + fractional bits needed to represent x in 1.x |
| add exponents | `exponent_A + exponent_B`, no bias removal needed yet | `exponent_A + exponent_B <= 126` for UE6M2, `<= 62` for UE5M3 |

*Significands, with the leading 1 written out:

| format | significand of one scale | worst product | bits |
|---|---|---|---|
| UE6M2 | `1.xx` = `[1, 1.75]` | `1.75 * 1.75 = 3.0625` = `11.0001` | **6** (2 int + 4 frac) |
| UE5M3 | `1.xxx` = `[1, 1.875]` | `1.875 * 1.875 = 3.5156` = `11.100001` | **8** (2 int + 6 frac) |

Question by Dzemo: We initially proposed UE6M2 and UE5M3 here because the logic to multiply two significands should **also be in a LUT/ROM** on an FPGA, not done with a real multiplier. Is this correct? Did we document this?

**Answer: yes to both, and no it was not documented.** Expand the product:

```
(1 + mA/2^M) * (1 + mW/2^M) = 1 + (mA + mW)/2^M + mA*mW/2^(2M)
```

The only multiply is `mA * mW`, which is 2 bits by 2 bits for UE6M2 (a 16-entry
table of 4 bits, 4 LUT6) and 3 bits by 3 bits for UE5M3 (a 64-entry table of 6
bits, 6 LUT6). The rest is shifts and adds. So the whole `P` unit is about
**20 LUT6**, with no DSP and no multiplier block. Section 9 writes it out.

### how two scale bytes become one `P`

**Step 1, what is inside one scale byte.** The two formats field their bytes
differently, and that is the only difference between them:

| format | byte | exponent field | mantissa field | value of one scale | significand `n` |
| :--- | :--- | ---: | ---: | :--- | ---: |
| UE5M3 | `(e << 3) \| m` | 5 bits, `0..31` | 3 bits, `0..7` | `n * 2^(e-15) / 8` | `n = 8 + m`, 4 bits, `8..15` |
| UE6M2 | `(e << 2) \| m` | 6 bits, `0..63` | 2 bits, `0..3` | `n * 2^(e-31) / 4` | `n = 4 + m`, 3 bits, `4..7` |

The significand is written `n = 2^M + m`, so `n` is the mantissa with its leading
one restored. `M` is the mantissa bits: 3 for UE5M3, 2 for UE6M2.

**Step 2, multiply two of them.**

```
s_A * s_W = n_A * n_W * 2^(e_A + e_W - 2*bias) / 2^(2*M)
          = N * 2^(Y + K0)

    N  = n_A * n_W        the two significands, multiplied
    Y  = e_A + e_W        the two exponent fields, added
    K0 = -(2*bias + 2*M)  a constant per format, no logic
```

**Step 3, the two fields of the 14-bit `P`.**

| format | `N` range | `x = N / 2^(2M)` | fraction bits of `x` | `Y` range | `Y` bits | `K0` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: |
| UE5M3 | 64 to 225 | 1.0 to 3.515625 | 6, so `x` is 2 integer + 6 fraction = **8 bits** | 0 to 62 | **6** | -36 |
| UE6M2 | 16 to 49 | 1.0 to 3.0625 | 4, so `x` is 2 integer + 4 fraction = **6 bits** | 0 to 126 | 7 | -66 |

So `P` is 14 bits laid out as:

| bits | field | content |
| :--- | :--- | :--- |
| 13 to 6 | `x` | `N`, unsigned, with `2M` fraction bits. UE6M2 occupies the low 6 bits and is zero extended to 8 |
| 5 to 0 | `Y` | `e_A + e_W`, **unsigned** |

And the definition of the value is:

```
P = x * 2^y        with     y = Y - 30
```

**Why `Y` is unsigned and not two's complement.** The exponent `y` reaches `+32`,
because `e_A = e_W = 31` gives `Y = 62` and `62 - 30 = 32`. Six bits of two's
complement stop at `+31`, so the largest legal exponent would not fit. Storing `Y`
as an unsigned `0..62` covers every legal value. Nothing is lost by dropping the
`-30`: it is a constant, so it joins `K0` and becomes part of the alignment shift.

**Verified, not assumed.** The identity was checked over all `255 * 255 = 65,025`
finite pairs of both formats, in exact arithmetic, against the scale tables of
`quant.py`. Every pair is exact with no rounding.

**Worked examples, UE5M3.**

| code A | code B | `n_A * n_W` = `N` | `x = N/64` | `Y` | `y = Y-30` | `P = x * 2^y` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 120, which is 1.0 | 120, which is 1.0 | 64 | 1.0 | 30 | 0 | **1.0** |
| 0, which is `2^-15` | 0, which is `2^-15` | 64 | 1.0 | 0 | -30 | **9.31323e-10** |
| 254, which is `1.75*2^16` | 0, which is `2^-15` | 112 | 1.75 | 31 | 1 | **3.5** |
| 254 | 254 | 196 | 3.0625 | 62 | 32 | **1.31533e10** |
| 120 | 254 | 112 | 1.75 | 46 | 16 | **114688** |

**Two measured facts that the tables do not show.**

1. `x` takes **35 distinct values** for UE5M3, not 162, because the products
   `n_A * n_W` of two numbers in `8..15` coincide often.
2. The ratio of the largest `P` to the smallest is **`2^63.61`** for UE5M3 and
   **`2^127.17`** for UE6M2. The naive value, the largest `x` times the largest
   `2^y`, gives `2^63.81` and `2^127.6`, and **it is not reachable**: the largest
   significand needs `m = 7`, and `m = 7` with `e = 31` is exactly code 255, which
   is NaN. So the largest significand and the largest exponent can never be in the
   same code, in either operand. This document used to state the naive ratio.

**`x` in logic, for UE5M3.** Expand the product of the significands:

```
N = (8 + m_A) * (8 + m_W) = 64 + 8*(m_A + m_W) + m_A*m_W
```

| piece | operation | cost, estimate |
| :--- | :--- | :--- |
| `m_A + m_W` | two 3-bit numbers, a 4-bit sum | 2 LUT6 |
| `m_A * m_W` | two 3-bit numbers, a 6-bit product, a 64-entry table | 6 LUT6 |
| `8 * (m_A + m_W)` | a shift by 3 | 0 LUT6 |
| add both, plus the constant 64 | one 9-bit add | about 9 LUT6 |
| **total** | | **about 17 LUT6** |

For UE6M2 the same four pieces are 2 bits wide, so the multiply is a 16-entry
table of 4 bits and the total is **about 12 LUT6**, estimate.

**`Y` in logic.** `e_A + e_W`, an adder and nothing else: two 5-bit numbers give a
6-bit sum, **5 LUT6** for UE5M3, and two 6-bit numbers give a 7-bit sum, **6 LUT6**
for UE6M2, estimate.

**A 14-bit `P` serves both formats, because 14 is the larger of the two splits.**
There is no hybrid. A `P` is the product of two scales of **one** format, so it is
never 8 significand bits and 7 exponent bits at the same time. Two cases only:

| build | significand `N` needs | exponent `Y` needs | total | inside a 14-bit `P` |
| :--- | ---: | ---: | ---: | :--- |
| UE5M3 | 8, because `15 * 15 = 225` | 6, because `31 + 31 = 62` | **14** | **0 spare bits** |
| UE6M2 | 6, because `7 * 7 = 49` | 7, because `63 + 63 = 126` | **13** | **1 spare bit** |

The two fields are never both at their widest in the same build, and the register is
14 bits wide in both. **So the field split must follow the format, and the two
formats split the same 14 bits differently:**

| build | significand field | exponent field | layout | significand bit pattern |
| :--- | ---: | ---: | :--- | :--- |
| UE5M3 | 8, and `225` needs all 8 | 6, and `62` needs all 6 | `[ 8 significand ][ 6 exponent ]` | `xx.yyyyyy`, 2 integer and 6 fraction, and `xx` is never `00` |
| UE6M2 | 7, and `49` needs only 6, so one bit is always 0 | 7, and `126` needs all 7 | `[ 7 significand ][ 7 exponent ]` | `0xx.yyyy`, 1 spare then 2 integer and 4 fraction, and `xx` is never `00` |

**One exponent field per build, and it holds one integer, the sum.** The inputs to
that sum are not the field width:

| build | one scale's exponent `e` | the sum `Y = e_A + e_W` | the field |
| :--- | ---: | ---: | ---: |
| UE5M3 | 5 bits, `0..31` | `0..62` | 6 bits |
| UE6M2 | 6 bits, `0..63` | `0..126` | 7 bits |

So the sum needs one more bit than each addend. It is not "two 7-bit integers": each
addend is 6 bits for UE6M2 and 5 bits for UE5M3.

For UE6M2 the alternative is the truthful `6 + 7 = 13` inside the same 14-bit
register, with the top bit of the significand field held at zero. Both are the same
register. The 7-bit significand field is chosen because it fills the 14 bits exactly.

Because the split differs, the binary point of the significand differs too: 6 fraction
bits for UE5M3 and 4 for UE6M2. That difference is one of the two parts of `K0`, so it
is a constant in the alignment shift and not a change of logic.

**The one code that no value uses.** In a UE5M3 build the exponent field is 6 bits
and the largest legal sum is 62, so the code 63 is unused. In a UE6M2 build the field
is 7 bits and the largest legal sum is 126, so the code 127 is unused. That is one
code in each build and not one bit of range. The significand field of UE5M3 has no
such code: `225` needs all 8 bits, and `255 - 225 = 30` unused codes is 0.13 of a bit.

**The 6-bit product that looks like a spare significand is `m_A * m_W`, not `N`.**
The mantissa product is 3 bits by 3 bits and reaches 49. It is one of three terms:

```
N = 64 + 8*(m_A + m_W) + m_A*m_W     for UE5M3, m_A = m_W = 7 gives 225 = 11100001
```

The leading `64` is the two restored ones, `1.xxx` times `1.xxx`, and it is what
pushes the significand from 6 bits to 8.

**`K0` in logic: nothing at all.** The shift that the accumulator applies is
`Y + K0 + (the constant that the accumulator's point contributes)`. All three terms
are constants except `Y`, so they fold into one adder constant. **The choice
between UE5M3 and UE6M2 therefore moves one constant by 30 and changes no logic.**
That is the reason a single 14-bit `P` register can serve either format while the
format decision is open.

Question by Dzemo: 15 bits or 14 bits? Why not have normalization? Seems very logical to have it. Is it integrated in the S*P calculation (the calculation doesn't need it normalized?)

**Answer: 14 bits, and no normalisation. It is not integrated in `S * P` and it
does not need to be.** Normalising means rewriting a value as `1.xxxx * 2^e'` by
shifting the significand right and adding 1 to the exponent. That changes the
**representation**, not the number:

```
sig * 2^e   and   (sig / 2) * 2^(e+1)   are the same value
```

`P` is consumed by `S * P` as a value and then dropped. It is never stored, never
re-packed into FP8 and never compared to another `P`. So there is nothing that a
normalised representation would help. Adding a normaliser costs a significand
test, one shift and an exponent increment, about 12 LUT6, and buys zero accuracy.
Measured support: E2 used a float32 scale product and reached 157.4 dB of layer-1
SQNR, which is 130 dB below the error of the operands themselves. The scale
product is not a limiting factor and cannot become one.

**The one condition that would flip this:** if a later rung keeps a matmul result
in block format without re-encoding it. Then `P` becomes a stored scale and must
be normalised. That is not this design.

#### does normalising `x` make `S * x` cheaper? No. Here is the count, because that is the real question.

Normalising `x` would mean forcing it into `[2, 4)`, so the most significant bit is
always 1 and could be implied instead of stored. The multiply that follows is
`S * x`, which is built as `S * N` and then a shift. `S` is 13 bits signed and `N`
is 8 bits.

| | `x` not normalised, which is the built form | `x` normalised to `[2, 4)` |
| :--- | :--- | :--- |
| `N` range | 64 to 225 | 128 to 255 |
| AND gates that form the partial products | `13 * 8 = 104` | `13 * 7 = 91`, the top bit is implied |
| partial product rows | 7 | 7 |
| **the adder tree, which is the cost** | **about 105 adders, about 105 LUT6** | **the same, about 105 adders** |
| AND gates in LUT6, two per LUT6 | 52 | 46 |
| the normaliser, once, shared by all lanes | 0 | an 8-bit 2-to-1 shift mux, 8 LUT6, plus a 6-bit exponent increment, 6 LUT6, so **14 LUT6** |
| rounding events inside the region that is otherwise exact | **0** | **1**, in about half of the block pairs |
| size of the product | 21 bits | 21 bits, **unchanged** |

Net at 4 lanes: 24 LUT6 saved and 14 LUT6 spent, so **about 10 LUT6 out of about
1,800, which is half of one percent**, all figures estimates until E9. The adder
tree does not change, and the adder tree is where the area is. The product width
does not change either, because `x` still spans `1` to `3.5` after normalisation. A
normalising shift moves the most significant bit, it does not remove a magnitude
range.

**The real price is not area, it is exactness.** In the built form `x = N / 2^(2M)`
is exact. `N` is an integer and `2^(2M)` is a binary point, so no bit is lost. That
exactness is what makes `S * P` exact, and what lets the whole reduction run as
integer arithmetic with one rounding at the very end. Normalising forces a right
shift on the half of the pairs that already sit above 2, and a right shift of an
odd `N` discards a bit. So normalising buys 0.5 percent of the multiply area and
pays with a rounding event in the one region this design keeps exact.

**The wider form does not need any help from the narrow one.** `S * N` with
`N` in `64..225` is a 13-by-8 multiply with 7 adder rows. `S * N` with `N` in
`128..255` is the same 7 adder rows and one fewer AND row. The multiplier does not
get shorter, because the significand still has 8 bits of value.

Mistakes made along the way:

1) **Do not add the fractional bit counts of the two operands.** The product can
exceed 2, so it needs two integer bits. That is the mistake that gives 11 and 12
instead of 13 and 14.

2) `P` is **never** packed back into FP8. It is consumed by `S * P` which accumulates later and that P is dropped.
There is no normalising shift, so the guard bit is not needed.
Question by Dzemo: The guard bit is not needed why exactly? This is tied to my last question for the 14 vs 15 bits for this product discussion.

**Answer: a guard bit exists to make a right shift round instead of truncate.
With no right shift, there is nothing to guard.** The guard bit is the 15th bit.
It is only needed when the significand must be shifted to become normalised, and
that shift is a rounding event. No normalisation, no shift, no rounding event, so
`P` stays at 14 bits. It is an 8-bit significand plus a 6-bit exponent for UE5M3.
Of those 8 significand bits, the top 2 are integer, so 6 are fraction, and
`8 + 6 = 14`. For UE6M2 it is a 6-bit significand plus a 7-bit exponent. The sign is
0 bits because both scales are unsigned.
### 4.5 Stage 5, the term and the layer accumulator

```
term = S * P = S * x * 2^y
```

| name | what it is | width |
|---|---|---|
| `S` | the block sum of 16 payload products | 13, signed, bound 2304 |
| `P` | the scale product, `P = x * 2^y`. It is never stored | 14 |
| `x` | the significand of `P`, `x = n_A * n_W`. Section 4.4 calls it `N` | 8 UE5M3, 6 UE6M2 |
| `y` | the exponent of one term, `y = Y + K0` | a position, not a length |
| `Y` | the unsigned exponent field, `Y = e_A + e_W` | 6 UE5M3, 7 UE6M2 |
| `K0` | a constant of the format: -36 for UE5M3, -66 for UE6M2 | 0, it is built in |
| the payload unit of 0.5 on each side | a free right shift by 2. It sits inside `K0` | 0 |

**The term is exact.** `S * x` is at most `2304 * 225 = 518,400`, which is 19
magnitude bits and 20 with the sign. That product is a 13-bit by 8-bit integer
multiply, and it does not round. `P` does not round either, because `x` and `y`
are exact fields. So the datapath has exactly two places where a bit can be lost:
**the alignment shift** below, and **the re-encode of the final value**.

#### A term has its own exponent. A sum has one.

| object | how many exponents it needs | why |
|---|---|---|
| one term | one per term | each block pair has its own scale |
| the sum of `nb` terms | **one for the whole sum** | the result is one value, so it has one scale |

So an accumulator is **one integer at one binary point**, and every one of its bits
is significand. The exponent of each term leaves the register and becomes a
**shift**. This is true for a fixed-point accumulator and for a floating-point one.
The float one does the same shift and then renormalises, and the renormalisation is
where it rounds, once per add.

`21 + 7 = 28` describes a **term word**: 21 significand bits and 7 exponent-code
bits. It cannot describe an accumulator, because the 7 exponent bits of an
accumulator would all hold the same build constant.

#### The decision

| item | value | where it lives |
|---|---|---|
| the format | **two's complement fixed point, one point per layer** | |
| the width | **48 bits, one width for the whole engine** | a SystemVerilog parameter |
| the point | **`p = min(y)` of that layer, the bottom of the scale window** | one field of the layer descriptor |
| growth allowance inside the 48 | **8 bits**, so `nb` up to 256 and `K` up to 4096 | implied by the width |
| distance allowance inside the 48 | **20 octaves** at `nb = 256`, **22** at `nb = 49` | implied by the width |
| truncation | one counter per layer | |
| overflow | one counter per layer. Wrap or saturate is still open | |

**The point is per layer, and it costs zero LUT6.** It is only a constant in the
shift-amount arithmetic. A point at one fixed absolute position, for example
`24.24`, is right for exactly one layer by luck, and it destroys the low bits of
every layer whose `min(y)` is lower.

**The width is not raised at run time.** The fabric cannot grow. A wider
accumulator is a new bitstream. The 8-bit growth allowance and the distance
allowance are what make one bitstream serve more than one model.

#### The `24.24` point, used for the experiments, and `min(y)` for the design

A point in the middle of 48 bits is `24.24`: 24 integer bits and 24 fraction bits,
so the point sits at `2^0` and the register reaches down to `2^-24`.

**Two tiers, and they are not the same decision:**

| tier | point | why |
| :--- | :--- | :--- |
| **the experiments**, L1e, L1f, L2 and later | **fixed `24.24`** | one constant, one build, and it is the point that the measurement produced |
| **the design**, the engine specification | **`p = min(y)` of the layer**, a field of the layer descriptor | it is correct for every layer. `24.24` is correct only for the layers that stay above `2^-24` |

**`24.24` comes from MNIST-1D layer 1, whose `min(y)` is exactly -24.** The two agree
there, and that is where the number came from.

**For this experiment `24.24` is exact on both datasets and both formats, all six
layers.** Two checks, and both are the inequalities of this section:

| dataset | layer | finest low bit of any term, `min(y)` | `24.24` truncates? | top of the largest sum | `24.24` overflows? |
| :--- | ---: | ---: | :--- | ---: | :--- |
| MNIST | 1 | `2^-22` | no, 2 bits of room | `2^5` | no, 18 bits of room |
| MNIST | 2 | `2^-12` | no, 12 bits of room | `2^6` | no, 17 bits of room |
| MNIST | 3 | `2^-10` | no, 14 bits of room | `2^5` | no, 18 bits of room |
| MNIST-1D | 1 | `2^-24` | **no, exactly on the point** | `2^7` | no, 16 bits of room |
| MNIST-1D | 2 | `2^-11` | no, 13 bits of room | `2^8` | no, 15 bits of room |
| MNIST-1D | 3 | `2^-10` | no, 14 bits of room | `2^8` | no, 15 bits of room |

**So no term loses a bit and no sum overflows in any of the six layers.** The
experiments therefore measure the datapath and not the choice of point, which is what
an experiment has to do.

**`24.24` stops being exact as soon as a tensor has scales below `2^-24`.** Then the
low bits of every term are cut, and nothing in the run says so:

| layer | `min(y)` | a `24.24` register |
| :--- | ---: | :--- |
| MNIST-1D layer 1 | -24 | lands exactly. Nothing is lost |
| MNIST layer 1 | -22 | 2 bits of the bottom are never used |
| a layer whose scales sit at `2^-30` | -30 | **6 low bits of every term are truncated** |
| a layer whose scales sit at `2^-16` | -16 | 8 bits are wasted, nothing is lost |

**That is why the design uses `min(y)` and not `24.24`.** The experiments run on a
fixed point because the probing procedure of the next subsection is not built yet. When
it is built, the layer descriptor carries `min(y)` and `24.24` becomes a value that no
build uses.

#### The rule that sets the width

```
width = 20  +  (max(y) - min(y))  +  ceil(log2 nb)
        ^        ^                     ^
        |        |                     the growth of a sum of nb terms
        |        the distance of the scale window, in octaves
        the signed product S*x: 19 magnitude bits plus the sign
```

| `nb` | growth | distance that 48 bits covers |
|---:|---:|---:|
| 49 | 6 | **22 octaves** |
| 256 | 8 | **20 octaves** |

**22 octaves is 6.6 decades.**

#### The only two ways to lose a bit, as two inequalities

```
truncation happens iff   p  >  min(y)
overflow   happens iff   p + width - 1  <  max(y) + 19 + growth + 1
```

With `p = min(y)` the first is false by definition of `min`. The second is false
when the width is at least `20 + distance + growth`. **So the bottom of the window
is the only choice where one failure is impossible by construction and the other is
impossible by the width.**

| choice of point | truncates | width it needs |
|---|---|---|
| **`p = min(y)`, chosen** | **never**, every shift is a left shift | `20 + distance + growth` |
| `p = (min(y) + max(y)) / 2` | every term below `p` | `20 + distance/2 + growth` |
| `p = max(y)` | every term below `p` | `20 + growth` |

**The middle point halves the width requirement and truncates the bottom half of the
terms.** For a distance of 16 octaves it saves 8 bits, which is about 9% of the
engine, and it costs the low bits of every block below the point. It is rejected
because the ruler of this project is exactness. The divide by 2 is a one-position
shift and it is easy in hardware. Easy is not the question.

#### The measurements behind the numbers

UE5M3, rule `L1c`, exact integer path, 10,000 test samples, from
`python/width_probe.py`. `top` is the top bit of the largest **sum**, not of the
largest term.

| dataset | layer | `nb` | top of largest sum | finest low bit | width needed | footprint `C` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| MNIST | 1 | 49 | `2^5` | `2^-22` | 28 | 25 |
| MNIST | 2 | 1 | `2^6` | `2^-12` | 19 | 19 |
| MNIST | 3 | 1 | `2^5` | `2^-10` | 16 | 16 |
| MNIST-1D | 1 | 49 | `2^7` | `2^-24` | **32** | 30 |
| MNIST-1D | 2 | 1 | `2^8` | `2^-11` | 20 | 20 |
| MNIST-1D | 3 | 1 | `2^8` | `2^-10` | 19 | 19 |

**`C` is the term footprint**: the positions from the finest low bit of any term to
the top of the largest term. `C - 20` is the distance the data used, about 5
octaves on MNIST layer 1 and about 10 on MNIST-1D layer 1. **The 48-bit width covers
22.** An earlier message gave 6 and 11 for those two numbers. It subtracted the 19
magnitude bits instead of the 20 signed bits.

**A legal tensor can still defeat any fixed width, and the failure is
self-limiting.** `width_probe.py` builds three legal UE5M3 tensors:

| tensor | span | what 32 bits does |
| :--- | ---: | :--- |
| trained weights, Gaussian input | 20 | nothing, exact |
| one block of both operands at the floor of the window | 40 | 8 bits lost from 1.7% of the terms, SQNR 191.2 dB |
| one block at the floor and another at the top | 51 | 19 bits lost from 1.7% of the terms, SQNR 230.1 dB |

The bits that are lost sit `span - width` octaves below the top of the largest
term, so the wider the spread the smaller the error it causes, and the SQNR goes up
as the span goes up. The one case that is not self-limiting is a large span **and**
cancellation between the large terms. A trained layer spans 4.5 decades, which is
15 octaves, and it does not cancel to 50 bits.

#### What the alternatives cost

| option | verdict |
| :--- | :--- |
| 32 bits fixed | covers a 6-octave distance. MNIST-1D layer 1 alone needs 30 positions of footprint, and the distance can exceed 6. **Rejected** |
| 38 bits fixed | it has no distance term, so it is a bet, and the bet does not say what it covers. **Rejected** |
| 150 bits, the whole format | the shifter is 7 stages times 150 bits, about 1,050 LUT6 per lane, which is 3.6 times the area for 5.6 times the range that no measured tensor reaches. **Rejected** |
| the mid-point `(min+max)/2` | see above. **Rejected** |
| a fixed `24.24` point | right for MNIST-1D layer 1 by luck. Truncates any layer below `2^-24`. **Rejected** |
| a floating-point accumulator | one rounding per add instead of one per output value. **Rejected** |
| raising the width during training | the fabric cannot grow. A new bitstream is the only way. **Rejected as a run-time mechanism** |

#### Cost, estimates, rung E9 measures them

| block | LUT6 per lane | flip-flops |
| :--- | ---: | ---: |
| barrel shifter, 21 bits in, 48 out, sign extended, 22-octave range, 5 stages | 210 to 250 | 0 |
| 48-bit adder, ripple through the carry chain | 48 | 0 |
| overflow detect | 2 | 0 |
| the accumulator register | 0 | **48** |
| **one lane** | **260 to 300** | **48** |
| **4 lanes** | **1,040 to 1,200** | **192** |

| whole engine, 4 lanes | LUT6 | share of an Artix-7 35T |
| :--- | ---: | ---: |
| the multiply path of section 4.5 and 9 | about 1,800 | 8.7% |
| the accumulator | about 1,100 | 5.3% |
| **total** | **about 2,900** | **14%** |

**The shifter is four times the adder.** The only lever on the shifter is the
declared distance.

#### The build procedure that produces the point

| step | what it does | where |
| :--- | :--- | :--- |
| 1 | cache `min(Y)` and `max(Y)` per layer while the quantiser builds the blocks. `Y` needs only the two scale codes, so it is available before any payload | `python/quant.py`, to be added |
| 2 | report `20 + (max(Y) - min(Y)) + ceil(log2 nb)` per layer | `python/width_probe.py` |
| 3 | compare with the width parameter. If a layer needs more than 48, raise the parameter and rebuild the bitstream | a script to write |
| 4 | after a run, read the truncation and overflow counters. A non-zero counter means the layer was wider than the declared distance | a script to write |

#### What this section used to say, and why each item fell

| withdrawn statement | why it fell |
| :--- | :--- |
| `32` bits, with two bits of margin | the top was taken from the largest **term** and not from the largest **sum**. At 32 bits MNIST-1D layer 1 has nothing spare |
| `F = ceil(log2(max absolute C)) - 31` | it was written here as a decision. It was never decided. It came from a script that needed a value |
| "stick to 32, it is the measured rule" | 32 is one point of a rule. It is not the design |
| "38 is not a candidate" | true for the rule of that revision, and that rule had no distance term. Superseded by the 48-bit decision |
| `width = 20 + ceil(log2 nb)` | it has no distance term, so it silently assumes that every block of the layer has the same scale |

Question from Dzemo, kept for the record: should we stick to 32 bits? If we use 38
bits we would have a catch-all, it is an integer adder, it does not make sense that
it would require **that** much more area. **Answer, revised: the distance term is
what was missing from that question, and it is the term that decides the width.
With it, 48 bits is the catch-all you were asking for, and 38 is not.**

#### One consequence that must be carried into 4.6

The bias is stored at the accumulator's width and at the accumulator's point,
because it is the register's initial value. **At 48 bits that is `42 * 48 = 2016`
bits, not 1344.** Section 4.6 states its percentages at 32 bits and they must be
recomputed.

### 4.6 Stage 6, bias and ReLU

**Decision: the bias is stored at the accumulator width, in the same fixed-point format and at the
same binary point as the layer accumulator, and it is loaded as the accumulator's
initial value.**

The bias is the only tensor in the model that enters **no multiply**. So it has no
reason to share the block format of the operands. It joins the accumulator
unchanged.

`b1` has shape `(k,)`. It has no reduction axis, so it is not blocked along `j` and
it never joins a block sum.

| step | operation | width |
|---|---|---|
| store | one signed fixed-point constant per bias value, binary point as the accumulator | **the accumulator width, which is 48 bits per value** |
| load | written into the accumulator as its **initial value**, before the first term | the accumulator width |
| ReLU | `max(0, acc)` after the last term | the accumulator width |

**The bias width is not a second decision.** It is the accumulator width, because the
bias is the accumulator's initial value. So the width of section 4.5 sets both, and
changing that width changes this one with it. **At 48 bits the bias store is 42 * 48 =
2016 bits.**

There is no decoder. In inference the bias is a constant in the bitstream, so the
load is a wire. Zero logic, zero cycles, zero MACs.

| bias storage | bits, 42 values | whole store | accuracy |
|---|---:|---:|---|
| **the accumulator width, 48-bit fixed point** | **2016** | 58,509 + 2016 = **60,525** | the `float64` column below |
| FP4 payload plus one scale per value | 504 | 59,013 | the `float64` column, plus or minus 0.10 |
| FP4 payload, one shared scale per layer | 4*16+8 and 4*10+8 = 120 | 58,629 | loses up to 1.14 points |

The weight store is 13,002 values in blocks of 16 payloads plus one 8-bit scale,
which is 4.5 bits per value, 58,509 bits. Two numbers matter, and they are
different numbers:

| number | bits | share of the 58,509-bit weight store |
|---|---:|---:|
| what the 48-bit bias store costs in total | 2016 | **3.4%** |
| what it costs **above** the FP4 per-value store of 504 bits | 1512 | **2.6%**, 189 bytes |

**189 bytes is the whole price of not having a second encoder rule for a 42-value
tensor.** The 2.6% is the price against the FP4 option. The 3.4% is the price of
the bias in the store, and it is paid whichever way the bias is stored, because
even the FP4 option costs 504 bits. **These figures moved when the accumulator went
from 32 bits to 48 bits in section 4.5.**

**And the claim in section 2 still holds.** It is about values that enter a
multiply. The bias enters none. 42 values of 13,002 are outside the block format
and 12,960 are inside it.

**Why not FP4: measured, and it was three bad options.** These are the numbers that
rejected the FP4 bias store. The `k` bias values of one layer span a wide dynamic
range:

| vector | n | amax | amin | within-vector range |
|---|---:|---:|---:|---:|
| `b1` | 16 | 0.243 to 0.364 | 0.0044 to 0.0276 | 8.8x to 75x |
| `b2` | 16 | 0.319 to 0.413 | 0.0003 to 0.0057 | 56x to **1195x** |
| `b3` | 10 | 0.296 to 0.359 | 0.0065 to 0.0284 | 12.7x to 45.6x |

Quantization error as a fraction of `amax`:

| format | one shared scale | one scale per value |
|---|---:|---:|
| UE6M2 | 6.2% to 15.4% | 1.9% to 9.1% |
| UE5M3 | 5.8% to 15.4% | 1.9% to 7.3% |

The weight error bound is 11.1% (UE6M2) and 5.88% (UE5M3), so a shared bias block is
worse than the weights it is added to. Accuracy, L1d configuration, rule `L1c`
(`four6`), full test sets, 3 seeds. The `bias float64` column reproduces the L1d
table of `experiment_L1.md` exactly, so it is the same harness.

| dataset | format | bias float64 | one shared block | one scale per value |
|---|---|---:|---:|---:|
| MNIST | UE6M2 | 91.50 | 91.34 | 91.50 |
| MNIST | UE5M3 | 91.52 | 91.64 | 91.53 |
| MNIST-1D | UE6M2 | 52.50 | 52.61 | 53.06 |
| MNIST-1D | UE5M3 | 53.87 | **52.73** | 53.77 |

So the three ways to put the bias on the FP4 grid are: lose up to 1.14 points with
one shared scale, or pay 42 extra scale constants and a block size of 1 for no
measurable gain, or store 840 extra bits and stop arguing. **The third one.**

**Rejected: folding the bias into the reduction.** The standard trick is to append
one element to the contraction: `A` gets a column of 1.0 and `B` gets a row equal
to the bias. Then the bias is one more block term and it joins the accumulator like
any other. Three costs:

| cost | layer 1 (`j` = 784) | layer 2 (`j` = 16) |
|---|---|---|
| `j` grows by 1 and must pad to a multiple of 16 | 785 to 800, `nb` 49 to 50, +2% | 17 to 32, `nb` 1 to 2, **+100%** |
| the bias must survive as a payload inside a block | it shares a scale with 15 other values | same |
| the 1.0 column must encode exactly as 1.0 | it shares a block scale with 15 real activations, so it is only exact for particular scales | same |

So the fold is worst exactly where it looks cheapest, on the two small layers. A
stored 48-bit bias costs no MACs and no growth.

`b3` is 10 values and it is stored as 10 values. There is nothing to pad, because
it is not a block. **Decided.**

Question by Dzemo: Why is bias not blocked by 16? Doesn't make *any sense* as we're about to block literally **everything** that the bias remains unblocked. If the bias is blocked as well, it can go into the GEMM calculation in hardware. We could accumulate all biases for that layer in parallel to the A@B weights, in the same manner actually as A@B weights, and just add it to the layer accumulator in the end as if it were just more additional layers and that's it. Am I missing something here? Something trivial and dumb? Thoughts?

**Answer: you are not missing anything. The fold is legal, and you named its cost
in your own sentence.** "Add it to the layer accumulator in the end as if it were
just more additional layers" is exactly the `nb` 1 to 2 cost on layers 2 and 3 in
the table above. You are right that the bias can be blocked, and right that
blocking it is more uniform. It is not worth a doubled contraction on two layers,
and it is not worth a bias that loses precision on the way in, when the bias has no
multiply for the block format to save.

**And your other instinct is right: the order was wrong.** Adding the bias after the
last term costs an extra adder. Making it the accumulator's **initial value** costs
nothing, because the register is loaded with a constant instead of with zero. It
also removes one rounding, because the bias never has to be rounded against a
partial sum. That is what this stage now does. 

### 4.7 Stage 7, encode C

`C` is now a tensor of 48-bit fixed-point values. It goes through the **same FP4 encoder** as `A` or `B`, with no special case.
Only then is it ready for the next layer.

| step | operation | code |
|---|---|---|
| 1 | `amax` over C's own block of 16 | `quantize_blocks` |
| 2 | pick the scale code by argmin over the table, ties to the even code | `encode_scale` |
| 3 | round each element to a payload code | `payload_index` |

`C` is encoded by the same encoder as `A`. Its blocks are new, they run along the
next layer's reduction axis, and its scale comes from `C`'s own `amax`, never from
`P`.
Question from Dzemo: This text about C just before this question sounds... redundant? Says literally nothing IMHO.

**Answer: yes. The paragraph said one thing three times and it is now one sentence
above.** "New blocks" and "not `P`" were the same statement: a block that is new
cannot carry the old scale.

---

### The accumulator: one scale per sum

**Section 4.5 holds the decision and the evidence. This block keeps only the two
sentences that the rest of the document refers to.**

A term has its own exponent. A sum has one. So an accumulator is one integer at one
binary point, and the exponent of each term becomes a shift. `21 + 7 = 28` is the
width of a term word and it cannot be the width of an accumulator.

Two inequalities, and no others:

```
no truncation at the bottom:  p  <= min(y)
no overflow at the top:       p + width - 1 >= max(y) + 19 + growth + 1
```

**Decided:** 48 bits, `p = min(y)` of the layer, one truncation counter and one
overflow counter per layer. **Open:** wrap or saturate when the overflow counter
fires.

## 5. The complete bit budget

### 5.1 One pair of blocks

| intermediate | width | count | exact? |
|---|---|---|---|
| payload code of A | 4 | 16 | n/a, stored |
| payload code of B | 4 | 16 | n/a, stored |
| scale code of A | 8 | 1 | n/a, stored |
| scale code of B | 8 | 1 | n/a, stored |
| payload integer | 4 magnitude | 32 | **yes**, a table lookup |
| LUT address | 8 | 16 | yes |
| LUT output, one payload product | 9 | 16 | **yes**, 5-bit by 5-bit integers, magnitude at most 144 |
| `S`, the block sum | **13** | 1 | **yes**, the sum of 16 integers, bound 2304 |
| `N`, the significand of `P` | 8 | 1 | **yes**, `n_A * n_W`, bound 225 |
| `Y`, the exponent of `P` | 7 | 1 | **yes**, `e_A + e_W`, bound 126 |
| `P` | **14** | 1 | **yes**, the 14 bits are `N` and `Y`, checked over all 65,025 pairs |
| `S * N` | 21 | 1 | **yes**, a 13-bit by 8-bit integer product, bound 518,400 |
| the alignment shift | the accumulator width | 1 | **exact, or truncating** when a term sits below the accumulator's point |
| the re-encode of the result | 4 + 8 | 1 | **1 rounding**, once per output value |

**The multiply path is exact.** Every intermediate from the 4-bit payload codes to
the 21-bit term is an integer product or an integer sum, so no bit is lost. `P` does
not round: it is `N * 2^(Y + K0)` with both fields exact. `S * P` does not round: it
is the exact product of a 13-bit integer and an 8-bit integer. The only rounding in
the datapath is the **re-encode of the final result, once per output value**, and
the only place a bit can be lost before that is the **alignment shift**, and only
when the accumulator's point sits above a term's low bit.

An earlier revision of this table counted `P`, `S * P` and the add as three rounding
events. That was wrong. Compare with 31 rounding events in a decoded float32 matmul,
16 multiplies plus 15 adds, each of which rounds.

### 5.2 One full layer, our model

Layer 1: `A = X (batch, 784)`, `B = W1 (784, 16)`. `i = batch`, `j = 784`, `k = 16`, `nb = 49`.

| resource | count per sample |
|---|---|
| A payload codes | 784 |
| A scale codes | 49 |
| B payload codes | 784 |
| B scale codes | 49 |
| LUT uses | 49 * 16 = 784 |
| integer adds into `S` | 49 * 15 = 735 |
| `S` values | 49 * 16 = 784 |
| `P` values | 49 * 16 = 784 |
| scale multiplies | 49 * 16 = 784 |
| `S * P` multiplies | 49 * 16 = 784 |
| adds into the layer accumulator | 49 * 16 = 784 |
| layer accumulator registers | `k` in a fully parallel engine, 16 here. **4 in the decided 4-lane engine**, each reused for one quarter of the output neurons and 49 blocks |
| bias constants, stored | 16, at 48 bits each, loaded into the accumulator as its initial value. Not a block, no scale, no decode |
| bias adds | 0. The load is a wire |

| layer | `j` | `nb` | layer accumulators |
|---|---:|---:|---:|
| 1 | 784 | 49 | 16 |
| 2 | 16 | 1 | 16 |
| 3 | 16 | 1 | 10 |

The number of layer accumulator registers is `k`, the number of output neurons,
not `k * nb`. The 49 block terms arrive one after another and reuse the same
register. That is why the adder tree depth to `S` matters for throughput but the
accumulator count does not.

The table above counts the arithmetic of one layer in this model. In general, a
layer of reduction `K` and output width `k` has `nb = K / 16` block pairs and `k`
accumulators. Nothing else in the table changes, because every quantity is either
proportional to `nb * k` or to `k`.

### 5.3 The encoder's storage

The encoder cannot stream a block. `amax` must see all 16 values before any one of
them can be encoded, because the scale comes from the maximum of the block. So one
block is buffered.

| encoder resource | size | note |
|---|---:|---|
| block buffer | **512 flip-flops** | 16 values at 32 bits. Fixed, independent of `i`, `j` and `k` |
| encoded block register | 72 flip-flops | 16 payload codes at 4 bits, one scale at 8 bits |
| blocks encoded per sample, this model | 51 | 49 for the input, 1 for each hidden layer |
| blocks encoded in general | `nb` per row of `A` | for a layer of reduction `K`, that is `K / 16` per row |

So the flip-flop count of the datapath is:

| register | count |
|---|---:|
| layer accumulators, one per lane, at the accumulator width of 48 bits. **The decided engine has 4 lanes** | **192**, and 768 if all 16 output neurons are in flight |
| encoder block buffer | 512 |
| encoded block register | 72 |
| **total** | **776** |

**The encoder buffer is 512 flip-flops against 192 for the four lane accumulators.**
The buffer is 2.7 times the whole accumulator state of the engine, and it is one block.
That is why the encoder is the next design problem.
The previous version of this section counted only the 512 of the accumulators.

---

## 6. One worked element

`X[0] @ W1[:, 0]`, that is `C[0, 0]`, UE6M2, rule `L1c`, seed 42, sample 0 of the
MNIST training set. Blocks 0 to 8 are zero because MNIST has a blank border.

| block `b` | `S` | `P` | `term = S*P/4` | layer accumulator |
|---:|---:|---:|---:|---:|
| 0 to 8 | 0 | ~1e-11 | 0 | 0 |
| 9 | 108 | 8.544922e-3 | +0.230713 | +0.230713 |
| 10 | 104 | 1.562500e-2 | +0.406250 | +0.636963 |
| 11 | 430 | 9.765625e-3 | +1.049805 | +1.686768 |
| 12 | 16 | 1.171875e-2 | +0.046875 | +1.733643 |
| 13 | 198 | 1.953125e-2 | +0.966797 | +2.700439 |
| 14 | 187 | 1.367188e-2 | +0.639160 | +3.339600 |
| 15 | 120 | 9.765625e-3 | +0.292969 | +3.632568 |
| 16 | 468 | 4.272461e-3 | +0.499878 | +4.132446 |
| 17 | 0 | ~1e-11 | 0 | +4.132446 |
| 18 | 140 | 6.103516e-3 | +0.213623 | +4.346069 |
| 19 | 48 | 5.859375e-3 | +0.070313 | +4.416382 |
| 20 | 8 | 1.953125e-2 | +0.039063 | +4.455444 |
| 21 | -100 | 1.367188e-2 | -0.341797 | +4.113647 |
| 22 | 0 | ~1e-11 | 0 | +4.113647 |
| 23 | -390 | 8.544922e-3 | -0.833130 | +3.280518 |
| 24 | 0 | ~1e-11 | 0 | +3.280518 |
| 25 | -352 | 8.544922e-3 | -0.751953 | +2.528564 |
| 26 | 0 | ~1e-11 | 0 | +2.528564 |
| 27 | -155 | 1.367188e-2 | -0.529785 | +1.998779 |
| 28 | -48 | 1.220703e-3 | -0.014648 | +1.984131 |
| 29 | -34 | 1.367188e-2 | -0.116211 | +1.867920 |
| 30 | -16 | 1.953125e-2 | -0.078125 | +1.789795 |
| 31 | -8 | 4.882812e-3 | -0.009766 | +1.780029 |
| 32 | 62 | 1.171875e-2 | +0.181641 | +1.961670 |
| 33 | 0 | ~1e-11 | 0 | +1.961670 |
| 34 | 115 | 6.835938e-3 | +0.196533 | +2.158203 |
| 35 | 175 | 1.171875e-2 | +0.512695 | +2.670898 |
| 36 | 100 | 7.812500e-3 | +0.195313 | +2.866211 |
| 37 | 326 | 1.367188e-2 | +1.114258 | +3.980469 |
| 38 | 28 | 5.981445e-3 | +0.041870 | +4.022339 |
| 39 | 378 | 7.812500e-3 | +0.738281 | +4.760620 |
| 40 | 88 | 1.953125e-2 | +0.429688 | +5.190308 |
| 41 | 38 | 6.835938e-3 | +0.064941 | +5.255249 |
| 42 | 152 | 5.859375e-3 | +0.222656 | +5.477905 |
| 43 to 48 | 0 | ~1e-11 | 0 | **+5.477905** |

`C[0, 0] = 5.4779052734375`. Then the bias, then ReLU, then encode.

The accumulator column above starts at **0** so that the terms are visible. In the
real datapath it starts at the decoded `b1[0]`, and every row of that column is
shifted by that constant. The order of the terms does not change.
Question from Dzemo: Again, for the bias, am I crazy here, or could b1 be folded into the C[0,0] as it was being accumulated and everything was in FP4. Am I making any sense?

**Answer: you are making sense, and the answer is yes with a cost.** `b1[0]` can be
the accumulator's first value, and in this design it is. It cannot be a *block term*
without the padding cost of section 4.6, because a term needs a partner block in `A`
and a spare slot in the contraction, and `j` = 784 has no spare slot. Folding it as a
term also means encoding the bias to FP4 with a scale shared with 15 other values,
which is the case that loses 1.14 points in the table in section 4.6.
The `P` of the zero blocks is tiny but not zero, because `amax = 0` gives the
smallest scale code, not a zero scale. A zero scale does not exist in this format.
`S = 0` makes the term zero, so the value is unaffected.
Question from Dzemo: Well, in logic when we get to it, our multipliers should have a case where one of the operands is 0, because then we can shorten the process and just write 0 out, no need to multiply by 0, right?

**Answer: right, and half of it is already free.**

| where | zero case | cost |
|---|---|---|
| payload product | address `0x0` of the table returns 0. There is no multiply to skip | **already free** |
| `P` | both scales are never zero, so `P` is never zero | nothing to skip |
| `S * P` | `S = 0` makes the whole term 0 | needs a detect: one 13-input OR tree, about 4 LUT6 |

Measured: `S = 0` in **41.5%** of the MNIST layer-1 activation blocks, 203545 of
490000. That number is the blank MNIST border and is not a property of the format.

**What the skip buys in a pipeline: power, not area and not latency.** The pipeline
depth does not change and there is no branch to take. You gate the multiply inputs
so the toggles stop, which is operand isolation, about one AND gate per operand bit.
Expect a power saving and no area saving.

---

## 7. What is decided, and what is open

| item | value | status |
|---|---|---|
| block size | 16, always along the reduction axis | **decided** |
| payload | 4 bits, E2M1 | **decided** |
| scale | unsigned 8 bits, UE6M2 or UE5M3 | format choice still open |
| `S` | 13 bits, exact | **decided**, proven from the format bounds |
| `P` | 14 bits, 2 integer + 8 significand, 7 exponent | **decided**, proven from the format bounds |
| scale product rounding | round to nearest, ties to even | **decided** |
| layer accumulator | two's complement fixed point, **48 bits, one width for the engine** | **decided**, section 4.5 |
| layer accumulator point | **`p = min(y)` of the layer**, a field of the layer descriptor. **The experiments use a fixed `24.24`**, which is exact on all six layers of the two datasets | **decided**, section 4.5 |
| layer accumulator width rule | `20 + (max(y) - min(y)) + ceil(log2(nb))`. Inside 48 bits that is a **22-octave distance at `nb = 49`**, and a **20-octave distance at `nb = 256`** | **decided**, section 4.5 |
| layer accumulator width, evidence | footprint `C` measured at 16 to 30 positions on the two datasets. Distance used by the data is about 5 octaves on MNIST layer 1 and about 10 on MNIST-1D layer 1 | **decided by the measurements**, and a fixed width cannot cover every legal tensor |
| layer accumulator overflow behaviour | wrap or saturate. Saturate and count is the recommendation | **open**, section 4.5 |
| bias storage | **at the accumulator width, signed fixed point, the accumulator's binary point.** Now 48 bits per value, so `42 * 48 = 2016` bits. Section 4.6 still states its percentages at 32 bits and they must be recomputed | **decided**, the FP4 store is rejected in 4.6 with its measurements. The percentages in 4.6 are stale |
| bias add | the accumulator width, fixed point, loaded as the accumulator's **initial value**. No decoder, no adder | **decided**, corrected from "added after the last term" |
| bias folded into the reduction | rejected. `nb` goes 1 to 2 on layers 2 and 3 | **decided against** |
| `b3` padding to 16 | not padded. It is not a block, it is 10 constants | **decided** |
| `P` normalisation | none. `P` is never stored, so no shift and no guard bit. 14 bits | **decided** |
| payload product | sign-magnitude table, 64 entries by 8 bits, 16 parallel copies, **144 LUT6** | **decided**, estimate, item measured in rung E9 |
| `P` circuit | two small tables plus shifts and adds, about **20 LUT6**, no DSP | **decided**, estimate |
| `P` unit count | one per `(i, b, k)`, so 16 for layer 1 at 320 LUT6, or fewer at a higher rate | **open**, a scheduling choice, not a format choice |
| `S * P` circuit | **13x8** shift-add multiply, about 135 LUT6, plus a 21-bit alignment shifter, about 105 LUT6, plus the `S` tree of 15 adders, about 161 LUT6. **About 450 LUT6 per lane for all three.** 4 lanes is about 1800 LUT6, which is 9% of an Artix-7 35T. Or 16 DSP48 if DSP hard blocks are allowed | **open**, rung E9 decides. The earlier "1600 LUT6 for 16 lanes" counted the multiplier only and left out the `S` tree and the shifter |
| encode `A` circuit | **not designed.** `amax` over 16 values, one scale encode, 16 payload quantisers, one pack into 72 bits | **open.** It is steps 1 and 9 of the engine scoreboard, it runs on both sides of every layer, and it is the largest unaccounted element in the datapath |
| encode cost, rule `amax/6` | one scale encode and one payload pass per block | estimate |
| encode cost, rule `four6` | two scale encodes, two payload passes, 16 subtractions, 16 absolute values, two adds, one compare. Close to **twice** the encoder | **estimate.** This is the hardware price of a rule that is still open, and the accuracy difference between the two rules is not yet visible on three seeds |
| encoder block buffer | `amax` needs all 16 values before any of them can be encoded, so the block cannot be streamed. One block of 16 values at 32 bits is **512 flip-flops**, counted in section 5.3 | **decided by the format** |
| encoder cycles | target **8 cycles or fewer per block**. The inference speed of the engine is set by this number, because the encoder feeds the MAC array | **open**, and it is the next design decision |
| figure of merit | about **28 LUT6 per payload-MAC per cycle**, per lane | estimate, `engine.md` section 4 |
| order of operations | integer block sum first, then one scale multiply | **decided**, this is the defer camp |
| C's scale | computed from C, never taken from `P` | **decided** |
| C's blocks | along the next layer's reduction axis | **decided** |

Question from Dzemo: Here, I am not so sure about this, does it make sense that it goes in this order or we push the biases with the A@B accumulation over the contractions?

**Answer: your doubt was correct, the order was wrong and it is now fixed.** The
bias is the accumulator's initial value, not a late add. Pushing it through the
contraction is a legal design and is rejected in section 4.6 with its three costs,
the decisive one being that `nb` doubles on layers 2 and 3.

---

## 8. What this document does not cover

The update path. Backpropagation has a different reduction axis, because the
weight gradient sums over the batch, not over `j`. The blocking rule for a weight
gradient is therefore different. That is rung L3 and later.

---

## 9. How the three multiplies are built

Three multiplies exist in this datapath. Only one of them is a real multiplier.

| multiply | where | size | how |
|---|---|---|---|
| payload product | 4.3 | 4 bits by 4 bits | **table** |
| significand product | 4.4 | 2 or 3 bits by 2 or 3 bits | **table** |
| term `S * P` | 4.5 | 13 bits by 14 bits | **real multiplier** |

Every LUT6 count below is an **estimate** from the table size, not a synthesis
result. Rung E9 measures them.

### 9.1 There is no ROM on an FPGA

A **LUT6** is the fabric primitive: 6 address inputs, 1 output, 64 stored bits. So
a LUT6 **is** a 64-word ROM of one bit. "Use a ROM" and "use a LUT" describe the
same wire. What decides the circuit is the **number of parallel ports**, because
the 16 products of one block all happen in the same cycle.

| form | stored bits | ports | serves 16 parallel lookups |
|---|---|---|---|
| block RAM | 2304 in one block | 2 | no |
| distributed RAM (LUTRAM) | 36 LUT6 | 1 or 2 | no |
| combinational LUT6 tables | any | as many as you copy | **yes** |

Any memory form must be copied 16 times to serve 16 lookups in one cycle, and 16
copies of a memory is more expensive than 16 copies of a 64-entry table. That is
the whole argument. It is also the answer to the question in 4.3.

### 9.2 The payload product

| implementation | stored bits | LUT6 per copy | copies | total LUT6 | latency |
|---|---:|---:|---:|---:|---|
| full signed table, 256 by 9 | 2304 | 36 | 16 | 576 | 0 |
| magnitude table, 64 by 8 | 512 | 8 | 16 | 128 | 0 |
| sign XOR, one per pair | 0 | 1 | 16 | 16 | 0 |
| *the XOR row is an upper bound. One LUT6 holds several 2-input XORs* | | | | | |
| **sign-magnitude, total** | | | | **144** | 0 |
| 16 signed 4x4 multipliers | 0 | 35 | 16 | 560 | 0 |

144 against 560. The sign-magnitude split saves 4 times the logic because 8
magnitudes need 3 bits and the product magnitude fits in 8 bits, so the table is
64 entries and not 256. The 37 distinct signed products do not need to be stored
as 37 entries, because a table is addressed and not searched.

### 9.3 The significand product inside `P`

```
(1 + mA/2^M) * (1 + mW/2^M) = 1 + (mA + mW)/2^M + mA*mW/2^(2M)
```

| format | `mA * mW` | table | LUT6 |
|---|---|---|---|
| UE6M2 | 2 bits by 2 bits | 16 entries of 4 bits | 4 |
| UE5M3 | 3 bits by 3 bits | 64 entries of 6 bits | 6 |

Then two shifts and a three-input add, about 10 LUT6. The exponent path is a
6-bit add, 7 LUT6. So one `P` unit is about **20 LUT6** and it contains no DSP.

How many `P` units: `P` depends on the triple `(i, b, k)`, so one A block needs
`k` of them, 16 for layer 1. Fully parallel that is 320 LUT6. One unit time
multiplexed is 20 LUT6 and 16 cycles per block. **Open, this is a scheduling
decision and not a format decision.**

### 9.4 The term `S * P` is the real multiplier

`P` is a significand times `2^exponent`. So

```
S * P = (S * significand) << exponent
```

The multiply is 13 bits by 8 bits, and the exponent is a shift into the
accumulator's binary point. Sizes:

| form | estimate |
|---|---|
| one 13-bit by 8-bit multiply | about 105 full adders, about 100 to 135 LUT6, or **one DSP48** |
| as a plain 13 by 14 multiply | about 200 LUT6 |
| the `S` tree above it, 15 adders, 9 bits widening to 13 | about **161 LUT6 per lane**, estimate |
| the alignment, shift the 21-bit product to the accumulator's binary point | about **60 to 105 LUT6 per lane**, estimate, depending on which side of the multiply the shift is applied to |
| **one complete lane**, `S` tree plus multiply plus align plus accumulate | **about 450 LUT6, range 350 to 550** |
| 4 lanes, in logic | about **1800 LUT6**, which is 9% of an Artix-7 35T |
| 16 lanes, in logic | about **7200 LUT6**, which is 35% of an Artix-7 35T. Too large |
| 16 lanes, one DSP48 each | **16 DSP48**, which is 18% of the DSP blocks of an Artix-7 35T |

**The multiplier is not the whole cost of the lane.** The payload table is 144 LUT6
for 16 lanes. The `S` tree is 161 LUT6 per lane, so it is larger than the
multiplier it feeds. A lane costs about 450 LUT6, and the figure of merit for the
engine is **about 28 LUT6 per payload-MAC per cycle**, which is 450 divided by the
16 payload-MACs of one block pair.

**The count of lanes is the main area decision, and it is a scheduling decision,
not a format decision.** 4 lanes costs 1800 LUT6 and needs 203 cycles per sample of
this model. 16 lanes costs 7200 LUT6 and needs 51 cycles, and the encoder cannot
feed it. Whether `S * P` runs in a DSP or in logic, and how many lanes exist, is
**open**. It is measured in rung E9.
