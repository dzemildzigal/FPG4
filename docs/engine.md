# The engine

This document states what FPG4 is, what the MLP model in this repository is for, and
the rule that every later circuit must obey. Read it before `datapath.md`.

---

## 1. What FPG4 is

FPG4 is a **general purpose FP4 engine for FPGA**. It does two jobs: it runs
inference, and it runs training.

The engine accepts weights in any format, from the outside world. For the PYNQ-Z2 this will be done via RAM, so PS<---RAM--->PL. 
It **always converts them** to the block-scaled FP4 format: one 4-bit E2M1 payload per value,
one unsigned 8-bit scale per block of 16 values along the reduction axis. The
scale is UE6M2 or UE5M3. That choice is still open.

The engine is a **GEMM engine**. A GEMM is `C = A @ B`. Every learned operation in
this project reduces to a GEMM plus a small number of elementwise steps for bias and activations.

| operation | how it uses the engine |
|---|---|
| dense layer | one GEMM, `A (i, j)` times `B (j, k)` |
| convolutional layer | a GEMM per patch group, or one reshaped GEMM |
| recurrent layer | several GEMMs per time step |
| attention | two GEMMs for the scores and the context, plus a softmax |
| backward pass | a GEMM per gradient, with a different reduction axis |

So the unit of design is the GEMM. The unit of cost is one **block-scaled MAC**.

| term | definition |
|---|---|
| **payload-MAC** | one 4-bit by 4-bit multiply, added into a sum. The atomic arithmetic operation of the engine |
| **block pair** | one block of 16 from `A` and one block of 16 from `B` that own the same 16 positions of the reduction axis |
| **block-scaled MAC** | the contribution of one block pair to one element of `C`. It is 16 payload-MACs, one significand multiply, one alignment shift, and one accumulate |
| **`nb`** | the number of block pairs in one reduction, that is `K / 16` |
| **`k`** | the number of output columns of the GEMM |

One contraction axis of the GEMM is multiple block-scaled MACs. 

---

## 2. What the model is

The repository contains a 784-16-16-10 multilayer perceptron. Two hidden layers of
16, ReLU, softmax with cross entropy, 13,002 parameters.

**The model is a test fixture.** It exists for three jobs:

| job | what it proves |
|---|---|
| correctness | the FP4 datapath produces the same numbers as float64, within a stated error |
| quantization damage | how many accuracy points the format costs, measured on two datasets |
| training | that a training step runs without a float32 master copy of the weights |

**The choice of model (and its size) is to keep track of the impact each step does to our datapath**. 
The model is small because a small model is quick to run and easy to check. 
It is not small because the engine targets small models.

### The rule for the RTL design at any level 

**All RTL for this project is to be architecture-agnostic**, thus *all* RTL 
is to be optimized for **general ML operations** like (GEMM, MAC, etc.)
and **not** optimized for a specific architecture.


| quantity | is it a constant | what it must be |
|---|---|---|
| block size | 16 | a constant of the format. This is the only fixed number |
| `nb` | 49, 1, 1 for this model | an input to the block-scaled MAC accumulator width rule, `20 + ceil(log2(nb))` rounded to a byte |
| `k`, the output width | 16, 10 in the model | a build parameter, the number of parallel lanes |
| `K`, the reduction length | 784, 16 in the model | free. A 4096-wide layer gives `nb = 256` |
| `i`, the batch | 10,000 in the test set | free. One row at a time in inference, small groups in training |

A layer with `K = 4096` has `nb = 256` and needs `20 + 8 = 28` accumulator bits,
rounded to **32**. A layer with `K = 131072` has `nb = 8192` and needs
`20 + 13 = 33`, rounded to **40**. The accumulator width is a parameter, not a
constant of 32. For now we assume 32, but the real width would be configured upon model arch decision and load to the FPGA.

---

## 3. The rule for every pipeline step

The engine is efficient only if **every step** is efficient. So every step gets
four metrics that are tracked during development.

| number | unit | why it matters |
|---|---|---|
| **area** | LUT6, and flip-flops separately | the fabric budget |
| **cycles** | cycles per block of 16 values | throughput, and where the pipeline stalls |
| **power** | switching activity per block | the power budget |
| **critical path** | combinational depth | how fast the clock can run |

A step that is cheap in area and slow in cycles is not efficient. A step that is
fast and large is not efficient. The step is efficient only when all four are
known and all four are optimized.

**Nomenclature about sources** A number from a table size
is labelled `estimate`. A number from synthesis is labelled `measured`. A number
from a formula is labelled `proven`.

---

## 4. The scoreboard

This table is the scoreboard for the whole engine. It is filled one row at a
time. Every number is an **estimate** until rung E9 synthesises the block.

| # | step | circuit | LUT6 | flip-flops | cycles per block | status |
|-:|---|---|---:|---:|---:|---|
| 1 | encode operand `A` | undesigned. `amax` over 16, scale encode, 16 payload quantisers, pack | not designed | not designed | target 8 or fewer | **open** |
| 2 | encode operand `B` | the same circuit, run off-line for weights | not designed | not designed | not on the critical path | **open** |
| 3 | payload product | sign-magnitude table, 64 entries by 8 bits, one copy per lane | 9 per lane | 0 | 0, combinational | estimate |
| 4 | block sum `S` | tree of 15 adders, 9 bits widening to 13 | 161 | 0 | 0, combinational | estimate |
| 5 | scale product `P` | two small tables, two shifts, one add, one exponent add | 20, shared by all lanes | 0 | 0, combinational | estimate |
| 6 | term `S * x * 2^y` | 13 bit by 8 bit shift-add multiply, then a shift to the accumulator's binary point | 135 + 105 | 0 | 0, combinational | estimate |
| 7 | accumulate | 32-bit carry-chain add into the layer accumulator | 32 | 32 | 1 | estimate |
| 8 | bias and ReLU | bias is the accumulator's initial value, so a wire. ReLU is a mux | 32 | 0 | 0 | decided, section 4.6 of `datapath.md` |
| 9 | re-encode `C` | the same circuit as step 1 | same | same | same | **open** |

| total per lane | value | note |
|---|---:|---|
| LUT6 | **about 450**, range 350 to 550 | excludes steps 1 and 9, which are undesigned |
| flip-flops | about 130 | the accumulator plus the pipeline registers |
| **LUT6 per payload-MAC per cycle** | **about 28** | 450 divided by the 16 payload-MACs of one block pair |

The earlier figure of 19 LUT6 per payload-MAC counted only the multiplier and the
shifter. It left out the `S` tree, which is 161 LUT6 and is larger than the
multiplier. **28 is the number to defend.** Rung E9 replaces it.

### What the scoreboard is for

Two consequences, and both are testable:

1. **The MAC core is not the whole engine.** Steps 1 and 9 are undesigned and
   they run on both sides of every layer. A total cost cannot be quoted while
   they are blank.
2. **The only way to beat 28 is to do fewer payload-MACs.** Area per MAC is set by
   the arithmetic. Fewer MACs means sparsity, and sparsity means skipped blocks.
   Skipping a zero block saves power, and saves cycles only if the schedule allows
   it.

---

## 5. Area and power, separately

Area and power are not the same problem and they are not solved by the same
change.

| change | area | power | cycles |
|---|---|---|---|
| more lanes | grows linearly | grows linearly | falls linearly |
| one shared lane, time multiplexed | falls linearly | falls | grows linearly |
| pipeline register | adds flip-flops | small | no change, allows a faster clock |
| operand isolation, gate the input of an idle unit | small | **falls** | no change |
| zero block skip | small, a 13-input OR tree | falls | falls only if the schedule permits |

**Operand isolation** = hold the inputs of a block of logic at zero when its
output is not used, so the logic stops toggling. Power in CMOS follows the number
of transitions. A unit that is not toggling does not burn switching power.

A caution that applies to every sparsity claim in this project. On MNIST, 41.5% of
the layer 1 activation blocks are all zero, because the border of every MNIST
image is blank. **That is a dataset artifact, not a property of the engine.** Do
not size hardware from it. Measure sparsity per workload, and treat the skip as a
power feature that is free to have and dangerous to rely on.

---

## 6. Inference and training, in parallel

The two jobs share the forward GEMM. They differ in what comes after it.

| | inference | training |
|---|---|---|
| forward GEMM | yes | yes |
| encode the activations | yes | yes |
| store the activations | no | yes, for the backward pass |
| backward GEMMs | no | yes, three per layer |
| weight gradient reduction axis | none | the batch, not `j` |
| optimizer state | none | yes, and it is the hard part |

The update problem is the known open risk of this project. The median optimizer
update is 1.431e-3 of the weight value, which is 349 times below one 4-bit step.
That is what rung L6 and the ECO candidate address.

So the engine is tested as an inference platform and as a training platform **with
the same circuits**. If a change helps one and damages the other, it is recorded
as a trade, not as a win.

---

## 7. What is out of scope

| item | why |
|---|---|
| softmax and cross entropy | elementwise or row-wise, outside the GEMM. They stay in float64 in the simulation and are not part of the engine claim |
| nonlinearity beyond ReLU | ReLU is a mux. Sigmoid, tanh and GELU need a table or a polynomial, and none of them is designed here |
| the optimizer | rung L6. It is a separate datapath with its own cost |
| a full accelerator | there is no direct memory access engine, no scheduler, and no host interface. The scoreboard costs arithmetic only |
| a comparison against a commercial FP4 tensor core | the numbers would not be comparable, because a tensor core hides its cost inside a hard block |

---

## 8. How this document relates to the others

| document | relation |
|---|---|
| `spec.md` | the normative definition of the format. Authority for every number. Version 0.7: the amendment of 2026-10-18 is applied, and the layer accumulator follows the `20 + ceil(log2(nb))` rule of `datapath.md`. Three documents it cites stay in the archive, and section 1 of `spec.md` says which and why |
| `datapath.md` | one GEMM, stage by stage, with the widths and the circuits. This document says why those stages are the ones that matter |
| `experiment_plan.md` | the rungs. Each rung changes one factor |
| `experiment_L1.md` | the measured accuracy damage on two datasets |
| `experiment_L2.md` | QAT on the weights only |
| rung E9 | **synthesis.** It converts every `estimate` in section 4 into a `measured` number |

The order of work that follows from this document:

1. Define the encoder as a unit. It is steps 1 and 9, it is half the arithmetic,
   and it is the only part of the pipeline that has never been designed.
2. Fill section 4 with real numbers, in the order 3, 4, 5, 6, 7, then 1, 9.
3. Then synthesise the whole block in rung E9.
