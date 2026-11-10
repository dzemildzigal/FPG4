# Delayed scaling. Experiment E8

This document records experiment E8. The date of the run is 2026-10-09. The code
is `ref/e8_delayed_scaling.py`. The command is in the last section.

The result is a rejection. Delayed scaling costs 7.5 dB on the real tensors and
up to 30 dB on the stress cases. The tolerance was 0.3 dB. The block buffer and
the current-block dependency stay in the design.

## 1. The question

The rule four6 needs two passes over a block. The first pass finds the maximum
and the two candidate scales. The second pass quantizes the block with both
candidates and keeps the one with the smaller absolute error. The pipeline
therefore holds a dependency. The scale of a block must be ready before that
block leaves the quantizer.

A delayed encoder removes the dependency. It computes the scale from the block
that it has just seen, and it applies that scale to the next block. The
quantizer then holds the scale in advance, and the encoder has a whole block of
time for its work instead of one cycle.

Three questions follow.

1. What accuracy does a stale scale cost?
2. What is the risk of saturation, because a stale scale can be too small?
3. Does the gain of the rule four6 survive when the choice is also one block old?

The third question is the one that pays. If the choice is stable, the encoder
can drop the second pass. If the choice moves every block, it cannot.

## 2. The four pipelines

| Rule | Scale from | Candidate choice | Dependency |
| :--- | :--- | :--- | :--- |
| `amax6_now` | the current block | one target, amax/6 | yes |
| `amax6_prev` | the previous block | one target, amax/6 | no |
| `four6_now` | the current block | tested on the current block | yes |
| `four6_prev` | the previous block | tested on the previous block | no |

Each tensor is one stream of blocks in memory order, which is the order that a
hardware encoder sees. Only the first block of a tensor has no predecessor, and
it uses its own maximum. The delay therefore walks along the memory layout.

The four rows separate two effects.

    the cost of a stale scale   amax6_now against amax6_prev, and four6_now against four6_prev
    the gain of the rule four6  amax6_now against four6_now, and amax6_prev against four6_prev

## 3. The result

Mean over the 8 real tensors from experiment E1, UE5M3, block 16.

| Rule | SNR dB | Cost of the stale scale dB | Gain of four6 dB | Saturated % | Error from saturation % |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `amax6_now` | 20.57 | 0.000 | 0.000 | 3.44 | 21.58 |
| `amax6_prev` | 12.84 | 7.734 | 0.000 | 5.83 | 84.36 |
| `four6_now` | 21.27 | 0.000 | 0.694 | 2.25 | 14.25 |
| `four6_prev` | 13.79 | 7.475 | 0.954 | 3.98 | 76.76 |

The stale scale costs 7.48 dB on the best rule. That is a factor of 5.6 in error
power, and it buys a buffer of 4 look-up tables for each lane.

The synthetic cases are worse.

| Case | `amax6_now` | `amax6_prev` | `four6_now` | `four6_prev` | Cost of the stale scale, four6 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| normal | 20.56 | 15.90 | 21.07 | 16.81 | 4.27 |
| spike 1 percent | 20.84 | 2.17 | 21.05 | 2.26 | 18.79 |
| one big value per block | 31.59 | 4.22 | 34.38 | 4.59 | 29.80 |
| half the tensor small | 20.58 | 15.56 | 21.16 | 16.38 | 4.78 |
| wide range, 12 decades | 30.44 | 0.17 | 30.56 | 0.17 | 30.39 |

The wide-range case falls to 0.17 dB, which is a total loss of the signal. The
error power grows by a factor of 1100.

## 4. Why it fails

### 4.1 The block maximum moves quickly

The table gives the ratio of the block maximum to the maximum of the block
before it.

| Case | Median | p01 | p99 |
| :--- | ---: | ---: | ---: |
| the 8 real tensors | 0.95 to 1.04 | 0.35 to 0.50 | 1.9 to 4.9 |
| spike 1 percent | 0.994 | 0.017 | 78.0 |
| wide range, 12 decades | 1.037 | 0.004 | 172.1 |

The median movement is almost nothing, which is why the idea looks attractive.
One percent of the blocks move by a factor of three or more on real data, and by
two orders of magnitude on the stress cases. Those blocks are the ones that
fail, and they carry the energy.

### 4.2 Saturation follows

A stale scale that is too small pushes the top of the block past the payload
range, and the payload saturates. The share of the error energy that comes from
saturated elements rises from 14.25 to 76.76 percent on the real tensors, and it
reaches 99.97 percent on the wide-range case.

| Case | `amax6_now` | `amax6_prev` | `four6_now` | `four6_prev` |
| :--- | ---: | ---: | ---: | ---: |
| the 8 real tensors | 3.44 | 5.83 | 2.25 | 3.98 |
| wide range, 12 decades | 3.37 | 5.49 | 2.45 | 5.16 |

### 4.3 The choice of four6 is not stable

| Case | Choice changed between neighbouring blocks % |
| :--- | ---: |
| The 8 real tensors | 38.1 to 54.8 |
| spike 1 percent | 43.1 |
| wide range, 12 decades | 37.5 |

The rule four6 changes its target on roughly every second block. A choice made
one block ahead is therefore wrong about half of the time. The third question of
section 1 has its answer, and the answer is no.

## 5. A conservative stale scale does not repair it

A smaller divisor makes the scale larger, so the block cannot saturate. The
price is one bit of payload precision for each factor of two.

| Divisor | Target code | Real tensors SNR dB | normal | spike 1 percent | wide range | Real saturated % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6.0 | 6.0 | 12.84 | 15.59 | 1.70 | 0.07 | 5.83 |
| 4.0 | 4.0 | 15.43 | 18.33 | 2.06 | 0.09 | 1.40 |
| 3.0 | 3.0 | 15.95 | 17.95 | 2.41 | 0.11 | 0.40 |
| 2.0 | 2.0 | 15.02 | 15.83 | 3.08 | 0.14 | 0.09 |
| 1.5 | 1.5 | 13.31 | 13.67 | 3.72 | 0.17 | 0.05 |

The best divisor is 3.0, and its result is 15.95 dB. The current-block rule
reaches 21.27 dB. The conservative stale scale removes the saturation and still
loses 5.32 dB. A larger scale cannot put precision back into a block after the
fact, and the payload is the precision.

The stress cases stay broken for every divisor. The spike case ranges from 1.70
to 3.72 dB against 21.05 dB.

## 6. What survives

The gain of the rule four6 survives the delay. It is 0.954 dB with a stale scale
and 0.694 dB with a current scale on the real tensors, and it grows to 0.908 dB
on the normal case.

So the rule is robust and the freshness of the scale is what matters. Those are
two different properties, and E8 separates them.

## 7. The decision

The block buffer stays, and the encoder sees the current block. The second pass
of the rule four6 stays.

The price of that decision is known and small. A block buffer holds 16 payload
codes for each lane, which is 64 bits. On a 7-series device a 16-deep shift
register is one look-up table for each bit, because one LUT6 holds a 32-deep
SRL. The buffer is therefore about 4 look-up tables for each lane, or 64
look-up tables for a 16-lane front end.

The exchange rate is the point. The buffer costs 4 look-up tables for each lane
and it protects 7.5 dB of accuracy on real data and 30 dB on the stress cases.

The timing question is separate from the area question. The dependency stays, so
the encoder must finish inside one block time. That is a pipeline and clock
question, and experiment E9 measures it.

## 8. What this does not measure

The area and the clock of the encoder. That is experiment E9.

## 9. The commands

    python3 ref/e8_delayed_scaling.py
    python3 ref/e8_delayed_scaling.py --blocks 4096
    python3 ref/e8_delayed_scaling.py --format e6m2

The run takes one second. It writes `out/e8_delayed_scaling.csv` and the tables
in this document. It needs `out/e1_tensors.npz` from experiment E1. Run E1
first.
