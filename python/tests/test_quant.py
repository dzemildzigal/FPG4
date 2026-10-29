import os
import sys
# This file lives in python/tests/. quant.py lives in python/. Put python/ on the import path so "import quant" works from any directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import quant as q

# Each format is a tuple: (name, scale table, quantize function, a value far above the top scale)
# Top scale: UE6M2 table[254] = 1.5 * 2^32 = 6.4e9, so 1e11 / 4 is above it. UE5M3 table[254] = 114688, so 1e6 / 4 is above it.
FORMATS = [
    ("UE6M2", q.e6m2, q.quantize_fp4_b_e6m2, 1e11),
    ("UE5M3", q.e5m3, q.quantize_fp4_b_e5m3, 1e6),
]
RULES = ["L1b", "L1c"]

# Test matrix: shape (K=32, N=1). The quantize functions transpose it to (1, 32) and cut it into 2 blocks of 16.
# Block 0 = base[0:16, 0], block 1 = base[16:32, 0]. The codes have shape (N, B) = (1, 2).
def make_base():
    return np.full((32, 1), 0.1)


# TEST 1: encode(table[c]) == c for every c in 0..254.
# If the input is exactly a table value, the encoder must return that value's code. No code may be lost or merged.
def test_1_round_trip(name, table):
    x = table[:255].reshape(255, 1, 1)                     # (N=255, B=1, 1): 255 blocks, each holds one scale value
    scales, codes = q.encode_scale(x, table)
    assert codes.shape == (255, 1), f"{name}: codes shape {codes.shape}"
    assert np.array_equal(codes[:, 0], np.arange(255)), f"{name}: round trip codes differ"
    assert np.array_equal(scales, x), f"{name}: round trip scales differ"


# TEST 2: the table matches numbers computed by hand.
# Test 1 uses the same table on both sides, so it cannot find a wrong table. These numbers are an independent check.
# s = (1 + m / 2^M) * 2^(e - bias). code 124 in UE6M2: e = 124 >> 2 = 31, m = 0 -> 2^(31-31) = 1.0
# code 120 in UE5M3: e = 120 >> 3 = 15, m = 0 -> 2^(15-15) = 1.0
def test_2_table(name, table, expected):
    assert np.all(np.diff(table[:255]) > 0), f"{name}: table is not strictly increasing"
    assert np.isnan(table[255]), f"{name}: table[255] is not NaN"
    assert not np.isnan(table[:255]).any(), f"{name}: NaN inside codes 0..254"
    for code, value in expected.items():
        assert table[code] == value, f"{name}: table[{code}] = {table[code]}, expected {value}"


# TEST 3: a zero block stays zero. amax = 0 -> code 0, payload 0 / scale = 0, output exactly 0.0, no NaN from 0/0.
def test_3_zero_block(name, table, fn, rule):
    W = make_base()
    W[:16, 0] = 0.0                                        # block 0 all zero, block 1 stays 0.1
    Wq, codes, _, counters = fn(W, rule=rule)
    assert codes[0, 0] == 0, f"{name} {rule}: zero block code {codes[0, 0]}"
    assert np.all(Wq[:16, 0] == 0.0), f"{name} {rule}: zero block output not 0.0"
    assert not np.isnan(Wq).any(), f"{name} {rule}: NaN in output"
    assert counters["zero_blocks"] == 1, f"{name} {rule}: zero_blocks = {counters['zero_blocks']}"


# TEST 4 and 5: a NaN or an infinity makes its whole block NaN (code 255). The other block must not change at all.
def test_4_5_nonfinite_block(name, table, fn, rule, bad):
    base = make_base()
    Wref, cref, _, _ = fn(base, rule=rule)                 # reference: the clean matrix
    W = base.copy()
    W[0, 0] = bad                                          # one bad element in block 0
    Wq, codes, _, counters = fn(W, rule=rule)
    assert codes[0, 0] == 255, f"{name} {rule} {bad}: code {codes[0, 0]}, expected 255"
    assert np.isnan(Wq[:16, 0]).all(), f"{name} {rule} {bad}: block 0 is not all NaN"
    assert np.array_equal(Wq[16:, 0], Wref[16:, 0]), f"{name} {rule} {bad}: block 1 changed"
    assert codes[0, 1] == cref[0, 1], f"{name} {rule} {bad}: block 1 code changed"
    assert counters["nonfinite_blocks"] == 1, f"{name} {rule} {bad}: nonfinite_blocks = {counters['nonfinite_blocks']}"


# TEST 6: a value above the top saturates. Code 254, the counter counts it, the largest output is 6 * table[254].
# The other 15 elements are 0.1, far below one step of the huge scale, so they round to 0.0.
def test_6_above_top(name, table, fn, rule, big):
    W = make_base()
    W[0, 0] = big
    Wq, codes, _, counters = fn(W, rule=rule)
    assert codes[0, 0] == 254, f"{name} {rule}: code {codes[0, 0]}, expected 254"
    assert counters["scale_saturated"] == 1, f"{name} {rule}: scale_saturated = {counters['scale_saturated']}"
    assert Wq[0, 0] == 6 * table[254], f"{name} {rule}: top output {Wq[0, 0]}, expected {6 * table[254]}"
    assert np.all(Wq[1:16, 0] == 0.0), f"{name} {rule}: small elements are not 0.0"


# TEST 6b (control): amax exactly 6 * table[254] fits, so it is NOT saturation. The counter must stay 0.
def test_6b_exactly_at_top(name, table, fn, rule):
    W = make_base()
    W[0, 0] = 6 * table[254]
    Wq, codes, _, counters = fn(W, rule=rule)
    assert codes[0, 0] == 254, f"{name} {rule}: code {codes[0, 0]}, expected 254"
    assert counters["scale_saturated"] == 0, f"{name} {rule}: scale_saturated = {counters['scale_saturated']}"
    assert Wq[0, 0] == 6 * table[254], f"{name} {rule}: output {Wq[0, 0]}"


# TEST 7: ties give the even code, and only exact ties.
# mid is exactly between table[c] and table[c+1]. Expected: c if c is even, else c + 1 (exactly one of the two is even).
# One step (np.nextafter = the next float) above mid is closer to c+1, one step below is closer to c. No tie there.
def test_7_ties(name, table):
    def encode_one(v):
        return int(q.encode_scale(np.array([[[v]]]), table)[1][0, 0])
    for c in range(254):
        mid = (table[c] + table[c + 1]) / 2
        even = c if c % 2 == 0 else c + 1
        assert encode_one(mid) == even, f"{name}: tie at c={c} gave {encode_one(mid)}, expected {even}"
        assert encode_one(np.nextafter(mid, np.inf)) == c + 1, f"{name}: just above tie at c={c}"
        assert encode_one(np.nextafter(mid, 0.0)) == c, f"{name}: just below tie at c={c}"


# EXTRA A: shape and dtype. The histogram and the RTL test vectors will read these codes.
def test_extra_a_shape_dtype(name, fn, rule):
    Wbig = np.random.default_rng(1).normal(size=(64, 8))   # K=64, N=8 -> 8 rows of 4 blocks
    Wq, codes, _, _ = fn(Wbig, rule=rule)
    assert codes.shape == (8, 4), f"{name} {rule}: codes shape {codes.shape}"
    assert codes.dtype == np.uint8, f"{name} {rule}: codes dtype {codes.dtype}"
    assert Wq.shape == Wbig.shape, f"{name} {rule}: Wq shape {Wq.shape}"
    assert not np.isnan(Wq).any(), f"{name} {rule}: NaN in output of finite input"


# EXTRA B: block independence. One block quantized alone must equal the same block inside a big matrix.
# A wrong axis in the L1c error sum would mix blocks and break this.
def test_extra_b_independence(name, fn, rule):
    Wbig = np.random.default_rng(1).normal(size=(64, 8))
    alone = fn(Wbig[16:32, 3:4], rule=rule)[0]             # one block, shape (16, 1)
    inside = fn(Wbig, rule=rule)[0][16:32, 3:4]            # the same block cut from the full result
    assert np.array_equal(alone, inside), f"{name} {rule}: block result depends on other blocks"


EXPECTED_TABLE = {
    "UE6M2": {124: 1.0, 0: 2.0**-31, 254: 1.5 * 2.0**32},
    "UE5M3": {120: 1.0, 0: 2.0**-15, 254: 1.75 * 2.0**16},   # = 114688.0
}


def run_all():
    count = 0
    for name, table, fn, big in FORMATS:
        # tests that do not use the rule: once per format
        test_1_round_trip(name, table)
        test_2_table(name, table, EXPECTED_TABLE[name])
        test_7_ties(name, table)
        count += 3
        print(f"{name}: tests 1, 2, 7 pass")
        for rule in RULES:
            test_3_zero_block(name, table, fn, rule)
            for bad in (np.nan, np.inf, -np.inf):
                test_4_5_nonfinite_block(name, table, fn, rule, bad)
            test_6_above_top(name, table, fn, rule, big)
            test_6b_exactly_at_top(name, table, fn, rule)
            test_extra_a_shape_dtype(name, fn, rule)
            test_extra_b_independence(name, fn, rule)
            count += 1 + 3 + 1 + 1 + 1 + 1
            print(f"{name} {rule}: tests 3, 4, 5, 6, 6b, extra A, extra B pass")
    print(f"All {count} checks passed.")


if __name__ == "__main__":
    run_all()
