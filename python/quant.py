import numpy as np

def sqnr(W, Wq):
    err = np.sum((W - Wq)**2)
    if err == 0:
        return np.inf  # identical tensors. The limit is infinite, and 0/0 is not an error here.
    return 10 * np.log10(np.sum(W**2) / err)

# The 8 magnitudes that the 4-bit payload can store. The payload is a sign bit plus the index into this list.
PAYLOAD_CODES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# The same 8 magnitudes as integers, in units of 0.5: 0,1,2,3,4,6,8,12. A product of two of them is an integer.
# The largest product is 12 * 12 = 144, which needs 8 bits. Nothing rounds in that product, so there is no FP4 multiplier.
PAYLOAD_INTS = (2 * PAYLOAD_CODES).astype(np.int32)

def payload_index(Wb, scales):
    # The payload code of every element: the index (0..7) of the closest PAYLOAD_CODES value.
    # Wb is (rows, blocks, 16), scales is (rows, blocks, 1), the result is (rows, blocks, 16) int64.
    r = np.abs(Wb) / scales #scale the absolute value of Wb by it's already calculated block scaling factors to get it in range of CODES.
    return np.argmin(np.abs(r[..., None] - PAYLOAD_CODES), axis=-1) #find the index of the code in CODES closest to the scaled absolute value of Wb, this is where the quantization rounds to nearest.

def decode_payload(sign_source, idx, scales):
    # Turn a payload code back into a number: sign * magnitude * scale.
    return np.sign(sign_source) * PAYLOAD_CODES[idx] * scales # (N, B, block) * (N, B, block) * (N, B, 1)

def quantize_payload(Wb, scales):
    # Round to nearest payload code. Returns the decoded values, not the codes. The codes are in payload_index().
    idx = payload_index(Wb, scales)
    return decode_payload(Wb, idx, scales)


def scale_table_builder(E, M, bias):
    #outputs a 256-entry float64 array. 1 for each possible 8-bit code representing the scale factor
    #from this we can build E6M2 or E5M3 by just changing E, M and bias in the call.
    #code range 0 to 254 -> (1 + m/2^M, m) * 2^(e-bias) | e = code >> M, m = code & (2^M - 1)
    #code 255 -> NaN
    # 2^M via bitshifts is 1 << M
    table = np.zeros(256, dtype=np.float64)
    for code in range(255):
        e = code >> M
        m = code & ((1 << M) - 1)
        table[code] = (1 + m / (1 << M)) * (2.0**(e - bias))
    table[255] = np.nan
    return table

e6m2 = scale_table_builder(6, 2, 31)
e5m3 = scale_table_builder(5, 3, 15)

def encode_scale(x, table):
    # GOAL: turn the real number x = amax/6 (or amax/4) into the 8-bit code of the closest scale that the format can store.
    # The format can store only 255 scales: table[0] ... table[254]. table[255] is the NaN code.
    # So this is "round to nearest" on a list of 255 allowed values, the same idea as quantize_payload but with 255 values instead of 8.
    #
    # x has shape (N, B, 1), table has shape (256,).
    # returns: scales (N, B, 1) float64, codes (N, B) uint8
    v = x[..., 0]                                          # (N, B). One number per block.

    # STEP 1: distance from every block's v to every allowed scale.
    # v[..., None] has shape (N, B, 1), table[:255] has shape (255,). NumPy broadcasts to (N, B, 255).
    # table[:255] leaves out code 255 on purpose: NaN is not a scale, so it must never win the search.
    dist = np.abs(v[..., None] - table[:255])

    # STEP 2: the closest scale is the one with the smallest distance. argmin gives its index. The index IS the code (table[code] is the scale).
    # Two cases need no extra code because argmin already does the right thing:
    #   v below table[0] (or exactly 0, a zero block): table[0] is the closest, code 0.
    #   v above table[254]: table[254] is the closest, code 254. This is saturation: the scale is too small for the block. scale_counters counts it.
    idx = np.argmin(dist, axis=-1)                         # (N, B)

    # STEP 3: ties. A tie means v is exactly in the middle of two neighbors: same distance to both.
    # argmin returns the FIRST minimum, so on a tie idx is the LOWER code c and the other tied code is c + 1.
    # Rule (spec): on a tie take the EVEN code (lowest bit 0).
    # Why even: "always lower" would push every tie in the same direction and the error would drift. "Even" sends half the ties up and half down.
    # Python and the RTL encoder must make the same choice, or their codes differ on tie inputs.
    # c and c+1 differ by 1, so exactly one of them is even:
    #   c even -> keep c (argmin already chose it).
    #   c odd  -> take c + 1.
    upper = np.clip(idx + 1, 0, 254)                       # the neighbor above. At 254 there is none, clip keeps it at 254.
    is_tie = (np.abs(v - table[upper]) == np.abs(v - table[idx])) & (idx < 254)
    idx = np.where(is_tie & (idx % 2 == 1), idx + 1, idx)

    # STEP 4: NaN and infinity. They have no closest scale. The spec says code 255, which decodes to NaN, so the whole block becomes NaN.
    # np.isfinite is False for NaN, +inf and -inf. This overrides whatever the search returned for those blocks.
    idx = np.where(np.isfinite(v), idx, 255)

    # STEP 5: look up the scale values. table[255] is NaN, so a block with code 255 gets scale NaN and every element of it becomes NaN.
    return table[idx][..., None], idx.astype(np.uint8)

def clamp_to_e6m2(x):
    return encode_scale(x, e6m2)

def clamp_to_e5m3(x):
    return encode_scale(x, e5m3)


def scale_counters(amax, table):
    # Counts on amax BEFORE the encoder, so they do not depend on the encoder code.
    # x = amax / 6 is the value the encoder gets under the plain amax/6 rule.
    x = amax / 6.0
    finite = np.isfinite(amax)
    return {
        "scale_saturated": int(np.sum(finite & (x > table[254]))),            # finite, above the top code
        "scale_underflow": int(np.sum(finite & (amax > 0) & (x < table[0]))), # nonzero, below the lowest code
        "zero_blocks": int(np.sum(amax == 0)),
        "nonfinite_blocks": int(np.sum(~finite)),                              # NaN or infinity -> code 255
    }

def four6(Wb, amax6, amax4, scale_code6, scale_code4, rule):
    # We make a decision for each block that minimizes the quantization error for the block
    # for Wb_hat6, quantize the block with the amax6 scale
    # for Wb_hat4, quantize the block with the amax4 scale
    # then compare sum(|Wb-Wb_hat|)for the two, keep the lower, tie goes to amax6
    #
    # Returns 4 values, because the datapath experiment (L1e) needs the payload codes and the scale that won:
    #   Wb_hat      (rows, blocks, 16)  the decoded values
    #   scale_codes (rows, blocks)      the 8-bit code of the scale that won
    #   idx         (rows, blocks, 16)  the payload code index (0..7) of every element
    #   scales      (rows, blocks, 1)   the scale value that won
    if rule not in ["L1b", "L1c"]:
        raise ValueError("Invalid rule specified for four6 quantization")
    idx6 = payload_index(Wb, amax6)
    Wb_hat6 = decode_payload(Wb, idx6, amax6)
    if rule == "L1b":
        # For rule "L1b", we return amax6-quantized Wb directly
        return Wb_hat6, scale_code6, idx6, amax6
    if rule == "L1c":
        # Wb_hat4, quantize Wb with the amax4 scale
        idx4 = payload_index(Wb, amax4)
        Wb_hat4 = decode_payload(Wb, idx4, amax4)
        # compute the quantization error for each block
        err6 = Wb - Wb_hat6
        err4 = Wb - Wb_hat4

        # Wb_hat6 and Wb_hat4 are used to compute the quantization error for each block
        # instead of returning the mask which prompts us to do another quantization, immediately return the quantized result here instead of the mask.
        mask = np.sum(np.abs(err6), axis=2, keepdims=True) <= np.sum(np.abs(err4), axis=2, keepdims=True)
        #we need to return Wb_hat, codes, sqnr
        Wb_hat = np.where(mask, Wb_hat6, Wb_hat4)
        scale_codes = np.where(mask[..., 0], scale_code6, scale_code4) # mask is (N, B, 1), the codes are (N, B)
        idx = np.where(mask, idx6, idx4)
        scales = np.where(mask, amax6, amax4)
        return Wb_hat, scale_codes, idx, scales


def quantize_blocks(B, table, rule="L1b"):
    # The one place where a block tensor becomes FP4. Weights and activations both come through here.
    # B is (rows, blocks, 16). The caller arranges the blocks, because a weight blocks along K
    # and an activation already has K in its last axis.
    # Returns (B_hat, scale_codes, idx, scales, counters):
    #   B_hat       (rows, blocks, 16)  decoded values, float64
    #   scale_codes (rows, blocks)      the 8-bit scale code of each block
    #   idx         (rows, blocks, 16)  the payload code index (0..7) of each element
    #   scales      (rows, blocks, 1)   the scale value of each block
    #   counters    dict                the 4 counters of scale_counters
    amax = np.max(np.abs(B), axis=2, keepdims=True) # the largest magnitude in each block, (rows, blocks, 1)
    scale6, scale_code6 = encode_scale(amax / 6.0, table) # block scale that puts amax at the top payload code 6
    scale4, scale_code4 = encode_scale(amax / 4.0, table) # block scale that puts amax at payload code 4
    counters = scale_counters(amax, table)
    B_hat, scale_codes, idx, scales = four6(B, scale6, scale4, scale_code6, scale_code4, rule)
    return B_hat, scale_codes, idx, scales, counters


def quantize_fp4_b_fp64(W, block=16):
    # -> (Wq, None, sqnr, None)  no 8-bit scale, so no codes and no counters

    Wt = W.T #(N, K)
    Wb = Wt.reshape(Wt.shape[0], Wt.shape[1] // block, block) #format it into blocks
    amax = np.max(np.abs(Wb), axis=2, keepdims=True) #/6 because magnitude of FP4 is max 6
    scales = np.where(amax > 0, amax / 6.0, 1.0)
    Wb_hat = quantize_payload(Wb, scales)
    sqnr_wb = sqnr(Wb, Wb_hat)

    Wq = Wb_hat.reshape(Wt.shape).T
    return Wq, None, sqnr_wb, None

def quantize_blocks_weights(W, table, rule="L1b", block=16):
    # Weights. W is (K, N): K is the accumulation axis, N is the number of outputs of the layer.
    # A block runs along K, so the matrix is transposed first: Wt is (N, K) and then K is cut into blocks.
    # This is the only place where the transpose happens. quantize_blocks then sees the same
    # (rows, blocks, 16) layout that an activation gives, so both operands share one core.
    # Returns (Wq, scale_codes, idx, scales, counters), with idx and scales in the (N, blocks, ...) layout.
    K, N = W.shape
    if K % block != 0:
        raise ValueError(f"K must be a multiple of {block}, got {K}")
    Wt = W.T
    Wb = Wt.reshape(N, K // block, block)
    Wb_hat, scale_codes, idx, scales, counters = quantize_blocks(Wb, table, rule)
    return Wb_hat.reshape(N, K).T, scale_codes, idx, scales, counters

def quantize_fp4_b_e6m2(W, block=16, rule="L1b"):
    # -> (Wq, codes, sqnr, counters)
    # difference to fp64 is now our block scaling factor is also fixed point 8-bit E6M2

    Wt = W.T #(N, K)
    Wb = Wt.reshape(Wt.shape[0], Wt.shape[1] // block, block) #format it into blocks. Now it's a 3D matrix, same rows, but columns are now (old_cols//block_size, block_size)
    #for E6M2, the scaling factor needs to be clamped to the closest to result of four6 value in E6M2 from amax
    Wb_hat, scale_codes, _, _, counters = quantize_blocks(Wb, e6m2, rule)
    Wq = Wb_hat.reshape(Wt.shape).T
    return Wq, scale_codes, sqnr(W, Wq), counters

def quantize_fp4_b_e5m3(W, block=16, rule="L1b"):
    # -> (Wq, codes, sqnr, counters)
    # difference to fp64 is now our block scaling factor is also fixed point 8-bit E5M3

    Wt = W.T #(N, K)
    Wb = Wt.reshape(Wt.shape[0], Wt.shape[1] // block, block) #format it into blocks
    Wb_hat, scale_codes, _, _, counters = quantize_blocks(Wb, e5m3, rule)
    Wq = Wb_hat.reshape(Wt.shape).T
    return Wq, scale_codes, sqnr(W, Wq), counters

def quantize_blocks_act(A, table, rule="L1b", block=16):
    # Activations. A is (N, K): one row per sample, K along the accumulation axis.
    # The block axis is K, the same axis the weights are blocked along, so the two operands of one MAC tile
    # share a 16-long slice and each has one scale. No transpose: unlike a weight (K, N), the activation
    # already has K in its last axis.
    # Returns (Aq, scale_codes, idx, scales, counters), the same 5 values as quantize_blocks.
    n, K = A.shape
    if K % block != 0:
        raise ValueError(f"K must be a multiple of {block}, got {K}")
    Ab = A.reshape(n, K // block, block)
    Ab_hat, scale_codes, idx, scales, counters = quantize_blocks(Ab, table, rule)
    return Ab_hat.reshape(n, K), scale_codes, idx, scales, counters

def quantize_fp4_b_act(A, table, rule="L1b", block=16):
    # Same interface as the weight functions: (Aq, codes, sqnr, counters).
    Aq, scale_codes, _, _, counters = quantize_blocks_act(A, table, rule, block)
    return Aq, scale_codes, sqnr(A, Aq), counters

def act_quantizer(table, rule="L1b"):
    # Closures for model.forward(p, X, quant=...). The hook gets one tensor and returns (quantized tensor, counters).
    def quant(A):
        Aq, _, _, counters = quantize_fp4_b_act(A, table, rule)
        return Aq, counters
    return quant

def test_quantize_fp4_b_fp64():
    from mnist import get_x_y_test
    from train import load_weights
    import os
    X_test, y_test = get_x_y_test(os.path.join(os.path.dirname(__file__), '..','data/mnist'))
    W = load_weights(42)
    W1 = W["W1"]
    b1 = W["b1"]
    W2 = W["W2"]
    b2 = W["b2"]
    W3 = W["W3"]
    b3 = W["b3"]
    Wq1, codes1, snr1, _ = quantize_fp4_b_fp64(W1)
    print("Quantized weights:", Wq1)
    print("Codes:", codes1)
    print("SQNR:", snr1)
    Wq2, codes2, snr2, _ = quantize_fp4_b_fp64(W2)
    print("Quantized weights:", Wq2)
    print("Codes:", codes2)
    print("SQNR:", snr2)
    Wq3, codes3, snr3, _ = quantize_fp4_b_fp64(W3)
    print("Quantized weights:", Wq3)
    print("Codes:", codes3)
    print("SQNR:", snr3)
    #we don't quantize the biases, for now we only look at the weights ideal quantization
    #now we do inference and run a test to see the performance of the test set on this quantized weights.
    from model import forward, loss, accuracy
    P_test, _ = forward({"W1": Wq1, "b1": b1, "W2": Wq2, "b2": b2, "W3": Wq3, "b3": b3}, X_test)
    test_loss = loss(P_test, y_test)
    test_acc = accuracy(P_test, y_test)
    print("Test loss on quantized weights:", test_loss)
    print("Test accuracy on quantized weights:", test_acc)


def test_quantize_fp4_b_e6m2(rule):
    from mnist import get_x_y_test
    from train import load_weights
    import os
    if rule not in ["L1b", "L1c"]:
        raise ValueError("Invalid rule specified for quantization")

    
    X_test, y_test = get_x_y_test(os.path.join(os.path.dirname(__file__), '..','data/mnist'))
    W = load_weights(42)
    W1 = W["W1"]
    b1 = W["b1"]
    W2 = W["W2"]
    b2 = W["b2"]
    W3 = W["W3"]
    b3 = W["b3"]
    Wq1, codes1, snr1, _ = quantize_fp4_b_e6m2(W1, rule=rule)
    print("Quantized weights:", Wq1)
    print("Codes:", codes1)
    print("SQNR:", snr1)
    Wq2, codes2, snr2, _ = quantize_fp4_b_e6m2(W2, rule=rule)
    print("Quantized weights:", Wq2)
    print("Codes:", codes2)
    print("SQNR:", snr2)
    Wq3, codes3, snr3, _ = quantize_fp4_b_e6m2(W3, rule=rule)
    print("Quantized weights:", Wq3)
    print("Codes:", codes3)
    print("SQNR:", snr3)
    from model import forward, loss, accuracy
    P_test, _ = forward({"W1": Wq1, "b1": b1, "W2": Wq2, "b2": b2, "W3": Wq3, "b3": b3}, X_test)
    test_loss = loss(P_test, y_test)
    test_acc = accuracy(P_test, y_test)
    print("Test loss on quantized weights:", test_loss)
    print("Test accuracy on quantized weights:", test_acc)

def test_quantize_fp4_b_e5m3(rule):
    print("Testing quantization with rule:", rule)
    from mnist import get_x_y_test
    from train import load_weights
    import os

    if rule not in ["L1b", "L1c"]:
        raise ValueError("Invalid rule specified for quantization")

    X_test, y_test = get_x_y_test(os.path.join(os.path.dirname(__file__), '..','data/mnist'))
    W = load_weights(42)
    W1 = W["W1"]
    b1 = W["b1"]
    W2 = W["W2"]
    b2 = W["b2"]
    W3 = W["W3"]
    b3 = W["b3"]
    Wq1, codes1, snr1, _ = quantize_fp4_b_e5m3(W1, rule=rule)
    print("Quantized weights:", Wq1)
    print("Codes:", codes1)
    print("SQNR:", snr1)
    Wq2, codes2, snr2, _ = quantize_fp4_b_e5m3(W2, rule=rule)
    print("Quantized weights:", Wq2)
    print("Codes:", codes2)
    print("SQNR:", snr2)
    Wq3, codes3, snr3, _ = quantize_fp4_b_e5m3(W3, rule=rule)
    print("Quantized weights:", Wq3)
    print("Codes:", codes3)
    print("SQNR:", snr3)
    from model import forward, loss, accuracy
    P_test, _ = forward({"W1": Wq1, "b1": b1, "W2": Wq2, "b2": b2, "W3": Wq3, "b3": b3}, X_test)
    test_loss = loss(P_test, y_test)
    test_acc = accuracy(P_test, y_test)
    print("Test loss on quantized weights:", test_loss)
    print("Test accuracy on quantized weights:", test_acc)

if __name__ == "__main__":
    test_quantize_fp4_b_fp64()
    test_quantize_fp4_b_e6m2("L1b")
    test_quantize_fp4_b_e6m2("L1c")
    test_quantize_fp4_b_e5m3("L1b")
    test_quantize_fp4_b_e5m3("L1c")
    print("All tests completed successfully.")
    