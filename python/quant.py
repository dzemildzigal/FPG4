import numpy as np

def sqnr(W, Wq):
    return 10 * np.log10(np.sum(W**2) / np.sum((W - Wq)**2))

def quantize_fp4(W, block=16):
    # -> (Wq, scales, sqnr)
    pass