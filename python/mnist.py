"""
MNIST dataset loader and utility functions.
Read the four IDX files. Return X_train(60000,784), fp32 in [0,1], y_train(60000,), X_test(10000,784), y_test(10000,)
"""
import numpy as np
import gzip
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mnist")

def get_x_y_train(path):
    with gzip.open(os.path.join(path, "train-images-idx3-ubyte.gz"), 'rb') as f:
        magic, size, nrows, ncols = np.frombuffer(f.read(16), dtype='>i4')
        assert magic == 2051, f"bad image magic {magic}"
        shape = (size, nrows * ncols)
        X_train = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape).astype(np.float32) / np.float32(255.0)
    with gzip.open(os.path.join(path, "train-labels-idx1-ubyte.gz"), 'rb') as f:
        magic, count = np.frombuffer(f.read(8), dtype='>i4')
        assert magic == 2049, f"bad label magic {magic}"
        shape = (count,)
        y_train = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape).copy()
    return X_train, y_train

def get_x_y_test(path):
    with gzip.open(os.path.join(path, "t10k-images-idx3-ubyte.gz"), 'rb') as f:
        magic, size, nrows, ncols = np.frombuffer(f.read(16), dtype='>i4')
        assert magic == 2051, f"bad image magic {magic}"
        shape = (size, nrows * ncols)
        X_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape).astype(np.float32) / np.float32(255.0)
    with gzip.open(os.path.join(path, "t10k-labels-idx1-ubyte.gz"), 'rb') as f:
        magic, count = np.frombuffer(f.read(8), dtype='>i4')
        assert magic == 2049, f"bad label magic {magic}"
        shape = (count,)
        y_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape).copy()
    return X_test, y_test