"""
MNIST-1D dataset: generator and loader. Numpy only.

MNIST-1D (Greydanus and Kobak, ICML 2024, https://github.com/greydanus/mnist1d, Apache-2.0).
10 classes. Each class has one hand-made template of 12 points. Every sample is a random
distortion of its template: pad, dilate, scale, shift, noise, shear, resample.

This file is a numpy rewrite of the generator in that repo (scipy removed, random numbers from
np.random.default_rng). It does NOT reproduce the random stream of the original, so samples differ
from the published file. The distribution is the same, except for the 2 settings below.
  - seq_len: 784 here (the original uses 40). 784 = 49 blocks of 16, the same K as MNIST.
    The signal before the last step has 72 points, so a length of 784 only stretches the same
    signal. Neighbor inputs correlate at 0.999. It is not a harder problem than length 40.
  - num_samples: 500,000 here (the original uses 5,000).

Output of generate(): 4 files in out_dir.
  x_train.npy (400000, 784) float32, y_train.npy (400000,) uint8,
  x_test.npy (100000, 784) float32, y_test.npy (100000,) uint8, meta.json.
X is standardized over the whole set: mean 0, std 1 (same as the original).

Run from python/:  uv run python mnist1d.py
"""
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mnist1d")

# Original settings of the generator (get_dataset_args in the repo).
TEMPLATE_LEN = 12
PADDING = (36, 60)         # zeros appended to the template: between 36 and 60
SCALE_COEFF = 0.4          # amplitude scaled by a factor in [0.8, 1.2]
MAX_TRANSLATION = 48       # circular shift of 0 to 47 points
CORR_NOISE_SCALE = 0.25    # smooth noise, only where the signal is zero (the padding)
IID_NOISE_SCALE = 0.02     # white noise everywhere
SHEAR_SCALE = 0.75         # linear ramp added to the signal
SIGMA = 2.0                # width of the smoothing filter for the correlated noise


def templates():
    # (10, 12). One row per digit class. Same numbers as the original.
    d = np.asarray([
        [5, 6, 6.5, 6.75, 7, 7, 7, 7, 6.75, 6.5, 6, 5],
        [5, 3, 3, 3.4, 3.8, 4.2, 4.6, 5, 5.4, 5.8, 5, 5],
        [5, 6, 6.5, 6.5, 6, 5.25, 4.75, 4, 3.5, 3.5, 4, 5],
        [5, 6, 6.5, 6.5, 6, 5, 5, 6, 6.5, 6.5, 6, 5],
        [5, 4.4, 3.8, 3.2, 2.6, 2.6, 5, 5, 5, 5, 5, 5],
        [5, 3, 3, 3, 3, 5, 6, 6.5, 6.5, 6, 4.5, 5],
        [5, 4, 3.5, 3.25, 3, 3, 3, 3, 3.25, 3.5, 4, 5],
        [5, 7, 7, 6.6, 6.2, 5.8, 5.4, 5, 4.6, 4.2, 5, 5],
        [5, 4, 3.5, 3.5, 4, 5, 5, 4, 3.5, 3.5, 4, 5],
        [5, 4, 3.5, 3.5, 4, 5, 5, 5, 5, 4.7, 4.3, 5],
    ], dtype=np.float64)
    d -= d.mean(axis=1, keepdims=True)    # whiten: mean 0 and std 1 per template
    d /= d.std(axis=1, keepdims=True)
    d -= d[:, :1]                         # the signal starts at 0 (and ends at 0)
    return d / 6.0


def resample(x, n):
    # Linear interpolation of x onto n points. Same result as scipy interp1d(kind="linear").
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


def smooth(noise):
    # Gaussian filter, sigma 2, cut at 4 sigma, mirrored edges. Same result as scipy gaussian_filter.
    r = int(4 * SIGMA + 0.5)                                   # kernel radius: 8
    kernel = np.exp(-0.5 * (np.arange(-r, r + 1) / SIGMA) ** 2)
    kernel /= kernel.sum()                                     # weights sum to 1, so a flat signal stays flat
    return np.convolve(np.pad(noise, r, mode="symmetric"), kernel, mode="valid")


def transform(x, rng, seq_len):
    # One template (12,) -> one sample (seq_len,). Every step is random, so no two samples are equal.
    # 1. pad: append 36 to 60 zeros. The 1e-8 marks the template, so step 5 can tell signal from padding.
    p = PADDING[0] + int(rng.random() * (PADDING[1] - PADDING[0] + 1))
    x = np.concatenate([x + 1e-8, np.zeros(p)])
    # 2. dilate: resample to a fixed 72 points (12 + 60). A long padding squeezes the template.
    x = resample(x, TEMPLATE_LEN + PADDING[1])
    # 3. scale: amplitude times a factor in [0.8, 1.2].
    x = x * (1 + SCALE_COEFF * (rng.random() - 0.5))
    # 4. translate: circular shift by 0 to 47 points.
    x = np.roll(x, rng.integers(MAX_TRANSLATION))
    # 5. noise: smooth noise replaces the zeros (the padding only), then white noise on everything.
    x = np.where(x != 0, x, smooth(CORR_NOISE_SCALE * rng.standard_normal(len(x))))
    x = x + IID_NOISE_SCALE * rng.standard_normal(len(x))
    # 6. shear: subtract a ramp with a random slope.
    x = x - SHEAR_SCALE * (rng.random() - 0.5) * np.linspace(-0.5, 0.5, len(x))
    # 7. resample 72 points to seq_len points.
    return resample(x, seq_len)


def generate(out_dir=DATA, num_samples=500_000, seq_len=784, train_frac=0.8, seed=42, chunk=50_000):
    # Chunked: only `chunk` rows live in float64 at once. The files are written through memmaps, so
    # peak memory is about chunk * seq_len * 8 bytes = 0.3 GB. One random generator runs through all
    # samples in order, so the output does not depend on `chunk`.
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    T = templates()

    # STEP 1: labels. num_samples // 10 per class, then shuffled so train and test hold all classes.
    y = np.repeat(np.arange(10, dtype=np.uint8), num_samples // 10)
    rng.shuffle(y)
    n = len(y)
    n_train = int(n * train_frac)

    x_train = np.lib.format.open_memmap(os.path.join(out_dir, "x_train.npy"), mode="w+", dtype=np.float32, shape=(n_train, seq_len))
    x_test = np.lib.format.open_memmap(os.path.join(out_dir, "x_test.npy"), mode="w+", dtype=np.float32, shape=(n - n_train, seq_len))

    # STEP 2: samples, chunk by chunk. Keep the sum and the sum of squares in float64 for STEP 3.
    total, total_sq = 0.0, 0.0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        buf = np.stack([transform(T[c], rng, seq_len) for c in y[s:e]])    # (e-s, seq_len) float64
        total += buf.sum()
        total_sq += (buf ** 2).sum()
        k = int(np.clip(n_train - s, 0, e - s))                           # rows of this chunk that belong to train
        x_train[s:s + k] = buf[:k]
        if k < e - s:
            x_test[s + k - n_train:e - n_train] = buf[k:]
        print(f"generated {e}/{n}", flush=True)

    # STEP 3: standardize over the whole set (train and test together), like the original.
    count = n * seq_len
    mean = total / count
    std = float(np.sqrt(total_sq / count - mean ** 2))
    for arr in (x_train, x_test):
        for s in range(0, len(arr), chunk):
            arr[s:s + chunk] = (arr[s:s + chunk] - mean) / std
        arr.flush()

    np.save(os.path.join(out_dir, "y_train.npy"), y[:n_train])
    np.save(os.path.join(out_dir, "y_test.npy"), y[n_train:])
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"num_samples": n, "seq_len": seq_len, "train_frac": train_frac, "seed": seed,
                   "raw_mean": mean, "raw_std": std}, f, indent=2)


# Same interface as mnist.py: path is the folder that holds the files.
def get_x_y_train(path):
    return np.load(os.path.join(path, "x_train.npy")), np.load(os.path.join(path, "y_train.npy"))


def get_x_y_test(path):
    return np.load(os.path.join(path, "x_test.npy")), np.load(os.path.join(path, "y_test.npy"))


if __name__ == "__main__":
    generate()
