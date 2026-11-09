import os
import sys
import tempfile
# This file lives in python/tests/. mnist1d.py lives in python/. Put python/ on the import path so "import mnist1d" works from any directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import mnist1d as m

# Run: from python/, "uv run python tests/test_mnist1d.py". It writes a small dataset to a temporary folder. It does not touch data/mnist1d.


# TEST 1: templates. 10 classes, 12 points, start at 0, all different.
def test_1_templates():
    T = m.templates()
    assert T.shape == (10, 12), f"templates shape {T.shape}"
    assert np.all(T[:, 0] == 0.0), "a template does not start at 0"
    for i in range(10):
        for j in range(i + 1, 10):
            assert not np.allclose(T[i], T[j]), f"templates {i} and {j} are equal"


# TEST 2: smooth(). A flat signal stays flat (the kernel sums to 1 and the edges are mirrored).
# White noise loses spread: the output std is sqrt(sum of squared kernel weights) = sqrt(1 / (2 * sqrt(pi) * sigma)) = 0.376 of the input std for sigma 2.
def test_2_smooth():
    flat = np.full(72, 0.7)
    assert np.allclose(m.smooth(flat), flat, atol=1e-15), "flat signal changed"
    noise = np.random.default_rng(0).standard_normal(10_000)
    out = m.smooth(noise)
    assert out.shape == noise.shape, f"smooth changed the length: {out.shape}"
    assert abs(out.std() / noise.std() - 0.376) < 0.02, f"std ratio {out.std() / noise.std()}, expected 0.376"


# TEST 3: resample(). Straight line in, straight line out. The end points stay.
def test_3_resample():
    x = np.linspace(2.0, 5.0, 72)
    y = m.resample(x, 784)
    assert y.shape == (784,)
    assert y[0] == 2.0 and y[-1] == 5.0, "end points moved"
    assert np.allclose(y, np.linspace(2.0, 5.0, 784), atol=1e-12), "a line is not a line after resampling"


# TEST 4: transform(). Right shape, finite, repeatable with the same generator state, different with a different one.
def test_4_transform():
    T = m.templates()
    a = m.transform(T[3], np.random.default_rng(1), 784)
    b = m.transform(T[3], np.random.default_rng(1), 784)
    c = m.transform(T[3], np.random.default_rng(2), 784)
    assert a.shape == (784,) and a.dtype == np.float64, f"{a.shape} {a.dtype}"
    assert np.all(np.isfinite(a)), "non-finite sample"
    assert np.array_equal(a, b), "same seed gave different samples"
    assert not np.array_equal(a, c), "different seeds gave the same sample"
    assert m.transform(T[3], np.random.default_rng(1), 40).shape == (40,), "seq_len argument ignored"


# TEST 5: generate() and the loader, on a small set (200 samples).
# Shapes, dtypes, 20 samples per class, standardized, and the same output for any chunk size.
def test_5_generate_and_load():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2, tempfile.TemporaryDirectory() as d3:
        m.generate(d1, num_samples=200, seq_len=784, seed=7, chunk=64)
        m.generate(d2, num_samples=200, seq_len=784, seed=7, chunk=200)   # one chunk: the straddle code never runs
        m.generate(d3, num_samples=200, seq_len=784, seed=8, chunk=64)
        Xtr, ytr = m.get_x_y_train(d1)
        Xte, yte = m.get_x_y_test(d1)
        assert Xtr.shape == (160, 784) and Xte.shape == (40, 784), f"{Xtr.shape} {Xte.shape}"
        assert Xtr.dtype == np.float32 and Xte.dtype == np.float32
        assert ytr.dtype == np.uint8 and yte.dtype == np.uint8 and ytr.shape == (160,) and yte.shape == (40,)
        assert Xtr.shape[1] % 16 == 0, "K is not a multiple of the block size 16"
        assert np.array_equal(np.bincount(np.concatenate([ytr, yte])), np.full(10, 20)), "classes are not balanced"
        both = np.concatenate([Xtr, Xte]).astype(np.float64)
        assert abs(both.mean()) < 1e-4 and abs(both.std() - 1.0) < 1e-4, f"mean {both.mean()} std {both.std()}"
        Xtr2, ytr2 = m.get_x_y_train(d2)
        assert np.array_equal(Xtr, Xtr2) and np.array_equal(ytr, ytr2), "chunk size changed the output"
        assert np.array_equal(m.get_x_y_test(d2)[0], Xte), "chunk size changed the test set"
        Xtr3, _ = m.get_x_y_train(d3)
        assert not np.array_equal(Xtr, Xtr3), "different seeds gave the same data"


# TEST 6: labels match samples. A nearest-class-mean classifier is weak here (the shift of up to 48 of 72 points blurs the class means),
# measured 20.5% on 2000 samples. With scrambled labels it sits at 10% +- 1% (1000 test samples). The limit 15% is 5 standard errors above chance.
def test_6_labels_match_samples():
    with tempfile.TemporaryDirectory() as d:
        m.generate(d, num_samples=5000, seq_len=784, seed=3, chunk=1000)
        X, y = m.get_x_y_train(d)
        Xt, yt = m.get_x_y_test(d)
        means = np.stack([X[y == c].mean(axis=0) for c in range(10)])           # (10, 784)
        dist = ((Xt[:, None, :] - means[None]) ** 2).sum(axis=2)                # (N, 10, 784) summed over 784 -> (N, 10)
        acc = np.mean(np.argmin(dist, axis=1) == yt)
        assert acc > 0.15, f"nearest class mean accuracy {acc}: labels do not match samples"


def run_all():
    tests = [test_1_templates, test_2_smooth, test_3_resample, test_4_transform, test_5_generate_and_load, test_6_labels_match_samples]
    for t in tests:
        t()
        print(f"{t.__name__} pass")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
