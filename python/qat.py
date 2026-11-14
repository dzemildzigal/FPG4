"""Rung L2: quantization-aware training, weights only.

Scope, fixed by docs/experiment_L2.md:

    forward pass   the three weight matrices are quantized to FP4 with an 8-bit scale,
                   decoded back to float, and used for the matrix multiplies
    backward pass  full float, straight-through estimator
    biases         never quantized
    activations    never quantized. That is rung L2a

What a run does, in order, every step:

    1. quantize the master weights  W  ->  Wq        (Wq is on the FP4 grid)
    2. forward with Wq                               (the cache holds the float activations)
    3. backward with Wq, at the master W             (STE: d Wq / d W := 1)
    4. Adam updates the master W, which is never on the grid

Step 3 is the whole idea and it needs no new code. model.backward(p, y, cache) reads
p["W2"] and p["W3"] to propagate the deltas, and it never reads a derivative of a
quantizer. So passing Wq to backward gives:

    dL/dA1 = dZ2 @ W2q.T        the correct derivative of A1 @ Q(W2) with respect to A1
    dW2    = A1.T @ dZ2         the STE gradient, which lands on the master W

That is the standard weight-quantization STE. It is why step 1 writes into a copy and
step 4 writes into the master: the two dictionaries must be different objects.
"""
import numpy as np
import quant as q
from model import init_params, adam, forward, backward, evaluate
from train import DTYPE

# The three matrices that are quantized. The biases are left out on purpose.
WEIGHTS = ("W1", "W2", "W3")

# The four configurations of the run matrix. The rule names are the ones in quant.py:
#   "L1b" = amax/6, one scale per block, the scale puts the block maximum at payload code 6
#   "L1c" = four6, two candidate scales, the one with the smaller absolute error wins
CONFIGS = {
    "a6-e6m2": (q.e6m2, "L1b"),
    "f6-e6m2": (q.e6m2, "L1c"),
    "a6-e5m3": (q.e5m3, "L1b"),
    "f6-e5m3": (q.e5m3, "L1c"),
}

COUNTERS = ("scale_saturated", "scale_underflow", "zero_blocks", "nonfinite_blocks")

# The three tensors that model.forward sends through the quantiser, in the order it sends them.
# The hook cannot see names, only calls, so the call count is the label.
ACT_SITES = ("input X", "A1", "A2")


class ActQuant:
    """The activation hook for L2a. One tensor in, one quantized tensor out, and a record.

    model.forward calls this three times per step, always in the same order: the input X, then
    A1, then A2. The order is the only name a hook gets, so it is the label of the record.

    Why the record is per tensor: the three tensors are not equally damaged by FP4. The input of
    MNIST-1D is a smooth continuous signal with a small range, A1 and A2 are sparse after ReLU.
    A single pooled number over all three hides which one carries the loss.
    """

    def __init__(self, table, rule, pre_quantized_input=False):
        self.table = table
        self.rule = rule
        # pre_quantized_input: the caller quantized the input tensor itself, once, before training.
        # A block of 16 values lives inside one row, so this is identical to quantizing each batch.
        # When True the input call returns the tensor unchanged and its SQNR comes from the caller.
        self.pre_quantized_input = pre_quantized_input
        self.fixed = {}
        self.reset()

    def reset(self):
        self.calls = 0
        self.sqnr_sum = [0.0] * len(ACT_SITES)
        self.n = [0] * len(ACT_SITES)
        self.counters = {k: 0 for k in COUNTERS}

    def __call__(self, A):
        i = self.calls % len(ACT_SITES)
        self.calls += 1
        if self.pre_quantized_input and i == 0:
            return A, {}
        Aq, _, _, _, counters = q.quantize_blocks_act(A, self.table, self.rule)
        self.sqnr_sum[i] += q.sqnr(A, Aq)
        self.n[i] += 1
        for k in COUNTERS:
            self.counters[k] += counters[k]
        return Aq, counters

    def report(self):
        # Mean SQNR per tensor over the epoch's steps. A tensor that was never called stays 0.0
        # and its count stays 0, which is visible in the output and not silently averaged.
        # A site in self.fixed was measured outside the loop (the pre-quantized input) and that
        # number is constant for the whole run, so it is reported as it is.
        out = {}
        for i, name in enumerate(ACT_SITES):
            if i in self.fixed:
                out[name] = self.fixed[i]
            else:
                out[name] = self.sqnr_sum[i] / self.n[i] if self.n[i] else float("nan")
        return out


def quantize_master(p, table, rule):
    """Return (pq, sqnrs, counters).

    pq is a new dict. The three weight matrices are replaced by Wq, the FP4 values decoded
    back to the master's float width. The biases are the same objects as in p, which is
    correct, because the bias is never quantized (docs/experiment_L1.md, item 5).

    sqnrs is per matrix, in dB, measured against the master. It is the same quantity that
    rung L1 reports, so the L2 number can be put next to the L1 number.
    counters sums the four scale counters over the three matrices. They are expected to
    be 0 on weights: an 8-bit scale covers 21 to 39 decades and a weight matrix uses a
    few octaves. They are logged anyway, because a zero is evidence too.
    """
    pq = {}
    sqnrs = {}
    counters = {k: 0 for k in COUNTERS}
    for k, v in p.items():
        if k in WEIGHTS:
            wq, _, _, _, c = q.quantize_blocks_weights(v, table, rule)
            pq[k] = wq
            sqnrs[k] = q.sqnr(v, wq)
            for name in COUNTERS:
                counters[name] += c[name]
        else:
            pq[k] = v
    return pq, sqnrs, counters


def train_qat(seed, epochs, batch_size, X, y, X_test, y_test, config,
              acts=False, X_pre_quantized=False, input_sqnr=None, dtype=DTYPE, verbose=True):
    """One L2 run. Returns {"params", "history", "config", "seed"}.

    The history has one entry per epoch, plus two per-step records that L1 could not
    produce, because L1 never had a moving weight:

        sqnr[epoch][matrix]            dB of Wq against the master at the end of the epoch
        counters[epoch]                the four scale counters, summed over the epoch
        frozen_steps[epoch][matrix]    steps in which Wq was bit-identical to the step
                                       before, so the quantizer could not follow the master
        stale_elements[epoch][matrix]  the mean over the epoch's steps of the fraction of
                                       elements whose FP4 value did not move. 0 means the
                                       quantizer followed every element every step

    Why those two matter: they are the smallest measurement of the training problem.
    If the master moves and Wq does not, the optimizer is spending steps on values that
    the fabric cannot see. That is the input to L3 and L7, and it is cheap to record.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}, expected one of {list(CONFIGS)}")
    table, rule = CONFIGS[config]

    rng = np.random.default_rng(seed)
    p = init_params(seed, dtype=dtype)
    hook = ActQuant(table, rule, pre_quantized_input=X_pre_quantized) if acts else None
    if hook is not None and input_sqnr is not None:
        hook.fixed[0] = float(input_sqnr)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    t = 1

    history = {
        "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
        "sqnr": [], "counters": [], "frozen_steps": [], "stale_elements": [],
        "act_sqnr": [], "act_counters": [],
    }
    prev_wq = None

    for epoch in range(epochs):
        frozen = {k: 0 for k in WEIGHTS}
        stale = {k: 0 for k in WEIGHTS}
        ep_counters = {k: 0 for k in COUNTERS}
        steps = 0
        if hook is not None:
            hook.reset()

        indices = np.arange(X.shape[0])
        rng.shuffle(indices)
        for start_idx in range(0, X.shape[0], batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            pq, _, counters = quantize_master(p, table, rule)
            for name in COUNTERS:
                ep_counters[name] += counters[name]

            if prev_wq is not None:
                for k in WEIGHTS:
                    same = pq[k] == prev_wq[k]
                    if same.all():
                        frozen[k] += 1
                    stale[k] += int(np.count_nonzero(same))
            prev_wq = {k: pq[k].copy() for k in WEIGHTS}

            P, cache = forward(pq, X_batch, quant=hook)
            grads = backward(pq, y_batch, cache)
            # The gradient is at the master, so Adam writes the master. p is mutated in place.
            p, m, v = adam(p, grads, m, v, t)
            t += 1
            steps += 1

        pq, sqnrs, _ = quantize_master(p, table, rule)
        # prev_wq is NOT refreshed here on purpose. pq at this point is the same quantized master as the
        # first comparison of the next epoch would be, so refreshing it would add one self-comparison per
        # epoch: a step counted as frozen that compared a tensor with itself. Left alone, the first step of
        # epoch n+1 compares Wq after the last Adam step of epoch n against Wq before it, which is a real
        # one-step comparison and belongs in the count.

        # The reported number is the DEPLOYED model, so it is pq and not the master.
        # pq is what the fabric holds. The master is an artefact of the training loop that the fabric
        # never sees, so its accuracy is not the accuracy of this rung.
        # With acts=True the evaluation also has to quantize the activations, and it uses its own hook
        # so that the per-tensor SQNR of the training steps stays a training measurement.
        eval_hook = ActQuant(table, rule, pre_quantized_input=X_pre_quantized) if acts else None
        train_loss, train_acc = evaluate(pq, X, y, quant=eval_hook)
        eval_hook = ActQuant(table, rule, pre_quantized_input=X_pre_quantized) if acts else None
        test_loss, test_acc = evaluate(pq, X_test, y_test, quant=eval_hook)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["sqnr"].append(sqnrs)
        history["counters"].append(ep_counters)
        history["frozen_steps"].append(frozen)
        history["stale_elements"].append({k: stale[k] / (steps * p[k].size) for k in WEIGHTS})
        if hook is not None:
            history["act_sqnr"].append(hook.report())
            history["act_counters"].append(dict(hook.counters))

        if verbose:
            print(f"{epoch:2d} loss {train_loss:.4f} acc {100 * train_acc:6.2f} "
                  f"test {test_loss:.4f} {100 * test_acc:6.2f} "
                  f"sqnr {sqnrs['W1']:5.2f} {sqnrs['W2']:5.2f} {sqnrs['W3']:5.2f} "
                  f"frozen {frozen['W1']:4d} {frozen['W2']:4d} {frozen['W3']:4d}")

    return {"params": p, "history": history, "config": config, "seed": seed}


def self_check(seed=42, config="a6-e5m3"):
    """Three checks that must hold before any L2 number means anything.

    1. The quantizer is idempotent on its own output. Take the master, quantize it to Wq,
       then put Wq back into the master and quantize again. The second result must be
       bit-identical to the first. If it is not, Wq drifts against the master every step
       even when the master stands still, and every L2 number is noise.

    2. The gradient that backward() returns at Wq is the gradient of the float loss at Wq.
       A finite difference on the float forward pass, with no quantizer in it, must agree
       with the analytic value. Check 1 makes L2 restartable: an L2 run that begins from
       an on-grid master takes the same step as L0 would, because Q(W) == W there.
       This check is what makes the step correct, not just identical.

    3. The STE gradient is not the gradient of the quantized loss, and it must not be.
       The quantized loss is a staircase, so its own gradient is 0 almost everywhere and
       nothing would train. This check records the size of the staircase so the
       distinction is on the record and not just in a comment.

    Returns a dict of the three results, all of them booleans except the third.
    """
    table, rule = CONFIGS[config]
    p = init_params(seed, dtype=np.float64)
    pq, _, _ = quantize_master(p, table, rule)

    # 1. idempotence
    with np.errstate(divide="ignore"):
        pq2, _, _ = quantize_master(pq, table, rule)
    idempotent = all(np.array_equal(pq[k], pq2[k]) for k in WEIGHTS)
    worst = max(float(np.max(np.abs(pq[k] - pq2[k]))) for k in WEIGHTS)

    # 2. the analytic gradient at Wq against a finite difference on the float loss
    from model import loss
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 784))
    y = rng.integers(0, 10, size=64)
    _, cache_q = forward(pq, X)
    grads_q = backward(pq, y, cache_q)
    h = 1e-6
    worst_grad = 0.0
    for key in ("W1", "W2", "W3"):
        # Why flatiter and not ravel(): quantize_blocks_weights returns Wb_hat.reshape(N, K).T, which is a
        # transposed view and therefore NOT contiguous. ravel() would return a copy and every perturbation
        # below would be thrown away. arr.flat[i] writes through to the real array in either case.
        arr = pq[key]
        g = grads_q[key].flat
        for i in rng.choice(arr.size, 3, replace=False):
            old = arr.flat[i]
            arr.flat[i] = old + h
            l_plus = loss(forward(pq, X)[0], y)
            arr.flat[i] = old - h
            l_minus = loss(forward(pq, X)[0], y)
            arr.flat[i] = old
            num = (l_plus - l_minus) / (2 * h)
            den = max(abs(num), abs(g[i]), 1e-12)
            worst_grad = max(worst_grad, abs(num - g[i]) / den)

    # 3. L2a: the forward pass must hand the backward pass the QUANTIZED activations.
    # This is the one thing that can go silently wrong in L2a. model.forward quantizes X, A1 and A2
    # and then stores them in the cache, and backward reads the cache. If forward stored the
    # unquantized values while the multiplies used the quantized ones, the training would still run,
    # still converge, and measure a model that does not exist. So: every cached activation tensor
    # must be a fixed point of its own quantizer. Q(A) == A holds only when A is already on the grid.
    # There is no finite-difference check here on purpose. The STE gradient is the gradient of a
    # surrogate, so it must NOT agree with a finite difference of the quantized loss. Section 4
    # measures how far the quantized loss is from a smooth function, which is the same fact.
    hook = ActQuant(table, rule)
    _, cache_a = forward(pq, X, quant=hook)
    on_grid = {}
    for name in ("X", "A1", "A2"):
        A = cache_a[name]
        Aq, _, _, _, _ = q.quantize_blocks_act(A, table, rule)
        on_grid[name] = bool(np.array_equal(A, Aq))

    # 4. how far the quantized loss is from a smooth function
    X_train, y_train, _, _ = _load("mnist")
    Xs = X_train[:256]
    ys = y_train[:256]
    base = _q_loss(p, table, rule, Xs, ys)
    eps = 1e-3
    moved = 0
    for i in rng.choice(p["W1"].size, 50, replace=False):
        w = p["W1"]
        old = w.flat[i]
        w.flat[i] = old + eps
        if abs(_q_loss(p, table, rule, Xs, ys) - base) > 0.0:
            moved += 1
        w.flat[i] = old
    return {"idempotent": idempotent, "worst_idempotence_error": worst,
            "worst_gradient_relative_error": worst_grad,
            "cached_activations_are_on_the_grid": on_grid,
            "of_50_perturbed_weights_that_moved_the_quantized_loss": moved}


def _q_loss(p, table, rule, X, y):
    from model import loss
    pq, _, _ = quantize_master(p, table, rule)
    P, _ = forward(pq, X)
    return loss(P, y)


def _load(dataset):
    from train import load_dataset
    return load_dataset(dataset)


if __name__ == "__main__":
    import sys
    print(self_check())
    dataset = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    config = sys.argv[2] if len(sys.argv) > 2 else "a6-e5m3"
    X_train, y_train, X_test, y_test = _load(dataset)
    result = train_qat(42, 10, 128, X_train, y_train, X_test, y_test, config)
    print(f"{dataset} {config} seed 42 -> {100 * result['history']['test_acc'][-1]:.2f}")
