"""
The second rounding of A and dZ, measured on real activations. No training.

Question: an operand of the dW GEMM must be blocked along the batch.
That blocking is a SECOND rounding of a tensor that is already FP4 along the features. How much does the second rounding
change the tensor, and how much does it change the gradient the GEMM produces?

route (i)  one rounding  : block the exact float tensor along the batch, round once.
route (ii) two roundings : take the FP4 value already stored (blocked along the features), block that along the batch, round again.
reference  exact         : the float tensor with no rounding along the batch.

Weights are FP4 along K (the L2 setting). Activations are float32 until this script rounds them.
Run from FPG4/python:  PYTHONPATH=. uv run python /tmp/shape/second_rounding.py
"""
import os
import numpy as np
import quant as q
from model import forward, backward, relu_derivative
from train import load_weights

HERE = os.path.dirname(os.path.abspath(q.__file__))
SEEDS = [42, 4242, 424242, 123, 456]
CONFIGS = {"a6-e6m2": (q.e6m2, "L1b"), "f6-e6m2": (q.e6m2, "L1c"), "a6-e5m3": (q.e5m3, "L1b"), "f6-e5m3": (q.e5m3, "L1c")}
BATCH, NBATCH = 128, 4
LAYERS = [("X", "layer 1, A is the 784-wide input"), ("A1", "layer 2, A is the 16-wide first hidden layer"),
          ("A2", "layer 3, A is the 16-wide second hidden layer")]


def round_along(T, axis, table, rule):
    # One scale per 16 values along `axis`. 0 = batch (samples), 1 = features (the K axis of the forward pass).
    # Zeros pad the width up to a multiple of 16 and never change amax.
    T = T if axis == 1 else T.T
    C = T.shape[1]
    pad = -(-C // 16) * 16 - C
    Tp = np.pad(T, ((0, 0), (0, pad))) if pad else T
    Th = q.quantize_blocks(Tp.reshape(Tp.shape[0], -1, 16), table, rule)[0]
    Th = Th.reshape(Tp.shape)[:, :C]
    return Th if axis == 1 else Th.T


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cos(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def metrics(A, dZ, table, rule):
    Af, dZf = round_along(A, 1, table, rule), round_along(dZ, 1, table, rule)    # stored: blocked along the features
    A1, dZ1 = round_along(A, 0, table, rule), round_along(dZ, 0, table, rule)    # route (i): one rounding
    A2, dZ2 = round_along(Af, 0, table, rule), round_along(dZf, 0, table, rule)  # route (ii): two roundings
    dW = A.T @ dZ
    dW1, dW2 = A1.T @ dZ1, A2.T @ dZ2
    return {
        "e1": rel(A1, A), "e2": rel(A2, A),                                        # error of the A operand
        "diff": float(np.mean(np.concatenate([(A1 != A2).ravel(), (dZ1 != dZ2).ravel()]))),
        "dw1": rel(dW1, dW), "dw2": rel(dW2, dW),                                  # error of the gradient, vs exact
        "ddw": rel(dW2, dW1),                                                      # what the second rounding changes
        "cos": cos(dW1, dW2),                                                      # angle between the two gradients
        "b1": rel(np.sum(dZ1, 0), np.sum(dZ, 0)), "b2": rel(np.sum(dZ2, 0), np.sum(dZ, 0)),
    }


def gain(cache, dZ3, p, table, rule, route):
    # The whole gradient of one batch, with both operands of every dW GEMM blocked along the batch by one route.
    r = (lambda T: round_along(T, 0, table, rule)) if route == 1 else (lambda T: round_along(round_along(T, 1, table, rule), 0, table, rule))
    dZ2 = (dZ3 @ p["W3"].T) * relu_derivative(cache["Z2"])
    dZ1 = (dZ2 @ p["W2"].T) * relu_derivative(cache["Z1"])
    g = {}
    for A, dZ, tag in ((cache["X"], dZ1, "W1"), (cache["A1"], dZ2, "W2"), (cache["A2"], dZ3, "W3")):
        g[tag] = r(A).T @ r(dZ)
    return np.concatenate([g["W1"].ravel(), g["W2"].ravel(), g["W3"].ravel()])


def main():
    for dataset in ("mnist", "mnist1d"):
        mod = __import__("mnist" if dataset == "mnist" else "mnist1d")
        X, y = mod.get_x_y_train(os.path.join(HERE, "..", "data", dataset))
        X = X.astype(np.float32)
        print(f"\n===== {dataset}: train {X.shape}, {len(SEEDS)} seeds, {NBATCH} batches of {BATCH}, weights FP4 along K =====")
        for name, (table, rule) in CONFIGS.items():
            res = {L: {k: [] for k in ("e1", "e2", "diff", "dw1", "dw2", "ddw", "cos", "b1", "b2")} for L, _ in LAYERS}
            gc = []
            for seed in SEEDS:
                p = load_weights(seed, dataset=dataset)
                for k in ("W1", "W2", "W3"):
                    p[k] = q.quantize_blocks_weights(p[k], table, rule)[0]
                for b in range(NBATCH):
                    sl = slice(b * BATCH, (b + 1) * BATCH)
                    P, cache = forward(p, X[sl])
                    dZ3 = (P - np.eye(10)[y[sl]]) / BATCH
                    dZ2 = (dZ3 @ p["W3"].T) * relu_derivative(cache["Z2"])
                    dZ1 = (dZ2 @ p["W2"].T) * relu_derivative(cache["Z1"])
                    for L, dZ in (("X", dZ1), ("A1", dZ2), ("A2", dZ3)):
                        m = metrics(cache[L], dZ, table, rule)
                        for k in res[L]:
                            res[L][k].append(m[k])
                    gc.append(cos(gain(cache, dZ3, p, table, rule, 1), gain(cache, dZ3, p, table, rule, 2)))
            print(f"\n  config {name}")
            print("  layer | A err one | A err two | values differ | dW err one | dW err two | dW two vs dW one | cos(dW one, two) | db err one | db err two")
            for L, _ in LAYERS:
                m = {k: np.mean(v) for k, v in res[L].items()}
                s = {k: np.std(v) for k, v in res[L].items()}
                print(f"  {L:>5} | {100*m['e1']:6.2f}% | {100*m['e2']:6.2f}% | {100*m['diff']:6.1f}% | {100*m['dw1']:6.2f}% | {100*m['dw2']:6.2f}% |"
                      f" {100*m['ddw']:6.2f}% | {m['cos']:.6f} | {100*m['b1']:6.2f}% | {100*m['b2']:6.2f}%")
            print(f"  whole gradient, cos(route one, route two) = {np.mean(gc):.6f}  (min {np.min(gc):.6f}, {len(gc)} batches)")


if __name__ == "__main__":
    main()
