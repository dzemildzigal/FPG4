"""
L1d sweep: FP4 on the weights AND on the activations, after training.

Run from python/:  uv run python l1d_sweep.py [mnist|mnist1d]
Needs the weights from:  uv run python train.py [mnist|mnist1d]

Difference to l1_sweep.py: there the matmul input X and the hidden activations A1, A2 stay in float64.
Here they are quantized too, with the same format and the same rule as the weights, so both operands
of every MAC are on the FP4 grid. The arithmetic is still float64 (no accumulator model), that is L1e.

For every config it prints three numbers: weights only (the L1 number), weights + activations (L1d),
and the extra drop that the activations cause.
"""
import sys

import numpy as np

import quant as q
from model import evaluate, forward
from train import load_weights, load_dataset

SEEDS = [42, 4242, 424242]
DATASET = sys.argv[1] if len(sys.argv) > 1 else "mnist"
SQNR_SAMPLES = 10_000        # the activation statistics are measured on one slice, in one chunk

# name, scale table, weight quantize function, rule
CONFIGS = [
    ("UE6M2 amax/6", q.e6m2, q.quantize_fp4_b_e6m2, "L1b"),
    ("UE6M2 four6", q.e6m2, q.quantize_fp4_b_e6m2, "L1c"),
    ("UE5M3 amax/6", q.e5m3, q.quantize_fp4_b_e5m3, "L1b"),
    ("UE5M3 four6", q.e5m3, q.quantize_fp4_b_e5m3, "L1c"),
]


def quantized_weights(P, fn, rule):
    # W1, W2, W3 go through the L1 quantizer. The biases stay float64, like L1.
    Q = dict(P)
    sqnr = {}
    for key in ("W1", "W2", "W3"):
        Q[key], _, sqnr[key], _ = fn(P[key], rule=rule)
    return Q, sqnr


def act_quantizer(table, rule):
    # The hook for model.forward: every activation tensor is quantized with the same table and rule as the weights.
    # It also records the SQNR and the counters of every call, in call order, so the caller can report them.
    log = {"sqnr": [], "counters": []}

    def quant(A):
        Aq, _, snr, counters = q.quantize_fp4_b_act(A, table, rule)
        log["sqnr"].append(snr)
        log["counters"].append(counters)
        return Aq, counters

    return quant, log


def act_stats(P, X, table, rule):
    # One single-chunk forward on a slice. forward calls the hook 3 times: X, then A1, then A2.
    # The weights here are the quantized ones, so the activations are the real ones that L1d produces.
    quant, log = act_quantizer(table, rule)
    forward(P, X[:SQNR_SAMPLES], quant=quant)
    names = ("X", "A1", "A2")
    assert len(log["sqnr"]) == 3, f"expected 3 hook calls, got {len(log['sqnr'])}"
    return {n: (log["sqnr"][i], log["counters"][i]) for i, n in enumerate(names)}


if __name__ == "__main__":
    _, _, X_test, y_test = load_dataset(DATASET)
    print(f"dataset {DATASET}: test set {X_test.shape}, seeds {SEEDS}")
    print(f"activation statistics on the first {SQNR_SAMPLES} test samples\n")

    base = np.array([100 * evaluate(load_weights(s, DATASET), X_test, y_test)[1] for s in SEEDS])
    print(f"L0 float            " + "".join(f"{a:>9.2f}" for a in base) + f"  mean {base.mean():.2f}")

    rows = {}
    for name, table, fn, rule in CONFIGS:
        w_only, w_act, details = [], [], []
        for seed in SEEDS:
            P = load_weights(seed, DATASET)
            Q, ws = quantized_weights(P, fn, rule)
            w_only.append(100 * evaluate(Q, X_test, y_test)[1])
            quant, _ = act_quantizer(table, rule)
            w_act.append(100 * evaluate(Q, X_test, y_test, quant=quant)[1])
            details.append(act_stats(Q, X_test, table, rule))
        rows[name] = (np.array(w_only), np.array(w_act), details)

        print(f"L1  {name:<15}" + "".join(f"{a:>9.2f}" for a in w_only) + f"  mean {np.mean(w_only):.2f}")
        print(f"L1d {name:<15}" + "".join(f"{a:>9.2f}" for a in w_act) + f"  mean {np.mean(w_act):.2f}")
        print(f"    {"extra drop":<15}" + "".join(f"{a - b:>9.2f}" for a, b in zip(w_only, w_act))
              + f"  mean {np.mean(w_only) - np.mean(w_act):.2f}")
        for key in ("X", "A1", "A2"):
            snr = [d[key][0] for d in details]
            sat = [d[key][1]["scale_saturated"] for d in details]
            und = [d[key][1]["scale_underflow"] for d in details]
            zero = [d[key][1]["zero_blocks"] for d in details]
            nf = [d[key][1]["nonfinite_blocks"] for d in details]
            print(f"    {key:<3} SQNR {np.mean(snr):6.2f} dB | counters saturated {sat} underflow {und} zero {zero} nonfinite {nf}")
        print()

    print("summary (mean accuracy over seeds)")
    print(f"{'config':<16}{'L0':>8}{'weights':>10}{'weights+acts':>14}{'extra drop':>12}")
    for name, (w_only, w_act, _) in rows.items():
        print(f"{name:<16}{base.mean():>8.2f}{w_only.mean():>10.2f}{w_act.mean():>14.2f}{w_only.mean() - w_act.mean():>12.2f}")
