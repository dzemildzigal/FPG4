"""
L1 sweep: post-training quantization of the trained weights, every rung, every seed.

Run from python/:  uv run python l1_sweep.py [mnist|mnist1d]
Needs the weights from:  uv run python train.py [mnist|mnist1d]

Prints one block per rung: accuracy, loss, drop against L0 and SQNR per seed, then a summary table.
Run it on both datasets after every change to quant.py. A change in a number is a regression.
"""
import sys

import numpy as np

import quant as q
from model import evaluate
from train import load_weights, load_dataset

SEEDS = [42, 4242, 424242]
DATASET = sys.argv[1] if len(sys.argv) > 1 else "mnist"

# rung name -> quantize function. L0 has none: the float weights are used as they are.
RUNGS = {
    "L0 float": None,
    "L1a fp64 scale": lambda W: q.quantize_fp4_b_fp64(W),
    "L1b UE6M2": lambda W: q.quantize_fp4_b_e6m2(W, rule="L1b"),
    "L1c UE6M2": lambda W: q.quantize_fp4_b_e6m2(W, rule="L1c"),
    "L1b UE5M3": lambda W: q.quantize_fp4_b_e5m3(W, rule="L1b"),
    "L1c UE5M3": lambda W: q.quantize_fp4_b_e5m3(W, rule="L1c"),
}


def run_rung(fn, X_test, y_test):
    # For each seed: quantize W1, W2, W3 (biases stay float), then evaluate on the test set.
    accs, losses, sqnr_range = [], [], []
    signal, noise = 0.0, 0.0                                  # pooled SQNR: sum of W^2 over sum of (W - Wq)^2
    for seed in SEEDS:
        P = load_weights(seed, DATASET)
        Q = dict(P)
        per_matrix = []
        if fn is not None:
            for key in ("W1", "W2", "W3"):
                Wq, _, snr, _ = fn(P[key])
                Q[key] = Wq
                per_matrix.append(snr)
                signal += np.sum(P[key] ** 2)
                noise += np.sum((P[key] - Wq) ** 2)
        test_loss, test_acc = evaluate(Q, X_test, y_test)
        accs.append(100 * test_acc)
        losses.append(test_loss)
        sqnr_range.append((min(per_matrix), max(per_matrix)) if per_matrix else None)
    pooled = 10 * np.log10(signal / noise) if fn is not None else None
    return np.array(accs), np.array(losses), sqnr_range, pooled


if __name__ == "__main__":
    _, _, X_test, y_test = load_dataset(DATASET)
    print(f"dataset {DATASET}: test set {X_test.shape}, seeds {SEEDS}\n")
    results = {name: run_rung(fn, X_test, y_test) for name, fn in RUNGS.items()}
    base = results["L0 float"][0]

    for name, (accs, losses, sqnr_range, pooled) in results.items():
        print(name)
        for i, seed in enumerate(SEEDS):
            line = f"  {seed:>6}: acc {accs[i]:.2f}  loss {losses[i]:.4f}  drop {base[i] - accs[i]:.2f}"
            if sqnr_range[i] is not None:
                line += f"  SQNR {sqnr_range[i][0]:.2f} to {sqnr_range[i][1]:.2f} dB"
            print(line)

    print("\nsummary (means over seeds)")
    print(f"{'rung':<16}" + "".join(f"{s:>9}" for s in SEEDS) + "  mean acc  mean loss  drop  pooled SQNR")
    for name, (accs, losses, _, pooled) in results.items():
        row = f"{name:<16}" + "".join(f"{a:>9.2f}" for a in accs)
        row += f"  {accs.mean():>8.2f}  {losses.mean():>9.4f}  {base.mean() - accs.mean():>4.2f}  "
        row += f"{pooled:>11.2f}" if pooled is not None else "       none"
        print(row)
