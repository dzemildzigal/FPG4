"""Does the simulation get the same answer in float32 as in float64?

The question exists because the fabric is float32 and the ladder is float64. If the
width changes halfway up the ladder, every gap that is measured against an earlier
rung stops comparing. So the width of the whole simulation is decided once, here.

What is measured: L0, the plain float training run with no quantization anywhere.
Same network, same Adam, same seeds, same epochs. Only the float width changes.
The float64 column is not re-run, because those numbers are already published in
docs/experiment_L2.md and they came from the same code path.

    uv run python fp32_probe.py

Read the output like this: if the float32 mean sits inside the float64 seed spread,
the width is not a variable in this project and the ladder moves to float32 for free.
"""
import numpy as np
from train import load_dataset, train

# The published L0 result, from docs/experiment_L2.md, "Starting baseline".
# Same seeds, same everything, dtype float64.
FP64 = {
    "mnist": {42: 94.69, 4242: 95.42, 424242: 95.44},
    "mnist1d": {42: 65.15, 4242: 64.87, 424242: 65.31},
}
SEEDS = [42, 4242, 424242]


def main():
    for dataset in ("mnist", "mnist1d"):
        X_train, y_train, X_test, y_test = load_dataset(dataset)
        print(f"=== {dataset} ===")
        print("seed   | float64 | float32 | difference")
        print("-------|---------|---------|-----------")
        d = []
        for seed in SEEDS:
            result = train(seed, 10, 128, X_train, y_train, X_test, y_test,
                           tag=f"{dataset}_fp32", dtype=np.float32, save_plot=False)
            acc = 100.0 * result["history"]["test_acc"][-1]
            ref = FP64[dataset][seed]
            d.append(acc - ref)
            print(f"{seed:6d} | {ref:7.2f} | {acc:7.2f} | {acc - ref:+9.2f}")
        d = np.array(d)
        ref = np.array([FP64[dataset][s] for s in SEEDS])
        print(f"mean   | {ref.mean():7.2f} | {ref.mean() + d.mean():7.2f} | {d.mean():+9.2f}")
        print(f"spread of the three float64 seeds: {ref.max() - ref.min():.2f} points")
        print(f"mean shift from the width alone:   {abs(d.mean()):.2f} points")
        print()


if __name__ == "__main__":
    main()
