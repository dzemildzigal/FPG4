import numpy as np
from model import init_params, adam, forward, backward, evaluate
import mnist
import mnist1d
import os

# The float width of every run in this project. See python/fp32_probe.py for the measurement against float64.
DTYPE = np.float32

# name -> (module with get_x_y_train / get_x_y_test, data folder). The name is part of every output file name.
DATASETS = {
    "mnist": (mnist, "data/mnist"),
    "mnist1d": (mnist1d, "data/mnist1d"),
}

def load_dataset(name):
    module, folder = DATASETS[name]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", folder)
    X_train, y_train = module.get_x_y_train(path)
    X_test, y_test = module.get_x_y_test(path)
    return X_train, y_train, X_test, y_test

def train(seed, epochs, batch_size, X, y, X_test, y_test, tag="mnist", dtype=DTYPE, save_plot=True):
    # dtype is the float width of the whole run, see model.init_params. Every number below follows from it.
    # save_plot=False is for sweeps: 120 identical curves are not evidence, and they cost disk time.
    rng = np.random.default_rng(seed)
    p = init_params(seed, dtype=dtype)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    t = 1
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in range(epochs):
        indices = np.arange(X.shape[0])
        rng.shuffle(indices)
        for start_idx in range(0, X.shape[0], batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            P, cache = forward(p, X_batch)
            grads = backward(p, y_batch, cache)
            p, m, v = adam(p, grads, m, v, t)
            t += 1

        train_loss, train_acc = evaluate(p, X, y)
        test_loss, test_acc = evaluate(p, X_test, y_test)

        print(epoch, train_loss, train_acc, test_loss, test_acc)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

    if not save_plot:
        return {"params": p, "history": history}

    import matplotlib.pyplot as plt
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["test_loss"], label="test")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.grid(True); ax1.legend()
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["test_acc"], label="test")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.grid(True); ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"experiment_L0_curve_{tag}_seed{seed}.png"), dpi=120)
    plt.close(fig)

    return {"params": p, "history": history}


def acceptance_test(seed, X_train, y_train, X_test, y_test, tag="mnist"):
    result = train(seed, epochs=10, batch_size=128, X=X_train, y=y_train, X_test=X_test, y_test=y_test, tag=tag)
    return result


def weights_path(seed, dataset):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", f"weights_{dataset}_seed{seed}.npy")


def save_weights(seed, dataset, data, dtype=DTYPE):
    X_train, y_train, X_test, y_test = data
    result = train(seed, epochs=10, batch_size=128, X=X_train, y=y_train, X_test=X_test, y_test=y_test,
                   tag=dataset, dtype=dtype)
    np.savez(weights_path(seed, dataset), result["params"], **result["params"])
    return result

def load_weights(seed, dataset="mnist"):
    data = np.load(weights_path(seed, dataset) + ".npz")
    params = {key: data[key] for key in data if key != "arr_0"}
    return params


if __name__ == "__main__":
    # uv run python train.py [mnist|mnist1d]   (default: mnist)
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    assert dataset in DATASETS, f"unknown dataset {dataset}, choose from {list(DATASETS)}"
    data = load_dataset(dataset)
    for seed in [42, 4242, 424242]:
        save_weights(seed, dataset, data)
