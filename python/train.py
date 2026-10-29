import numpy as np
from model import init_params, adam, forward, backward, loss, accuracy
from mnist import get_x_y_train, get_x_y_test
import os
X_test, y_test = get_x_y_test(os.path.join(os.path.dirname(__file__),"..", "data/mnist"))
X_train, y_train = get_x_y_train(os.path.join(os.path.dirname(__file__),"..", "data/mnist"))

def train(seed, epochs, batch_size, X, y, X_test, y_test):
    rng = np.random.default_rng(seed)
    p = init_params(seed)
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

        P_train, _ = forward(p, X)
        train_loss = loss(P_train, y)
        train_acc = accuracy(P_train, y)
        P_test, _ = forward(p, X_test)  # Replace with actual test set
        test_loss = loss(P_test, y_test)  # Replace with actual test set
        test_acc = accuracy(P_test, y_test)  # Replace with actual test set

        print(epoch, train_loss, train_acc, test_loss, test_acc)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

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
    fig.savefig(os.path.join(out_dir, f"l0_curve_seed{seed}.png"), dpi=120)
    plt.close(fig)

    return {"params": p, "history": history}


def acceptance_test(seed, X_train, y_train, X_test, y_test):
    result = train(seed, epochs=10, batch_size=128, X=X_train, y=y_train, X_test=X_test, y_test=y_test)
    return result


def save_weights(seed):
    result = train(seed, epochs=10, batch_size=128, X=X_train, y=y_train, X_test=X_test, y_test=y_test)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", f"weights_seed{seed}.npy"), result["params"], **result["params"])
    return result

def load_weights(seed):
    data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", f"weights_seed{seed}.npy.npz"))
    params = {key: data[key] for key in data if key != "arr_0"}
    return params


if __name__ == "__main__":
    for seed in [42, 4242, 424242]:
        save_weights(seed)
        