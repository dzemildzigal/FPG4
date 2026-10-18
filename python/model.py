import numpy as np

def onehot(y, num_classes=10):
    return np.eye(num_classes)[y]


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def init_params(seed):
    # Initialize model parameters with the given random seed, return them as a dictionary.
    rng = np.random.default_rng(seed)
    std1 = np.sqrt(2.0 / 784)  # He initialization
    std2 = np.sqrt(2.0 / 16)
    std3 = np.sqrt(2.0 / 16)
    params = {
        "W1": rng.standard_normal((784, 16)) * std1,
        "b1": np.zeros(16),
        "W2": rng.standard_normal((16, 16)) * std2,
        "b2": np.zeros(16),
        "W3": rng.standard_normal((16, 10)) * std3,
        "b3": np.zeros(10)
    }
    return params

def forward(p, X):
    # Perform the forward pass using the parameters p and input X.
    X = X.astype(p["W1"].dtype)

    Z1 = X @ p["W1"] + p["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ p["W2"] + p["b2"]
    A2 = relu(Z2)
    Z3 = A2 @ p["W3"] + p["b3"]
    P = softmax(Z3)
    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "P": P
    }
    return P, cache

def backward(p, y, cache):
    # Perform the backward pass using the parameters p, ground truth y, and cached values.
    Y = onehot(y)

    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]
    Z3 = cache["Z3"]
    P = cache["P"]

    B = y.shape[0]
    dZ3 = (P - Y) / B
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0) # (10,)

    dA2 = dZ3 @ p["W3"].T
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0) # (16,)

    dA1 = dZ2 @ p["W2"].T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0) # (16,) 

    grads = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3
    }
    return grads

def loss(P, y):
    # loss = -mean(log(P[arange(B),y]))
    B = y.shape[0]
    return -np.sum(onehot(y) * np.log(P)) / B

def accuracy(P, y):
    # accuracy = mean(argmax(P, axis=1) == y)
    return np.mean(np.argmax(P, axis=1) == y)


def adam(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    # perform adam optimizer on parameters using the gradients
    for k in params:
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
        m_hat = m[k] / (1 - beta1 ** t)
        v_hat = v[k] / (1 - beta2 ** t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return params, m, v


def acceptance_test(seed):
    params = init_params(seed)
    rng = np.random.default_rng(seed)
    B=8
    X = rng.standard_normal((B, 784))
    y = rng.integers(0, 10, size=B)
    P, cache = forward(params, X)
    grads = backward(params, y, cache)
    #pick 5 random entries in each of the six arrays
    for key in grads:
        g = grads[key].ravel()        # analytic, read only
        w = params[key].ravel()       # view into the parameter
        idx = np.random.choice(w.size, 5, replace=False)
        #for each entry, add eps=1e-6, compute the loss, subtract eps, compute the loss. divide the difference by 2*eps. that is the numerical gradient.
        eps = 1e-6
        for i in idx:
            old_val = w[i]
            w[i] = old_val + eps
            P_plus, _ = forward(params, X)
            loss_plus = -np.sum(onehot(y) * np.log(P_plus)) / B
            w[i] = old_val - eps
            P_minus, _ = forward(params, X)
            loss_minus = -np.sum(onehot(y) * np.log(P_minus)) / B
            w[i] = old_val
            num_grad = (loss_plus - loss_minus) / (2 * eps)
            print(f"{key}[{i}] - numerical gradient: {num_grad}, backward gradient: {g[i]}")
            #compare, rel=|analytic-numeric|/max(|analytic|, |numeric|, 1e-8)
            rel_error = abs(g[i] - num_grad) / max(abs(g[i]), abs(num_grad), 1e-8)
            print(f"{key}[{i}] - relative error: {rel_error}")

if __name__ == "__main__":
    acceptance_test(42)
    acceptance_test(4242)
    acceptance_test(424242)