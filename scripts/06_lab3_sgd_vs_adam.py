"""
Hands-on Lab 3: Train a Neural Network with SGD vs Adam

Trains the same small network (2-4-1) on a XOR-style dataset twice -
once with plain SGD, once with Adam - and plots the loss/accuracy
convergence curves side by side.
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# XOR dataset (a few repeats so it behaves like a mini training set)
X = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float), (25, 1))
y = np.tile(np.array([[0], [1], [1], [0]], dtype=float), (25, 1))


def init_params(seed):
    rng = np.random.RandomState(seed)
    return {
        'W1': rng.randn(2, 4) * 0.5, 'b1': np.zeros((1, 4)),
        'W2': rng.randn(4, 1) * 0.5, 'b2': np.zeros((1, 1)),
    }


def forward(params, X):
    z1 = X @ params['W1'] + params['b1']
    a1 = sigmoid(z1)
    z2 = a1 @ params['W2'] + params['b2']
    a2 = sigmoid(z2)
    return a1, a2


def backward(params, X, y, a1, a2):
    delta2 = (a2 - y) * a2 * (1 - a2)
    dW2 = a1.T @ delta2 / len(X)
    db2 = np.mean(delta2, axis=0, keepdims=True)
    delta1 = (delta2 @ params['W2'].T) * a1 * (1 - a1)
    dW1 = X.T @ delta1 / len(X)
    db1 = np.mean(delta1, axis=0, keepdims=True)
    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}


def bce_loss(y, a2):
    eps = 1e-9
    return -np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))


def accuracy(y, a2):
    return np.mean((a2 > 0.5).astype(float) == y)


def train(optimizer, epochs=300, lr=0.5):
    params = init_params(seed=3)
    # Adam state
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    losses, accs = [], []

    for t in range(1, epochs + 1):
        a1, a2 = forward(params, X)
        grads = backward(params, X, y, a1, a2)
        losses.append(bce_loss(y, a2))
        accs.append(accuracy(y, a2))

        for k in params:
            if optimizer == 'sgd':
                params[k] -= lr * grads[k]
            elif optimizer == 'adam':
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
                m_hat = m[k] / (1 - beta1 ** t)
                v_hat = v[k] / (1 - beta2 ** t)
                params[k] -= lr * 0.1 * m_hat / (np.sqrt(v_hat) + eps)  # smaller lr scale for Adam
    return losses, accs


sgd_losses, sgd_accs = train('sgd')
adam_losses, adam_accs = train('adam')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(sgd_losses, label='SGD')
axes[0].plot(adam_losses, label='Adam')
axes[0].set_title('Loss convergence'); axes[0].set_xlabel('epoch'); axes[0].set_ylabel('BCE loss')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(sgd_accs, label='SGD')
axes[1].plot(adam_accs, label='Adam')
axes[1].set_title('Accuracy convergence'); axes[1].set_xlabel('epoch'); axes[1].set_ylabel('accuracy')
axes[1].legend(); axes[1].grid(True)
plt.tight_layout(); plt.show()

print(f'Final SGD  -> loss: {sgd_losses[-1]:.4f}, accuracy: {sgd_accs[-1]:.2f}')
print(f'Final Adam -> loss: {adam_losses[-1]:.4f}, accuracy: {adam_accs[-1]:.2f}')
