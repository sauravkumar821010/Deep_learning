"""
Animation 3: SGD vs Adam convergence race

Trains a small network (2-4-1) on an XOR-style dataset with both SGD and Adam,
then animates the two loss curves racing to convergence frame by frame.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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


def train(optimizer, epochs=300, lr=0.5):
    params = init_params(seed=3)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    losses = []

    for t in range(1, epochs + 1):
        a1, a2 = forward(params, X)
        grads = backward(params, X, y, a1, a2)
        losses.append(bce_loss(y, a2))

        for k in params:
            if optimizer == 'sgd':
                params[k] -= lr * grads[k]
            elif optimizer == 'adam':
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
                m_hat = m[k] / (1 - beta1 ** t)
                v_hat = v[k] / (1 - beta2 ** t)
                params[k] -= lr * 0.1 * m_hat / (np.sqrt(v_hat) + eps)
    return losses


sgd_losses = train('sgd')
adam_losses = train('adam')

fig_c, ax_c = plt.subplots(figsize=(7, 4))
line_sgd, = ax_c.plot([], [], label='SGD', color='tab:blue')
line_adam, = ax_c.plot([], [], label='Adam', color='tab:orange')
ax_c.set_xlim(0, len(sgd_losses))
ax_c.set_ylim(0, max(max(sgd_losses), max(adam_losses)) * 1.05)
ax_c.set_xlabel('epoch'); ax_c.set_ylabel('BCE loss')
ax_c.legend(); ax_c.grid(True)

step_c = 4  # skip frames so the animation stays short and smooth
frame_indices_c = list(range(0, len(sgd_losses), step_c))


def update_c(i):
    frame = frame_indices_c[i]
    line_sgd.set_data(range(frame + 1), sgd_losses[:frame + 1])
    line_adam.set_data(range(frame + 1), adam_losses[:frame + 1])
    ax_c.set_title(f'SGD vs Adam — epoch {frame}')
    return line_sgd, line_adam


anim_c = animation.FuncAnimation(fig_c, update_c, frames=len(frame_indices_c), interval=60, blit=False)
plt.show()
