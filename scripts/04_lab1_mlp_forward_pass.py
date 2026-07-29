"""
Hands-on Lab 1: Build a Simple MLP - Forward Pass & Loss Calculation

Manually implements forward propagation on a tiny dataset and computes
both MSE and Cross-Entropy losses, so students can compare the numbers directly.
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Small dataset: 4 samples, 2 features each (mimics XOR-style pattern), binary labels
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
y = np.array([[0.0], [1.0], [1.0], [0.0]])  # XOR labels

# Network architecture: 2 (input) -> 3 (hidden) -> 1 (output)
np.random.seed(1)
W1 = np.random.randn(2, 3) * 0.5
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.5
b2 = np.zeros((1, 1))

# ---- Forward pass (manual, step by step) ----
Z1 = X @ W1 + b1        # linear step 1
A1 = sigmoid(Z1)        # activation 1
Z2 = A1 @ W2 + b2       # linear step 2
y_hat = sigmoid(Z2)     # final prediction (output activation)

print('Z1 (pre-activation, hidden layer):\n', np.round(Z1, 3))
print('\nA1 (activation, hidden layer):\n', np.round(A1, 3))
print('\nZ2 (pre-activation, output layer):\n', np.round(Z2, 3))
print('\nPredicted y_hat:\n', np.round(y_hat, 3))
print('\nTrue labels y:\n', y)

# ---- Loss calculation ----
mse = np.mean((y - y_hat) ** 2)
eps = 1e-9  # avoid log(0)
cross_entropy = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

print(f'\nMSE Loss           : {mse:.4f}')
print(f'Cross-Entropy Loss : {cross_entropy:.4f}')

plt.bar(['MSE', 'Cross-Entropy'], [mse, cross_entropy], color=['steelblue', 'orange'])
plt.title('Loss comparison on the SAME predictions (untrained network)')
plt.ylabel('Loss value'); plt.show()
