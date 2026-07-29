"""
Hands-on Lab 2: Backpropagation Demo (2-2-1 network)

A tiny 2 input -> 2 hidden -> 1 output network. Computes the forward pass, then
manually derives and prints every gradient in the backward pass, and finally
runs several gradient-descent updates to observe how the gradients shrink.
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_deriv(a):
    # derivative w.r.t. z, expressed using the activation a = sigmoid(z)
    return a * (1 - a)


# Single training example for clarity
x = np.array([[0.5, 0.8]])   # shape (1, 2)
y_true = np.array([[1.0]])   # target

np.random.seed(7)
W1 = np.random.randn(2, 2) * 0.5   # input -> hidden (2-2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1) * 0.5   # hidden -> output (2-1)
b2 = np.zeros((1, 1))

lr = 0.5
loss_history = []
grad_norm_history = []

print('Initial W1:\n', np.round(W1, 3), '\nInitial W2:\n', np.round(W2, 3), '\n')

for epoch in range(10):
    # ---- Forward pass ----
    z1 = x @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)          # prediction

    loss = np.mean((y_true - a2) ** 2)

    # ---- Backward pass (manual chain rule) ----
    dLoss_da2 = -(y_true - a2)                 # dL/da2  (MSE derivative)
    delta2 = dLoss_da2 * sigmoid_deriv(a2)      # delta at output layer
    dW2 = a1.T @ delta2
    db2 = delta2

    delta1 = (delta2 @ W2.T) * sigmoid_deriv(a1)  # delta at hidden layer
    dW1 = x.T @ delta1
    db1 = delta1

    grad_norm = np.linalg.norm(dW1) + np.linalg.norm(dW2)
    loss_history.append(loss)
    grad_norm_history.append(grad_norm)

    print(f'Epoch {epoch+1:2d} | loss={loss:.5f} | dW2={np.round(dW2.ravel(),4)} | dW1_norm={np.linalg.norm(dW1):.4f}')

    # ---- Gradient descent update ----
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(loss_history, 'o-')
axes[0].set_title('Loss per epoch'); axes[0].set_xlabel('epoch'); axes[0].set_ylabel('MSE loss'); axes[0].grid(True)
axes[1].plot(grad_norm_history, 'o-', color='red')
axes[1].set_title('Gradient magnitude per epoch (observe it shrinking)')
axes[1].set_xlabel('epoch'); axes[1].set_ylabel('||dW1|| + ||dW2||'); axes[1].grid(True)
plt.tight_layout(); plt.show()
