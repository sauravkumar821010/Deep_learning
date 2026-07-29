"""
Hands-on Lab 4: Loss Function Comparison - MSE vs MAE

Fits a simple linear regression model (y_hat = wx + b) with gradient descent
on noisy data that contains a couple of outliers, once using MSE and once
using MAE, and compares the resulting fit lines and error curves.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)
x_data = np.linspace(0, 10, 30)
y_data = 2.5 * x_data + 5 + rng.normal(0, 1.5, size=x_data.shape)

# Inject a couple of strong outliers
y_data[5] += 30
y_data[20] -= 25


def train_regression(loss_type, epochs=500, lr=0.01):
    w, b = 0.0, 0.0
    history = []
    for _ in range(epochs):
        y_pred = w * x_data + b
        error = y_pred - y_data
        if loss_type == 'mse':
            loss = np.mean(error ** 2)
            dw = np.mean(2 * error * x_data)
            db = np.mean(2 * error)
        else:  # mae
            loss = np.mean(np.abs(error))
            dw = np.mean(np.sign(error) * x_data)
            db = np.mean(np.sign(error))
        w -= lr * dw
        b -= lr * db
        history.append(loss)
    return w, b, history


w_mse, b_mse, mse_history = train_regression('mse')
w_mae, b_mae, mae_history = train_regression('mae')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].scatter(x_data, y_data, label='data (with outliers)', color='gray')
axes[0].plot(x_data, w_mse * x_data + b_mse, color='steelblue', label=f'MSE fit (w={w_mse:.2f})')
axes[0].plot(x_data, w_mae * x_data + b_mae, color='orange', label=f'MAE fit (w={w_mae:.2f})')
axes[0].set_title('MSE vs MAE fitted line (outliers pull MSE line more)')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); axes[0].legend(); axes[0].grid(True)

axes[1].plot(mse_history, label='MSE loss')
axes[1].plot(mae_history, label='MAE loss')
axes[1].set_title('Training error curves'); axes[1].set_xlabel('epoch'); axes[1].set_ylabel('loss')
axes[1].legend(); axes[1].grid(True)

plt.tight_layout(); plt.show()

print(f'MSE fit -> slope: {w_mse:.3f}, intercept: {b_mse:.3f}')
print(f'MAE fit -> slope: {w_mae:.3f}, intercept: {b_mae:.3f}')
print('True underlying slope/intercept: 2.5 / 5 (MAE should stay closer to this despite outliers)')
