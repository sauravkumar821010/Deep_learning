"""
Animation 4: MSE vs MAE - watching the fitted line adjust

Both regression lines start flat at the origin and adjust every frame.
Watch the MSE line get pulled towards the outliers while the MAE line
stays closer to the true trend.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

rng = np.random.RandomState(0)
x_data = np.linspace(0, 10, 30)
y_data = 2.5 * x_data + 5 + rng.normal(0, 1.5, size=x_data.shape)

# Inject a couple of strong outliers
y_data[5] += 30
y_data[20] -= 25


def train_regression_track(loss_type, epochs=150, lr=0.01):
    w, b = 0.0, 0.0
    params_hist = []
    for _ in range(epochs):
        y_pred = w * x_data + b
        error = y_pred - y_data
        if loss_type == 'mse':
            dw = np.mean(2 * error * x_data)
            db = np.mean(2 * error)
        else:  # mae
            dw = np.mean(np.sign(error) * x_data)
            db = np.mean(np.sign(error))
        w -= lr * dw
        b -= lr * db
        params_hist.append((w, b))
    return params_hist


mse_params_hist = train_regression_track('mse')
mae_params_hist = train_regression_track('mae')

fig_d, ax_d = plt.subplots(figsize=(7, 5))
ax_d.scatter(x_data, y_data, color='gray', label='data (with outliers)')
line_mse_d, = ax_d.plot([], [], color='steelblue', label='MSE fit')
line_mae_d, = ax_d.plot([], [], color='orange', label='MAE fit')
ax_d.set_xlabel('x'); ax_d.set_ylabel('y')
ax_d.legend(); ax_d.grid(True)


def update_d(frame):
    w_m, b_m = mse_params_hist[frame]
    w_a, b_a = mae_params_hist[frame]
    line_mse_d.set_data(x_data, w_m * x_data + b_m)
    line_mae_d.set_data(x_data, w_a * x_data + b_a)
    ax_d.set_title(f'MSE vs MAE fit — epoch {frame}')
    return line_mse_d, line_mae_d


anim_d = animation.FuncAnimation(fig_d, update_d, frames=len(mse_params_hist), interval=80, blit=False)
plt.show()
