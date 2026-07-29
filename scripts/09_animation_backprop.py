"""
Animation 2: Backpropagation - loss & weight evolution

Same 2-2-1 network as Lab 2, trained for more epochs. Left panel: loss curve
being drawn live. Right panel: the output-layer weights W2 visibly moving
as gradient descent updates them.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_deriv(a):
    return a * (1 - a)


x_b = np.array([[0.5, 0.8]])
y_b = np.array([[1.0]])
rng_b = np.random.RandomState(7)
W1_b = rng_b.randn(2, 2) * 0.5
b1_b = np.zeros((1, 2))
W2_b = rng_b.randn(2, 1) * 0.5
b2_b = np.zeros((1, 1))
lr_b = 0.5

anim_loss_hist = []
anim_w2_hist = []

for epoch in range(60):
    z1 = x_b @ W1_b + b1_b
    a1 = sigmoid(z1)
    z2 = a1 @ W2_b + b2_b
    a2 = sigmoid(z2)
    loss = np.mean((y_b - a2) ** 2)

    delta2 = -(y_b - a2) * sigmoid_deriv(a2)
    dW2 = a1.T @ delta2
    db2 = delta2
    delta1 = (delta2 @ W2_b.T) * sigmoid_deriv(a1)
    dW1 = x_b.T @ delta1
    db1 = delta1

    anim_loss_hist.append(loss)
    anim_w2_hist.append(W2_b.ravel().copy())

    W2_b -= lr_b * dW2
    b2_b -= lr_b * db2
    W1_b -= lr_b * dW1
    b1_b -= lr_b * db1

anim_w2_hist = np.array(anim_w2_hist)

fig_b, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(11, 4))
line_b, = ax_b1.plot([], [], 'o-')
ax_b1.set_xlim(0, len(anim_loss_hist))
ax_b1.set_ylim(0, max(anim_loss_hist) * 1.1)
ax_b1.set_xlabel('epoch'); ax_b1.set_ylabel('MSE loss'); ax_b1.set_title('Loss decreasing')
ax_b1.grid(True)

bars_b = ax_b2.bar(['W2[0]', 'W2[1]'], anim_w2_hist[0])
ax_b2.set_ylim(anim_w2_hist.min() - 0.5, anim_w2_hist.max() + 0.5)
ax_b2.set_title('Output-layer weights changing')


def update_b(frame):
    line_b.set_data(range(frame + 1), anim_loss_hist[:frame + 1])
    for bar, h in zip(bars_b, anim_w2_hist[frame]):
        bar.set_height(h)
    fig_b.suptitle(f'Backpropagation — epoch {frame}')
    return (line_b, *bars_b)


anim_b = animation.FuncAnimation(fig_b, update_b, frames=len(anim_loss_hist), interval=150, blit=False)
plt.show()
