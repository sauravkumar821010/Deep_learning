"""
Gradient Descent demo on f(x) = (x - 3)^2 + 2
Visualizes gradient descent following the slope downhill to the minimum.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

f = lambda x: (x - 3) ** 2 + 2
grad_f = lambda x: 2 * (x - 3)

x = -6.0          # starting point
lr = 0.2          # learning rate
history = [x]

for step in range(20):
    x = x - lr * grad_f(x)
    history.append(x)

xs = np.linspace(-7, 9, 200)
plt.plot(xs, f(xs), label='f(x) = (x-3)^2 + 2')
plt.plot(history, [f(v) for v in history], 'o-', color='red', label='GD path')
plt.scatter([history[0]], [f(history[0])], color='black', zorder=5, label='start')
plt.xlabel('x'); plt.ylabel('f(x)'); plt.title('Gradient Descent following the slope downhill')
plt.legend(); plt.grid(True); plt.show()

print(f'Start x = {history[0]:.3f} -> after 20 steps x = {history[-1]:.3f} (true minimum x=3)')
