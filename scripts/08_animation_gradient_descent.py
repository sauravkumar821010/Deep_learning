"""
Animation 1: Gradient Descent path animation

Watch a point roll downhill towards the minimum of f(x) = (x-3)^2 + 2,
one gradient step per frame.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

f = lambda x: (x - 3) ** 2 + 2
grad_f = lambda x: 2 * (x - 3)

x0 = -6.0
lr_anim = 0.2
gd_path = [x0]
for _ in range(25):
    x0 = x0 - lr_anim * grad_f(x0)
    gd_path.append(x0)

xs_anim = np.linspace(-7, 9, 200)
fig_a, ax_a = plt.subplots(figsize=(7, 4))
ax_a.plot(xs_anim, f(xs_anim), label='f(x) = (x-3)^2 + 2')
point_a, = ax_a.plot([], [], 'ro', markersize=10, label='current position')
trail_a, = ax_a.plot([], [], 'r--', alpha=0.6)
ax_a.set_xlabel('x'); ax_a.set_ylabel('f(x)')
ax_a.legend(); ax_a.grid(True)


def update_a(frame):
    point_a.set_data([gd_path[frame]], [f(gd_path[frame])])
    trail_a.set_data(gd_path[:frame + 1], [f(v) for v in gd_path[:frame + 1]])
    ax_a.set_title(f'Gradient Descent — step {frame}/{len(gd_path) - 1}, x = {gd_path[frame]:.3f}')
    return point_a, trail_a


anim_a = animation.FuncAnimation(fig_a, update_a, frames=len(gd_path), interval=300, blit=False)
plt.show()
