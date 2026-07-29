"""
Regularization demo: Dropout & L2 weight decay.
"""
import numpy as np

np.random.seed(42)

# Tiny demo: what dropout does to an activation vector
activations = np.array([1.2, 0.8, 2.5, 0.3, 1.7, 0.9, 2.1, 1.0])
dropout_rate = 0.4

mask = (np.random.rand(*activations.shape) > dropout_rate).astype(float)
dropped = activations * mask / (1 - dropout_rate)   # inverted dropout scaling

print('Original activations :', activations)
print('Dropout mask (kept=1):', mask)
print('After dropout (scaled):', np.round(dropped, 2))

# L2 penalty demo: larger weights -> larger penalty
weights_small = np.array([0.1, -0.2, 0.15])
weights_large = np.array([2.0, -3.0, 2.5])
lam = 0.01
l2_small = lam * np.sum(weights_small ** 2)
l2_large = lam * np.sum(weights_large ** 2)
print(f'\nL2 penalty (small weights): {l2_small:.5f}')
print(f'L2 penalty (large weights): {l2_large:.5f}  <- pushed down harder by L2')
