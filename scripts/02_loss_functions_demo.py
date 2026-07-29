"""
Loss Functions demo: MSE, MAE, and Cross-Entropy.
Visualizes how each loss penalizes error differently.
"""
import numpy as np
import matplotlib.pyplot as plt

# Visualize how each loss penalizes error differently for a single prediction
error = np.linspace(-3, 3, 200)
mse_loss = error ** 2
mae_loss = np.abs(error)

plt.plot(error, mse_loss, label='MSE: (y - y_hat)^2')
plt.plot(error, mae_loss, label='MAE: |y - y_hat|')
plt.xlabel('Prediction error (y - y_hat)'); plt.ylabel('Loss value')
plt.title('MSE vs MAE penalty as error grows')
plt.legend(); plt.grid(True); plt.show()

# Cross-entropy for a binary classification example (true label y=1)
p_hat = np.linspace(0.01, 0.99, 200)   # predicted probability of class 1
ce_loss = -np.log(p_hat)               # since y = 1

plt.plot(p_hat, ce_loss, color='green')
plt.xlabel('Predicted probability for true class (y=1)'); plt.ylabel('Cross-Entropy Loss')
plt.title('Cross-Entropy blows up as confidence in the WRONG answer increases')
plt.grid(True); plt.show()
