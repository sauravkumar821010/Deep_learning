# Deep Learning

> **Hands-on Deep Learning lecture notebooks covering MLP, backpropagation, optimizers, loss functions & regularization — with animated NumPy demos.**

A hands-on classroom notebook covering the fundamentals of Deep Learning — from the transition out of the classic MLP all the way through backpropagation, optimizers, loss functions, and regularization — with runnable NumPy demos and live **animations** designed to be projected and run interactively during a lecture.

## Contents

| File | Description |
|---|---|
| [`Deep_Learning_Intro_MLP_Backprop.ipynb`](./Deep_Learning_Intro_MLP_Backprop.ipynb) | Main lecture notebook: theory + hands-on labs + animations |
| [`sum_two_numbers.ipynb`](./sum_two_numbers.ipynb) | Minimal "hello world" notebook example |

## `Deep_Learning_Intro_MLP_Backprop.ipynb`

### Theory sections
1. **Introduction to Deep Learning & scope** — ML vs DL, transition from a single Perceptron → MLP → Deep Neural Networks.
2. **The Backpropagation algorithm** — chain-rule derivation of gradients layer by layer.
3. **Optimization: Gradient Descent** — the update rule, plus a visual demo of GD minimizing a 1-D function.
4. **Loss Functions** — MSE, MAE, and Cross-Entropy, with plots showing how each penalizes error differently.
5. **Regularization** — Dropout and L2 weight decay, with small numeric demos.

### Hands-on labs (separate, runnable code cells)
1. **Build a Simple MLP** — manual forward pass on a small dataset; computes and compares MSE vs Cross-Entropy loss on the same predictions.
2. **Backpropagation Demo (2-2-1 network)** — manually derives and prints every gradient in the backward pass, then runs gradient descent while tracking how the gradients shrink.
3. **Train NN with SGD vs Adam** — trains the same small network with both optimizers on an XOR-style dataset and plots loss/accuracy convergence side by side.
4. **Loss Function Comparison: MSE vs MAE** — fits a linear regression model on noisy data with outliers under both losses and compares the resulting fit lines and error curves.

### Animated visualizations
Each animation renders as an interactive HTML5/JS player (play, pause, frame scrubber) via `matplotlib.animation.FuncAnimation` + `IPython.display.HTML`:
1. Gradient Descent rolling down the loss curve, step by step.
2. Backpropagation: loss curve being drawn live + output-layer weights updating each epoch.
3. SGD vs Adam: loss curves racing to convergence.
4. MSE vs MAE: the fitted regression line settling into place (watch MSE get pulled toward outliers vs MAE staying robust).

## Getting started

```bash
pip install numpy matplotlib ipykernel
```

Open `Deep_Learning_Intro_MLP_Backprop.ipynb` in VS Code (or Jupyter), select a Python kernel, and run the cells top to bottom. Each hands-on lab and animation is self-contained in its own cell so you can pause on any concept during a live session.
