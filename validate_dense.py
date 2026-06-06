"""
validate_dense.py
-----------------
Compares Mamdani vs NN on the dense grid and reports RMSE and
max |Δ| for Kp, Ki, Kd.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio

# ------------- Network must match train_network_dense.py -------------
class FuzzyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# ------------- Load weights -------------
weights = sio.loadmat("nn_weights.mat")
model = FuzzyNet()
with torch.no_grad():
    model.fc1.weight.copy_(torch.from_numpy(weights["fc1_weight"]).float())
    model.fc1.bias  .copy_(torch.from_numpy(weights["fc1_bias"]).float().squeeze())
    model.fc2.weight.copy_(torch.from_numpy(weights["fc2_weight"]).float())
    model.fc2.bias  .copy_(torch.from_numpy(weights["fc2_bias"]).float().squeeze())
    model.fc3.weight.copy_(torch.from_numpy(weights["fc3_weight"]).float())
    model.fc3.bias  .copy_(torch.from_numpy(weights["fc3_bias"]).float().squeeze())
model.eval()

# ------------- Load dense Mamdani data -------------
data = pd.read_csv("training_data_dense.csv", header=None).values
X_np = data[:, 0:2].astype(np.float32)
y_np = data[:, 2:5]

X = torch.from_numpy(X_np)

with torch.no_grad():
    y_pred_np = model(X).numpy()

# ------------- Build a coarse 31x31 view for plotting -------------
# (purely for nicer plots; all metrics use full dense set)
N_plot = 31
e_plot = np.linspace(-1, 1, N_plot)
de_plot = np.linspace(-1, 1, N_plot)
E_plot, DE_plot = np.meshgrid(e_plot, de_plot)
grid_plot = np.column_stack([E_plot.ravel(), DE_plot.ravel()]).astype(np.float32)

with torch.no_grad():
    y_plot_pred = model(torch.from_numpy(grid_plot)).numpy()

# interpolate Mamdani targets to this coarse grid for plotting
# (nearest neighbour from the dense dataset)
dense_inputs = X_np
dense_targets = y_np

def nn_interpolate(inputs_dense, targets_dense, points):
    # brute-force nearest neighbour (OK for our size)
    out = np.zeros((points.shape[0], targets_dense.shape[1]))
    for i, p in enumerate(points):
        d2 = ((inputs_dense - p) ** 2).sum(axis=1)
        j = np.argmin(d2)
        out[i] = targets_dense[j]
    return out

y_plot_true = nn_interpolate(dense_inputs, dense_targets, grid_plot)

names = ["Kp", "Ki", "Kd"]
fig = plt.figure(figsize=(15, 12))
for k in range(3):
    M = y_plot_true[:, k].reshape(N_plot, N_plot)
    A = y_plot_pred[:, k].reshape(N_plot, N_plot)
    D = M - A

    ax = fig.add_subplot(3, 3, k*3 + 1, projection="3d")
    ax.plot_surface(E_plot, DE_plot, M, cmap="viridis")
    ax.set_title(f"Mamdani {names[k]}")
    ax.set_xlabel("error"); ax.set_ylabel("d_error")

    ax = fig.add_subplot(3, 3, k*3 + 2, projection="3d")
    ax.plot_surface(E_plot, DE_plot, A, cmap="viridis")
    ax.set_title(f"Neural Net {names[k]}")
    ax.set_xlabel("error"); ax.set_ylabel("d_error")

    ax = fig.add_subplot(3, 3, k*3 + 3, projection="3d")
    ax.plot_surface(E_plot, DE_plot, D, cmap="coolwarm")
    ax.set_title(f"Difference (M - NN)")
    ax.set_xlabel("error"); ax.set_ylabel("d_error")

plt.tight_layout()
plt.savefig("comparison_dense.png", dpi=120)
plt.show()

# ------------- Numerical summary on full dense set -------------
print("\nNumerical summary on dense dataset:")
for k, name in enumerate(names):
    M = y_np[:, k]
    A = y_pred_np[:, k]
    rmse = np.sqrt(np.mean((M - A) ** 2))
    max_abs = np.max(np.abs(M - A))
    rel_max = max_abs / (M.max() - M.min() + 1e-8) * 100.0
    print(
        f"  {name}: RMSE = {rmse:.8f}, "
        f"max |Δ| = {max_abs:.8f} "
        f"({rel_max:.2f}% of {name} range)"
    )