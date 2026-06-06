"""
validate.py (v3 - 128-neuron layers, all-point comparison)
----------------------------------------------------------
Compares Mamdani surfaces to the NN outputs on the full grid and
prints RMSE + max |Δ| for Kp, Ki, Kd.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio

# Architecture MUST match train_network.py
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

# Load saved weights
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

# Load Mamdani data (full grid)
data = pd.read_csv("training_data.csv", header=None).values
X = data[:, 0:2].astype(np.float32)
y = data[:, 2:5]

with torch.no_grad():
    y_pred = model(torch.from_numpy(X)).numpy()

# Reshape to grid
N = int(np.sqrt(X.shape[0]))
E  = X[:, 0].reshape(N, N)
DE = X[:, 1].reshape(N, N)

names = ["Kp", "Ki", "Kd"]
fig = plt.figure(figsize=(15, 12))
for k in range(3):
    M = y[:, k].reshape(N, N)
    A = y_pred[:, k].reshape(N, N)
    D = M - A

    ax = fig.add_subplot(3, 3, k*3 + 1, projection="3d")
    ax.plot_surface(E, DE, M, cmap="viridis")
    ax.set_title(f"Mamdani {names[k]}")
    ax.set_xlabel("error")
    ax.set_ylabel("d_error")

    ax = fig.add_subplot(3, 3, k*3 + 2, projection="3d")
    ax.plot_surface(E, DE, A, cmap="viridis")
    ax.set_title(f"Neural Net {names[k]}")
    ax.set_xlabel("error")
    ax.set_ylabel("d_error")

    ax = fig.add_subplot(3, 3, k*3 + 3, projection="3d")
    ax.plot_surface(E, DE, D, cmap="coolwarm")
    ax.set_title(f"Difference (M - NN), max |Δ| = {np.max(np.abs(D)):.4f}")
    ax.set_xlabel("error")
    ax.set_ylabel("d_error")

plt.tight_layout()
plt.savefig("comparison.png", dpi=120)
plt.show()

print("\nNumerical summary:")
for k, name in enumerate(names):
    M = y[:, k]
    A = y_pred[:, k]
    rmse = np.sqrt(np.mean((M - A) ** 2))
    max_abs = np.max(np.abs(M - A))
    print(f"  {name}: RMSE = {rmse:.8f}, max |Δ| = {max_abs:.8f}")