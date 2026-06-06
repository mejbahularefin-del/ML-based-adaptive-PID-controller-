"""
train_network_multipower.py
---------------------------
Trains a 3-input neural network: (error, d_error, power) -> (Kp_eff, Ki_eff, Kd_eff).
Same Ki-weighted, center-weighted loss as the dense single-power version.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- Load data ----------------
print("Loading multipower data...")
data = pd.read_csv("training_data_multipower.csv", header=None).values
print(f"Loaded {data.shape[0]} samples with {data.shape[1]} columns.")

X_np = data[:, 0:3].astype(np.float32)   # error, d_error, power
y_np = data[:, 3:6].astype(np.float32)   # normalised Kp, Ki, Kd
X = torch.from_numpy(X_np); y = torch.from_numpy(y_np)

# Precompute center-region mask (small |error| and |d_error| at any power)
e  = X[:, 0]
de = X[:, 1]
center_mask = ((e.abs() <= 0.25) & (de.abs() <= 0.25)).float()

# ---------------- Network: 3 -> 128 -> 128 -> 3 ----------------
class FuzzyNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = FuzzyNet3()
print("\nNetwork architecture:"); print(model)

# ---------------- Custom weighted loss ----------------
ki_weight    = 4.0
center_boost = 4.0
def weighted_loss(pred, target):
    sq = (pred - target) ** 2
    w_outputs = torch.tensor([1.0, ki_weight, 1.0], dtype=pred.dtype)
    sq_weighted = sq * w_outputs
    per_sample = sq_weighted.sum(dim=1)
    w_center = 1.0 + center_boost * center_mask
    return (per_sample * w_center).mean()

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[4000, 8000, 12000], gamma=0.3
)

# ---------------- Train ----------------
EPOCHS = 15000
losses = []

# Quick scale info for printing
ki_max = y_np[:, 1].max()

print(f"\nTraining for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    pred = model(X); loss = weighted_loss(pred, y)
    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
    losses.append(loss.item())
    if (epoch + 1) % 500 == 0:
        with torch.no_grad():
            ki_err = (pred[:,1] - y[:,1]).abs()
            print(f"Epoch {epoch+1:5d}/{EPOCHS}  loss = {loss.item():.8f}  "
                  f"max |ΔKi| ≈ {ki_err.max().item():.5f} "
                  f"({100*ki_err.max().item()/ki_max:.2f}% of Ki range)")

print("Training finished.")

# ---------------- Plots ----------------
plt.figure(figsize=(8,5))
plt.plot(losses, label="Weighted training loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.yscale("log")
plt.title("Multipower training curve"); plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig("training_curves_multipower.png", dpi=120); plt.close()

# ---------------- Save weights ----------------
weights = {}
for name, param in model.named_parameters():
    weights[name.replace(".", "_")] = param.detach().numpy().astype(np.float64)
sio.savemat("nn_weights_3in.mat", weights)
print("Saved trained weights to nn_weights_3in.mat")
print("Weight names:", list(weights.keys()))