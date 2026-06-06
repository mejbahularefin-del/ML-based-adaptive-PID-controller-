"""
train_network_dense.py
----------------------
Trains a 2-128-128-3 network on a densely sampled grid, using a
Ki-weighted and center-weighted MSE loss to minimize Ki error
(< 5% of full scale) as much as possible.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- Load dense data ----------------
print("Loading data (dense grid)...")
data = pd.read_csv("training_data_dense.csv", header=None).values
print(f"Loaded {data.shape[0]} samples, each with {data.shape[1]} columns.")

# Inputs: error, d_error
X_np = data[:, 0:2].astype(np.float32)
# Targets: Kp, Ki, Kd
y_np = data[:, 2:5].astype(np.float32)

X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)

# Precompute masks for center region and full-scale factor for "5%" target
e = X[:, 0]
de = X[:, 1]
center_mask = ((e.abs() <= 0.25) & (de.abs() <= 0.25)).float()  # 1 in center, 0 outside

# Rough scale of Ki to interpret 5% (max of target Ki)
ki_max = y[:, 1].max().item()
print(f"Max Ki in dataset: {ki_max:.4f}")

# ---------------- Network: 2 -> 128 -> 128 -> 3 ----------------
class FuzzyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))   # outputs in [0, 1]
        return x

model = FuzzyNet()
print("\nNetwork architecture:")
print(model)

# ---------------- Custom weighted loss ----------------
# - Ki error weighted more than Kp, Kd
# - Center region around (0,0) weighted more, because Ki spike is there
base_criterion = nn.MSELoss(reduction="none")
ki_weight = 4.0       # emphasise Ki vs Kp/Kd
center_boost = 4.0    # emphasise center samples vs boundary

def weighted_loss(pred, target):
    sq = (pred - target) ** 2              # [N,3]
    # base per-output weights: [1, ki_weight, 1]
    w_outputs = torch.tensor([1.0, ki_weight, 1.0], dtype=pred.dtype, device=pred.device)
    sq_weighted = sq * w_outputs           # [N,3]
    per_sample = sq_weighted.sum(dim=1)    # [N]

    # center weighting: 1 outside center, (1+center_boost) inside
    w_center = 1.0 + center_boost * center_mask.to(pred.device)
    per_sample = per_sample * w_center

    return per_sample.mean()

# ---------------- Optimiser & schedule ----------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[4000, 8000, 12000],
    gamma=0.3,
)

# ---------------- Train (very long) ----------------
EPOCHS = 15000        # long training; you said time is fine
losses = []

print(f"\nTraining for {EPOCHS} epochs on ALL dense data...")
for epoch in range(EPOCHS):
    model.train()
    pred = model(X)
    loss = weighted_loss(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if (epoch + 1) % 500 == 0:
        # quick Ki metrics on the fly (absolute error in Ki)
        with torch.no_grad():
            ki_pred = pred[:, 1]
            ki_err = (ki_pred - y[:, 1]).abs()
            ki_max_err = ki_err.max().item()
            ki_max_pct = 100.0 * ki_max_err / ki_max
        print(
            f"Epoch {epoch+1:5d}/{EPOCHS}  "
            f"loss = {loss.item():.8f}  "
            f"max |ΔKi| ≈ {ki_max_err:.5f} ({ki_max_pct:.2f}% of Ki range)"
        )

print("Training finished.")

# ---------------- Plot training curve ----------------
plt.figure(figsize=(8, 5))
plt.plot(losses, label="Weighted training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Training curve (dense, Ki-weighted)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves_dense.png", dpi=120)
plt.close()

# ---------------- Save weights ----------------
weights = {}
for name, param in model.named_parameters():
    safe_name = name.replace(".", "_")
    weights[safe_name] = param.detach().numpy().astype(np.float64)
sio.savemat("nn_weights.mat", weights)
print("Saved trained weights to nn_weights.mat")
print("Weight names:", list(weights.keys()))

print("\nAll done. Now run validate_dense.py to see exact surfaces and max |Δ|.")