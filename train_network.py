"""
train_network.py (v3 - aggressive fit)
--------------------------------------
Trains a larger neural network that maps (error, d_error) -> (Kp, Ki, Kd)
and uses ALL samples for training to match the Mamdani surfaces as closely
as possible.
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

# ---------------- Load data (all points used for training) ----------------
print("Loading data...")
data = pd.read_csv("training_data.csv", header=None).values
print(f"Loaded {data.shape[0]} samples, each with {data.shape[1]} columns.")

X_np = data[:, 0:2].astype(np.float32)   # error, d_error
y_np = data[:, 2:5].astype(np.float32)   # Kp, Ki, Kd

X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)

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
        x = torch.sigmoid(self.fc3(x))   # keep outputs in [0, 1]
        return x

model = FuzzyNet()
print("\nNetwork architecture:")
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
# Reduce LR as training progresses for fine-tuning
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.3)

# ---------------- Train ----------------
EPOCHS = 8000
train_losses = []

print(f"\nTraining for {EPOCHS} epochs on ALL data (no hold-out set)...")
for epoch in range(EPOCHS):
    model.train()
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    train_losses.append(loss.item())

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1:5d}/{EPOCHS}  train MSE = {loss.item():.8f}")

print("Training finished.")

# ---------------- Plot training curve ----------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.yscale("log")
plt.title("Training curve (2-128-128-3 network, all data)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=120)
plt.close()

# ---------------- Save weights ----------------
weights = {}
for name, param in model.named_parameters():
    safe_name = name.replace(".", "_")
    weights[safe_name] = param.detach().numpy().astype(np.float64)
sio.savemat("nn_weights.mat", weights)
print("Saved trained weights to nn_weights.mat")
print("Weight names:", list(weights.keys()))

# ---------------- Surface plots ----------------
N = 41
e_axis  = np.linspace(-1, 1, N)
de_axis = np.linspace(-1, 1, N)
EE, DEE = np.meshgrid(e_axis, de_axis)
grid_inputs = np.column_stack([EE.ravel(), DEE.ravel()]).astype(np.float32)

model.eval()
with torch.no_grad():
    grid_outputs = model(torch.from_numpy(grid_inputs)).numpy()

Kp_surf = grid_outputs[:, 0].reshape(N, N)
Ki_surf = grid_outputs[:, 1].reshape(N, N)
Kd_surf = grid_outputs[:, 2].reshape(N, N)

fig = plt.figure(figsize=(15, 4))
for i, (Z, name) in enumerate(
        [(Kp_surf, "Kp"), (Ki_surf, "Ki"), (Kd_surf, "Kd")]):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.plot_surface(EE, DEE, Z, cmap="viridis")
    ax.set_xlabel("error")
    ax.set_ylabel("d_error")
    ax.set_zlabel(name)
    ax.set_title(f"NN learned {name}")
plt.tight_layout()
plt.savefig("nn_surfaces.png", dpi=120)
plt.close()

print("\nAll done. Now run validate.py to compare with the Mamdani surfaces.")