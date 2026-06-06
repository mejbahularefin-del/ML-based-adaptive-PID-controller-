"""compare_with_matlab.py — runs the same test inputs through PyTorch
   so you can verify they match the MATLAB output."""

import numpy as np, torch, torch.nn as nn, scipy.io as sio

class FuzzyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)
    def forward(self, x):
        return torch.sigmoid(self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x))))))

w = sio.loadmat("nn_weights.mat")
model = FuzzyNet()
with torch.no_grad():
    model.fc1.weight.copy_(torch.from_numpy(w["fc1_weight"]).float())
    model.fc1.bias  .copy_(torch.from_numpy(w["fc1_bias"]).float().squeeze())
    model.fc2.weight.copy_(torch.from_numpy(w["fc2_weight"]).float())
    model.fc2.bias  .copy_(torch.from_numpy(w["fc2_bias"]).float().squeeze())
    model.fc3.weight.copy_(torch.from_numpy(w["fc3_weight"]).float())
    model.fc3.bias  .copy_(torch.from_numpy(w["fc3_bias"]).float().squeeze())
model.eval()

tests = np.array([[0,0],[0.5,0],[-0.5,0.3],[0.8,-0.6],[0,-0.5],[-0.3,0.4]],
                 dtype=np.float32)
with torch.no_grad():
    pred = model(torch.from_numpy(tests)).numpy()

print("Python neural network forward pass:")
print(f"{'Input':<22}  {'Kp':<10}  {'Ki':<10}  {'Kd':<10}")
print("-" * 60)
for inp, out in zip(tests, pred):
    print(f"({inp[0]:+.2f}, {inp[1]:+.2f})            "
          f"{out[0]:.6f}    {out[1]:.6f}    {out[2]:.6f}")