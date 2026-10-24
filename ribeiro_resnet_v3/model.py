"""1-D ResNet (Ribeiro-style) for 3-class mitral valve classification from a 12-lead signal, plus the input normalisation and the 5-fold ensemble loader.

Classes: 0 = Control, 1 = MR (significant mitral regurgitation), 2 = MS (significant mitral stenosis).
Input: (B, 12, 5000), the signal of one 10 s ECG at 500 Hz, lead order I, II, III, aVR, aVL, aVF, V1-V6, normalised with `normalise`.
The interval regression head (`n_reg=6`) exists only because the checkpoints were trained with it as an auxiliary task; `forward` ignores it unless `aux=True`.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ("Control", "MR", "MS")


class ResidualBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=1, k=17, dropout=0.0):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=p, bias=False); self.bn1 = nn.BatchNorm1d(cout); self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(cout, cout, k, padding=p, bias=False); self.bn2 = nn.BatchNorm1d(cout)
        self.down = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout)) if stride != 1 or cin != cout else None

    def forward(self, x):
        o = self.act(self.bn1(self.conv1(x))); o = self.bn2(self.conv2(self.drop(o)))
        return self.act(o + (self.down(x) if self.down else x))


class PlainRibeiro(nn.Module):
    """Stem (64 channels, kernel 17), 8 residual blocks (64 -> 256 channels), global average pooling, dropout 0.3, linear class head. 7.9 M parameters."""

    def __init__(self, cin=12, ncls=3, dropout=0.30, n_reg=0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(cin, 64, 17, padding=8, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            ResidualBlock1D(64, 64, 1, 17, 0.00), ResidualBlock1D(64, 64, 1, 17, 0.00),
            ResidualBlock1D(64, 128, 2, 17, 0.05), ResidualBlock1D(128, 128, 1, 17, 0.05),
            ResidualBlock1D(128, 196, 2, 17, 0.10), ResidualBlock1D(196, 196, 1, 17, 0.10),
            ResidualBlock1D(196, 256, 2, 17, 0.15), ResidualBlock1D(256, 256, 1, 17, 0.15))
        self.pool = nn.AdaptiveAvgPool1d(1); self.drop = nn.Dropout(dropout); self.head = nn.Linear(256, ncls)
        self.reg_head = nn.Linear(256, n_reg) if n_reg else None

    def forward(self, x, aux=False):
        z = self.pool(self.blocks(self.stem(x))).squeeze(-1); out = self.head(self.drop(z))
        return (out, self.reg_head(z)) if aux else out


def normalise(sig):
    """(N, 5000, 12) signal with NaN outside each lead's own window -> (N, 12, 5000) float32.
    Each lead is centred and scaled by the mean and standard deviation of its finite samples; missing samples become 0."""
    x = np.asarray(sig, np.float32); fin = np.isfinite(x); vc = np.maximum(fin.sum(1, keepdims=True).astype(np.float32), 1.0)
    mu = np.where(fin, x, 0.0).sum(1, keepdims=True) / vc; c = np.where(fin, x - mu, 0.0)
    std = np.sqrt(np.maximum((c ** 2).sum(1, keepdims=True) / vc, 1e-6)); std = np.where(fin.sum(1, keepdims=True) == 0, 1.0, std)
    return np.transpose(c / std, (0, 2, 1)).astype(np.float32)


def load_ensemble(weights_dir, device="cpu"):
    """The five fold networks stored in `weights_dir` (fold0.pt ... fold4.pt, fp16 weights of the weight-averaged (EMA) model at epoch 20)."""
    nets = []
    for f in range(5):
        sd = torch.load(Path(weights_dir) / f"fold{f}.pt", map_location="cpu", weights_only=True)
        sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
        m = PlainRibeiro(n_reg=6); m.load_state_dict(sd, strict=True); nets.append(m.to(device).eval())
    return nets


@torch.no_grad()
def predict_proba(nets, x):
    """Mean of the softmax outputs of the fold networks. x: (B, 12, 5000) float tensor on the same device. No weights, no thresholds: the class is the argmax."""
    return torch.stack([F.softmax(m(x).float(), 1) for m in nets]).mean(0).cpu().numpy()
