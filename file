"""
Oil-Spill-Detection-using-SAR-Data-Deep-Learning (Synthetic Data)
------------------------------------------------------------------
- Generates synthetic SAR-like grayscale images with speckle noise.
- Injects "oil spills" as dark elliptical regions (reduced backscatter).
- Trains a lightweight U-Net for binary segmentation (spill vs. background).
- Evaluates with Dice and IoU, and saves sample predictions.
- Designed to run on CPU (works on GPU if available).

Dependencies:
    - numpy, matplotlib, torch, torchvision (optional for transforms), tqdm
Install (if needed):
    pip install numpy matplotlib torch torchvision tqdm
"""
import os
import math
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------
# Synthetic SAR generator
# ---------------------------
def rayleigh_speckle(shape: Tuple[int, int], scale: float = 0.6) -> np.ndarray:
    """Rayleigh-distributed noise commonly used to mimic SAR speckle."""
    return np.random.rayleigh(scale=scale, size=shape).astype(np.float32)


def draw_ellipse_mask(h: int, w: int, center=None, axes=None, angle=None) -> np.ndarray:
    """
    Create a boolean ellipse mask using the analytic ellipse equation.
    angle in radians, axes is (a,b).
    """
    if center is None:
        center = (np.random.randint(int(0.3*w), int(0.7*w)),
                  np.random.randint(int(0.3*h), int(0.7*h)))
    if axes is None:
        a = np.random.randint(int(0.08*w), int(0.2*w))
        b = np.random.randint(int(0.04*h), int(0.15*h))
        axes = (a, b)
    if angle is None:
        angle = np.deg2rad(np.random.uniform(0, 180.0))

    yy, xx = np.mgrid[0:h, 0:w]
    x0, y0 = center
    a, b = axes

    # Rotate coordinates
    cos_t, sin_t = np.cos(angle), np.sin(angle)
    x = (xx - x0) * cos_t + (yy - y0) * sin_t
    y = -(xx - x0) * sin_t + (yy - y0) * cos_t

    # Ellipse inequality
    mask = (x**2) / (a**2 + 1e-6) + (y**2) / (b**2 + 1e-6) <= 1.0
    return mask.astype(np.float32)


def generate_sar_sample(img_size: int = 96, spill_prob: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one synthetic SAR image and corresponding binary mask.
    - Background: Rayleigh speckle simulating sea clutter
    - Spill: dark ellipse (reduced backscatter due to capillary wave damping)
    """
    img = rayleigh_speckle((img_size, img_size), scale=np.random.uniform(0.45, 0.8))

    mask = np.zeros_like(img, dtype=np.float32)

    if np.random.rand() < spill_prob:
        # Optionally add 1–2 spill blobs to mimic fragmented slicks
        n_blobs = np.random.choice([1, 1, 2])
        for _ in range(n_blobs):
            e_mask = draw_ellipse_mask(img_size, img_size)
            mask = np.maximum(mask, e_mask)

        # Apply darkening inside spill (reduced backscatter)
        damp_factor = np.random.uniform(0.25, 0.45)  # lower is darker
        img = img * (1 - 0.85 * mask) + damp_factor * mask

        # Add mild blur-like effect by averaging with a shifted version
        img_shift = np.roll(img, shift=np.random.randint(-2, 3), axis=np.random.choice([0, 1]))
        img = 0.7 * img + 0.3 * img_shift

    # Normalize to [0, 1]
    img = img - img.min()
    img = img / (img.max() + 1e-8)

    return img.astype(np.float32), mask.astype(np.float32)


def make_dataset(n_samples: int = 1500, img_size: int = 96):
    """
    Create arrays of synthetic images and masks.
    Ensures at least >100 samples as requested by the user.
    """
    assert n_samples > 100, "Please request more than 100 samples."
    X, Y = [], []
    for _ in tqdm(range(n_samples), desc="Generating synthetic SAR dataset"):
        img, m = generate_sar_sample(img_size=img_size, spill_prob=np.random.uniform(0.4, 0.7))
        X.append(img)
        Y.append(m)
    X = np.stack(X)[..., None]  # (N, H, W, 1)
    Y = np.stack(Y)[..., None]  # (N, H, W, 1)
    return X, Y


# ---------------------------
# PyTorch Dataset
# ---------------------------
class SARDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].transpose(2, 0, 1)  # (1, H, W)
        y = self.Y[idx].transpose(2, 0, 1)  # (1, H, W)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


# ---------------------------
# Tiny U-Net model
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetTiny(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        bn = self.bottleneck(p3)

        u3 = self.up3(bn)
        u3 = torch.cat([u3, d3], dim=1)
        c3 = self.conv3(u3)

        u2 = self.up2(c3)
        u2 = torch.cat([u2, d2], dim=1)
        c2 = self.conv2(u2)

        u1 = self.up1(c2)
        u1 = torch.cat([u1, d1], dim=1)
        c1 = self.conv1(u1)

        logits = self.outc(c1)
        return logits


# ---------------------------
# Losses and metrics
# ---------------------------
def dice_coef(pred, target, eps=1e-6):
    # pred, target in {0,1}, dims: (N,1,H,W)
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    den = pred.sum(dim=1) + target.sum(dim=1)
    return ((2 * inter + eps) / (den + eps)).mean()


def iou_score(pred, target, eps=1e-6):
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean()


# ---------------------------
# Training
# ---------------------------
def train_unet():
    # Config
    img_size = 96
    n_samples = 1500   # > 100 samples as requested
    val_ratio = 0.15
    batch_size = 16
    epochs = 10
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    X, Y = make_dataset(n_samples=n_samples, img_size=img_size)

    # Train/val split
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    n_val = int(val_ratio * n_samples)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    ds_train = SARDataset(X_train, Y_train)
    ds_val = SARDataset(X_val, Y_val)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = UNetTiny(in_ch=1, out_ch=1, base=16).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_iou = 0.0
    os.makedirs("outputs", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(dl_train, desc=f"Epoch {epoch}/{epochs} [train]"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(ds_train)

        # Validation
        model.eval()
        with torch.no_grad():
            preds, gts = [], []
            val_loss = 0.0
            for xb, yb in tqdm(dl_val, desc=f"Epoch {epoch}/{epochs} [val]"):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                prob = torch.sigmoid(logits)
                pred = (prob > 0.5).float()
                preds.append(pred.cpu())
                gts.append(yb.cpu())
            preds = torch.cat(preds, dim=0)
            gts = torch.cat(gts, dim=0)
            dice = dice_coef(preds, gts).item()
            iou = iou_score(preds, gts).item()

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  val_loss={val_loss/len(ds_val):.4f}  Dice={dice:.4f}  IoU={iou:.4f}")

        # Save best
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), os.path.join("outputs", "best_unet_tiny.pt"))

    # Save a few qualitative results
    model.eval()
    n_show = 6
    xs, ys, ps = [], [], []
    with torch.no_grad():
        for i in range(n_show):
            x = torch.from_numpy(X_val[i].transpose(2,0,1)).unsqueeze(0).float().to(device)
            y = torch.from_numpy(Y_val[i].transpose(2,0,1)).unsqueeze(0).float().to(device)
            pr = torch.sigmoid(model(x))
            xs.append(x.cpu().numpy()[0,0])
            ys.append(y.cpu().numpy()[0,0])
            ps.append((pr.cpu().numpy()[0,0] > 0.5).astype(np.float32))

    fig_h = 3 * n_show
    fig, axes = plt.subplots(n_show, 3, figsize=(9, fig_h))
    for i in range(n_show):
        axes[i,0].imshow(xs[i], cmap='gray', vmin=0, vmax=1)
        axes[i,0].set_title("SAR (synthetic)")
        axes[i,1].imshow(ys[i], cmap='gray', vmin=0, vmax=1)
        axes[i,1].set_title("Mask (GT)")
        axes[i,2].imshow(ps[i], cmap='gray', vmin=0, vmax=1)
        axes[i,2].set_title("Pred (U-Net tiny)")
        for j in range(3):
            axes[i,j].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join("outputs", "qualitative_results.png"), dpi=160)
    plt.close(fig)

    print("Training complete.")
    print("Best model saved to: outputs/best_unet_tiny.pt")
    print("Sample predictions saved to: outputs/qualitative_results.png")


if __name__ == "__main__":
    train_unet()
