"""SimCLR self-supervised contrastive pre-training (2-D slice level)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ── Encoder ───────────────────────────────────────────────────────────────────

class SimCLREncoder(nn.Module):
    """ResNet-18 backbone adapted for 1-channel medical images.

    Replaces the first conv (3-ch) with a 1-channel version.
    Output: global-average-pooled 512-d feature vector.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        base = resnet18(weights=None if not pretrained else "IMAGENET1K_V1")
        # Adapt first conv for 1-channel input
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove classification head
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # → (B, 512, 1, 1)
        self.out_dim  = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)   # (B, 512)


# ── Projection head ───────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """2-layer MLP with BN: 512 → 256 → 128."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)  # L2-normalised


# ── Full SimCLR model ─────────────────────────────────────────────────────────

class SimCLR(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.encoder    = SimCLREncoder(pretrained=pretrained)
        self.projector  = ProjectionHead(in_dim=self.encoder.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ── NT-Xent loss ──────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """Normalised temperature-scaled cross-entropy (SimCLR contrastive loss)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z1, z2: (B, D) L2-normalised projections."""
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)                    # (2B, D)
        sim = (z @ z.T) / self.tau                         # (2B, 2B)

        # Mask diagonal (self-similarity)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        targets = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
        loss = F.cross_entropy(sim, targets)
        return loss


# ── Training utilities ────────────────────────────────────────────────────────

def train_simclr_epoch(
    model: SimCLR,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: NTXentLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for v1, v2 in loader:
        v1, v2 = v1.to(device), v2.to(device)
        z1 = model(v1)
        z2 = model(v2)
        loss = criterion(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def extract_slice_embeddings(
    model: SimCLR,
    patient_ids: list,
    dicom_root,
    device: torch.device,
    target_h: int = 128,
    target_w: int = 128,
) -> dict:
    """Run encoder over all T2 slices for each patient.

    Returns {patient_id: np.ndarray (D, 512)}.
    """
    import numpy as np
    import cv2
    import pydicom
    from src.dataset import load_volume, TARGET_D

    model.eval()
    embeddings = {}

    with torch.no_grad():
        for pid in patient_ids:
            pt_str = f"{pid:03d}"
            from src.dataset import load_volume
            vol = load_volume(pt_str, "AX_T2", dicom_root)
            if vol is None:
                embeddings[pid] = np.zeros((TARGET_D, 512), dtype=np.float32)
                continue

            slices = []
            for sl in vol:
                sl_r = cv2.resize(sl, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                mu, sigma = sl_r.mean(), sl_r.std() + 1e-6
                sl_r = (sl_r - mu) / sigma
                slices.append(sl_r)

            # Pad/crop depth
            D = TARGET_D
            if len(slices) >= D:
                slices = slices[(len(slices)-D)//2: (len(slices)-D)//2 + D]
            else:
                slices = slices + [np.zeros_like(slices[0])] * (D - len(slices))

            batch = torch.tensor(np.stack(slices)[:, None], dtype=torch.float32).to(device)  # (D, 1, H, W)
            feats = model.encode(batch).cpu().numpy()    # (D, 512)
            embeddings[pid] = feats

    return embeddings
