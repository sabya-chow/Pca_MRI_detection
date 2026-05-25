"""3D classifier: SimCLR encoder + temporal aggregator + CORAL ordinal head."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CORAL ordinal head ────────────────────────────────────────────────────────

class CORALHead(nn.Module):
    """Consistent Rank Logits (CORAL) ordinal regression head.

    Outputs K-1 cumulative probability logits via shared weights + bias.
    Predicted class = number of thresholds where P > 0.5.
    """

    def __init__(self, in_features: int, num_classes: int = 5):
        super().__init__()
        self.fc   = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x) + self.bias)  # (B, K-1)


def coral_loss(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Binary CE over K-1 ordinal thresholds.

    preds  : (B, K-1) sigmoid probabilities
    targets: (B,) integer labels in [0, K-1]
    """
    K = num_classes
    ordinal = torch.zeros(len(targets), K - 1, device=preds.device)
    for i, t in enumerate(targets):
        ordinal[i, :int(t)] = 1.0
    return F.binary_cross_entropy(preds, ordinal)


def coral_predict(probs: torch.Tensor) -> torch.Tensor:
    """Convert cumulative probs → integer class labels."""
    return (probs > 0.5).sum(dim=1).long()


# ── Temporal aggregator ───────────────────────────────────────────────────────

class TemporalAggregator(nn.Module):
    """Aggregate a sequence of per-slice embeddings into a single vector.

    Architecture:
        1×D conv → ReLU → 1×D conv → GlobalAvgPool → LayerNorm → Linear
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 256):
        super().__init__()
        # Treat sequence of embeddings as 1-D signal over depth
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm  = nn.LayerNorm(hidden_dim)
        self.proj  = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, in_dim) → (B, out_dim)."""
        x = x.permute(0, 2, 1)          # (B, in_dim, D)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=-1)              # (B, hidden_dim) global avg over D
        x = self.norm(x)
        return self.proj(x)             # (B, out_dim)


# ── Full classifier: encoder + aggregator + CORAL ────────────────────────────

class ProstateCancerClassifier(nn.Module):
    """SimCLR 2D encoder applied slice-wise + temporal aggregator + CORAL head.

    Forward input: (B, C, D, H, W) where C=1 (T2 only) or C=3 (multimodal).
    Forward output: (B, K-1) cumulative ordinal probabilities.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 512,
        num_classes: int = 5,
        freeze_encoder: bool = False,
        in_channels: int = 1,
    ):
        super().__init__()
        self.encoder   = encoder
        self.aggregator = TemporalAggregator(in_dim=encoder_dim, hidden_dim=256, out_dim=256)
        self.coral     = CORALHead(in_features=256, num_classes=num_classes)
        self.dropout   = nn.Dropout(0.3)
        self.in_channels = in_channels

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        # Reshape to process each slice independently
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)  # (B*D, C, H, W)
        feats = self.encoder(x_2d)                                  # (B*D, 512)
        feats = feats.reshape(B, D, -1)                            # (B, D, 512)
        agg   = self.aggregator(feats)                              # (B, 256)
        agg   = self.dropout(agg)
        return self.coral(agg)                                       # (B, K-1)


# ── Lightweight 3D CNN fallback (no SimCLR, direct 3D conv) ──────────────────

class Direct3DCNN(nn.Module):
    """Compact 3-D CNN for when SimCLR pre-training is skipped."""

    def __init__(self, in_channels: int = 1, num_classes: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.dropout = nn.Dropout(0.4)
        self.coral   = CORALHead(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x).flatten(1)   # (B, 128)
        return self.coral(self.dropout(feat))
