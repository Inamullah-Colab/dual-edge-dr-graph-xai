from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTokenAttentionClassifier(nn.Module):
    """Semantic attention over two modality tokens: X12 and X34."""

    def __init__(self, x12_dim: int, x34_dim: int, hidden_dim: int = 192, n_classes: int = 5):
        super().__init__()
        self.x12_proj = nn.Sequential(nn.Linear(x12_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.15))
        self.x34_proj = nn.Sequential(nn.Linear(x34_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.15))
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x12: torch.Tensor, x34: torch.Tensor):
        t12 = self.x12_proj(x12)
        t34 = self.x34_proj(x34)
        tokens = torch.stack([t12, t34], dim=1)
        attn = F.softmax(self.score(tokens).squeeze(-1), dim=1)
        fused = torch.sum(tokens * attn.unsqueeze(-1), dim=1)
        return self.classifier(fused), attn, fused


class NumpyTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x12: np.ndarray, x34: np.ndarray, y: np.ndarray):
        self.x12 = torch.tensor(x12, dtype=torch.float32)
        self.x34 = torch.tensor(x34, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x12[idx], self.x34[idx], self.y[idx]
