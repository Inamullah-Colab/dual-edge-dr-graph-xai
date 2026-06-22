from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class JacobianMapper(nn.Module):
    """Differentiable mapper used for X3 -> X4 sensitivity analysis."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class X34JacobianBuilder:
    """Builds a true X3-X4 Jacobian representation without image dependencies.

    A differentiable mapper f: X3 -> X4 is fitted. For each sample, autograd computes
    J_i = df(X3_i) / dX3_i. The stored X34 vector contains reduced X3, reduced X4,
    Jacobian norm/alignment, output biomarker sensitivities, and input sensitivities.
    """

    def __init__(self, x3_dim: int = 128, x4_dim: int = 32, random_state: int = 42):
        self.x3_dim = x3_dim
        self.x4_dim = x4_dim
        self.random_state = random_state
        self.x3_scaler = StandardScaler()
        self.x4_scaler = StandardScaler()
        self.x3_pca: PCA | None = None
        self.x4_pca: PCA | None = None
        self.model: JacobianMapper | None = None

    def reduce(self, x3: np.ndarray, x4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x3z = self.x3_scaler.fit_transform(x3)
        x4z = self.x4_scaler.fit_transform(x4)
        self.x3_pca = PCA(n_components=min(self.x3_dim, x3z.shape[1]), random_state=self.random_state).fit(x3z)
        self.x4_pca = PCA(n_components=min(self.x4_dim, x4z.shape[1]), random_state=self.random_state).fit(x4z)
        return self.x3_pca.transform(x3z).astype(np.float32), self.x4_pca.transform(x4z).astype(np.float32)

    def fit_mapper(self, x3: np.ndarray, x4: np.ndarray, epochs: int = 80, batch_size: int = 256) -> JacobianMapper:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = JacobianMapper(x3.shape[1], x4.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        xt = torch.tensor(x3, dtype=torch.float32, device=device)
        yt = torch.tensor(x4, dtype=torch.float32, device=device)
        for _ in range(epochs):
            perm = torch.randperm(xt.shape[0], device=device)
            for start in range(0, xt.shape[0], batch_size):
                idx = perm[start:start + batch_size]
                loss = F.smooth_l1_loss(model(xt[idx]), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.model = model
        return model

    def jacobian_summary(self, x3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("fit_mapper must be called before jacobian_summary")
        device = next(self.model.parameters()).device
        self.model.eval()
        frob, diag, out_sens, in_sens = [], [], [], []
        for row in x3:
            x = torch.tensor(row[None, :], dtype=torch.float32, device=device, requires_grad=True)
            y = self.model(x)[0]
            grads = []
            for j in range(y.numel()):
                g = torch.autograd.grad(y[j], x, retain_graph=True)[0][0].detach().cpu().numpy()
                grads.append(g)
            j_mat = np.stack(grads).astype(np.float32)
            absj = np.abs(j_mat)
            d = min(j_mat.shape)
            frob.append(float(np.linalg.norm(j_mat, ord="fro")))
            diag.append(float(np.abs(np.diag(j_mat[:d, :d])).mean()))
            out_sens.append(absj.mean(axis=1))
            in_sens.append(absj.mean(axis=0))
        return np.asarray(frob), np.asarray(diag), np.vstack(out_sens), np.vstack(in_sens)
