"""
Self-supervised learning components for vibration spectrograms.

Implements a SimSiam-style pretraining pipeline using the ResNetVibCNN encoder.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrogramAugmentations(nn.Module):
    """
    Stochastic augmentations for vibration spectrograms of shape (B, C, F, T).
    """

    def __init__(
        self,
        freq_mask_param: int = 24,
        time_mask_param: int = 3,
        noise_std: float = 0.01,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.noise_std = noise_std
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, C, F, T = x.shape

        # Frequency masking
        if torch.rand(1).item() < self.p:
            f = torch.randint(1, self.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(1, F - f), (1,)).item()
            x = x.clone()
            x[:, :, f0 : f0 + f, :] = 0.0

        # Time masking
        if torch.rand(1).item() < self.p:
            t = torch.randint(1, self.time_mask_param + 1, (1,)).item()
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            x = x.clone()
            x[:, :, :, t0 : t0 + t] = 0.0

        # Gaussian noise
        if torch.rand(1).item() < self.p:
            x = x + torch.randn_like(x) * self.noise_std

        return x


class SimSiamVib(nn.Module):
    """
    SimSiam network for self-supervised pretraining of the ResNetVibCNN encoder.

    Architecture:
        encoder -> projector -> predictor
        Loss: negative cosine similarity between predictor(z1) and stop_grad(projector(z2))
    """

    def __init__(
        self,
        encoder: nn.Module,
        proj_dim: int = 128,
        pred_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = encoder

        # Determine encoder output dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 257, 13)
            feat = self.encoder(dummy)
            encoder_dim = feat.shape[-1]

        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim, proj_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = (
            self._negative_cosine_similarity(p1, z2.detach())
            + self._negative_cosine_similarity(p2, z1.detach())
        ) / 2.0
        return loss, z1, z2

    @staticmethod
    def _negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()


def build_ssl_encoder(dropout: float = 0.3, hidden_dim: int = 64) -> nn.Module:
    """
    Build the encoder part of ResNetVibCNN as a standalone feature extractor.
    """
    from grinding_physic_fusion.models.architectures import ResNetVibCNN

    model = ResNetVibCNN(dropout=dropout, hidden_dim=hidden_dim)

    class Encoder(nn.Module):
        def __init__(self, base: ResNetVibCNN) -> None:
            super().__init__()
            self.stem = base.stem
            self.res1 = base.res1
            self.res2 = base.res2
            self.gap = base.gap
            self.fc = base.fc

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.stem(x)
            x = self.res1(x)
            x = self.res2(x)
            x = self.gap(x)
            x = self.fc(x)
            return x

    return Encoder(model)
