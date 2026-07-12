"""
Advanced Model Zoo for Grinding Surface Roughness Prediction.

Supports multi-modal fusion of:
- AE spectrograms:    (B, 2, 300, 47)
- Vib spectrograms:   (B, 3, 257, 13)
- Physics features:   (B, 44)  (aggregated: 11 features x 4 stats)
- Physics time-series:(B, 11, T)  (variable length, padded)
- Process parameters: (B, 3)  (wheel_speed, workpiece_speed, grinding_depth)
- BDI/ST:             (B, 2)  (mean_bdi, mean_st)

All PyTorch models return predictions of shape (B,) via forward().
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    import torchvision.models as tvmodels
except ImportError:  # pragma: no cover
    tvmodels = None  # type: ignore

# ---------------------------------------------------------------------------
# Optional third-party ML wrappers
# ---------------------------------------------------------------------------

_MISSING_SKLEARN = False
_MISSING_XGB = False
_MISSING_LGBM = False

try:
    from sklearn.ensemble import RandomForestRegressor  # noqa: F401
except Exception:  # pragma: no cover
    _MISSING_SKLEARN = True

try:
    import xgboost as xgb  # noqa: F401
except Exception:  # pragma: no cover
    _MISSING_XGB = True

try:
    import lightgbm as lgb  # noqa: F401
except Exception:  # pragma: no cover
    _MISSING_LGBM = True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sparsemax (for TabNet)
# ---------------------------------------------------------------------------

class Sparsemax(nn.Module):
    """Sparsemax activation: a sparse alternative to softmax."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.view(-1, x.shape[self.dim])
        sorted_x, _ = torch.sort(x, dim=1, descending=True)
        cumsum = torch.cumsum(sorted_x, dim=1)
        k = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1)
        condition = sorted_x - (cumsum - 1) / k > 0
        # Find largest k satisfying condition
        support = condition.sum(dim=1)
        tau = (cumsum[torch.arange(x.size(0)), support - 1] - 1) / support
        output = torch.clamp(x - tau.unsqueeze(1), min=0.0)
        return output.view(original_shape)


# ---------------------------------------------------------------------------
# SE Block (Squeeze-and-Excitation)
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ---------------------------------------------------------------------------
# CBAM Block (Channel + Spatial Attention)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 4, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

    def get_attention_weights(self) -> dict[str, torch.Tensor | None]:
        """Return None placeholder; attention maps can be extracted via hooks if needed."""
        return {"channel": None, "spatial": None}


# ---------------------------------------------------------------------------
# Residual Block with SE
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block with BatchNorm, optional SE, and dropout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        use_se: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


# ---------------------------------------------------------------------------
# TCN Components
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    """Chomp padding for causal convolutions."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single TCN residual block with dilated causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# ---------------------------------------------------------------------------
# TabNet Components
# ---------------------------------------------------------------------------

class GLUBlock(nn.Module):
    """Gated Linear Unit block for TabNet feature transformer."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x[:, : x.size(1) // 2] * torch.sigmoid(x[:, x.size(1) // 2 :])


class FeatureTransformerBlock(nn.Module):
    """A single feature transformer block with private decision steps."""

    def __init__(self, in_dim: int, out_dim: int, n_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = in_dim
        for _ in range(n_layers):
            self.layers.append(GLUBlock(current_dim, out_dim))
            current_dim = out_dim
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.bn(x)


class TabNetEncoder(nn.Module):
    """Simplified TabNet encoder with attention masking over features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 3,
        gamma: float = 1.3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        self.feature_transformers = nn.ModuleList()
        self.attention_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.feature_transformers.append(FeatureTransformerBlock(input_dim, n_d + n_a, n_layers=4))
            self.attention_transformers.append(nn.Sequential(nn.Linear(n_d, input_dim), Sparsemax(dim=-1)))

        self.bn = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(n_d, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.bn(x)
        batch_size = x.size(0)
        prior = torch.ones(batch_size, self.input_dim, device=x.device)
        d = torch.zeros(batch_size, self.n_d, device=x.device)

        attention_masks = []
        for step in range(self.n_steps):
            attn = self.attention_transformers[step](d)
            M = prior * attn
            attention_masks.append(M)

            masked_x = M * x
            out = self.feature_transformers[step](masked_x)
            d, a = out[:, : self.n_d], out[:, self.n_d :]
            prior = prior * (self.gamma - M)

        d = self.dropout(d)
        return self.fc_out(d), torch.stack(attention_masks, dim=1)


# ---------------------------------------------------------------------------
# Traditional ML Wrappers
# ---------------------------------------------------------------------------

class _SklearnWrapper(nn.Module):
    """Base wrapper to give sklearn/xgboost/lightgbm models a torch-like interface."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Any = None
        self.fitted = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            warnings.warn("Model has not been fitted yet. Returning zeros.")
            return torch.zeros(x.size(0), device=x.device)
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
        preds = self.predict(x_np)
        return torch.tensor(preds, dtype=torch.float32, device=x.device).squeeze()

    def get_attention_weights(self) -> None:
        return None


class RandomForestModel(_SklearnWrapper):
    """Random Forest regressor wrapper for physics features + process params."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 8, n_jobs: int = 1, random_state: int = 42, **kwargs: Any) -> None:
        super().__init__()
        if _MISSING_SKLEARN:
            raise ImportError("scikit-learn is required for RandomForestModel")
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, random_state=random_state, **kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)
        self.fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


class XGBoostModel(_SklearnWrapper):
    """XGBoost regressor wrapper for physics features + process params."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if _MISSING_XGB:
            raise ImportError("xgboost is required for XGBoostModel")
        import xgboost as xgb

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)
        self.fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


class LightGBMModel(_SklearnWrapper):
    """LightGBM regressor wrapper for physics features + process params."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbosity: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if _MISSING_LGBM:
            raise ImportError("lightgbm is required for LightGBMModel")
        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbosity=verbosity,
            **kwargs,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)
        self.fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


class RidgeRegressionModel(_SklearnWrapper):
    """Ridge-regression wrapper for flattened spectrogram / tabular inputs."""

    def __init__(self, alpha: float = 1.0, random_state: int = 42, **kwargs: Any) -> None:
        super().__init__()
        if _MISSING_SKLEARN:
            raise ImportError("scikit-learn is required for RidgeRegressionModel")
        from sklearn.linear_model import Ridge

        self.model = Ridge(alpha=alpha, random_state=random_state, **kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)
        self.fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


class ShallowMLPModel(_SklearnWrapper):
    """Small feed-forward MLP baseline for flattened inputs."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 500,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if _MISSING_SKLEARN:
            raise ImportError("scikit-learn is required for ShallowMLPModel")
        from sklearn.neural_network import MLPRegressor

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)
        self.fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


# ---------------------------------------------------------------------------
# ResNet-style CNN Encoders (shared building blocks)
# ---------------------------------------------------------------------------

class ResNetAECNN(nn.Module):
    """
    ResNet-style CNN for AE spectrogram (B, 2, 300, 47).
    Uses 2 residual blocks with BatchNorm, SE blocks, and global average pooling.
    """

    def __init__(self, dropout: float = 0.3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(16, 32, stride=2, use_se=True, dropout=dropout)
        self.res2 = ResidualBlock(32, 64, stride=2, use_se=True, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Linear(hidden_dim, 1)
        self.physics_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 44, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.gap(x)
        feat = self.fc(x)
        if physics is not None:
            feat = torch.cat([feat, physics], dim=-1)
            x = self.physics_fusion(feat)
        else:
            x = self.regressor(feat)
        return x.squeeze(-1)

    def get_attention_weights(self) -> None:
        return None


class ResNetVibCNN(nn.Module):
    """
    ResNet-style CNN for vibration spectrogram (B, 3, 257, 13).
    Uses 2 residual blocks with BatchNorm, SE blocks, and global average pooling.
    """

    def __init__(self, dropout: float = 0.3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(16, 32, stride=2, use_se=True, dropout=dropout)
        self.res2 = ResidualBlock(32, 64, stride=2, use_se=True, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Linear(hidden_dim, 1)
        self.physics_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 44, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.gap(x)
        feat = self.fc(x)
        if physics is not None:
            feat = torch.cat([feat, physics], dim=-1)
            x = self.physics_fusion(feat)
        else:
            x = self.regressor(feat)
        return x.squeeze(-1)

    def get_attention_weights(self) -> None:
        return None


class ResNetFusion(nn.Module):
    """
    ResNet-based multi-modal fusion of AE and VIB spectrograms.
    Uses the same ResNet-style encoders as ResNetAECNN and ResNetVibCNN,
    concatenates their embeddings, and regresses surface roughness.
    Input shapes: ae_spec (B, 2, 300, 47), vib_spec (B, 3, 257, 13).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.physics_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2 + 44, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ae_spec is None and vib_spec is None:
            raise ValueError("At least one of ae_spec or vib_spec must be provided.")
        embeddings = []
        if ae_spec is not None:
            embeddings.append(self.ae_enc(ae_spec))
        else:
            embeddings.append(torch.zeros(vib_spec.size(0), 64, device=vib_spec.device))
        if vib_spec is not None:
            embeddings.append(self.vib_enc(vib_spec))
        else:
            embeddings.append(torch.zeros(ae_spec.size(0), 64, device=ae_spec.device))
        fused = torch.cat(embeddings, dim=-1)
        if physics is not None:
            fused = torch.cat([fused, physics], dim=-1)
            out = self.physics_fusion(fused)
        else:
            out = self.fusion_mlp(fused)
        return out.squeeze(-1)

    def get_attention_weights(self) -> None:
        return None


# ---------------------------------------------------------------------------
# TabNet Regressor
# ---------------------------------------------------------------------------

class TabNetRegressor(nn.Module):
    """
    Attention-based tabular deep learning for variable-dim tabular inputs.
    Lazily initializes encoder on first forward to match actual input dimension.
    """

    def __init__(
        self,
        input_dim: int = 47,
        n_d: int = 24,
        n_a: int = 24,
        n_steps: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.dropout = dropout
        self.encoder: nn.Module | None = None
        self.regressor: nn.Module | None = None
        self._attention_masks: torch.Tensor | None = None

    def _build(self, dim: int) -> None:
        self.encoder = TabNetEncoder(
            input_dim=dim,
            output_dim=32,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            dropout=self.dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(16, 1),
        )
        device = next(self.parameters(), torch.tensor(0)).device
        self.encoder = self.encoder.to(device)
        self.regressor = self.regressor.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is None or x.size(-1) != self.input_dim:
            self._build(x.size(-1))
            self.input_dim = x.size(-1)
        feat, masks = self.encoder(x)
        self._attention_masks = masks
        out = self.regressor(feat)
        return out.squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return attention masks of shape (B, n_steps, input_dim)."""
        return self._attention_masks


# ---------------------------------------------------------------------------
# LSTM Physics Model
# ---------------------------------------------------------------------------

class LSTMPhysicsModel(nn.Module):
    """
    Bidirectional LSTM on raw physics time-series (B, 11, T).
    Uses pack_padded_sequence for variable-length sequences.
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 48,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 11, T) or (B, T, 11). We expect (B, 11, T) and transpose internally.
            lengths: (B,) actual sequence lengths before padding.
        Returns:
            (B,) predictions.
        """
        if x.dim() == 2:
            # Fallback: treat as flat feature vector
            x = x.unsqueeze(1)
        if x.size(1) == 11 and x.size(2) != 11:
            x = x.transpose(1, 2)  # -> (B, T, 11)

        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (hn, _) = self.lstm(packed)
            # hn: (num_layers*2, B, hidden)
            # Take final forward and backward hidden states from last layer
            forward = hn[-2, :, :]
            backward = hn[-1, :, :]
            hidden = torch.cat([forward, backward], dim=1)
        else:
            out, _ = self.lstm(x)  # (B, T, hidden*2)
            hidden = out[:, -1, :]  # last timestep

        out = self.fc(hidden)
        return out.squeeze(-1)

    def get_attention_weights(self) -> None:
        return None


# ---------------------------------------------------------------------------
# TCN Physics Model
# ---------------------------------------------------------------------------

class TCNPhysicsModel(nn.Module):
    """
    Temporal Convolutional Network on physics time-series (B, 11, T).
    Uses dilated causal convolutions with residual connections.
    """

    def __init__(
        self,
        input_size: int = 11,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_channels is None:
            num_channels = [32, 48, 48]
        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 11, T) or (B, T, 11). Expects (B, 11, T) as channels-first.
        Returns:
            (B,) predictions.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.size(1) != 11 and x.size(2) == 11:
            x = x.transpose(1, 2)  # ensure (B, 11, T)
        out = self.network(x)
        out = self.gap(out).squeeze(-1)
        out = self.fc(out)
        return out.squeeze(-1)

    def get_attention_weights(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Modality Encoders (for fusion models)
# ---------------------------------------------------------------------------

class _ModalityAECNN(nn.Module):
    """AE spectrogram encoder -> feature vector."""

    def __init__(self, output_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(16, 32, stride=2, use_se=True, dropout=dropout)
        self.res2 = ResidualBlock(32, 48, stride=2, use_se=True, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.gap(x)
        return self.fc(x)


class _ModalityVibCNN(nn.Module):
    """Vibration spectrogram encoder -> feature vector."""

    def __init__(self, output_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(16, 32, stride=2, use_se=True, dropout=dropout)
        self.res2 = ResidualBlock(32, 48, stride=2, use_se=True, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.gap(x)
        return self.fc(x)


class _ModalityPhysics(nn.Module):
    """Physics feature encoder -> feature vector (simplified TabNet-style)."""

    def __init__(self, input_dim: int = 44, output_dim: int = 48, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = TabNetEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=24,
            n_a=24,
            n_steps=2,
            gamma=1.3,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, _ = self.encoder(x)
        return feat


class _ModalityParams(nn.Module):
    """Process params encoder -> feature vector."""

    def __init__(self, output_dim: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Cross-Modal Transformer
# ---------------------------------------------------------------------------

class CrossModalTransformer(nn.Module):
    """
    Multi-modal fusion using a Transformer encoder over stacked modality vectors.
    - AE spectrogram -> ResNet CNN -> vector
    - Vib spectrogram -> ResNet CNN -> vector
    - Physics features -> TabNet-style -> vector
    - Params -> MLP -> vector
    Stack as (B, 4, D), apply Transformer, pool, regress.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._attn_weights: torch.Tensor | None = None

    def forward(
        self,
        ae_spec: torch.Tensor,
        vib_spec: torch.Tensor,
        physics: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        ae_feat = self.ae_enc(ae_spec)      # (B, D)
        vib_feat = self.vib_enc(vib_spec)   # (B, D)
        phy_feat = self.physics_enc(physics)  # (B, D)
        par_feat = self.params_enc(params)    # (B, D)

        seq = torch.stack([ae_feat, vib_feat, phy_feat, par_feat], dim=1)  # (B, 4, D)
        out = self.transformer(seq)  # (B, 4, D)

        # Store attention weights via hook if needed; for now store pooled representation
        pooled = self.pool(out.transpose(1, 2)).squeeze(-1)  # (B, D)
        pred = self.regressor(pooled)
        return pred.squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Placeholder: Transformer attention maps require forward hooks for extraction."""
        return self._attn_weights


# ---------------------------------------------------------------------------
# Physics-Informed Fusion Net
# ---------------------------------------------------------------------------

class PhysicsInformedFusionNet(nn.Module):
    """
    Physics-informed multi-task network.
    Same encoders as CrossModalTransformer, but adds auxiliary predictions for BDI and ST.
    Main output: surface roughness.
    Auxiliary outputs: BDI, ST (brittle-ductile index and scratch threshold).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        shared_dim = embed_dim
        self.roughness_head = nn.Sequential(
            nn.Linear(shared_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.bdi_head = nn.Sequential(
            nn.Linear(shared_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        self.st_head = nn.Sequential(
            nn.Linear(shared_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        ae_spec: torch.Tensor,
        vib_spec: torch.Tensor,
        physics: torch.Tensor,
        params: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ae_feat = self.ae_enc(ae_spec)
        vib_feat = self.vib_enc(vib_spec)
        phy_feat = self.physics_enc(physics)
        par_feat = self.params_enc(params)

        seq = torch.stack([ae_feat, vib_feat, phy_feat, par_feat], dim=1)
        out = self.transformer(seq)
        pooled = self.pool(out.transpose(1, 2)).squeeze(-1)

        roughness = self.roughness_head(pooled).squeeze(-1)
        bdi = self.bdi_head(pooled).squeeze(-1)
        st = self.st_head(pooled).squeeze(-1)
        return {
            "roughness": roughness,
            "bdi": bdi,
            "st": st,
        }

    def predict_roughness(
        self,
        ae_spec: torch.Tensor,
        vib_spec: torch.Tensor,
        physics: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method returning only roughness predictions of shape (B,)."""
        return self.forward(ae_spec, vib_spec, physics, params)["roughness"]

    def get_attention_weights(self) -> torch.Tensor | None:
        return None


# ---------------------------------------------------------------------------
# Self-Supervised Pretrained CNN
# ---------------------------------------------------------------------------

class SelfSupervisedPretrainedCNN(nn.Module):
    """
    Architecture supporting contrastive pre-training (SimCLR-style).
    Contains a CNN trunk, a projector head for contrastive learning,
    and a predictor head for downstream regression.
    Supports both AE and Vib spectrograms via modality flag.
    """

    def __init__(
        self,
        modality: str = "ae",  # "ae" or "vib"
        embed_dim: int = 64,
        proj_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.modality = modality
        if modality == "ae":
            self.trunk = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
            in_ch = 2
        else:
            self.trunk = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
            in_ch = 3

        # Projector for contrastive learning (SimCLR)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

        # Predictor for downstream regression
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, mode: str = "predict") -> torch.Tensor:
        """
        Args:
            x: spectrogram input.
            mode: "project" returns contrastive projection (B, proj_dim);
                  "predict" returns regression output (B,).
        """
        feat = self.trunk(x)
        if mode == "project":
            return self.projector(feat)
        out = self.predictor(feat)
        return out.squeeze(-1)

    def get_attention_weights(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Gated Multi-modal Fusion Net
# ---------------------------------------------------------------------------

class GatedMultimodalFusionNet(nn.Module):
    """
    Gated fusion mechanism where each modality has a learned gate.
    Gates are computed from concatenated modality features.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        total_dim = embed_dim * 4
        # Gating network: outputs 4 sigmoid gates
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, 4),
            nn.Sigmoid(),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._gates: torch.Tensor | None = None

    def forward(
        self,
        ae_spec: torch.Tensor,
        vib_spec: torch.Tensor,
        physics: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        ae_feat = self.ae_enc(ae_spec)
        vib_feat = self.vib_enc(vib_spec)
        phy_feat = self.physics_enc(physics)
        par_feat = self.params_enc(params)

        concat = torch.cat([ae_feat, vib_feat, phy_feat, par_feat], dim=-1)
        gates = self.gate_net(concat)  # (B, 4)
        self._gates = gates

        # Apply gates per modality
        ae_g = ae_feat * gates[:, 0:1]
        vib_g = vib_feat * gates[:, 1:2]
        phy_g = phy_feat * gates[:, 2:3]
        par_g = par_feat * gates[:, 3:4]

        fused = torch.cat([ae_g, vib_g, phy_g, par_g], dim=-1)
        out = self.fusion_mlp(fused)
        return out.squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return gate values of shape (B, 4)."""
        return self._gates


# ---------------------------------------------------------------------------
# Multi-scale Spectrogram CNN
# ---------------------------------------------------------------------------

class MultiscaleSpectrogramCNN(nn.Module):
    """
    Multi-scale CNN for spectrograms with parallel conv branches
    (3x3, 5x5, 7x7), concatenation, and SE blocks.
    Works for both AE and Vib modalities.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.branch3 = self._make_branch(in_channels, base_channels, 3)
        self.branch5 = self._make_branch(in_channels, base_channels, 5)
        self.branch7 = self._make_branch(in_channels, base_channels, 7)

        total_ch = base_channels * 3
        self.se = SEBlock(total_ch, reduction=4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_ch, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.physics_fusion = nn.Sequential(
            nn.Linear(total_ch + 44, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _make_branch(in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)

        # Ensure same spatial size
        target_h = min(b3.size(2), b5.size(2), b7.size(2))
        target_w = min(b3.size(3), b5.size(3), b7.size(3))
        if b3.size(2) != target_h or b3.size(3) != target_w:
            b3 = F.adaptive_avg_pool2d(b3, (target_h, target_w))
        if b5.size(2) != target_h or b5.size(3) != target_w:
            b5 = F.adaptive_avg_pool2d(b5, (target_h, target_w))
        if b7.size(2) != target_h or b7.size(3) != target_w:
            b7 = F.adaptive_avg_pool2d(b7, (target_h, target_w))

        out = torch.cat([b3, b5, b7], dim=1)
        out = self.se(out)
        out = self.gap(out)
        out = self.fc(out)
        return out.squeeze(-1)

    def get_attention_weights(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Channel Attention CNN (CBAM-style)
# ---------------------------------------------------------------------------

class ChannelAttentionCNN(nn.Module):
    """
    CBAM-style CNN with channel attention + spatial attention blocks.
    Applied to AE or Vib spectrograms.
    """

    def __init__(
        self,
        in_channels: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.cbam1 = CBAM(16, reduction=4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.cbam2 = CBAM(32, reduction=4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.cbam3 = CBAM(48, reduction=4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.physics_fusion = nn.Sequential(
            nn.Linear(48 + 44, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv1(x)
        x = self.cbam1(x)
        x = self.conv2(x)
        x = self.cbam2(x)
        x = self.conv3(x)
        x = self.cbam3(x)
        x = self.gap(x)
        feat = self.fc[0](x)  # Flatten
        feat = feat.view(feat.size(0), -1)
        if physics is not None:
            feat = torch.cat([feat, physics], dim=-1)
            x = self.physics_fusion(feat)
        else:
            x = self.fc(feat)
        return x.squeeze(-1)

    def get_attention_weights(self) -> dict[str, torch.Tensor | None]:
        return {
            "cbam1": self.cbam1.get_attention_weights(),
            "cbam2": self.cbam2.get_attention_weights(),
            "cbam3": self.cbam3.get_attention_weights(),
        }


# ---------------------------------------------------------------------------
# Legacy / Baseline Models (kept for backward compatibility)
# ---------------------------------------------------------------------------

class _AECNNEncoder(nn.Module):
    """CNN encoder for AE spectrogram input of shape (B, 2, 300, 47)."""

    def __init__(self, output_dim: int = 32) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        return self.fc(x)


class _VibCNNEncoder(nn.Module):
    """CNN encoder for vibration spectrogram input of shape (B, 3, 257, 13)."""

    def __init__(self, output_dim: int = 32) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        return self.fc(x)


class _PhysicsMLPEncoder(nn.Module):
    """MLP encoder for physics features input of shape (B, input_dim).
    Lazily initializes weights on first forward pass so input_dim is inferred."""

    def __init__(self, input_dim: int | None = None, output_dim: int = 16) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp: nn.Module | None = None

    def _build(self, dim: int) -> None:
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(32, self.output_dim),
            nn.ReLU(inplace=True),
        )
        # Move to same device as parameters would be
        if next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
            self.mlp = self.mlp.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp is None:
            self._build(x.size(-1))
        return self.mlp(x)


class _ParamsMLPEncoder(nn.Module):
    """MLP encoder for process parameters input of shape (B, 3)."""

    def __init__(self, output_dim: int = 8) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AECNN(nn.Module):
    """CNN predicting surface roughness from AE spectrogram (B, 2, 300, 47)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _AECNNEncoder(output_dim=32)
        self.regressor = nn.Linear(32, 1)
        self.physics_fusion = nn.Sequential(
            nn.Linear(32 + 44, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.encoder(x)
        if physics is not None:
            feat = torch.cat([feat, physics], dim=-1)
            x = self.physics_fusion(feat)
        else:
            x = self.regressor(feat)
        return x.squeeze(-1)

    def get_attention_weights(self):
        return None


class VibCNN(nn.Module):
    """CNN predicting surface roughness from vibration spectrogram (B, 3, 257, 13)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _VibCNNEncoder(output_dim=32)
        self.regressor = nn.Linear(32, 1)
        self.physics_fusion = nn.Sequential(
            nn.Linear(32 + 44, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.encoder(x)
        if physics is not None:
            feat = torch.cat([feat, physics], dim=-1)
            x = self.physics_fusion(feat)
        else:
            x = self.regressor(feat)
        return x.squeeze(-1)

    def get_attention_weights(self):
        return None


class PhysicsMLP(nn.Module):
    """MLP predicting surface roughness from physics features (B, 44)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _PhysicsMLPEncoder(output_dim=16)
        self.regressor = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.regressor(x)
        return x.squeeze(-1)

    def get_attention_weights(self):
        return None


class ParamsMLP(nn.Module):
    """MLP predicting surface roughness from process parameters (B, 3)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ParamsMLPEncoder(output_dim=8)
        self.regressor = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.regressor(x)
        return x.squeeze(-1)

    def get_attention_weights(self):
        return None


class FusionModel(nn.Module):
    """
    Full multi-modal fusion model.
    Uses AECNN + VibCNN + PhysicsMLP + ParamsMLP encoders,
    concatenates embeddings, and predicts surface roughness via a final MLP.
    Optional flags allow ablation studies by dropping modalities.
    """

    def __init__(
        self,
        use_ae_spec: bool = True,
        use_vib_spec: bool = True,
        use_physics: bool = True,
        use_params: bool = True,
    ) -> None:
        super().__init__()
        self.use_ae_spec = use_ae_spec
        self.use_vib_spec = use_vib_spec
        self.use_physics = use_physics
        self.use_params = use_params

        self.ae_encoder = _AECNNEncoder(output_dim=32) if use_ae_spec else None
        self.vib_encoder = _VibCNNEncoder(output_dim=32) if use_vib_spec else None
        self.physics_encoder = _PhysicsMLPEncoder(output_dim=16) if use_physics else None
        self.params_encoder = _ParamsMLPEncoder(output_dim=8) if use_params else None

        total_dim = 0
        if use_ae_spec:
            total_dim += 32
        if use_vib_spec:
            total_dim += 32
        if use_physics:
            total_dim += 16
        if use_params:
            total_dim += 8

        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = []
        if self.use_ae_spec and ae_spec is not None:
            embeddings.append(self.ae_encoder(ae_spec))
        if self.use_vib_spec and vib_spec is not None:
            embeddings.append(self.vib_encoder(vib_spec))
        if self.use_physics and physics is not None:
            embeddings.append(self.physics_encoder(physics))
        if self.use_params and params is not None:
            embeddings.append(self.params_encoder(params))

        if not embeddings:
            raise ValueError("At least one modality must be enabled and provided.")

        x = torch.cat(embeddings, dim=-1)
        x = self.fusion_mlp(x)
        return x.squeeze(-1)

    def get_attention_weights(self):
        return None


class FeatureOnlyModel(nn.Module):
    """
    Lightweight model using only physics features + process parameters.
    Intended for edge-deployment scenarios.
    """

    def __init__(self) -> None:
        super().__init__()
        self.physics_encoder = _PhysicsMLPEncoder(output_dim=16)
        self.params_encoder = _ParamsMLPEncoder(output_dim=8)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(16 + 8, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def forward(self, physics: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.physics_encoder(physics), self.params_encoder(params)], dim=-1)
        x = self.fusion_mlp(x)
        return x.squeeze(-1)

    def get_attention_weights(self):
        return None


# ---------------------------------------------------------------------------
# Supported Configurations
# ---------------------------------------------------------------------------

_ALL_SUPPORTED_CONFIGS = [
    "ae", "vib", "physics", "params",
    "ae_vib", "ae_physics", "ae_params",
    "vib_physics", "vib_params", "physics_params",
    "ae_vib_physics", "ae_vib_params", "ae_vib_physics_params",
]


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention Building Block
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Standard PyTorch multi-head self-attention with pre-norm and dropout."""

    def __init__(self, d_model: int = 64, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        out, weights = self.mha(x, x, x, attn_mask=attn_mask, need_weights=True)
        self._attn_weights = weights
        out = self.norm(x + self.dropout(out))
        return out

    def get_attention_weights(self) -> torch.Tensor | None:
        return self._attn_weights


# ---------------------------------------------------------------------------
# Transformer Encoder Fusion
# ---------------------------------------------------------------------------

class TransformerEncoderFusion(nn.Module):
    """
    Advanced transformer-based fusion with learnable modality embeddings and CLS token.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        self.modality_embeddings = nn.Parameter(torch.randn(4, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        device = next(self.parameters()).device
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        feats: list[torch.Tensor] = []
        modalities: list[int] = []
        if ae_spec is not None:
            feats.append(self.ae_enc(ae_spec))
            modalities.append(0)
        if vib_spec is not None:
            feats.append(self.vib_enc(vib_spec))
            modalities.append(1)
        if physics is not None:
            feats.append(self.physics_enc(physics))
            modalities.append(2)
        if params is not None:
            feats.append(self.params_enc(params))
            modalities.append(3)

        seq = torch.stack(feats, dim=1)  # (B, N, D)
        embs = torch.stack([self.modality_embeddings[m] for m in modalities], dim=0).unsqueeze(0).expand(B, -1, -1)
        seq = seq + embs

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, seq], dim=1)  # (B, 1+N, D)

        out = self.transformer(seq)
        cls_out = self.norm(out[:, 0])
        return self.regressor(cls_out).squeeze(-1)

    def get_attention_weights(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Cross-Attention Fusion
# ---------------------------------------------------------------------------

class CrossAttentionFusionV1(nn.Module):
    """
    Cross-modal attention where each modality attends to all other modalities.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._attn_weights: dict[str, torch.Tensor] = {}

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        feat_map: dict[str, torch.Tensor] = {}
        if ae_spec is not None:
            feat_map["ae"] = self.ae_enc(ae_spec)
        if vib_spec is not None:
            feat_map["vib"] = self.vib_enc(vib_spec)
        if physics is not None:
            feat_map["physics"] = self.physics_enc(physics)
        if params is not None:
            feat_map["params"] = self.params_enc(params)

        keys = list(feat_map.keys())
        if len(keys) == 1:
            q = feat_map[keys[0]].unsqueeze(1)
            out, attn = self.cross_attn(q, q, q, need_weights=True)
            fused = self.norm(q + self.dropout(out)).squeeze(1)
            self._attn_weights = {keys[0]: attn}
            return self.regressor(fused).squeeze(-1)

        attended: list[torch.Tensor] = []
        self._attn_weights = {}
        for q_name in keys:
            q = feat_map[q_name].unsqueeze(1)  # (B, 1, D)
            kv = torch.stack([feat_map[k] for k in keys if k != q_name], dim=1)  # (B, M-1, D)
            out, attn = self.cross_attn(q, kv, kv, need_weights=True)
            out = self.norm(q + self.dropout(out)).squeeze(1)
            attended.append(out)
            self._attn_weights[q_name] = attn

        fused = torch.stack(attended, dim=1).mean(dim=1)  # (B, D)
        return self.regressor(fused).squeeze(-1)

    def get_attention_weights(self) -> dict[str, torch.Tensor]:
        return self._attn_weights


# ---------------------------------------------------------------------------
# Squeeze-Excitation Fusion
# ---------------------------------------------------------------------------

class SqueezeExcitationFusionV1(nn.Module):
    """
    SE blocks applied to spectrogram CNN features before fusion.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 64,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        # AE encoder with SE
        self.ae_stem = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.ae_res1 = ResidualBlock(16, 32, stride=2, use_se=False, dropout=dropout)
        self.ae_res2 = ResidualBlock(32, 48, stride=2, use_se=False, dropout=dropout)
        self.ae_se = SEBlock(48, reduction=4)
        self.ae_gap = nn.AdaptiveAvgPool2d(1)
        self.ae_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Vib encoder with SE
        self.vib_stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.vib_res1 = ResidualBlock(16, 32, stride=2, use_se=False, dropout=dropout)
        self.vib_res2 = ResidualBlock(32, 48, stride=2, use_se=False, dropout=dropout)
        self.vib_se = SEBlock(48, reduction=4)
        self.vib_gap = nn.AdaptiveAvgPool2d(1)
        self.vib_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._se_weights: dict[str, torch.Tensor] = {}

    @staticmethod
    def _se_forward(x: torch.Tensor, se_block: SEBlock) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply SE block and return output + attention weights."""
        b, c, _, _ = x.size()
        y = se_block.avg_pool(x).view(b, c)
        for layer in se_block.fc:
            y = layer(y)
        return x * y.view(b, c, 1, 1), y

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        device = next(self.parameters()).device
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        feats: list[torch.Tensor] = []

        if ae_spec is not None:
            x = self.ae_stem(ae_spec)
            x = self.ae_res1(x)
            x = self.ae_res2(x)
            x, w = self._se_forward(x, self.ae_se)
            self._se_weights["ae"] = w
            x = self.ae_gap(x)
            feats.append(self.ae_fc(x))
        else:
            feats.append(torch.zeros(B, 64, device=device))

        if vib_spec is not None:
            x = self.vib_stem(vib_spec)
            x = self.vib_res1(x)
            x = self.vib_res2(x)
            x, w = self._se_forward(x, self.vib_se)
            self._se_weights["vib"] = w
            x = self.vib_gap(x)
            feats.append(self.vib_fc(x))
        else:
            feats.append(torch.zeros(B, 64, device=device))

        if physics is not None:
            feats.append(self.physics_enc(physics))
        else:
            feats.append(torch.zeros(B, 64, device=device))

        if params is not None:
            feats.append(self.params_enc(params))
        else:
            feats.append(torch.zeros(B, 64, device=device))

        fused = torch.cat(feats, dim=-1)
        return self.fusion(fused).squeeze(-1)

    def get_attention_weights(self) -> dict[str, torch.Tensor]:
        return self._se_weights


# ---------------------------------------------------------------------------
# Attention MLP
# ---------------------------------------------------------------------------

class AttentionMLP(nn.Module):
    """
    Feature-wise attention-weighted MLP for tabular / encoded features.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(self, dropout: float = 0.3, config: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.ae_proj = nn.Linear(2, 8)
        self.vib_proj = nn.Linear(3, 8)
        self.phy_proj = nn.Linear(44, 16)
        self.par_proj = nn.Linear(3, 8)

        total_dim = 8 + 8 + 16 + 8
        self.attention = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._attn_weights: torch.Tensor | None = None

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        device = next(self.parameters()).device
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one input must be provided.")

        parts: list[torch.Tensor] = []
        if ae_spec is not None:
            x = F.adaptive_avg_pool2d(ae_spec, (1, 1)).view(B, -1)
            parts.append(self.ae_proj(x))
        else:
            parts.append(torch.zeros(B, 8, device=device))

        if vib_spec is not None:
            x = F.adaptive_avg_pool2d(vib_spec, (1, 1)).view(B, -1)
            parts.append(self.vib_proj(x))
        else:
            parts.append(torch.zeros(B, 8, device=device))

        if physics is not None:
            parts.append(self.phy_proj(physics))
        else:
            parts.append(torch.zeros(B, 16, device=device))

        if params is not None:
            parts.append(self.par_proj(params))
        else:
            parts.append(torch.zeros(B, 8, device=device))

        x = torch.cat(parts, dim=-1)
        attn = self.attention(x)
        self._attn_weights = attn
        out = self.mlp(x * attn)
        return out.squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        return self._attn_weights


# ---------------------------------------------------------------------------
# Modality-Gated Transformer
# ---------------------------------------------------------------------------

class ModalityGatedTransformer(nn.Module):
    """
    Gated mechanism + transformer with CLS token.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 4),
            nn.Sigmoid(),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._gates: torch.Tensor | None = None

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        device = next(self.parameters()).device
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        ae_f = self.ae_enc(ae_spec) if ae_spec is not None else torch.zeros(B, 64, device=device)
        vib_f = self.vib_enc(vib_spec) if vib_spec is not None else torch.zeros(B, 64, device=device)
        phy_f = self.physics_enc(physics) if physics is not None else torch.zeros(B, 64, device=device)
        par_f = self.params_enc(params) if params is not None else torch.zeros(B, 64, device=device)

        concat = torch.cat([ae_f, vib_f, phy_f, par_f], dim=-1)
        gates = self.gate_net(concat)  # (B, 4)
        self._gates = gates

        ae_g = ae_f * gates[:, 0:1]
        vib_g = vib_f * gates[:, 1:2]
        phy_g = phy_f * gates[:, 2:3]
        par_g = par_f * gates[:, 3:4]

        seq = torch.stack([ae_g, vib_g, phy_g, par_g], dim=1)  # (B, 4, D)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, seq], dim=1)

        out = self.transformer(seq)
        cls_out = self.norm(out[:, 0])
        return self.regressor(cls_out).squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        return self._gates


# ---------------------------------------------------------------------------
# Spectrogram ViT
# ---------------------------------------------------------------------------

class SpectrogramViT(nn.Module):
    """
    Vision Transformer for spectrograms with patch embedding and positional encoding.
    Supports configs containing at least one spectrogram modality.
    """

    supported_configs = [
        "ae", "vib", "ae_vib",
        "ae_physics", "vib_physics", "ae_vib_physics",
        "ae_params", "vib_params", "ae_vib_params",
        "ae_vib_physics_params",
    ]

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        # AE: 300x47 -> patch 15x15 -> 20x3 patches
        self.ae_patch = nn.Conv2d(2, embed_dim, kernel_size=(15, 15), stride=(15, 15))
        # Vib: 257x13 -> patch 10x10 -> 25x1 patches (with floor)
        self.vib_patch = nn.Conv2d(3, embed_dim, kernel_size=(10, 10), stride=(10, 10))

        self.ae_pos = nn.Parameter(torch.randn(1, 60, embed_dim) * 0.02)
        self.vib_pos = nn.Parameter(torch.randn(1, 30, embed_dim) * 0.02)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one input must be provided.")

        tokens: list[torch.Tensor] = []
        if ae_spec is not None:
            x = self.ae_patch(ae_spec)
            x = x.flatten(2).transpose(1, 2)
            N = x.size(1)
            x = x + self.ae_pos[:, :N]
            tokens.append(x)
        if vib_spec is not None:
            x = self.vib_patch(vib_spec)
            x = x.flatten(2).transpose(1, 2)
            N = x.size(1)
            x = x + self.vib_pos[:, :N]
            tokens.append(x)

        if not tokens:
            raise ValueError("At least one spectrogram modality must be provided.")

        seq = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, seq], dim=1)

        out = self.transformer(seq)
        cls_out = self.norm(out[:, 0])
        return self.regressor(cls_out).squeeze(-1)

    def get_attention_weights(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Physics-Guided Attention Net
# ---------------------------------------------------------------------------

class PhysicsGuidedAttentionNet(nn.Module):
    """
    Physics-guided attention where BDI and ST modulate fusion attention.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        self.physics_guide = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.Tanh(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._guide: torch.Tensor | None = None

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        device = next(self.parameters()).device
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        ae_f = self.ae_enc(ae_spec) if ae_spec is not None else torch.zeros(B, 64, device=device)
        vib_f = self.vib_enc(vib_spec) if vib_spec is not None else torch.zeros(B, 64, device=device)
        phy_f = self.physics_enc(physics) if physics is not None else torch.zeros(B, 64, device=device)
        par_f = self.params_enc(params) if params is not None else torch.zeros(B, 64, device=device)

        seq = torch.stack([ae_f, vib_f, phy_f, par_f], dim=1)  # (B, 4, D)

        if bdi_st is not None:
            guide = self.physics_guide(bdi_st).unsqueeze(1)  # (B, 1, D)
            self._guide = guide
            seq = seq * (1.0 + guide)

        out = self.transformer(seq)
        pooled = out.mean(dim=1)
        pooled = self.norm(pooled)
        return self.regressor(pooled).squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        return self._guide


# ---------------------------------------------------------------------------
# Hierarchical Fusion Net
# ---------------------------------------------------------------------------

class HierarchicalFusionNet(nn.Module):
    """
    Hierarchical multi-scale fusion with low-level CNN, mid-level MLP,
    high-level cross-attention, and skip connections.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 48,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        # Low-level spectrogram CNNs
        self.ae_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.vib_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
            nn.ReLU(inplace=True),
        )

        # Mid-level feature MLPs
        self.physics_mlp = nn.Sequential(
            nn.Linear(44, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.params_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, embed_dim),
            nn.ReLU(inplace=True),
        )

        # High-level cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Skip projection
        self.skip_proj = nn.Linear(embed_dim * 4, embed_dim)

        # Final regressor
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        device = next(self.parameters()).device
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        ae_low = self.ae_cnn(ae_spec) if ae_spec is not None else torch.zeros(B, 48, device=device)
        vib_low = self.vib_cnn(vib_spec) if vib_spec is not None else torch.zeros(B, 48, device=device)
        phy_mid = self.physics_mlp(physics) if physics is not None else torch.zeros(B, 48, device=device)
        par_mid = self.params_mlp(params) if params is not None else torch.zeros(B, 48, device=device)

        present: list[torch.Tensor] = []
        for feat in (ae_low, vib_low, phy_mid, par_mid):
            if feat.abs().sum() > 0:
                present.append(feat)

        if len(present) > 1:
            seq = torch.stack(present, dim=1)
            attn_out, _ = self.cross_attn(seq, seq, seq)
            high = self.cross_norm(attn_out + seq).mean(dim=1)
        else:
            high = present[0] if present else torch.zeros(B, 48, device=device)

        skip = torch.cat([ae_low, vib_low, phy_mid, par_mid], dim=-1)
        skip_proj = self.skip_proj(skip)

        fused = torch.cat([high, skip_proj], dim=-1)
        return self.regressor(fused).squeeze(-1)

    def get_attention_weights(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Dynamic Weighted Fusion
# ---------------------------------------------------------------------------

class DynamicWeightedFusion(nn.Module):
    """
    Dynamic modality weights computed by a meta-network.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 64,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        self.meta_net = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 4),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._weights: torch.Tensor | None = None

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        device = next(self.parameters()).device
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        ae_f = self.ae_enc(ae_spec) if ae_spec is not None else torch.zeros(B, 64, device=device)
        vib_f = self.vib_enc(vib_spec) if vib_spec is not None else torch.zeros(B, 64, device=device)
        phy_f = self.physics_enc(physics) if physics is not None else torch.zeros(B, 64, device=device)
        par_f = self.params_enc(params) if params is not None else torch.zeros(B, 64, device=device)

        concat = torch.cat([ae_f, vib_f, phy_f, par_f], dim=-1)
        logits = self.meta_net(concat)
        weights = F.softmax(logits, dim=-1)  # (B, 4)
        self._weights = weights

        fused = (
            ae_f * weights[:, 0:1]
            + vib_f * weights[:, 1:2]
            + phy_f * weights[:, 2:3]
            + par_f * weights[:, 3:4]
        )
        return self.fusion_mlp(fused).squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        return self._weights



# ---------------------------------------------------------------------------
# Attention-based and Advanced Fusion Models (New)
# ---------------------------------------------------------------------------

class _MHAFusionBlock(nn.Module):
    """Multi-Head Self-Attention block with FFN and residual connections."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, attn = self.mha(x, x, x, need_weights=True, average_attn_weights=False)
        out = self.norm1(x + out)
        out2 = self.ffn(out)
        out = self.norm2(out + out2)
        return out, attn


class _CrossAttnBlock(nn.Module):
    """Cross-attention block with residual connections and LayerNorm."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, attn = self.cross(query, kv, kv, need_weights=True, average_attn_weights=False)
        out = self.norm1(query + out)
        out2 = self.ffn(out)
        out = self.norm2(out + out2)
        return out, attn


class _LowRankBilinear(nn.Module):
    """Low-rank bilinear pooling for a pair of modality vectors."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.U = nn.Linear(dim, rank, bias=False)
        self.V = nn.Linear(dim, rank, bias=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.U(x1) * self.V(x2)


class _GATLayer(nn.Module):
    """Graph Attention Network layer with multi-head attention."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.residual = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.attn_src = nn.Parameter(torch.randn(num_heads, self.head_dim, 1))
        self.attn_dst = nn.Parameter(torch.randn(num_heads, self.head_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, M, _ = x.shape
        h = self.W(x)
        h = h.view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        src = torch.matmul(h, self.attn_src).squeeze(-1)
        dst = torch.matmul(h, self.attn_dst).squeeze(-1)
        scores = src.unsqueeze(-1) + dst.unsqueeze(-2)
        scores = self.leaky_relu(scores)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, h)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, M, self.out_dim)
        out = self.norm(self.residual(x) + out)
        return out, attn


class _IntraAttnAECNN(nn.Module):
    """AE CNN encoder with spatial attention pooling."""

    def __init__(self, output_dim: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(16, 32, stride=2, use_se=True, dropout=dropout)
        self.res2 = ResidualBlock(32, 48, stride=2, use_se=True, dropout=dropout)
        self.spatial_attn = nn.Conv2d(48, 1, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        attn_map = self.spatial_attn(x)
        B, C, H, W = x.shape
        attn = F.softmax(attn_map.view(B, 1, -1), dim=-1)
        x_flat = x.view(B, C, -1)
        pooled = torch.bmm(x_flat, attn.transpose(1, 2)).squeeze(-1)
        return self.fc(pooled), attn.view(B, H, W)


class _IntraAttnVibCNN(nn.Module):
    """Vib CNN encoder with spatial attention pooling."""

    def __init__(self, output_dim: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(16, 32, stride=2, use_se=True, dropout=dropout)
        self.res2 = ResidualBlock(32, 48, stride=2, use_se=True, dropout=dropout)
        self.spatial_attn = nn.Conv2d(48, 1, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        attn_map = self.spatial_attn(x)
        B, C, H, W = x.shape
        attn = F.softmax(attn_map.view(B, 1, -1), dim=-1)
        x_flat = x.view(B, C, -1)
        pooled = torch.bmm(x_flat, attn.transpose(1, 2)).squeeze(-1)
        return self.fc(pooled), attn.view(B, H, W)


class _IntraAttnPhysics(nn.Module):
    """Physics encoder with feature-wise attention."""

    def __init__(self, input_dim: int = 44, output_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_attn = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(x)
        attn = torch.sigmoid(self.feature_attn(out))
        return out * attn, attn


class _IntraAttnParams(nn.Module):
    """Params encoder with feature-wise attention."""

    def __init__(self, output_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, output_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_attn = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(x)
        attn = torch.sigmoid(self.feature_attn(out))
        return out * attn, attn


# ---------------------------------------------------------------------------
# 1. MultiHeadAttentionFusion
# ---------------------------------------------------------------------------

class MultiHeadAttentionFusion(nn.Module):
    """
    Multi-Head Self-Attention over stacked modality feature vectors.
    Uses learned positional embeddings for each modality.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        self.pos_embed = nn.Parameter(torch.randn(4, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            _MHAFusionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self._attn_weights: torch.Tensor | None = None

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        feats: list[torch.Tensor] = []
        if ae_spec is not None:
            feats.append(self.ae_enc(ae_spec))
        if vib_spec is not None:
            feats.append(self.vib_enc(vib_spec))
        if physics is not None:
            feats.append(self.physics_enc(physics))
        if params is not None:
            feats.append(self.params_enc(params))

        seq = torch.stack(feats, dim=1)
        M = seq.size(1)
        seq = seq + self.pos_embed[:M].unsqueeze(0)

        attn_weights = []
        out = seq
        for block in self.blocks:
            out, attn = block(out)
            attn_weights.append(attn)

        self._attn_weights = torch.stack(attn_weights, dim=1).mean(dim=1) if attn_weights else None
        pooled = self.pool(out.transpose(1, 2)).squeeze(-1)
        return self.regressor(pooled).squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return attention map of shape (B, num_heads, num_modalities, num_modalities)."""
        return self._attn_weights


# ---------------------------------------------------------------------------
# 2. CrossAttentionFusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Designate one modality as query and others as key/value.
    Multiple cross-attention layers with residual connections and LayerNorm.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
        query_modality: str = "physics",
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.query_modality = query_modality
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        self.blocks = nn.ModuleList([
            _CrossAttnBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self._attn_weights: torch.Tensor | None = None

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        encoders = {"ae": self.ae_enc, "vib": self.vib_enc, "physics": self.physics_enc, "params": self.params_enc}
        inputs = {"ae": ae_spec, "vib": vib_spec, "physics": physics, "params": params}
        encoded: dict[str, torch.Tensor] = {}
        for name, tensor in inputs.items():
            if tensor is not None:
                encoded[name] = encoders[name](tensor)

        query_name = self.query_modality
        if query_name not in encoded:
            query_name = next(iter(encoded))

        query = encoded[query_name].unsqueeze(1)
        kv_names = [n for n in encoded if n != query_name]
        if kv_names:
            kv = torch.stack([encoded[n] for n in kv_names], dim=1)
        else:
            kv = query

        out = query
        attn_weights = []
        for block in self.blocks:
            out, attn = block(out, kv)
            attn_weights.append(attn)

        self._attn_weights = torch.stack(attn_weights, dim=1).mean(dim=1) if attn_weights else None
        out = out.squeeze(1)
        return self.regressor(out).squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return cross-attention map of shape (B, num_heads, 1, num_kv_modalities)."""
        return self._attn_weights


# ---------------------------------------------------------------------------
# 3. SqueezeExcitationFusion
# ---------------------------------------------------------------------------

class SqueezeExcitationFusion(nn.Module):
    """
    Apply SE blocks across modalities after encoding each to a vector.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 32,
        se_reduction: int = 4,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        self.se_fc = nn.Sequential(
            nn.Linear(4, 4 // se_reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4 // se_reduction, 4, bias=False),
            nn.Sigmoid(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        device = next(self.parameters()).device
        slots = torch.zeros(B, 4, self.embed_dim, device=device)
        mask = torch.zeros(B, 4, device=device)

        encoders = [self.ae_enc, self.vib_enc, self.physics_enc, self.params_enc]
        inputs = [ae_spec, vib_spec, physics, params]
        for i, (enc, inp) in enumerate(zip(encoders, inputs)):
            if inp is not None:
                slots[:, i] = enc(inp)
                mask[:, i] = 1

        pooled = slots.mean(dim=-1)
        gates = self.se_fc(pooled)
        slots = slots * gates.unsqueeze(-1)
        slots = slots * mask.unsqueeze(-1)

        fused = slots.view(B, -1)
        return self.fusion_mlp(fused).squeeze(-1)

    def get_attention_weights(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 4. BilinearFusionNetwork
# ---------------------------------------------------------------------------

class BilinearFusionNetwork(nn.Module):
    """
    Low-rank bilinear pooling for pairs of modalities + final MLP.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 32,
        rank: int = 16,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.rank = rank
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        self.pairs = [
            ("ae", "vib"), ("ae", "physics"), ("ae", "params"),
            ("vib", "physics"), ("vib", "params"), ("physics", "params"),
        ]
        self.bilinear = nn.ModuleDict({
            f"{a}_{b}": _LowRankBilinear(embed_dim, rank) for a, b in self.pairs
        })

        self.fusion_mlp = nn.Sequential(
            nn.Linear(rank * len(self.pairs) + embed_dim * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        device = next(self.parameters()).device
        encoded: dict[str, torch.Tensor] = {}
        if ae_spec is not None:
            encoded["ae"] = self.ae_enc(ae_spec)
        if vib_spec is not None:
            encoded["vib"] = self.vib_enc(vib_spec)
        if physics is not None:
            encoded["physics"] = self.physics_enc(physics)
        if params is not None:
            encoded["params"] = self.params_enc(params)

        bilinear_feats = []
        for a, b in self.pairs:
            if a in encoded and b in encoded:
                bilinear_feats.append(self.bilinear[f"{a}_{b}"](encoded[a], encoded[b]))
            else:
                bilinear_feats.append(torch.zeros(B, self.rank, device=device))

        individual = torch.zeros(B, 4 * self.embed_dim, device=device)
        slot_map = {"ae": 0, "vib": 1, "physics": 2, "params": 3}
        for k, v in encoded.items():
            individual[:, slot_map[k] * self.embed_dim:(slot_map[k] + 1) * self.embed_dim] = v

        fused = torch.cat(bilinear_feats + [individual], dim=-1)
        return self.fusion_mlp(fused).squeeze(-1)

    def get_attention_weights(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 5. GraphNeuralNetworkFusion
# ---------------------------------------------------------------------------

class GraphNeuralNetworkFusion(nn.Module):
    """
    Treat modalities as nodes in a fully-connected graph.
    Uses GAT-style message passing and global graph pooling.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        self.gat_layers = nn.ModuleList([
            _GATLayer(embed_dim if i == 0 else hidden_dim, hidden_dim, num_heads, dropout)
            for i in range(num_layers)
        ])
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self._attn_weights: torch.Tensor | None = None

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        encoded: dict[str, torch.Tensor] = {}
        if ae_spec is not None:
            encoded["ae"] = self.ae_enc(ae_spec)
        if vib_spec is not None:
            encoded["vib"] = self.vib_enc(vib_spec)
        if physics is not None:
            encoded["physics"] = self.physics_enc(physics)
        if params is not None:
            encoded["params"] = self.params_enc(params)

        seq = torch.stack(list(encoded.values()), dim=1)

        attn_weights = []
        out = seq
        for layer in self.gat_layers:
            out, attn = layer(out)
            attn_weights.append(attn)

        self._attn_weights = torch.stack(attn_weights, dim=1).mean(dim=1) if attn_weights else None
        pooled = out.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return learned adjacency attention weights of shape (B, num_heads, M, M)."""
        return self._attn_weights


# ---------------------------------------------------------------------------
# 6. HierarchicalAttentionNetwork
# ---------------------------------------------------------------------------

class HierarchicalAttentionNetwork(nn.Module):
    """
    Two-level attention: intra-modal attention before encoding,
    inter-modal attention after encoding.
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ae_enc = _IntraAttnAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _IntraAttnVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _IntraAttnPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _IntraAttnParams(output_dim=embed_dim, dropout=dropout)

        self.pos_embed = nn.Parameter(torch.randn(4, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            _MHAFusionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self._intra_attn: dict[str, torch.Tensor] = {}
        self._inter_attn: torch.Tensor | None = None

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        encoders = {
            "ae": self.ae_enc, "vib": self.vib_enc,
            "physics": self.physics_enc, "params": self.params_enc,
        }
        inputs = {"ae": ae_spec, "vib": vib_spec, "physics": physics, "params": params}

        feats = []
        self._intra_attn = {}
        for name, tensor in inputs.items():
            if tensor is not None:
                enc_out, attn = encoders[name](tensor)
                feats.append(enc_out)
                self._intra_attn[name] = attn

        seq = torch.stack(feats, dim=1)
        M = seq.size(1)
        seq = seq + self.pos_embed[:M].unsqueeze(0)

        out = seq
        inter_attns = []
        for block in self.blocks:
            out, attn = block(out)
            inter_attns.append(attn)

        self._inter_attn = torch.stack(inter_attns, dim=1).mean(dim=1) if inter_attns else None
        pooled = out.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)

    def get_attention_weights(self) -> dict[str, Any]:
        """Return dict with intra-modal and inter-modal attention weights."""
        return {"intra": self._intra_attn, "inter": self._inter_attn}


# ---------------------------------------------------------------------------
# 7. PhysicsGuidedAttentionNetwork
# ---------------------------------------------------------------------------

class PhysicsGuidedAttentionNetwork(nn.Module):
    """
    Use BDI and ST as attention guides over modalities.
    Soft attention: softmax(MLP(concat(features, physics_indicators))).
    Supports all 13 input configurations.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        embed_dim: int = 32,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.ae_enc = _ModalityAECNN(output_dim=embed_dim, dropout=dropout)
        self.vib_enc = _ModalityVibCNN(output_dim=embed_dim, dropout=dropout)
        self.physics_enc = _ModalityPhysics(output_dim=embed_dim, dropout=dropout)
        self.params_enc = _ModalityParams(output_dim=embed_dim, dropout=dropout)

        self.guide_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4 + 2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 4),
        )
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self._attn_weights: torch.Tensor | None = None

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B: int | None = None
        for inp in (ae_spec, vib_spec, physics, params):
            if inp is not None:
                B = inp.size(0)
                break
        if B is None:
            raise ValueError("At least one modality must be provided.")

        device = next(self.parameters()).device
        slots = torch.zeros(B, 4, self.embed_dim, device=device)
        mask = torch.zeros(B, 4, device=device)

        encoders = [self.ae_enc, self.vib_enc, self.physics_enc, self.params_enc]
        inputs = [ae_spec, vib_spec, physics, params]
        for i, (enc, inp) in enumerate(zip(encoders, inputs)):
            if inp is not None:
                slots[:, i] = enc(inp)
                mask[:, i] = 1

        if bdi_st is not None:
            guide = bdi_st
        elif physics is not None:
            guide = physics[:, :2]
        else:
            guide = torch.zeros(B, 2, device=device)

        combined = torch.cat([slots.view(B, -1), guide], dim=-1)
        modality_attn = F.softmax(self.guide_mlp(combined), dim=-1)
        modality_attn = modality_attn * mask
        modality_attn = modality_attn / (modality_attn.sum(dim=-1, keepdim=True) + 1e-8)

        self._attn_weights = modality_attn.unsqueeze(1)
        fused = (slots * modality_attn.unsqueeze(-1)).sum(dim=1)
        return self.regressor(fused).squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return physics-guided attention map of shape (B, 1, 4)."""
        return self._attn_weights


# ---------------------------------------------------------------------------
# 8. TabTransformerRegressor
# ---------------------------------------------------------------------------

class TabTransformerRegressor(nn.Module):
    """
    Transformer encoder on column embeddings for tabular features.
    Supports variable tabular inputs: physics, params, bdi_st.
    """

    supported_configs = _ALL_SUPPORTED_CONFIGS

    def __init__(
        self,
        max_features: int = 49,
        embed_dim: int = 16,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
        config: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.max_features = max_features
        self.embed_dim = embed_dim

        self.embeddings = nn.Parameter(torch.randn(max_features, embed_dim) * 0.1)
        self.biases = nn.Parameter(torch.zeros(max_features, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                batch_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self._attn_weights: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor | None = None,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
        bdi_st: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x is None:
            parts: list[torch.Tensor] = []
            if physics is not None:
                parts.append(physics)
            if params is not None:
                parts.append(params)
            if bdi_st is not None:
                parts.append(bdi_st)
            if not parts:
                raise ValueError("At least one tabular input must be provided.")
            x = torch.cat(parts, dim=-1)

        B, N = x.shape
        if N > self.max_features:
            raise ValueError(f"Input has {N} features but max_features={self.max_features}")

        x = x.unsqueeze(-1)
        emb = x * self.embeddings[:N].unsqueeze(0) + self.biases[:N].unsqueeze(0)
        cls = self.cls_token.expand(B, -1, -1)
        emb = torch.cat([cls, emb], dim=1)

        out = emb
        attn_weights = []
        for layer in self.layers:
            src2, attn = layer.self_attn(out, out, out, need_weights=True, average_attn_weights=False)
            out = out + layer.dropout1(src2)
            out = layer.norm1(out)
            src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(out))))
            out = out + layer.dropout2(src2)
            out = layer.norm2(out)
            attn_weights.append(attn)

        self._attn_weights = torch.stack(attn_weights, dim=1).mean(dim=1) if attn_weights else None
        cls_out = self.norm(out[:, 0])
        return self.regressor(cls_out).squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return attention over features from CLS token: (B, num_heads, N)."""
        if self._attn_weights is None:
            return None
        return self._attn_weights[:, :, 0, 1:]


class _TransferSpectrogramEncoder(nn.Module):
    """Pretrained ResNet18 encoder adapted to variable-channel spectrograms."""

    def __init__(
        self,
        in_channels: int,
        output_dim: int = 128,
        freeze_blocks: int = 3,
    ) -> None:
        super().__init__()
        if tvmodels is None:
            raise ImportError("torchvision is required for TransferLearningFusionNet")

        backbone = tvmodels.resnet18(weights="IMAGENET1K_V1")
        self.output_dim = output_dim

        # Adapt first conv to accept in_channels while preserving pretrained weights.
        original_conv1 = backbone.conv1
        if in_channels != 3:
            self.input_proj = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            # Initialise roughly as a channel-wise average of pretrained weights.
            with torch.no_grad():
                pretrained_weights = original_conv1.weight.data
                self.input_proj.weight.data = pretrained_weights.mean(dim=1, keepdim=True).expand(
                    -1, in_channels, -1, -1
                ).contiguous()
        else:
            self.input_proj = original_conv1

        # Remove final FC; keep layers up to avgpool.
        self.layer0 = nn.Sequential(
            self.input_proj,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # Freeze early blocks to avoid overfitting on small grinding data.
        for block_idx, block in enumerate([self.layer0, self.layer1, self.layer2, self.layer3, self.layer4], start=1):
            if block_idx <= freeze_blocks:
                for p in block.parameters():
                    p.requires_grad = False

        self.projection = nn.Linear(512, output_dim)

        # ImageNet normalisation constants.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> resize to 224x224 for pretrained ResNet.
        if x.size(-2) != 224 or x.size(-1) != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalise assuming 3 effective channels after projection.
        if x.size(1) == 3:
            x = (x - self.mean) / self.std

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.projection(x)


class TransferLearningFusionNet(nn.Module):
    """
    Transfer-learning fusion network for grinding roughness prediction.

    Uses ImageNet-pretrained ResNet18 encoders for AE and vibration spectrograms,
    a small tabular MLP for features/process parameters, and a final fusion MLP.
    Designed as a side test to see whether transfer learning can outperform
    tree ensembles and from-scratch deep fusion models on the small VibeGrinding
    dataset under condition-level (LOGO) generalisation.
    """

    def __init__(
        self,
        use_ae_spec: bool = True,
        use_vib_spec: bool = True,
        use_ae_features: bool = True,
        use_vib_features: bool = True,
        use_physics: bool = True,
        use_params: bool = True,
        ae_spec_channels: int = 2,
        vib_spec_channels: int = 3,
        spec_embed_dim: int = 128,
        tab_embed_dim: int = 64,
        freeze_blocks: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.use_ae_spec = use_ae_spec
        self.use_vib_spec = use_vib_spec
        self.use_ae_features = use_ae_features
        self.use_vib_features = use_vib_features
        self.use_physics = use_physics
        self.use_params = use_params

        self.ae_encoder: nn.Module | None = None
        self.vib_encoder: nn.Module | None = None
        if use_ae_spec:
            self.ae_encoder = _TransferSpectrogramEncoder(
                in_channels=ae_spec_channels,
                output_dim=spec_embed_dim,
                freeze_blocks=freeze_blocks,
            )
        if use_vib_spec:
            self.vib_encoder = _TransferSpectrogramEncoder(
                in_channels=vib_spec_channels,
                output_dim=spec_embed_dim,
                freeze_blocks=freeze_blocks,
            )

        tab_input_dim = 0
        if use_ae_features:
            tab_input_dim += 8
        if use_vib_features:
            tab_input_dim += 12
        if use_physics:
            tab_input_dim += 44
        if use_params:
            tab_input_dim += 3

        self.tab_input_dim = tab_input_dim
        self.tab_embed_dim = tab_embed_dim
        self.tabular_encoder: nn.Module | None = None
        # Lazily built on first forward to match the actual tabular inputs
        # provided by the current config.

        self.fusion_dim = 0
        if use_ae_spec:
            self.fusion_dim += spec_embed_dim
        if use_vib_spec:
            self.fusion_dim += spec_embed_dim
        if tab_input_dim > 0:
            self.fusion_dim += tab_embed_dim

        if self.fusion_dim == 0:
            raise ValueError("At least one input modality must be enabled.")

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        ae_spec: torch.Tensor | None = None,
        vib_spec: torch.Tensor | None = None,
        ae_features: torch.Tensor | None = None,
        vib_features: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []

        if self.use_ae_spec and ae_spec is not None:
            embeddings.append(self.ae_encoder(ae_spec))  # type: ignore[arg-type]
        if self.use_vib_spec and vib_spec is not None:
            embeddings.append(self.vib_encoder(vib_spec))  # type: ignore[arg-type]

        tab_parts: list[torch.Tensor] = []
        if self.use_ae_features and ae_features is not None:
            tab_parts.append(ae_features)
        if self.use_vib_features and vib_features is not None:
            tab_parts.append(vib_features)
        if self.use_physics and physics is not None:
            tab_parts.append(physics)
        if self.use_params and params is not None:
            tab_parts.append(params)

        if tab_parts:
            tab_x = torch.cat(tab_parts, dim=-1)
            if self.tabular_encoder is None:
                actual_dim = tab_x.size(-1)
                self.tabular_encoder = nn.Sequential(
                    nn.Linear(actual_dim, self.tab_embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.fusion_mlp[2].p),
                    nn.Linear(self.tab_embed_dim, self.tab_embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.fusion_mlp[2].p),
                ).to(tab_x.device)
            embeddings.append(self.tabular_encoder(tab_x))

        x = torch.cat(embeddings, dim=-1)
        return self.fusion_mlp(x).squeeze(-1)

    def get_attention_weights(self) -> torch.Tensor | None:
        return None


# ---------------------------------------------------------------------------
# Temporal trajectory models (WP-7 side test)
# ---------------------------------------------------------------------------

class _MaskedTemporalPool(nn.Module):
    """Apply masked average or max pooling over the time dimension."""

    def __init__(self, pool: str = "avg") -> None:
        super().__init__()
        self.pool = pool

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        lengths: (B,) actual sequence lengths
        Returns: (B, D)
        """
        B, T, D = x.shape
        device = x.device
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
        mask = mask.unsqueeze(-1).float()  # (B, T, 1)

        if self.pool == "avg":
            summed = (x * mask).sum(dim=1)  # (B, D)
            return summed / lengths.unsqueeze(1).clamp(min=1)
        elif self.pool == "max":
            x_masked = x.masked_fill(mask == 0, -1e9)
            return x_masked.max(dim=1).values
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")


class _TrajectoryEncoderLSTM(nn.Module):
    """Single-modality LSTM encoder with dynamic-length handling."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Sort by length descending for pack_padded_sequence
        sorted_lengths, sort_idx = torch.sort(lengths, descending=True)
        x_sorted = x[sort_idx]

        packed = pack_padded_sequence(
            x_sorted, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, (hn, _) = self.lstm(packed)

        # hn shape: (num_layers * num_directions, B, hidden_size)
        if self.bidirectional:
            # Take final forward and backward hidden states from last layer
            hn = hn.view(self.num_layers, 2, -1, self.hidden_size)
            final = torch.cat([hn[-1, 0], hn[-1, 1]], dim=-1)
        else:
            final = hn[-1]

        # Unsort
        _, unsort_idx = torch.sort(sort_idx)
        return final[unsort_idx]


class TrajectoryLSTM(nn.Module):
    """
    LSTM-based sequence model for time-resolved sub-band energy trajectories.
    Supports variable-length sequences and multimodal fusion (AE + vibration).
    """

    def __init__(
        self,
        ae_input_dim: int = 6,
        vib_input_dim: int = 9,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_ae: bool = True,
        use_vib: bool = True,
        use_pp: bool = False,
        pp_dim: int = 3,
    ) -> None:
        super().__init__()
        self.use_ae = use_ae
        self.use_vib = use_vib
        self.use_pp = use_pp

        embed_dim = 0
        if use_ae:
            self.ae_encoder = _TrajectoryEncoderLSTM(
                ae_input_dim, hidden_size, num_layers, dropout
            )
            embed_dim += self.ae_encoder.output_dim
        if use_vib:
            self.vib_encoder = _TrajectoryEncoderLSTM(
                vib_input_dim, hidden_size, num_layers, dropout
            )
            embed_dim += self.vib_encoder.output_dim
        if use_pp:
            embed_dim += pp_dim

        if embed_dim == 0:
            raise ValueError("At least one input modality must be enabled.")

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        ae_trajectory: torch.Tensor | None = None,
        vib_trajectory: torch.Tensor | None = None,
        lengths_ae: torch.Tensor | None = None,
        lengths_vib: torch.Tensor | None = None,
        pp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []

        if self.use_ae and ae_trajectory is not None:
            if lengths_ae is None:
                lengths_ae = torch.full(
                    (ae_trajectory.size(0),), ae_trajectory.size(1), device=ae_trajectory.device
                )
            embeddings.append(self.ae_encoder(ae_trajectory, lengths_ae))

        if self.use_vib and vib_trajectory is not None:
            if lengths_vib is None:
                lengths_vib = torch.full(
                    (vib_trajectory.size(0),), vib_trajectory.size(1), device=vib_trajectory.device
                )
            embeddings.append(self.vib_encoder(vib_trajectory, lengths_vib))

        if self.use_pp and pp is not None:
            embeddings.append(pp)

        x = torch.cat(embeddings, dim=-1)
        return self.fusion_mlp(x).squeeze(-1)


class TrajectoryGRU(TrajectoryLSTM):
    """GRU variant of TrajectoryLSTM."""

    def __init__(self, **kwargs: Any) -> None:
        # Replace LSTM encoder with GRU by overriding after super init
        super().__init__(**kwargs)
        hidden_size = kwargs.get("hidden_size", 128)
        num_layers = kwargs.get("num_layers", 2)
        dropout = kwargs.get("dropout", 0.5)
        ae_input_dim = kwargs.get("ae_input_dim", 6)
        vib_input_dim = kwargs.get("vib_input_dim", 9)

        if self.use_ae:
            self.ae_encoder = nn.GRU(
                ae_input_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        if self.use_vib:
            self.vib_encoder = nn.GRU(
                vib_input_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # GRU forward through _TrajectoryEncoderLSTM expects LSTM interface;
        # simplest is to reuse parent but it won't work because encoders changed.
        # Implement dedicated forward below.
        raise NotImplementedError("Use TrajectoryGRUDedicated instead.")


class TrajectoryGRUDedicated(nn.Module):
    """GRU-based sequence model for trajectories."""

    def __init__(
        self,
        ae_input_dim: int = 6,
        vib_input_dim: int = 9,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_ae: bool = True,
        use_vib: bool = True,
        use_pp: bool = False,
        pp_dim: int = 3,
    ) -> None:
        super().__init__()
        self.use_ae = use_ae
        self.use_vib = use_vib
        self.use_pp = use_pp

        embed_dim = 0
        if use_ae:
            self.ae_encoder = nn.GRU(
                ae_input_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            embed_dim += hidden_size
        if use_vib:
            self.vib_encoder = nn.GRU(
                vib_input_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            embed_dim += hidden_size
        if use_pp:
            embed_dim += pp_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _encode(self, encoder: nn.GRU, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        sorted_lengths, sort_idx = torch.sort(lengths, descending=True)
        x_sorted = x[sort_idx]
        packed = pack_padded_sequence(
            x_sorted, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, hn = encoder(packed)
        final = hn[-1]
        _, unsort_idx = torch.sort(sort_idx)
        return final[unsort_idx]

    def forward(
        self,
        ae_trajectory: torch.Tensor | None = None,
        vib_trajectory: torch.Tensor | None = None,
        lengths_ae: torch.Tensor | None = None,
        lengths_vib: torch.Tensor | None = None,
        pp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []

        if self.use_ae and ae_trajectory is not None:
            if lengths_ae is None:
                lengths_ae = torch.full(
                    (ae_trajectory.size(0),), ae_trajectory.size(1), device=ae_trajectory.device
                )
            embeddings.append(self._encode(self.ae_encoder, ae_trajectory, lengths_ae))

        if self.use_vib and vib_trajectory is not None:
            if lengths_vib is None:
                lengths_vib = torch.full(
                    (vib_trajectory.size(0),), vib_trajectory.size(1), device=vib_trajectory.device
                )
            embeddings.append(self._encode(self.vib_encoder, vib_trajectory, lengths_vib))

        if self.use_pp and pp is not None:
            embeddings.append(pp)

        x = torch.cat(embeddings, dim=-1)
        return self.fusion_mlp(x).squeeze(-1)


class _TCNResidualBlock(nn.Module):
    """Residual block for TCN with dilation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv1(x)
        # Trim causal padding
        out = out[:, :, : x.size(-1)]
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, : x.size(-1)]
        out = self.relu2(out)
        out = self.dropout2(out)

        residual = x if self.downsample is None else self.downsample(x)
        return out + residual


class _TrajectoryEncoderTCN(nn.Module):
    """TCN encoder for one modality."""

    def __init__(
        self,
        input_dim: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]

        layers: list[nn.Module] = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            layers.append(
                _TCNResidualBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]
        self.pool = _MaskedTemporalPool(pool="avg")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        features = self.network(x)  # (B, C, T)
        features = features.transpose(1, 2)  # (B, T, C)
        return self.pool(features, lengths)


class TrajectoryTCN(nn.Module):
    """TCN-based sequence model for trajectories with multimodal fusion."""

    def __init__(
        self,
        ae_input_dim: int = 6,
        vib_input_dim: int = 9,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.5,
        use_ae: bool = True,
        use_vib: bool = True,
        use_pp: bool = False,
        pp_dim: int = 3,
    ) -> None:
        super().__init__()
        self.use_ae = use_ae
        self.use_vib = use_vib
        self.use_pp = use_pp

        if num_channels is None:
            num_channels = [64, 64]

        embed_dim = 0
        if use_ae:
            self.ae_encoder = _TrajectoryEncoderTCN(ae_input_dim, num_channels, kernel_size, dropout)
            embed_dim += self.ae_encoder.output_dim
        if use_vib:
            self.vib_encoder = _TrajectoryEncoderTCN(vib_input_dim, num_channels, kernel_size, dropout)
            embed_dim += self.vib_encoder.output_dim
        if use_pp:
            embed_dim += pp_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        ae_trajectory: torch.Tensor | None = None,
        vib_trajectory: torch.Tensor | None = None,
        lengths_ae: torch.Tensor | None = None,
        lengths_vib: torch.Tensor | None = None,
        pp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []

        if self.use_ae and ae_trajectory is not None:
            if lengths_ae is None:
                lengths_ae = torch.full(
                    (ae_trajectory.size(0),), ae_trajectory.size(1), device=ae_trajectory.device
                )
            embeddings.append(self.ae_encoder(ae_trajectory, lengths_ae))

        if self.use_vib and vib_trajectory is not None:
            if lengths_vib is None:
                lengths_vib = torch.full(
                    (vib_trajectory.size(0),), vib_trajectory.size(1), device=vib_trajectory.device
                )
            embeddings.append(self.vib_encoder(vib_trajectory, lengths_vib))

        if self.use_pp and pp is not None:
            embeddings.append(pp)

        x = torch.cat(embeddings, dim=-1)
        return self.fusion_mlp(x).squeeze(-1)


class _TrajectoryEncoderCNN(nn.Module):
    """1D-CNN encoder for one modality with masked pooling."""

    def __init__(
        self,
        input_dim: int,
        channels: list[int] | None = None,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64]

        layers: list[nn.Module] = []
        in_ch = input_dim
        for out_ch in channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.output_dim = channels[-1]
        self.pool = _MaskedTemporalPool(pool="avg")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, D, T)
        features = self.network(x)  # (B, C, T')
        features = features.transpose(1, 2)  # (B, T', C)
        # Update lengths after pooling
        pooled_lengths = lengths // 2
        pooled_lengths = torch.clamp(pooled_lengths, min=1)
        return self.pool(features, pooled_lengths)


class TrajectoryCNN(nn.Module):
    """1D-CNN sequence model for trajectories with multimodal fusion."""

    def __init__(
        self,
        ae_input_dim: int = 6,
        vib_input_dim: int = 9,
        channels: list[int] | None = None,
        kernel_size: int = 5,
        dropout: float = 0.5,
        use_ae: bool = True,
        use_vib: bool = True,
        use_pp: bool = False,
        pp_dim: int = 3,
    ) -> None:
        super().__init__()
        self.use_ae = use_ae
        self.use_vib = use_vib
        self.use_pp = use_pp

        if channels is None:
            channels = [32, 64]

        embed_dim = 0
        if use_ae:
            self.ae_encoder = _TrajectoryEncoderCNN(ae_input_dim, channels, kernel_size, dropout)
            embed_dim += self.ae_encoder.output_dim
        if use_vib:
            self.vib_encoder = _TrajectoryEncoderCNN(vib_input_dim, channels, kernel_size, dropout)
            embed_dim += self.vib_encoder.output_dim
        if use_pp:
            embed_dim += pp_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        ae_trajectory: torch.Tensor | None = None,
        vib_trajectory: torch.Tensor | None = None,
        lengths_ae: torch.Tensor | None = None,
        lengths_vib: torch.Tensor | None = None,
        pp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []

        if self.use_ae and ae_trajectory is not None:
            if lengths_ae is None:
                lengths_ae = torch.full(
                    (ae_trajectory.size(0),), ae_trajectory.size(1), device=ae_trajectory.device
                )
            embeddings.append(self.ae_encoder(ae_trajectory, lengths_ae))

        if self.use_vib and vib_trajectory is not None:
            if lengths_vib is None:
                lengths_vib = torch.full(
                    (vib_trajectory.size(0),), vib_trajectory.size(1), device=vib_trajectory.device
                )
            embeddings.append(self.vib_encoder(vib_trajectory, lengths_vib))

        if self.use_pp and pp is not None:
            embeddings.append(pp)

        x = torch.cat(embeddings, dim=-1)
        return self.fusion_mlp(x).squeeze(-1)


class TrajectoryAttention(nn.Module):
    """
    Temporal attention model for trajectories.

    Learns a scalar importance weight for each time step, pools the sequence
    using a weighted average, and regresses the target. Much faster than RNNs
    while still capturing which time segments are informative.
    """

    def __init__(
        self,
        ae_input_dim: int = 6,
        vib_input_dim: int = 9,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        use_ae: bool = True,
        use_vib: bool = True,
        use_pp: bool = False,
        pp_dim: int = 3,
    ) -> None:
        super().__init__()
        self.use_ae = use_ae
        self.use_vib = use_vib
        self.use_pp = use_pp

        # Context vector for attention
        self.ae_attention: nn.Module | None = None
        self.vib_attention: nn.Module | None = None

        embed_dim = 0
        if use_ae:
            self.ae_proj = nn.Sequential(
                nn.Linear(ae_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.ae_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            embed_dim += hidden_dim
        if use_vib:
            self.vib_proj = nn.Sequential(
                nn.Linear(vib_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.vib_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            embed_dim += hidden_dim
        if use_pp:
            embed_dim += pp_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _attn_pool(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        attn_mlp: nn.Module,
        proj: nn.Module,
    ) -> torch.Tensor:
        # x: (B, T, D)
        h = proj(x)  # (B, T, H)
        scores = attn_mlp(h).squeeze(-1)  # (B, T)

        # Mask padded positions
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (h * weights).sum(dim=1)  # (B, H)
        return pooled

    def forward(
        self,
        ae_trajectory: torch.Tensor | None = None,
        vib_trajectory: torch.Tensor | None = None,
        lengths_ae: torch.Tensor | None = None,
        lengths_vib: torch.Tensor | None = None,
        pp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []

        if self.use_ae and ae_trajectory is not None:
            if lengths_ae is None:
                lengths_ae = torch.full(
                    (ae_trajectory.size(0),), ae_trajectory.size(1), device=ae_trajectory.device
                )
            embeddings.append(
                self._attn_pool(ae_trajectory, lengths_ae, self.ae_attention, self.ae_proj)
            )

        if self.use_vib and vib_trajectory is not None:
            if lengths_vib is None:
                lengths_vib = torch.full(
                    (vib_trajectory.size(0),), vib_trajectory.size(1), device=vib_trajectory.device
                )
            embeddings.append(
                self._attn_pool(vib_trajectory, lengths_vib, self.vib_attention, self.vib_proj)
            )

        if self.use_pp and pp is not None:
            embeddings.append(pp)

        x = torch.cat(embeddings, dim=-1)
        return self.fusion_mlp(x).squeeze(-1)


class TransferFeatureMLP(nn.Module):
    """
    CPU-feasible transfer-learning fusion model for WP-6 side test.

    Uses pre-extracted frozen ResNet18 embeddings for AE and vibration
    spectrograms, plus optional process parameters, fused through a small MLP.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        use_ae: bool = True,
        use_vib: bool = True,
        use_pp: bool = False,
        pp_dim: int = 3,
    ) -> None:
        super().__init__()
        self.use_ae = use_ae
        self.use_vib = use_vib
        self.use_pp = use_pp

        input_dim = 0
        if use_ae:
            input_dim += embed_dim
        if use_vib:
            input_dim += embed_dim
        if use_pp:
            input_dim += pp_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        ae_embed: torch.Tensor | None = None,
        vib_embed: torch.Tensor | None = None,
        pp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.use_ae and ae_embed is not None:
            parts.append(ae_embed)
        if self.use_vib and vib_embed is not None:
            parts.append(vib_embed)
        if self.use_pp and pp is not None:
            parts.append(pp)
        x = torch.cat(parts, dim=-1)
        return self.mlp(x).squeeze(-1)


class MeanTrajectoryMLP(nn.Module):
    """
    Baseline that takes the time-averaged trajectory and passes it through an MLP.
    Tests whether temporal structure adds value beyond the mean representation.
    """

    def __init__(
        self,
        ae_input_dim: int = 6,
        vib_input_dim: int = 9,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        use_ae: bool = True,
        use_vib: bool = True,
        use_pp: bool = False,
        pp_dim: int = 3,
    ) -> None:
        super().__init__()
        self.use_ae = use_ae
        self.use_vib = use_vib
        self.use_pp = use_pp

        input_dim = 0
        if use_ae:
            input_dim += ae_input_dim
        if use_vib:
            input_dim += vib_input_dim
        if use_pp:
            input_dim += pp_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        ae_trajectory: torch.Tensor | None = None,
        vib_trajectory: torch.Tensor | None = None,
        lengths_ae: torch.Tensor | None = None,
        lengths_vib: torch.Tensor | None = None,
        pp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        if self.use_ae and ae_trajectory is not None:
            if lengths_ae is None:
                mean_ae = ae_trajectory.mean(dim=1)
            else:
                mask = torch.arange(ae_trajectory.size(1), device=ae_trajectory.device).unsqueeze(
                    0
                ) < lengths_ae.unsqueeze(1)
                masked = ae_trajectory * mask.unsqueeze(-1).float()
                mean_ae = masked.sum(dim=1) / lengths_ae.unsqueeze(1).clamp(min=1)
            parts.append(mean_ae)

        if self.use_vib and vib_trajectory is not None:
            if lengths_vib is None:
                mean_vib = vib_trajectory.mean(dim=1)
            else:
                mask = torch.arange(vib_trajectory.size(1), device=vib_trajectory.device).unsqueeze(
                    0
                ) < lengths_vib.unsqueeze(1)
                masked = vib_trajectory * mask.unsqueeze(-1).float()
                mean_vib = masked.sum(dim=1) / lengths_vib.unsqueeze(1).clamp(min=1)
            parts.append(mean_vib)

        if self.use_pp and pp is not None:
            parts.append(pp)

        x = torch.cat(parts, dim=-1)
        return self.mlp(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    # Traditional ML
    "RandomForestModel": RandomForestModel,
    "XGBoostModel": XGBoostModel,
    "LightGBMModel": LightGBMModel,
    "RidgeRegressionModel": RidgeRegressionModel,
    "ShallowMLPModel": ShallowMLPModel,
    # Deep Learning - Single modality
    "AECNN": AECNN,
    "VibCNN": VibCNN,
    "PhysicsMLP": PhysicsMLP,
    "ParamsMLP": ParamsMLP,
    "ResNetAECNN": ResNetAECNN,
    "ResNetVibCNN": ResNetVibCNN,
    "ResNetFusion": ResNetFusion,
    "MultiscaleSpectrogramCNN": MultiscaleSpectrogramCNN,
    "ChannelAttentionCNN": ChannelAttentionCNN,
    "SelfSupervisedPretrainedCNN": SelfSupervisedPretrainedCNN,
    # Deep Learning - Tabular / Sequence
    "TabNetRegressor": TabNetRegressor,
    "LSTMPhysicsModel": LSTMPhysicsModel,
    "TCNPhysicsModel": TCNPhysicsModel,
    # Deep Learning - Fusion
    "FusionModel": FusionModel,
    "FeatureOnlyModel": FeatureOnlyModel,
    "CrossModalTransformer": CrossModalTransformer,
    "PhysicsInformedFusionNet": PhysicsInformedFusionNet,
    "GatedMultimodalFusionNet": GatedMultimodalFusionNet,
    # Attention & Multi-modal Fusion
    "MultiHeadSelfAttention": MultiHeadSelfAttention,
    "TransformerEncoderFusion": TransformerEncoderFusion,
    "CrossAttentionFusionV1": CrossAttentionFusionV1,
    "SqueezeExcitationFusionV1": SqueezeExcitationFusionV1,
    "AttentionMLP": AttentionMLP,
    "ModalityGatedTransformer": ModalityGatedTransformer,
    "SpectrogramViT": SpectrogramViT,
    "PhysicsGuidedAttentionNet": PhysicsGuidedAttentionNet,
    "HierarchicalFusionNet": HierarchicalFusionNet,
    "DynamicWeightedFusion": DynamicWeightedFusion,
    # Attention-based and Advanced Fusion
    "MultiHeadAttentionFusion": MultiHeadAttentionFusion,
    "CrossAttentionFusion": CrossAttentionFusion,
    "SqueezeExcitationFusion": SqueezeExcitationFusion,
    "BilinearFusionNetwork": BilinearFusionNetwork,
    "GraphNeuralNetworkFusion": GraphNeuralNetworkFusion,
    "HierarchicalAttentionNetwork": HierarchicalAttentionNetwork,
    "PhysicsGuidedAttentionNetwork": PhysicsGuidedAttentionNetwork,
    "TabTransformerRegressor": TabTransformerRegressor,
    "TransferLearningFusionNet": TransferLearningFusionNet,
    "TransferFeatureMLP": TransferFeatureMLP,
    "TrajectoryLSTM": TrajectoryLSTM,
    "TrajectoryGRUDedicated": TrajectoryGRUDedicated,
    "TrajectoryTCN": TrajectoryTCN,
    "TrajectoryCNN": TrajectoryCNN,
    "TrajectoryAttention": TrajectoryAttention,
    "MeanTrajectoryMLP": MeanTrajectoryMLP,
}


def model_factory(model_name: str, **kwargs: Any) -> nn.Module:
    """
    Instantiate a model by name from the registry.

    Args:
        model_name: One of the keys in MODEL_REGISTRY.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)
