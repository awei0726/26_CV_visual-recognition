"""A framewise CNN and length-preserving TCN for visual CTC recognition.

No pretrained backbone, audio, or ESPnet dependency is required.  Every frame
uses the same spatial encoder.  Temporal convolutions are non-causal: this
model recognizes complete clips, rather than claiming streaming inference.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Integral, Real

import torch
from torch import Tensor, nn


def _groups(channels: int) -> int:
    """Choose a small GroupNorm group count which divides the channel count."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class SpatialEncoder(nn.Module):
    """Extract spatial features independently for each grayscale frame."""

    def __init__(self, channels: Sequence[int], feature_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = 1
        for width in channels:
            # GroupNorm sees one frame at a time. Unlike BatchNorm, padded
            # frames cannot change the statistics of the real frames.
            for stride in (2, 1):
                layers.extend(
                    [
                        nn.Conv2d(previous, width, 3, stride=stride, padding=1, bias=False),
                        nn.GroupNorm(_groups(width), width),
                        nn.GELU(),
                    ]
                )
                previous = width
        self.features = nn.Sequential(*layers)
        self.pool = SpatialMeanPool()
        self.projection = nn.Linear(previous * 4, feature_dim)

    def forward(self, frames: Tensor) -> Tensor:
        return self.projection(self.pool(self.features(frames)).flatten(1))


class SpatialMeanPool(nn.Module):
    """Pool powers-of-two feature maps into 2x2 without CUDA atomic backward.

    The fixed 64x64 input and stride-two CNN produce powers of two (or 1).
    Reshape + mean is equivalent to adaptive average pooling for these sizes,
    while supporting strict deterministic CUDA gradients.
    """

    def forward(self, features: Tensor) -> Tensor:
        batch, channels, height, width = features.shape
        if height == 1 and width == 1:
            return features.expand(batch, channels, 2, 2)
        if height % 2 or width % 2:
            raise ValueError("Spatial encoder output must be divisible into 2x2 bins")
        return features.reshape(batch, channels, 2, height // 2, 2, width // 2).mean(dim=(3, 5))


class TemporalBlock(nn.Module):
    """Two dilated convolutions, with channel-only normalization per frame."""

    def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(width, width, kernel_size, padding=padding, dilation=dilation),
                nn.Conv1d(width, width, kernel_size, padding=padding, dilation=dilation),
            ]
        )
        self.normalizations = nn.ModuleList([nn.LayerNorm(width), nn.LayerNorm(width)])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        residual = features
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            features = convolution(features)
            features = normalization(features.transpose(1, 2)).transpose(1, 2)
            features = self.dropout(self.activation(features))
            # Mask after EVERY convolution. Biases/normalization on padded
            # positions must not leak back into valid positions in later layers.
            features = features.masked_fill(~mask, 0)
        return (residual + features).masked_fill(~mask, 0)


class StudentModel(nn.Module):
    """Tiny CNN + residual dilated TCN + per-frame token classification.

    Args:
        vocab_size: Number of Teacher-compatible output classes (GRID: 41).
        cnn_channels: Spatial widths; each stage halves spatial resolution only.
        feature_dim: Framewise CNN feature size.
        hidden_dim: TCN channel count.
        tcn_layers: Number of two-convolution blocks, dilations 1, 2, 4, ... .
        kernel_size: An odd temporal kernel size to preserve the frame count.
        dropout: Temporal dropout probability.

    ``forward`` accepts normalized ``[B, T, 1, 64, 64]`` and integer valid
    lengths ``[B]``. It returns unnormalized logits ``[B, T, vocab_size]``.
    Lengths stay unchanged. Padded output logits are zero and must still be
    excluded from losses and decoding with the valid lengths.
    """

    def __init__(
        self,
        vocab_size: int = 41,
        cnn_channels: Sequence[int] = (16, 32, 64),
        feature_dim: int = 128,
        hidden_dim: int = 128,
        tcn_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if (not isinstance(cnn_channels, Sequence) or isinstance(cnn_channels, (str, bytes))
                or not cnn_channels or any(isinstance(c, bool) or not isinstance(c, Integral) or c <= 0 for c in cnn_channels)):
            raise ValueError("cnn_channels must contain positive integers")
        sizes = (vocab_size, feature_dim, hidden_dim, tcn_layers, kernel_size)
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in sizes):
            raise ValueError("vocab_size, feature_dim, hidden_dim, tcn_layers and kernel_size must be integers")
        if vocab_size < 2 or feature_dim < 1 or hidden_dim < 1 or tcn_layers < 1:
            raise ValueError("vocab_size >= 2 and positive feature/hidden/layer sizes are required")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        vocab_size, feature_dim, hidden_dim, tcn_layers, kernel_size = map(int, sizes)
        cnn_channels = tuple(map(int, cnn_channels))
        dropout = float(dropout)
        self._config = {
            "vocab_size": vocab_size,
            "cnn_channels": list(cnn_channels),
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "tcn_layers": tcn_layers,
            "kernel_size": kernel_size,
            "dropout": dropout,
        }
        self.spatial_encoder = SpatialEncoder(cnn_channels, feature_dim)
        self.temporal_projection = nn.Linear(feature_dim, hidden_dim)
        self.temporal_encoder = nn.ModuleList(
            [TemporalBlock(hidden_dim, kernel_size, 2**i, dropout) for i in range(tcn_layers)]
        )
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def get_config(self) -> dict:
        """Return a JSON-serializable configuration for checkpoint reconstruction."""
        return {**self._config, "cnn_channels": list(self._config["cnn_channels"])}

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, frames: Tensor, input_lengths: Tensor) -> Tensor:
        if frames.ndim != 5 or frames.shape[2:] != (1, 64, 64):
            raise ValueError("frames must have shape [B, T, 1, 64, 64]")
        batch, time = frames.shape[:2]
        if batch < 1 or time < 1 or not frames.is_floating_point():
            raise ValueError("frames must be a nonempty floating-point tensor")
        if input_lengths.ndim != 1 or input_lengths.shape[0] != batch:
            raise ValueError("input_lengths must have shape [B]")
        if input_lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_lengths must be an int32 or int64 tensor")
        lengths = input_lengths.to(device=frames.device)
        if bool(((lengths < 1) | (lengths > time)).any()):
            raise ValueError("every input length must be between 1 and padded T")
        valid = torch.arange(time, device=frames.device)[None, :] < lengths[:, None]
        # Avoid spending CNN computation on padding. GroupNorm remains per-frame,
        # so this selection also cannot couple samples through batch statistics.
        flat_frames = frames.reshape(batch * time, 1, 64, 64)
        valid_flat = valid.reshape(-1)
        valid_features = self.spatial_encoder(flat_frames[valid_flat])
        spatial = valid_features.new_zeros(batch * time, self._config["feature_dim"])
        spatial[valid_flat] = valid_features
        features = self.temporal_projection(spatial.reshape(batch, time, -1))
        mask = valid[:, None, :]
        features = features.transpose(1, 2).masked_fill(~mask, 0)
        for block in self.temporal_encoder:
            features = block(features, mask)
        logits = self.classifier(features.transpose(1, 2))
        return logits.masked_fill(~valid[:, :, None], 0)
