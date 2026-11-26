"""Compact 1D Convolutional Neural Networks trained from scratch on inertial windows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


class CompactCNN1D(nn.Module):
    """Lightweight sequential temporal 1D-CNN for ``(B, C_in, T)`` IMU windows."""

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 6,
        conv_channels: Sequence[int] = (32, 64, 128),
        kernel_sizes: Sequence[int] = (7, 5, 3),
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        head_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if len(conv_channels) != len(kernel_sizes) or not conv_channels:
            raise ValueError(
                "conv_channels and kernel_sizes must be non-empty and equal length."
            )

        blocks: list[nn.Module] = []
        prev_c = in_channels
        num_blocks = len(conv_channels)
        for idx, (out_c, k_size) in enumerate(zip(conv_channels, kernel_sizes)):
            padding = k_size // 2
            blocks.append(
                nn.Conv1d(
                    prev_c,
                    out_c,
                    kernel_size=k_size,
                    padding=padding,
                    bias=not use_batchnorm,
                )
            )
            if use_batchnorm:
                blocks.append(nn.BatchNorm1d(out_c))
            blocks.append(nn.ReLU(inplace=True))
            if idx < num_blocks - 1:
                blocks.append(nn.MaxPool1d(kernel_size=2))
            prev_c = out_c

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = _build_head(prev_c, num_classes, dropout, head_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = self.global_pool(z)
        return self.classifier(z)


class _ResidualBlock1D(nn.Module):
    """Two-layer temporal 1D residual block with optional projection shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        bias = not use_batchnorm

        layers1: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        ]
        if use_batchnorm:
            layers1.append(nn.BatchNorm1d(out_channels))
        layers1.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers1.append(nn.Dropout(p=dropout))
        layers1.append(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=bias)
        )
        if use_batchnorm:
            layers1.append(nn.BatchNorm1d(out_channels))
        self.body = nn.Sequential(*layers1)

        if in_channels != out_channels:
            proj: list[nn.Module] = [nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)]
            if use_batchnorm:
                proj.append(nn.BatchNorm1d(out_channels))
            self.shortcut: nn.Module = nn.Sequential(*proj)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.shortcut(x))


class CompactResCNN1D(nn.Module):
    """Compact residual 1D-CNN for ``(B, C_in, T)`` inertial time-series windows."""

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 6,
        conv_channels: Sequence[int] = (32, 64, 128),
        kernel_sizes: Sequence[int] = (7, 5, 3),
        dropout: float = 0.3,
        block_dropout: float = 0.1,
        use_batchnorm: bool = True,
        head_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if len(conv_channels) != len(kernel_sizes) or not conv_channels:
            raise ValueError(
                "conv_channels and kernel_sizes must be non-empty and equal length."
            )

        stages: list[nn.Module] = []
        prev_c = in_channels
        num_blocks = len(conv_channels)
        for idx, (out_c, k_size) in enumerate(zip(conv_channels, kernel_sizes)):
            stages.append(
                _ResidualBlock1D(
                    in_channels=prev_c,
                    out_channels=out_c,
                    kernel_size=k_size,
                    dropout=block_dropout,
                    use_batchnorm=use_batchnorm,
                )
            )
            if idx < num_blocks - 1:
                stages.append(nn.MaxPool1d(kernel_size=2))
            prev_c = out_c

        self.features = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = _build_head(prev_c, num_classes, dropout, head_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = self.global_pool(z)
        return self.classifier(z)


def _build_head(
    in_dim: int,
    num_classes: int,
    dropout: float,
    head_hidden: int | None,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Flatten()]
    if dropout > 0.0:
        layers.append(nn.Dropout(p=dropout))
    if head_hidden is not None and head_hidden > 0:
        layers.append(nn.Linear(in_dim, head_hidden))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(head_hidden, num_classes))
    else:
        layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)


def build_cnn_model(
    model_cfg: dict[str, Any],
    in_channels: int,
    num_classes: int = 6,
) -> nn.Module:
    """Instantiate a configured 1D-CNN model from YAML parameters."""
    name = str(model_cfg["name"])
    common = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        conv_channels=model_cfg["conv_channels"],
        kernel_sizes=model_cfg["kernel_sizes"],
        dropout=float(model_cfg["dropout"]),
        use_batchnorm=bool(model_cfg.get("use_batchnorm", True)),
        head_hidden=model_cfg.get("head_hidden"),
    )
    if name == "CompactCNN1D":
        return CompactCNN1D(**common)
    if name == "CompactResCNN1D":
        return CompactResCNN1D(
            **common,
            block_dropout=float(model_cfg.get("block_dropout", 0.1)),
        )
    raise ValueError(f"Unsupported model name '{name}'")
