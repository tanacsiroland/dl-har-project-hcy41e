"""Leak-free per-window summary feature extraction for classical HAR baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.core.data import SIGNAL_CHANNELS

SUPPORTED_STATS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "iqr",
    "rms",
    "mean_abs_diff",
)


def _compute_single_stat(x_raw: np.ndarray, stat_name: str) -> np.ndarray:
    """Compute one statistic independently for each window & channel -> shape (N, C)."""
    if stat_name == "mean":
        return np.mean(x_raw, axis=-1)
    if stat_name == "std":
        return np.std(x_raw, axis=-1)
    if stat_name == "min":
        return np.min(x_raw, axis=-1)
    if stat_name == "max":
        return np.max(x_raw, axis=-1)
    if stat_name == "median":
        return np.median(x_raw, axis=-1)
    if stat_name == "iqr":
        q75, q25 = np.percentile(x_raw, [75.0, 25.0], axis=-1)
        return q75 - q25
    if stat_name == "rms":
        return np.sqrt(np.mean(np.square(x_raw), axis=-1))
    if stat_name == "mean_abs_diff":
        return np.mean(np.abs(np.diff(x_raw, axis=-1)), axis=-1)
    raise ValueError(
        f"Unsupported summary stat '{stat_name}'. Supported: {SUPPORTED_STATS}"
    )


def extract_window_summary_features(
    x_raw: np.ndarray,
    stats: Sequence[str] = SUPPORTED_STATS,
) -> np.ndarray:
    """Compute ``stats`` along the time axis of ``(N, C, T)``; returns ``(N, C * len(stats))``."""
    if x_raw.ndim != 3:
        raise ValueError(f"Expected 3D tensor (N, C, T), got shape {x_raw.shape}")
    if not stats:
        raise ValueError("At least one summary statistic must be specified.")

    blocks = [_compute_single_stat(x_raw, name).astype(np.float32) for name in stats]
    # Interleave by channel so features for the same channel stay grouped together.
    stacked = np.stack(blocks, axis=-1)  # (N, C, num_stats)
    features = stacked.reshape(x_raw.shape[0], -1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Extracted summary features contain NaN or Inf values.")
    return features


def get_summary_feature_names(
    channels: Sequence[str] = SIGNAL_CHANNELS,
    stats: Sequence[str] = SUPPORTED_STATS,
) -> list[str]:
    """Return ordered feature names matching ``extract_window_summary_features``."""
    return [f"{ch}__{stat}" for ch in channels for stat in stats]
