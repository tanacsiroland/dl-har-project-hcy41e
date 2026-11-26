"""Dataset loading and strictly subject-disjoint splitting utilities for UCI HAR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SIGNAL_CHANNELS: tuple[str, ...] = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)

ACTIVITY_NAMES: tuple[str, ...] = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)

EXPECTED_TIMESTEPS = 128


@dataclass(frozen=True)
class DatasetSplit:
    """Immutable container for one subject-disjoint raw inertial dataset split."""

    name: str
    x_raw: np.ndarray  # Shape: (N, C, T), float32
    y: np.ndarray  # Shape: (N,), int64 in [0, 5]
    subjects: np.ndarray  # Shape: (N,), int64

    def __post_init__(self) -> None:
        n = self.x_raw.shape[0]
        if self.x_raw.ndim != 3 or self.x_raw.shape[1:] != (
            len(SIGNAL_CHANNELS),
            EXPECTED_TIMESTEPS,
        ):
            raise ValueError(
                f"Expected x_raw shape (N, {len(SIGNAL_CHANNELS)}, {EXPECTED_TIMESTEPS}), "
                f"got {self.x_raw.shape}"
            )
        if self.y.shape != (n,) or self.subjects.shape != (n,):
            raise ValueError(
                f"Shape mismatch in split '{self.name}': "
                f"x_raw={self.x_raw.shape}, y={self.y.shape}, subjects={self.subjects.shape}"
            )
        if not np.isfinite(self.x_raw).all():
            raise ValueError(f"Non-finite values found in split '{self.name}' x_raw.")
        if self.y.min() < 0 or self.y.max() >= len(ACTIVITY_NAMES):
            raise ValueError(f"Labels out of [0, {len(ACTIVITY_NAMES) - 1}] range.")

    @property
    def unique_subjects(self) -> list[int]:
        return [int(s) for s in np.unique(self.subjects).tolist()]

    def subset(self, indices: np.ndarray, name: str) -> "DatasetSplit":
        return DatasetSplit(
            name=name,
            x_raw=self.x_raw[indices],
            y=self.y[indices],
            subjects=self.subjects[indices],
        )


def load_subject_ids(data_dir: Path, split: str) -> set[int]:
    """Read only subject IDs for a split (used to assert test-set disjointness without loading test data)."""
    path = data_dir / split / f"subject_{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing subject file: {path}")
    arr = np.loadtxt(path, dtype=np.int64)
    return {int(s) for s in np.unique(arr).tolist()}


def load_raw_split(data_dir: Path, split: str) -> DatasetSplit:
    """Load raw 9-channel inertial signals (N, 9, 128), zero-indexed labels, and subjects."""
    if split not in {"train", "test"}:
        raise ValueError(f"Split must be 'train' or 'test', got '{split}'")

    split_dir = data_dir / split
    inertial_dir = split_dir / "Inertial Signals"
    channels = [
        np.loadtxt(inertial_dir / f"{ch}_{split}.txt", dtype=np.float32)
        for ch in SIGNAL_CHANNELS
    ]
    x_raw = np.stack(channels, axis=1)
    # UCI HAR labels are 1..6; convert to 0..5 for PyTorch/sklearn consistency.
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=np.int64)
    return DatasetSplit(name=split, x_raw=x_raw, y=y, subjects=subjects)


def make_subject_disjoint_split(
    train_split: DatasetSplit,
    n_splits: int = 4,
    val_fold: int = 0,
    seed: int = 42,
    locked_test_subjects: set[int] | None = None,
) -> tuple[DatasetSplit, DatasetSplit]:
    """Partition official train subjects into strictly subject-disjoint train and val splits."""
    if not (0 <= val_fold < n_splits):
        raise ValueError(f"val_fold={val_fold} must be in [0, {n_splits - 1}]")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(train_split.y)), train_split.y, train_split.subjects))
    train_idx, val_idx = folds[val_fold]

    train_out = train_split.subset(train_idx, name="train")
    val_out = train_split.subset(val_idx, name="val")

    assert_subject_disjointness(
        train_subjects=set(train_out.unique_subjects),
        val_subjects=set(val_out.unique_subjects),
        test_subjects=locked_test_subjects,
    )
    return train_out, val_out


def assert_subject_disjointness(
    train_subjects: set[int],
    val_subjects: set[int] | None = None,
    test_subjects: set[int] | None = None,
) -> None:
    """Assert zero subject_id overlap across all provided dataset splits."""
    val_set = val_subjects or set()
    test_set = test_subjects or set()

    if train_subjects & val_set:
        raise AssertionError(
            f"Subject leakage between train and val: {sorted(train_subjects & val_set)}"
        )
    if train_subjects & test_set:
        raise AssertionError(
            f"Subject leakage between train and test: {sorted(train_subjects & test_set)}"
        )
    if val_set & test_set:
        raise AssertionError(
            f"Subject leakage between val and test: {sorted(val_set & test_set)}"
        )


def resolve_channel_indices(channel_names: list[str] | None) -> list[int]:
    """Map optional channel names to canonical indices in SIGNAL_CHANNELS."""
    if channel_names is None:
        return list(range(len(SIGNAL_CHANNELS)))
    name_to_idx = {name: idx for idx, name in enumerate(SIGNAL_CHANNELS)}
    indices: list[int] = []
    for name in channel_names:
        if name not in name_to_idx:
            raise ValueError(
                f"Unknown channel '{name}'. Valid channels: {SIGNAL_CHANNELS}"
            )
        indices.append(name_to_idx[name])
    if not indices:
        raise ValueError("Channel list cannot be empty.")
    return indices


def compute_channel_standardizer(
    x_train: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std strictly from training windows ``(N, C, T)``."""
    if x_train.ndim != 3:
        raise ValueError(f"Expected 3D tensor (N, C, T), got {x_train.shape}")
    mean = x_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = x_train.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.where(std < eps, np.float32(1.0), std).astype(np.float32)
    return mean, std


def apply_channel_standardizer(
    x_raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply precomputed train-fit per-channel standardization to ``(N, C, T)``."""
    standardized = ((x_raw - mean) / std).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError("Standardized tensor contains non-finite values.")
    return standardized
