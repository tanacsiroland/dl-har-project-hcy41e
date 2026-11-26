"""PyTorch Dataset wrapper for standardized raw inertial windows."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class InertialWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory PyTorch Dataset returning ``(window[C, T], label)`` pairs."""

    def __init__(self, x_tensor: np.ndarray, y: np.ndarray) -> None:
        if x_tensor.ndim != 3:
            raise ValueError(f"Expected 3D array (N, C, T), got {x_tensor.shape}")
        if len(x_tensor) != len(y):
            raise ValueError(
                f"Length mismatch: x={len(x_tensor)} vs y={len(y)}"
            )
        self.x = torch.from_numpy(np.ascontiguousarray(x_tensor, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
