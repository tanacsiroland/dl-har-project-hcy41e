"""Shared seed, device, and runtime environment inspection helpers."""

from __future__ import annotations

import importlib
import os
import platform
import random
import sys
from typing import Any

import numpy as np
import yaml


def _get_package_version(import_name: str) -> str | None:
    try:
        mod = importlib.import_module(import_name)
    except ImportError:
        return None
    return str(getattr(mod, "__version__", "installed"))


def has_cuda() -> bool:
    """Return True only when CUDA is available AND at least one device is visible."""
    import torch

    # Guard against CUDA_VISIBLE_DEVICES="" leaving is_available() True with 0 devices.
    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


def resolve_device() -> Any:
    """Return CUDA device when visible, otherwise CPU."""
    import torch

    return torch.device("cuda" if has_cuda() else "cpu")


def set_seeds(seed: int, use_torch: bool = True, deterministic_cuda: bool = True) -> None:
    """Set Python, NumPy, and optionally PyTorch/cuDNN random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    if not use_torch:
        return
    import torch

    torch.manual_seed(seed)
    if deterministic_cuda and has_cuda():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collect_runtime_metadata(
    device: Any | None = None,
    include_torch: bool = True,
    include_sklearn: bool = False,
) -> dict[str, Any]:
    """Collect OS, Python, device, and package versions for run JSON metadata."""
    packages: dict[str, Any] = {}
    if include_torch:
        import torch

        packages["torch"] = torch.__version__
        packages["cuda"] = torch.version.cuda
    packages["numpy"] = np.__version__
    if include_sklearn:
        import sklearn

        packages["scikit-learn"] = sklearn.__version__
    packages["pyyaml"] = yaml.__version__

    info: dict[str, Any] = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_logical_count": os.cpu_count(),
        "packages": packages,
    }
    if device is not None:
        import torch

        info["device"] = str(device)
        info["gpu_name"] = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    return info


def collect_environment_info(strict_env: bool = False) -> dict[str, Any]:
    """Collect OS, Python, package, and GPU/CUDA runtime details for Phase 1 audit."""
    packages = {
        "torch": _get_package_version("torch"),
        "numpy": _get_package_version("numpy"),
        "scikit-learn": _get_package_version("sklearn"),
        "pyyaml": _get_package_version("yaml"),
        "matplotlib": _get_package_version("matplotlib"),
    }

    missing = [name for name, ver in packages.items() if ver is None]
    if strict_env and missing:
        raise RuntimeError(
            f"Missing required Python packages in current environment: {missing}"
        )

    cuda_info: dict[str, Any] = {
        "torch_installed": packages["torch"] is not None,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
        "cuda_tensor_sanity_ok": None,
        "cuda_sanity_error": None,
    }
    if packages["torch"] is not None:
        import torch

        cuda_available = has_cuda()
        cuda_info["cuda_available"] = cuda_available
        cuda_info["cuda_version"] = torch.version.cuda
        cuda_info["cudnn_version"] = (
            torch.backends.cudnn.version()
            if cuda_available and torch.backends.cudnn.is_available()
            else None
        )
        cuda_info["device_count"] = int(torch.cuda.device_count())
        devices = []
        for idx in range(cuda_info["device_count"]):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
        cuda_info["devices"] = devices

        if cuda_available:
            try:
                x = torch.ones((2, 9, 128), device="cuda", dtype=torch.float32)
                y = (x * 2.0).sum().item()
                cuda_info["cuda_tensor_sanity_ok"] = bool(y == 2 * 9 * 128 * 2.0)
            except Exception as exc:  # pragma: no cover - hardware dependent
                cuda_info["cuda_tensor_sanity_ok"] = False
                cuda_info["cuda_sanity_error"] = str(exc)

    return {
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "in_virtualenv": sys.prefix != sys.base_prefix,
        "cpu_logical_count": os.cpu_count(),
        "packages": packages,
        "missing_packages": missing,
        "cuda": cuda_info,
    }
