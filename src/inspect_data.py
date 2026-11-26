"""Phase 1: check the Python environment and the raw UCI HAR files.

Checks that all files exist, the tensor shapes are (N, C, T), there are no NaN/Inf
values, the labels are valid and no subject is in both train and test.
Writes the summary to outputs/dataset_summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.data import EXPECTED_TIMESTEPS, SIGNAL_CHANNELS

EXPECTED_FEATURE_DIM = 561
EXPECTED_TOTAL_SUBJECTS = 30
REQUIRED_ROOT_FILES: tuple[str, ...] = (
    "activity_labels.txt",
    "features.txt",
    "features_info.txt",
    "README.txt",
)


def _get_package_version(import_name: str) -> str | None:
    try:
        mod = importlib.import_module(import_name)
    except ImportError:
        return None
    return str(getattr(mod, "__version__", "installed"))


def collect_environment_info(strict_env: bool = False) -> dict[str, Any]:
    """Collect OS, Python, package, and GPU/CUDA runtime details."""
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

        cuda_available = bool(torch.cuda.is_available())
        cuda_info["cuda_available"] = cuda_available
        cuda_info["cuda_version"] = torch.version.cuda
        cuda_info["cudnn_version"] = (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
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


def _load_activity_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            idx_str, name = stripped.split(maxsplit=1)
            labels[int(idx_str)] = name
    return labels


def _hash_windows(arr: np.ndarray) -> set[bytes]:
    # Hash each (C, T) window's raw bytes to detect exact duplicates without high memory use.
    return {hashlib.blake2b(arr[i].tobytes(), digest_size=16).digest() for i in range(arr.shape[0])}


def inspect_split(
    data_dir: Path,
    split: str,
    activity_labels: dict[int, str],
) -> tuple[dict[str, Any], np.ndarray, set[int], set[bytes]]:
    """Load, validate, and summarize one UCI HAR split ('train' or 'test')."""
    split_dir = data_dir / split
    inertial_dir = split_dir / "Inertial Signals"
    required_files = [
        split_dir / f"X_{split}.txt",
        split_dir / f"y_{split}.txt",
        split_dir / f"subject_{split}.txt",
        *(inertial_dir / f"{ch}_{split}.txt" for ch in SIGNAL_CHANNELS),
    ]
    missing_files = [str(p) for p in required_files if not p.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files for split '{split}': {missing_files}"
        )

    channel_arrays: list[np.ndarray] = []
    for ch in SIGNAL_CHANNELS:
        ch_path = inertial_dir / f"{ch}_{split}.txt"
        arr = np.loadtxt(ch_path, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_TIMESTEPS:
            raise ValueError(
                f"Unexpected shape in {ch_path}: {arr.shape}, expected (N, {EXPECTED_TIMESTEPS})"
            )
        channel_arrays.append(arr)

    x_raw = np.stack(channel_arrays, axis=1)  # (N, C, T)
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64)
    subjects = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=np.int64)
    x_feat = np.loadtxt(split_dir / f"X_{split}.txt", dtype=np.float32)

    n_samples = int(x_raw.shape[0])
    if y.shape != (n_samples,) or subjects.shape != (n_samples,):
        raise ValueError(
            f"Row count mismatch in '{split}': x_raw={x_raw.shape}, y={y.shape}, subjects={subjects.shape}"
        )
    if x_feat.shape != (n_samples, EXPECTED_FEATURE_DIM):
        raise ValueError(
            f"Feature matrix shape mismatch in '{split}': {x_feat.shape}, "
            f"expected ({n_samples}, {EXPECTED_FEATURE_DIM})"
        )

    raw_nan_count = int(np.isnan(x_raw).sum())
    raw_inf_count = int(np.isinf(x_raw).sum())
    feat_nan_count = int(np.isnan(x_feat).sum())
    feat_inf_count = int(np.isinf(x_feat).sum())
    if raw_nan_count or raw_inf_count or feat_nan_count or feat_inf_count:
        raise ValueError(
            f"Non-finite values detected in '{split}': "
            f"raw_nan={raw_nan_count}, raw_inf={raw_inf_count}, "
            f"feat_nan={feat_nan_count}, feat_inf={feat_inf_count}"
        )

    label_ids, label_counts = np.unique(y, return_counts=True)
    expected_label_ids = sorted(activity_labels.keys())
    if label_ids.tolist() != expected_label_ids:
        raise ValueError(
            f"Label set mismatch in '{split}': found {label_ids.tolist()}, expected {expected_label_ids}"
        )

    class_distribution = {
        activity_labels[int(lbl)]: {
            "label_id": int(lbl),
            "count": int(cnt),
            "fraction": round(float(cnt) / n_samples, 6),
        }
        for lbl, cnt in zip(label_ids, label_counts)
    }

    subj_ids, subj_counts = np.unique(subjects, return_counts=True)
    windows_per_subject = {
        str(int(sid)): int(cnt) for sid, cnt in zip(subj_ids, subj_counts)
    }

    per_channel_stats = {}
    for idx, ch_name in enumerate(SIGNAL_CHANNELS):
        ch_data = x_raw[:, idx, :]
        per_channel_stats[ch_name] = {
            "min": round(float(ch_data.min()), 6),
            "max": round(float(ch_data.max()), 6),
            "mean": round(float(ch_data.mean()), 6),
            "std": round(float(ch_data.std()), 6),
        }

    window_hashes = _hash_windows(x_raw)
    summary = {
        "num_windows": n_samples,
        "raw_shape_nct": list(x_raw.shape),
        "raw_dtype": str(x_raw.dtype),
        "feature_shape": list(x_feat.shape),
        "feature_dtype": str(x_feat.dtype),
        "label_dtype": str(y.dtype),
        "subject_dtype": str(subjects.dtype),
        "raw_nan_count": raw_nan_count,
        "raw_inf_count": raw_inf_count,
        "feature_nan_count": feat_nan_count,
        "feature_inf_count": feat_inf_count,
        "exact_unique_windows": len(window_hashes),
        "exact_duplicate_windows_within_split": n_samples - len(window_hashes),
        "num_subjects": int(len(subj_ids)),
        "subject_ids": [int(s) for s in subj_ids.tolist()],
        "windows_per_subject": windows_per_subject,
        "windows_per_subject_min": int(subj_counts.min()),
        "windows_per_subject_max": int(subj_counts.max()),
        "class_distribution": class_distribution,
        "per_channel_stats": per_channel_stats,
    }
    return summary, x_raw, set(int(s) for s in subj_ids.tolist()), window_hashes


def audit_dataset(data_dir: Path, strict_env: bool = False) -> dict[str, Any]:
    """Run full environment and UCI HAR dataset verification."""
    t0 = time.perf_counter()
    env_info = collect_environment_info(strict_env=strict_env)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    missing_root = [
        str(data_dir / fname)
        for fname in REQUIRED_ROOT_FILES
        if not (data_dir / fname).is_file()
    ]
    if missing_root:
        raise FileNotFoundError(f"Missing required root dataset files: {missing_root}")

    activity_labels = _load_activity_labels(data_dir / "activity_labels.txt")
    with (data_dir / "features.txt").open("r", encoding="utf-8") as handle:
        feature_names = [line.strip() for line in handle if line.strip()]
    if len(feature_names) != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_DIM} entries in features.txt, found {len(feature_names)}"
        )

    train_summary, _, train_subjects, train_hashes = inspect_split(
        data_dir, "train", activity_labels
    )
    test_summary, _, test_subjects, test_hashes = inspect_split(
        data_dir, "test", activity_labels
    )

    overlap_subjects = sorted(train_subjects & test_subjects)
    all_subjects = sorted(train_subjects | test_subjects)
    if overlap_subjects:
        raise AssertionError(
            f"Subject leakage between official train and test splits: {overlap_subjects}"
        )
    if len(all_subjects) != EXPECTED_TOTAL_SUBJECTS:
        raise AssertionError(
            f"Expected {EXPECTED_TOTAL_SUBJECTS} unique subjects, found {len(all_subjects)}"
        )

    cross_split_exact_dups = len(train_hashes & test_hashes)
    elapsed = round(time.perf_counter() - t0, 3)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "dataset_path": str(data_dir.as_posix()),
        "signal_channels": list(SIGNAL_CHANNELS),
        "sampling_rate_hz": 50,
        "window_duration_sec": 2.56,
        "window_overlap_fraction": 0.5,
        "timesteps_per_window": EXPECTED_TIMESTEPS,
        "activity_labels": {str(k): v for k, v in sorted(activity_labels.items())},
        "total_windows": train_summary["num_windows"] + test_summary["num_windows"],
        "total_subjects": len(all_subjects),
        "subject_disjointness": {
            "strictly_disjoint": True,
            "overlapping_subject_ids": overlap_subjects,
            "cross_split_exact_duplicate_windows": cross_split_exact_dups,
        },
        "splits": {
            "train": train_summary,
            "test": test_summary,
        },
        "environment": env_info,
    }


def _print_console_summary(report: dict[str, Any]) -> None:
    env = report["environment"]
    cuda = env["cuda"]
    train = report["splits"]["train"]
    test = report["splits"]["test"]

    print("=== Phase 1: dataset audit ===")
    print(f"OS: {env['os']} | CPU logical cores: {env['cpu_logical_count']}")
    print(
        f"Python: {env['python_version']} ({env['python_executable']}) | "
        f"venv={env['in_virtualenv']}"
    )
    print(f"Packages: {env['packages']}")
    print(
        f"CUDA available: {cuda['cuda_available']} | CUDA version: {cuda['cuda_version']} | "
        f"Devices: {cuda['devices']} | Tensor test OK: {cuda['cuda_tensor_sanity_ok']}"
    )
    print(
        f"Train raw shape: {tuple(train['raw_shape_nct'])} ({train['raw_dtype']}) | "
        f"subjects ({train['num_subjects']}): {train['subject_ids']}"
    )
    print(
        f"Test  raw shape: {tuple(test['raw_shape_nct'])} ({test['raw_dtype']}) | "
        f"subjects ({test['num_subjects']}): {test['subject_ids']}"
    )
    print(
        f"Total windows: {report['total_windows']} | Total subjects: {report['total_subjects']} | "
        f"Strictly disjoint: {report['subject_disjointness']['strictly_disjoint']} | "
        f"Cross-split exact duplicates: {report['subject_disjointness']['cross_split_exact_duplicate_windows']}"
    )
    print(f"Audit runtime: {report['runtime_seconds']}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit environment and UCI HAR dataset integrity."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/UCI HAR Dataset"),
        help="Path to extracted 'UCI HAR Dataset' directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dataset_summary.json"),
        help="Output JSON path for verified audit statistics.",
    )
    parser.add_argument(
        "--strict-env",
        action="store_true",
        help="Fail if any package in requirements.txt is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_dataset(args.data_dir, strict_env=args.strict_env)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    _print_console_summary(report)
    print(f"Saved audit report to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
