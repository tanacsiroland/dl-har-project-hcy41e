"""Phase 3: train the 1D-CNN on one train/validation split.

The best epoch and early stopping are chosen by validation Macro-F1.
Test data is not loaded here.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.core.data import (
    SIGNAL_CHANNELS,
    load_raw_split,
    load_subject_ids,
    make_subject_disjoint_split,
    resolve_channel_indices,
)
from src.core.env import collect_runtime_metadata, resolve_device
from src.core.trainer import run_single_training


def train_and_evaluate(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Train configured 1D-CNN from scratch and write Phase 3 JSON report."""
    t_start = time.perf_counter()
    seed = int(cfg["seed"])
    device = resolve_device()

    data_dir = Path(cfg["data_dir"])
    locked_test_subjects = load_subject_ids(data_dir, "test")
    full_train = load_raw_split(data_dir, "train")

    split_cfg = cfg["split"]
    train_split, val_split = make_subject_disjoint_split(
        full_train,
        n_splits=int(split_cfg["n_splits"]),
        val_fold=int(split_cfg["val_fold"]),
        seed=seed,
        locked_test_subjects=locked_test_subjects,
    )

    ch_indices = resolve_channel_indices(cfg.get("channels"))
    selected_channel_names = [SIGNAL_CHANNELS[i] for i in ch_indices]

    out_cfg = cfg["outputs"]
    ckpt_path = Path(out_cfg["checkpoint_path"])
    run_res = run_single_training(
        train_split=train_split,
        val_split=val_split,
        model_cfg=cfg["model"],
        train_cfg=cfg["training"],
        ch_indices=ch_indices,
        seed=seed,
        device=device,
        checkpoint_path=ckpt_path,
        aug_cfg=cfg.get("augmentation"),
        compute_latency=True,
    )

    best_epoch = run_res["best_epoch"]
    best_val_metrics = run_res["val_metrics"]
    best_train_metrics = run_res["train_metrics"]
    model_cfg = cfg["model"]

    baseline_comparison: dict[str, Any] | None = None
    baselines_json_path = Path(cfg.get("baselines_json", ""))
    if baselines_json_path.is_file():
        with baselines_json_path.open("r", encoding="utf-8") as handle:
            b_data = json.load(handle)
        lr_best = b_data["models"]["logistic_regression"]["best_on_val"]
        rf_best = b_data["models"]["random_forest"]["best_on_val"]
        baseline_comparison = {
            "logistic_regression_val_macro_f1": lr_best["val_metrics"]["macro_f1"],
            "random_forest_val_macro_f1": rf_best["val_metrics"]["macro_f1"],
            "cnn_minus_logreg_macro_f1": round(
                best_val_metrics["macro_f1"] - lr_best["val_metrics"]["macro_f1"], 6
            ),
            "cnn_minus_rf_macro_f1": round(
                best_val_metrics["macro_f1"] - rf_best["val_metrics"]["macro_f1"], 6
            ),
        }

    elapsed = round(time.perf_counter() - t_start, 3)

    return {
        "phase": "phase_3_cnn_baseline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "config_path": str(config_path.as_posix()),
        "checkpoint_path": str(ckpt_path.as_posix()),
        "seed": seed,
        "determinism_note": "Seeds and cuDNN deterministic flags set; exact bit-level reproducibility across hardware/CUDA kernels is not guaranteed.",
        "test_set_status": "LOCKED_UNTOUCHED",
        "environment": collect_runtime_metadata(device=device, include_torch=True),
        "splits": {
            "train": {
                "num_windows": int(len(train_split.y)),
                "num_subjects": len(train_split.unique_subjects),
                "subject_ids": train_split.unique_subjects,
            },
            "val": {
                "num_windows": int(len(val_split.y)),
                "num_subjects": len(val_split.unique_subjects),
                "subject_ids": val_split.unique_subjects,
            },
            "locked_test_subject_ids": sorted(locked_test_subjects),
            "strictly_disjoint": True,
        },
        "input_spec": {
            "channels": selected_channel_names,
            "num_channels": len(ch_indices),
            "timesteps": int(train_split.x_raw.shape[2]),
            "scaler_fit_split": "train_only",
            "scaler_mean": run_res["scaler_mean"],
            "scaler_std": run_res["scaler_std"],
        },
        "model": {
            "architecture": model_cfg,
            "parameters": run_res["param_counts"],
            "inference_latency_ms": run_res["inference_latency_ms"],
        },
        "training": {
            **cfg["training"],
            "epochs_completed": run_res["epochs_completed"],
            "best_epoch": best_epoch,
            "early_stopped": run_res["early_stopped"],
        },
        "best_checkpoint_metrics": {
            "train": best_train_metrics,
            "val": best_val_metrics,
        },
        "comparison_vs_phase2_baselines": baseline_comparison,
        "history": run_res["history"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate PyTorch 1D-CNN on subject-disjoint UCI HAR."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cnn_baseline.yaml"),
        help="Path to YAML training configuration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    report = train_and_evaluate(cfg, args.config)
    output_path = Path(cfg["outputs"]["results_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    val_m = report["best_checkpoint_metrics"]["val"]
    tr_m = report["best_checkpoint_metrics"]["train"]
    params = report["model"]["parameters"]["trainable"]
    print("=== Phase 3: 1D-CNN baseline ===")
    print(
        f"Device: {report['environment']['device']} ({report['environment']['gpu_name']}) | "
        f"Trainable params: {params:,}"
    )
    print(
        f"Best epoch: {report['training']['best_epoch']} / {report['training']['epochs_completed']} | "
        f"Train Macro-F1={tr_m['macro_f1']:.4f} | "
        f"Val Macro-F1={val_m['macro_f1']:.4f} | Val Acc={val_m['accuracy']:.4f}"
    )
    if report["comparison_vs_phase2_baselines"] is not None:
        comp = report["comparison_vs_phase2_baselines"]
        print(
            f"Delta vs Phase 2 val Macro-F1: "
            f"{comp['cnn_minus_logreg_macro_f1']:+.4f} vs LogReg, "
            f"{comp['cnn_minus_rf_macro_f1']:+.4f} vs RandomForest"
        )
    print(f"Saved results: {output_path} | runtime={report['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
