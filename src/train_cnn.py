"""Config-driven PyTorch 1D-CNN training and validation utilities (Phase 3+).

Provides both the reusable single-split trainer ``run_single_training`` (shared
by Phase 4 hyperparameter tuning/GroupCV) and the standalone CLI entrypoint.
Checkpointing and early stopping are governed strictly by validation Macro-F1;
the official 9 test subjects remain locked and untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data import (
    ACTIVITY_NAMES,
    SIGNAL_CHANNELS,
    DatasetSplit,
    apply_channel_standardizer,
    assert_subject_disjointness,
    compute_channel_standardizer,
    load_raw_split,
    load_subject_ids,
    make_subject_disjoint_split,
    resolve_channel_indices,
)
from src.metrics import (
    compute_classification_metrics,
    save_confusion_matrices_figure,
    save_per_subject_f1_figure,
    save_training_curves_figure,
)
from src.models import build_cnn_model
from src.torch_data import InertialWindowDataset


def set_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def apply_train_augmentation(
    x_batch: torch.Tensor,
    aug_cfg: dict[str, Any] | None,
) -> torch.Tensor:
    """Apply lightweight sensor augmentation strictly during training steps."""
    if not aug_cfg or not aug_cfg.get("enabled", False):
        return x_batch

    out = x_batch
    scale_std = float(aug_cfg.get("scale_std", 0.0))
    if scale_std > 0.0:
        # Per-sample, per-channel multiplicative amplitude jitter around 1.0.
        scales = 1.0 + torch.randn(
            (out.size(0), out.size(1), 1), device=out.device, dtype=out.dtype
        ) * scale_std
        out = out * scales

    jitter_std = float(aug_cfg.get("jitter_std", 0.0))
    if jitter_std > 0.0:
        out = out + torch.randn_like(out) * jitter_std

    return out


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    subjects: np.ndarray | None = None,
    return_preds: bool = False,
) -> dict[str, Any]:
    """Evaluate loss and classification metrics on a DataLoader."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=False)
        y_batch = y_batch.to(device, non_blocking=False)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        bs = int(y_batch.size(0))
        total_loss += float(loss.item()) * bs
        total_count += bs
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    metrics = compute_classification_metrics(y_true, y_pred, subjects=subjects)
    metrics["loss"] = round(total_loss / max(total_count, 1), 6)
    if return_preds:
        metrics["y_pred"] = y_pred
    return metrics


@torch.no_grad()
def benchmark_inference_latency(
    model: nn.Module,
    sample_shape: tuple[int, int, int],
    device: torch.device,
    n_runs: int = 100,
) -> dict[str, float]:
    """Measure average forward latency (ms/window) on active device and CPU."""
    results: dict[str, float] = {}
    x_cpu = torch.randn(sample_shape, dtype=torch.float32)

    if device.type == "cuda":
        model.eval().to(device)
        x_gpu = x_cpu.to(device)
        for _ in range(10):
            _ = model(x_gpu)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x_gpu)
        torch.cuda.synchronize()
        results["gpu_single_window_ms"] = round(
            ((time.perf_counter() - t0) / n_runs) * 1000.0, 4
        )

    model_cpu = model.eval().to("cpu")
    for _ in range(10):
        _ = model_cpu(x_cpu)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model_cpu(x_cpu)
    results["cpu_single_window_ms"] = round(
        ((time.perf_counter() - t0) / n_runs) * 1000.0, 4
    )
    model.to(device)
    return results


def _build_optimizer(model: nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg["lr"])
    wd = float(train_cfg["weight_decay"])
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer '{opt_name}'")


def _build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler | None:
    sched_name = str(train_cfg.get("scheduler", "cosine")).lower()
    epochs = int(train_cfg["epochs"])
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=1e-5
        )
    if sched_name == "none":
        return None
    raise ValueError(f"Unsupported scheduler '{sched_name}'")


def run_single_training(
    train_split: DatasetSplit,
    val_split: DatasetSplit,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    ch_indices: list[int],
    seed: int,
    device: torch.device,
    checkpoint_path: Path | None = None,
    aug_cfg: dict[str, Any] | None = None,
    compute_latency: bool = False,
    return_val_preds: bool = False,
) -> dict[str, Any]:
    """Train and validate a PyTorch 1D-CNN on one subject-disjoint train/val split."""
    assert_subject_disjointness(
        set(train_split.unique_subjects), set(val_split.unique_subjects)
    )
    set_seeds(seed)
    t_run = time.perf_counter()

    x_tr_sel = train_split.x_raw[:, ch_indices, :]
    x_va_sel = val_split.x_raw[:, ch_indices, :]

    # Per-channel mean/std fit strictly on training subjects.
    ch_mean, ch_std = compute_channel_standardizer(x_tr_sel)
    x_tr_std = apply_channel_standardizer(x_tr_sel, ch_mean, ch_std)
    x_va_std = apply_channel_standardizer(x_va_sel, ch_mean, ch_std)

    train_ds = InertialWindowDataset(x_tr_std, train_split.y)
    val_ds = InertialWindowDataset(x_va_std, val_split.y)

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    dl_gen = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=dl_gen,
        num_workers=num_workers,
    )
    train_eval_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = build_cnn_model(
        model_cfg=model_cfg,
        in_channels=len(ch_indices),
        num_classes=len(ACTIVITY_NAMES),
    ).to(device)
    param_counts = count_parameters(model)

    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = _build_optimizer(model, train_cfg)
    scheduler = _build_scheduler(optimizer, train_cfg)

    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["early_stopping_patience"])

    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=False)
            y_batch = y_batch.to(device, non_blocking=False)
            x_batch = apply_train_augmentation(x_batch, aug_cfg)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            bs = int(y_batch.size(0))
            running_loss += float(loss.item()) * bs
            seen += bs

        if scheduler is not None:
            scheduler.step()

        # Skip per-subject breakdowns inside epochs to keep multi-fold tuning fast.
        train_eval = evaluate_model(model, train_eval_loader, criterion, device, subjects=None)
        val_eval = evaluate_model(model, val_loader, criterion, device, subjects=None)

        epoch_row = {
            "epoch": epoch,
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            "train_opt_loss": round(running_loss / max(seen, 1), 6),
            "train_loss": train_eval["loss"],
            "train_macro_f1": train_eval["macro_f1"],
            "train_accuracy": train_eval["accuracy"],
            "val_loss": val_eval["loss"],
            "val_macro_f1": val_eval["macro_f1"],
            "val_accuracy": val_eval["accuracy"],
        }
        history.append(epoch_row)

        if val_eval["macro_f1"] > best_val_f1:
            best_val_f1 = val_eval["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    assert best_state_dict is not None
    model.load_state_dict(best_state_dict)

    best_train_metrics = evaluate_model(
        model, train_eval_loader, criterion, device, subjects=train_split.subjects
    )
    best_val_metrics = evaluate_model(
        model,
        val_loader,
        criterion,
        device,
        subjects=val_split.subjects,
        return_preds=return_val_preds,
    )

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": best_epoch,
                "val_macro_f1": best_val_metrics["macro_f1"],
                "val_accuracy": best_val_metrics["accuracy"],
                "model_state_dict": best_state_dict,
                "model_config": model_cfg,
                "training_config": train_cfg,
                "augmentation_config": aug_cfg,
                "channel_indices": ch_indices,
                "channel_names": [SIGNAL_CHANNELS[i] for i in ch_indices],
                "scaler_mean": ch_mean,
                "scaler_std": ch_std,
                "seed": seed,
            },
            checkpoint_path,
        )

    latency_ms = (
        benchmark_inference_latency(
            model, sample_shape=(1, len(ch_indices), x_tr_std.shape[2]), device=device
        )
        if compute_latency
        else None
    )

    return {
        "runtime_seconds": round(time.perf_counter() - t_run, 3),
        "param_counts": param_counts,
        "scaler_mean": [round(float(v), 6) for v in ch_mean.reshape(-1)],
        "scaler_std": [round(float(v), 6) for v in ch_std.reshape(-1)],
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "early_stopped": len(history) < epochs,
        "history": history,
        "train_metrics": best_train_metrics,
        "val_metrics": best_val_metrics,
        "inference_latency_ms": latency_ms,
    }


def train_and_evaluate(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Train configured 1D-CNN from scratch and write full Phase 3 JSON + figures."""
    t_start = time.perf_counter()
    seed = int(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    figures_dir = Path(out_cfg["figures_dir"])
    fig_prefix = str(out_cfg["figure_prefix"])
    curves_path = figures_dir / f"{fig_prefix}_curves.png"
    cm_path = figures_dir / f"{fig_prefix}_val_cm.png"
    subj_path = figures_dir / f"{fig_prefix}_val_per_subject_f1.png"

    save_training_curves_figure(
        run_res["history"],
        best_epoch=best_epoch,
        title=f"{model_cfg['name']} Training Curves (Best Val Macro-F1={best_val_metrics['macro_f1']:.4f} @ Ep {best_epoch})",
        output_path=curves_path,
    )
    save_confusion_matrices_figure(
        [
            (
                f"{model_cfg['name']} (val, ep={best_epoch})\n"
                f"Macro-F1={best_val_metrics['macro_f1']:.4f}",
                np.asarray(best_val_metrics["confusion_matrix"], dtype=np.int64),
            )
        ],
        output_path=cm_path,
    )

    per_subj_comparison: dict[str, dict[str, dict[str, Any]]] = {
        f"1D-CNN (ep {best_epoch})": best_val_metrics["per_subject"]
    }
    baseline_comparison: dict[str, Any] | None = None
    baselines_json_path = Path(cfg.get("baselines_json", ""))
    if baselines_json_path.is_file():
        with baselines_json_path.open("r", encoding="utf-8") as handle:
            b_data = json.load(handle)
        lr_best = b_data["models"]["logistic_regression"]["best_on_val"]
        rf_best = b_data["models"]["random_forest"]["best_on_val"]
        per_subj_comparison[f"LogReg (C={lr_best['params']['C']})"] = lr_best[
            "val_metrics"
        ]["per_subject"]
        per_subj_comparison[
            f"RandomForest (depth={rf_best['params']['max_depth']})"
        ] = rf_best["val_metrics"]["per_subject"]
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

    save_per_subject_f1_figure(
        per_subj_comparison,
        title="Validation Per-Subject Macro-F1: 1D-CNN vs. Phase 2 Baselines",
        output_path=subj_path,
    )

    elapsed = round(time.perf_counter() - t_start, 3)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None

    return {
        "status": "VERIFIED",
        "phase": "phase_3_cnn_baseline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "config_path": str(config_path.as_posix()),
        "checkpoint_path": str(ckpt_path.as_posix()),
        "seed": seed,
        "determinism_note": "Seeds and cuDNN deterministic flags set; exact bit-level reproducibility across hardware/CUDA kernels is not guaranteed.",
        "test_set_status": "LOCKED_UNTOUCHED",
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "cpu_logical_count": os.cpu_count(),
            "device": str(device),
            "gpu_name": gpu_name,
            "packages": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "numpy": np.__version__,
                "pyyaml": yaml.__version__,
            },
        },
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
        "figures": [
            str(curves_path.as_posix()),
            str(cm_path.as_posix()),
            str(subj_path.as_posix()),
        ],
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
    print("=== Phase 3 Compact 1D-CNN Baseline [VERIFIED] ===")
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
