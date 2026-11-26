"""Reusable PyTorch 1D-CNN single-split training, evaluation, and latency helpers."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.core.data import (
    ACTIVITY_NAMES,
    SIGNAL_CHANNELS,
    DatasetSplit,
    apply_channel_standardizer,
    assert_subject_disjointness,
    compute_channel_standardizer,
)
from src.core.datasets import InertialWindowDataset
from src.core.env import set_seeds
from src.core.metrics import compute_classification_metrics
from src.models import build_cnn_model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
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
