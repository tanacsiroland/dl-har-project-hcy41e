"""Shared evaluation metrics and diagnostic figure utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import ACTIVITY_NAMES


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray | None = None,
    class_names: tuple[str, ...] = ACTIVITY_NAMES,
) -> dict[str, Any]:
    """Compute Macro-F1, Accuracy, per-class F1, confusion matrix, and per-subject scores."""
    labels = list(range(len(class_names)))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    per_class_vals = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_f1 = {
        name: round(float(val), 6) for name, val in zip(class_names, per_class_vals)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    result: dict[str, Any] = {
        "num_samples": int(len(y_true)),
        "macro_f1": round(macro_f1, 6),
        "accuracy": round(acc, 6),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
    }

    if subjects is not None:
        per_subj: dict[str, dict[str, Any]] = {}
        subj_f1s: list[float] = []
        subj_accs: list[float] = []
        for sid in np.unique(subjects):
            mask = subjects == sid
            yt_s = y_true[mask]
            yp_s = y_pred[mask]
            present_labels = sorted(set(yt_s.tolist()) | set(yp_s.tolist()))
            s_f1 = float(
                f1_score(
                    yt_s,
                    yp_s,
                    labels=present_labels,
                    average="macro",
                    zero_division=0,
                )
            )
            s_acc = float(accuracy_score(yt_s, yp_s))
            subj_f1s.append(s_f1)
            subj_accs.append(s_acc)
            per_subj[str(int(sid))] = {
                "num_windows": int(mask.sum()),
                "macro_f1": round(s_f1, 6),
                "accuracy": round(s_acc, 6),
            }

        result["per_subject"] = per_subj
        result["per_subject_summary"] = {
            "num_subjects": len(subj_f1s),
            "macro_f1_mean": round(float(np.mean(subj_f1s)), 6),
            "macro_f1_std": round(float(np.std(subj_f1s)), 6),
            "macro_f1_min": round(float(np.min(subj_f1s)), 6),
            "macro_f1_max": round(float(np.max(subj_f1s)), 6),
            "accuracy_mean": round(float(np.mean(subj_accs)), 6),
        }

    return result


def save_confusion_matrices_figure(
    named_cms: list[tuple[str, np.ndarray]],
    output_path: Path,
    class_names: tuple[str, ...] = ACTIVITY_NAMES,
    ncols: int | None = None,
) -> None:
    """Plot row-normalized confusion matrices with raw counts in a grid."""
    short_names = [
        name.replace("WALKING_UPSTAIRS", "WALK_UP")
        .replace("WALKING_DOWNSTAIRS", "WALK_DOWN")
        for name in class_names
    ]
    n_plots = len(named_cms)
    cols = ncols if ncols is not None else (2 if n_plots == 4 else n_plots)
    rows = int(np.ceil(n_plots / max(cols, 1)))
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.8 * cols, 4.9 * rows), squeeze=False
    )
    flat_axes = axes.flatten()

    for ax, (title, cm) in zip(flat_axes, named_cms):
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_norm = cm / row_sums
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10.5, pad=8)
        ticks = np.arange(len(short_names))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticklabels(short_names, fontsize=8.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm_norm[i, j] > 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7.5,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in flat_axes[n_plots:]:
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_per_subject_f1_figure(
    model_per_subject: dict[str, dict[str, dict[str, Any]]],
    title: str,
    output_path: Path,
) -> None:
    """Plot per-subject Macro-F1 bar comparison across models."""
    first_model = next(iter(model_per_subject.values()))
    subject_ids = sorted(first_model.keys(), key=lambda s: int(s))
    x = np.arange(len(subject_ids))
    n_models = len(model_per_subject)
    width = 0.78 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * len(subject_ids)), 4.9))
    for idx, (model_name, subj_map) in enumerate(model_per_subject.items()):
        scores = [subj_map[sid]["macro_f1"] for sid in subject_ids]
        offset = (idx - (n_models - 1) / 2.0) * width
        bars = ax.bar(x + offset, scores, width=width, label=model_name)
        for bar, val in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.012,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Subj {sid}" for sid in subject_ids])
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Macro-F1")
    ax.set_title(title, pad=28)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(n_models, 4),
        frameon=True,
        fontsize=8.5,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_per_class_f1_figure(
    model_per_class: dict[str, dict[str, float]],
    title: str,
    output_path: Path,
    class_names: tuple[str, ...] = ACTIVITY_NAMES,
) -> None:
    """Plot per-class F1 bar comparison across locked models on test."""
    short_names = [
        name.replace("WALKING_UPSTAIRS", "WALK_UP")
        .replace("WALKING_DOWNSTAIRS", "WALK_DOWN")
        for name in class_names
    ]
    x = np.arange(len(class_names))
    n_models = len(model_per_class)
    width = 0.78 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    for idx, (model_name, cls_map) in enumerate(model_per_class.items()):
        scores = [cls_map[c] for c in class_names]
        offset = (idx - (n_models - 1) / 2.0) * width
        bars = ax.bar(x + offset, scores, width=width, label=model_name)
        for bar, val in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.012,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("F1 Score")
    ax.set_title(title, pad=28)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(n_models, 4),
        frameon=True,
        fontsize=8.5,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_training_curves_figure(
    history: list[dict[str, Any]],
    best_epoch: int,
    title: str,
    output_path: Path,
) -> None:
    """Plot train/val loss and Macro-F1 across training epochs."""
    epochs = [row["epoch"] for row in history]
    tr_loss = [row["train_loss"] for row in history]
    va_loss = [row["val_loss"] for row in history]
    tr_f1 = [row["train_macro_f1"] for row in history]
    va_f1 = [row["val_macro_f1"] for row in history]

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax_loss.plot(epochs, tr_loss, label="Train Loss", linewidth=1.8)
    ax_loss.plot(epochs, va_loss, label="Val Loss", linewidth=1.8)
    ax_loss.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.2, label=f"Best epoch ({best_epoch})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Loss Curve")
    ax_loss.grid(alpha=0.3, linestyle="--")
    ax_loss.legend()

    ax_f1.plot(epochs, tr_f1, label="Train Macro-F1", linewidth=1.8)
    ax_f1.plot(epochs, va_f1, label="Val Macro-F1", linewidth=1.8)
    ax_f1.axvline(best_epoch, color="gray", linestyle="--", linewidth=1.2, label=f"Best epoch ({best_epoch})")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Macro-F1")
    ax_f1.set_ylim(0.0, 1.02)
    ax_f1.set_title("Macro-F1 Curve")
    ax_f1.grid(alpha=0.3, linestyle="--")
    ax_f1.legend(loc="lower right")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tuning_comparison_figure(
    cv_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot 5-fold GroupCV Macro-F1 (mean +/- std) and val_fold=0 Macro-F1 across models."""
    labels = [row["label"] for row in cv_rows]
    cv_means = [row["cv_macro_f1_mean"] for row in cv_rows]
    cv_stds = [row["cv_macro_f1_std"] for row in cv_rows]
    val0_scores = [row["val0_macro_f1"] for row in cv_rows]

    y_pos = np.arange(len(labels))
    height = 0.36

    fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.72 * len(labels))))
    bars_cv = ax.barh(
        y_pos + height / 2.0,
        cv_means,
        height=height,
        xerr=cv_stds,
        capsize=4,
        label="5-Fold GroupCV Macro-F1 (mean ± std)",
    )
    bars_v0 = ax.barh(
        y_pos - height / 2.0,
        val0_scores,
        height=height,
        label="Primary Val Split (5 subj) Macro-F1",
    )

    for bar, mean, std in zip(bars_cv, cv_means, cv_stds):
        ax.text(
            mean + std + 0.004,
            bar.get_y() + bar.get_height() / 2.0,
            f"{mean:.4f} ± {std:.4f}",
            va="center",
            ha="left",
            fontsize=8,
        )
    for bar, val in zip(bars_v0, val0_scores):
        ax.text(
            val + 0.004,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.75, 1.09)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Phase 4 Model Comparison — Primary Val vs. 5-Fold Subject-Disjoint GroupCV", pad=28)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_ablation_figure(
    ablation_rows: list[dict[str, Any]],
    title: str,
    output_path: Path,
) -> None:
    """Plot channel/component ablation Macro-F1 and static-posture F1 breakdown."""
    labels = [row["label"] for row in ablation_rows]
    overall = [row["cv_macro_f1_mean"] for row in ablation_rows]
    overall_std = [row["cv_macro_f1_std"] for row in ablation_rows]
    sitting = [row["oof_per_class_f1"]["SITTING"] for row in ablation_rows]
    standing = [row["oof_per_class_f1"]["STANDING"] for row in ablation_rows]
    laying = [row["oof_per_class_f1"]["LAYING"] for row in ablation_rows]

    x = np.arange(len(labels))
    width = 0.19

    fig, ax = plt.subplots(figsize=(max(8.5, 2.5 * len(labels)), 4.9))
    b1 = ax.bar(x - 1.5 * width, overall, width, yerr=overall_std, capsize=3, label="5-Fold CV Macro-F1")
    b2 = ax.bar(x - 0.5 * width, sitting, width, label="OOF SITTING F1")
    b3 = ax.bar(x + 0.5 * width, standing, width, label="OOF STANDING F1")
    b4 = ax.bar(x + 1.5 * width, laying, width, label="OOF LAYING F1")

    for idx_bar, bar in enumerate(b1):
        val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + overall_std[idx_bar] + 0.014,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    for bars in (b2, b3, b4):
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.012,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.11)
    ax.set_ylabel("F1 Score")
    ax.set_title(title, pad=28)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=True, fontsize=8.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
