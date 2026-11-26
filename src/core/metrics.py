"""Classification and statistical comparison metrics shared across phases."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.core.data import ACTIVITY_NAMES


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


def mcnemar_comparison(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> dict[str, Any]:
    """Paired McNemar test comparing correctness of model A vs model B on test."""
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true
    n_10 = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    n_01 = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    n_discordant = n_10 + n_01

    if n_discordant == 0:
        p_value = 1.0
        chi2_cc = 0.0
    else:
        # Exact binomial two-sided p-value in log-space to avoid float underflow.
        k = min(n_10, n_01)
        log_half_n = -n_discordant * math.log(2.0)
        log_n_fact = math.lgamma(n_discordant + 1)
        tail = sum(
            math.exp(log_n_fact - math.lgamma(i + 1) - math.lgamma(n_discordant - i + 1) + log_half_n)
            for i in range(k + 1)
        )
        p_value = min(1.0, 2.0 * tail)
        chi2_cc = ((abs(n_10 - n_01) - 1.0) ** 2) / n_discordant

    return {
        "a_correct_b_wrong": n_10,
        "a_wrong_b_correct": n_01,
        "discordant_total": n_discordant,
        "chi2_continuity_corrected": round(float(chi2_cc), 4),
        "exact_two_sided_p_value": float(p_value),
    }
