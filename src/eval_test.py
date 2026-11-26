"""Phase 5: evaluate the four final models once on the official test subjects.

Models: Logistic Regression and Random Forest (settings from Phase 2, refit on the
16 training subjects), the Phase 3 CNN checkpoint and the Phase 4 CNN checkpoint.
Nothing is tuned or selected here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from src.data import (
    ACTIVITY_NAMES,
    apply_channel_standardizer,
    assert_subject_disjointness,
    compute_channel_standardizer,
    load_raw_split,
    make_subject_disjoint_split,
)
from src.features import extract_window_summary_features
from src.metrics import compute_classification_metrics
from src.models import build_cnn_model
from src.torch_data import InertialWindowDataset
from src.train_cnn import benchmark_inference_latency, count_parameters, evaluate_model


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _extract_top_confusions(
    cm: np.ndarray, class_names: tuple[str, ...] = ACTIVITY_NAMES, top_k: int = 5
) -> list[dict[str, Any]]:
    total_errors = int(cm.sum() - np.trace(cm))
    pairs: list[dict[str, Any]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append(
                {
                    "true": class_names[i],
                    "predicted": class_names[j],
                    "count": int(cm[i, j]),
                    "share_of_errors": round(
                        float(cm[i, j]) / max(total_errors, 1), 6
                    ),
                }
            )
    pairs.sort(key=lambda r: r["count"], reverse=True)
    return pairs[:top_k]


def _mcnemar_comparison(
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


def _evaluate_cnn_checkpoint(
    ckpt_path: Path,
    train_x_raw: np.ndarray,
    test_x_raw: np.ndarray,
    test_y: np.ndarray,
    test_subjects: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    """Load locked checkpoint, verify train-only scaler, and evaluate once on test."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ch_indices = list(ckpt["channel_indices"])

    expected_mean, expected_std = compute_channel_standardizer(
        train_x_raw[:, ch_indices, :]
    )
    saved_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    saved_std = np.asarray(ckpt["scaler_std"], dtype=np.float32)
    if not (
        np.allclose(saved_mean, expected_mean, atol=1e-5)
        and np.allclose(saved_std, expected_std, atol=1e-5)
    ):
        raise AssertionError(
            f"Scaler mismatch in checkpoint {ckpt_path}; expected train-only stats."
        )

    x_te_std = apply_channel_standardizer(
        test_x_raw[:, ch_indices, :], saved_mean, saved_std
    )
    test_ds = InertialWindowDataset(x_te_std, test_y)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    model = build_cnn_model(
        model_cfg=ckpt["model_config"],
        in_channels=len(ch_indices),
        num_classes=len(ACTIVITY_NAMES),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    t_inf0 = time.perf_counter()
    eval_res = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        subjects=test_subjects,
        return_preds=True,
    )
    batch_eval_sec = round(time.perf_counter() - t_inf0, 4)
    y_pred = eval_res.pop("y_pred")
    latency = benchmark_inference_latency(
        model, sample_shape=(1, len(ch_indices), x_te_std.shape[2]), device=device
    )

    cm = np.asarray(eval_res["confusion_matrix"], dtype=np.int64)
    summary = {
        "checkpoint_path": str(ckpt_path.as_posix()),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_val_macro_f1": float(ckpt["val_macro_f1"]),
        "model_config": ckpt["model_config"],
        "parameters": count_parameters(model),
        "inference_latency_ms": latency,
        "test_batch_eval_seconds": batch_eval_sec,
        "test_metrics": eval_res,
        "total_test_errors": int(cm.sum() - np.trace(cm)),
        "top_confusions": _extract_top_confusions(cm),
    }
    return summary, y_pred


def run_final_test_evaluation(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Run single locked test-set evaluation across all finalized models."""
    t_start = time.perf_counter()
    seed = int(cfg["seed"])
    _set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(cfg["data_dir"])
    full_train = load_raw_split(data_dir, "train")
    test_split = load_raw_split(data_dir, "test")

    split_cfg = cfg["split"]
    train_split, val_split = make_subject_disjoint_split(
        full_train,
        n_splits=int(split_cfg["n_splits"]),
        val_fold=int(split_cfg["val_fold"]),
        seed=seed,
        locked_test_subjects=set(test_split.unique_subjects),
    )
    assert_subject_disjointness(
        set(full_train.unique_subjects),
        test_subjects=set(test_split.unique_subjects),
    )

    # Read locked baseline hyperparameters & feature specs from Phase 2 report
    with Path(cfg["baselines_results"]).open("r", encoding="utf-8") as handle:
        base_res = json.load(handle)
    feat_stats = list(base_res["features"]["stats"])
    lr_params = base_res["models"]["logistic_regression"]["best_on_val"]["params"]
    rf_params = base_res["models"]["random_forest"]["best_on_val"]["params"]

    x_tr_feat = extract_window_summary_features(train_split.x_raw, stats=feat_stats)
    x_te_feat = extract_window_summary_features(test_split.x_raw, stats=feat_stats)

    # 1. Locked Logistic Regression (fit strictly on 16-subject train)
    t_lr = time.perf_counter()
    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(lr_params["C"]),
                    max_iter=int(lr_params["max_iter"]),
                    random_state=seed,
                ),
            ),
        ]
    )
    lr_pipe.fit(x_tr_feat, train_split.y)
    lr_preds = lr_pipe.predict(x_te_feat)
    lr_metrics = compute_classification_metrics(
        test_split.y, lr_preds, subjects=test_split.subjects
    )
    lr_cm = np.asarray(lr_metrics["confusion_matrix"], dtype=np.int64)
    lr_summary = {
        "params": lr_params,
        "num_features": int(x_tr_feat.shape[1]),
        "val_macro_f1_reference": base_res["models"]["logistic_regression"][
            "best_on_val"
        ]["val_metrics"]["macro_f1"],
        "fit_plus_eval_seconds": round(time.perf_counter() - t_lr, 4),
        "test_metrics": lr_metrics,
        "total_test_errors": int(lr_cm.sum() - np.trace(lr_cm)),
        "top_confusions": _extract_top_confusions(lr_cm),
    }

    # 2. Locked Random Forest (fit strictly on 16-subject train)
    t_rf = time.perf_counter()
    rf_clf = RandomForestClassifier(
        n_estimators=int(rf_params["n_estimators"]),
        max_depth=rf_params["max_depth"],
        min_samples_leaf=int(rf_params["min_samples_leaf"]),
        random_state=seed,
        n_jobs=-1,
    )
    rf_clf.fit(x_tr_feat, train_split.y)
    rf_preds = rf_clf.predict(x_te_feat)
    rf_metrics = compute_classification_metrics(
        test_split.y, rf_preds, subjects=test_split.subjects
    )
    rf_cm = np.asarray(rf_metrics["confusion_matrix"], dtype=np.int64)
    rf_summary = {
        "params": rf_params,
        "num_features": int(x_tr_feat.shape[1]),
        "val_macro_f1_reference": base_res["models"]["random_forest"]["best_on_val"][
            "val_metrics"
        ]["macro_f1"],
        "fit_plus_eval_seconds": round(time.perf_counter() - t_rf, 4),
        "test_metrics": rf_metrics,
        "total_test_errors": int(rf_cm.sum() - np.trace(rf_cm)),
        "top_confusions": _extract_top_confusions(rf_cm),
    }

    # 3. Phase 3 CompactCNN1D Baseline Checkpoint
    cnn_base_summary, cnn_base_preds = _evaluate_cnn_checkpoint(
        ckpt_path=Path(cfg["locked_models"]["cnn_baseline_checkpoint"]),
        train_x_raw=train_split.x_raw,
        test_x_raw=test_split.x_raw,
        test_y=test_split.y,
        test_subjects=test_split.subjects,
        device=device,
    )

    # 4. Phase 4 Locked Winner CompactCNN1D Checkpoint
    cnn_tuned_summary, cnn_tuned_preds = _evaluate_cnn_checkpoint(
        ckpt_path=Path(cfg["locked_models"]["cnn_tuned_checkpoint"]),
        train_x_raw=train_split.x_raw,
        test_x_raw=test_split.x_raw,
        test_y=test_split.y,
        test_subjects=test_split.subjects,
        device=device,
    )

    # Paired McNemar statistical comparisons on the 2,947 test windows
    paired_tests = {
        "tuned_cnn_vs_logreg": _mcnemar_comparison(
            test_split.y, cnn_tuned_preds, lr_preds
        ),
        "tuned_cnn_vs_random_forest": _mcnemar_comparison(
            test_split.y, cnn_tuned_preds, rf_preds
        ),
        "tuned_cnn_vs_baseline_cnn": _mcnemar_comparison(
            test_split.y, cnn_tuned_preds, cnn_base_preds
        ),
    }

    elapsed = round(time.perf_counter() - t_start, 3)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None

    return {
        "phase": "phase_5_locked_test_evaluation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "config_path": str(config_path.as_posix()),
        "seed": seed,
        "test_set_protocol": "SINGLE_LOCKED_PASS_NO_TUNING_ON_TEST",
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
                "scikit-learn": sklearn.__version__,
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
            "test": {
                "num_windows": int(len(test_split.y)),
                "num_subjects": len(test_split.unique_subjects),
                "subject_ids": test_split.unique_subjects,
            },
            "strictly_disjoint": True,
        },
        "models": {
            "logistic_regression": lr_summary,
            "random_forest": rf_summary,
            "cnn_baseline": cnn_base_summary,
            "cnn_tuned_winner": cnn_tuned_summary,
        },
        "paired_statistical_comparisons": paired_tests,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single locked Phase 5 evaluation on the 9 unseen test subjects."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_eval.yaml"),
        help="Path to Phase 5 evaluation YAML configuration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    report = run_final_test_evaluation(cfg, args.config)
    output_path = Path(cfg["outputs"]["results_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    test_info = report["splits"]["test"]
    print("=== Phase 5: test evaluation ===")
    print(
        f"Test split: {test_info['num_windows']} windows from "
        f"{test_info['num_subjects']} subjects {test_info['subject_ids']}"
    )
    for key, label in [
        ("logistic_regression", "Logistic Regression"),
        ("random_forest", "Random Forest"),
        ("cnn_baseline", "CompactCNN1D (Phase 3)"),
        ("cnn_tuned_winner", "CompactCNN1D (Phase 4, tuned)"),
    ]:
        m = report["models"][key]["test_metrics"]
        errs = report["models"][key]["total_test_errors"]
        subj = m["per_subject_summary"]
        print(
            f"  {label:30s} | Test Macro-F1={m['macro_f1']:.4f} | "
            f"Acc={m['accuracy']:.4f} | Errors={errs}/{test_info['num_windows']} | "
            f"Subj Mean={subj['macro_f1_mean']:.4f} (min={subj['macro_f1_min']:.4f})"
        )
    mc = report["paired_statistical_comparisons"]["tuned_cnn_vs_logreg"]
    print(
        f"McNemar (Tuned CNN vs LogReg): CNN+/{mc['a_correct_b_wrong']} vs "
        f"LogReg+/{mc['a_wrong_b_correct']}, exact p={mc['exact_two_sided_p_value']:.3e}"
    )
    print(f"Saved report: {output_path} | runtime={report['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
