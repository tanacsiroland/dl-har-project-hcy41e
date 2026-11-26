"""Phase 2: Logistic Regression and Random Forest on per-window summary statistics.

Hyperparameters are chosen on the validation subjects, then the chosen settings are
checked with 5-fold StratifiedGroupKFold over the 21 training subjects.
Test data is not loaded here.
"""

from __future__ import annotations

import argparse
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
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import (
    ACTIVITY_NAMES,
    DatasetSplit,
    assert_subject_disjointness,
    load_raw_split,
    load_subject_ids,
    make_subject_disjoint_split,
)
from src.features import extract_window_summary_features, get_summary_feature_names
from src.metrics import compute_classification_metrics


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_logreg_pipeline(c_val: float, max_iter: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_val,
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def _build_rf_model(
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    seed: int,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )


def _run_group_cv(
    x_feat: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    n_folds: int,
    seed: int,
    estimator_factory: Any,
) -> dict[str, Any]:
    """Run leak-free StratifiedGroupKFold CV over the 21 official training subjects."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results: list[dict[str, Any]] = []
    oof_preds = np.empty_like(y)

    for fold_idx, (tr_idx, va_idx) in enumerate(sgkf.split(x_feat, y, groups=subjects)):
        tr_subj = set(np.unique(subjects[tr_idx]).tolist())
        va_subj = set(np.unique(subjects[va_idx]).tolist())
        assert_subject_disjointness(tr_subj, va_subj)

        model = estimator_factory()
        model.fit(x_feat[tr_idx], y[tr_idx])
        preds = model.predict(x_feat[va_idx])
        oof_preds[va_idx] = preds

        fold_metrics = compute_classification_metrics(
            y[va_idx], preds, subjects=subjects[va_idx]
        )
        fold_results.append(
            {
                "fold": fold_idx,
                "train_windows": int(len(tr_idx)),
                "val_windows": int(len(va_idx)),
                "val_subjects": sorted(va_subj),
                "macro_f1": fold_metrics["macro_f1"],
                "accuracy": fold_metrics["accuracy"],
            }
        )

    macro_f1_vals = [f["macro_f1"] for f in fold_results]
    acc_vals = [f["accuracy"] for f in fold_results]
    oof_metrics = compute_classification_metrics(y, oof_preds, subjects=subjects)

    return {
        "num_folds": n_folds,
        "fold_macro_f1_mean": round(float(np.mean(macro_f1_vals)), 6),
        "fold_macro_f1_std": round(float(np.std(macro_f1_vals)), 6),
        "fold_accuracy_mean": round(float(np.mean(acc_vals)), 6),
        "fold_accuracy_std": round(float(np.std(acc_vals)), 6),
        "oof_metrics": oof_metrics,
        "folds": fold_results,
    }


def _top_features_from_logreg(
    pipeline: Pipeline, feature_names: list[str], top_k: int = 10
) -> list[dict[str, Any]]:
    clf: LogisticRegression = pipeline.named_steps["clf"]
    # Mean absolute standardized coefficient across the 6 activity classes.
    mean_abs_coef = np.mean(np.abs(clf.coef_), axis=0)
    top_idx = np.argsort(mean_abs_coef)[::-1][:top_k]
    return [
        {
            "feature": feature_names[int(i)],
            "mean_abs_coef": round(float(mean_abs_coef[int(i)]), 6),
        }
        for i in top_idx
    ]


def _top_features_from_rf(
    model: RandomForestClassifier, feature_names: list[str], top_k: int = 10
) -> list[dict[str, Any]]:
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_k]
    return [
        {
            "feature": feature_names[int(i)],
            "gini_importance": round(float(importances[int(i)]), 6),
        }
        for i in top_idx
    ]


def run_baselines(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Execute Phase 2 classical ML baselines strictly on validation and group CV."""
    t0 = time.perf_counter()
    seed = int(cfg["seed"])
    _set_seeds(seed)

    data_dir = Path(cfg["data_dir"])
    locked_test_subjects = load_subject_ids(data_dir, "test")
    full_train: DatasetSplit = load_raw_split(data_dir, "train")

    split_cfg = cfg["split"]
    train_split, val_split = make_subject_disjoint_split(
        full_train,
        n_splits=int(split_cfg["n_splits"]),
        val_fold=int(split_cfg["val_fold"]),
        seed=seed,
        locked_test_subjects=locked_test_subjects,
    )

    stats = list(cfg["features"]["stats"])
    feature_names = get_summary_feature_names(stats=stats)

    x_tr_feat = extract_window_summary_features(train_split.x_raw, stats=stats)
    x_va_feat = extract_window_summary_features(val_split.x_raw, stats=stats)
    x_all_tr_feat = extract_window_summary_features(full_train.x_raw, stats=stats)

    # 1. Logistic Regression (train-fit StandardScaler inside Pipeline; select C strictly on val)
    lr_cfg = cfg["models"]["logistic_regression"]
    lr_max_iter = int(lr_cfg["max_iter"])
    lr_trials: list[dict[str, Any]] = []
    best_lr: dict[str, Any] | None = None

    for c_val in lr_cfg["c_candidates"]:
        c_float = float(c_val)
        t_fit = time.perf_counter()
        pipe = _build_logreg_pipeline(c_float, lr_max_iter, seed)
        pipe.fit(x_tr_feat, train_split.y)
        val_preds = pipe.predict(x_va_feat)
        fit_sec = round(time.perf_counter() - t_fit, 3)

        val_metrics = compute_classification_metrics(
            val_split.y, val_preds, subjects=val_split.subjects
        )
        trial = {
            "params": {"C": c_float, "max_iter": lr_max_iter},
            "fit_seconds": fit_sec,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
        }
        lr_trials.append(trial)
        if best_lr is None or val_metrics["macro_f1"] > best_lr["val_metrics"]["macro_f1"]:
            best_lr = {
                "params": trial["params"],
                "fit_seconds": fit_sec,
                "train_metrics": compute_classification_metrics(
                    train_split.y,
                    pipe.predict(x_tr_feat),
                    subjects=train_split.subjects,
                ),
                "val_metrics": val_metrics,
                "top_features": _top_features_from_logreg(pipe, feature_names),
            }

    assert best_lr is not None
    lr_cv = _run_group_cv(
        x_all_tr_feat,
        full_train.y,
        full_train.subjects,
        n_folds=int(split_cfg["cv_folds"]),
        seed=seed,
        estimator_factory=lambda: _build_logreg_pipeline(
            float(best_lr["params"]["C"]), lr_max_iter, seed
        ),
    )

    # 2. Random Forest (select hyperparameters strictly on val Macro-F1)
    rf_cfg = cfg["models"]["random_forest"]
    n_estimators = int(rf_cfg["n_estimators"])
    rf_trials: list[dict[str, Any]] = []
    best_rf: dict[str, Any] | None = None

    for cand in rf_cfg["candidates"]:
        max_depth = None if cand["max_depth"] is None else int(cand["max_depth"])
        min_samples_leaf = int(cand["min_samples_leaf"])
        t_fit = time.perf_counter()
        rf = _build_rf_model(n_estimators, max_depth, min_samples_leaf, seed)
        rf.fit(x_tr_feat, train_split.y)
        val_preds = rf.predict(x_va_feat)
        fit_sec = round(time.perf_counter() - t_fit, 3)

        val_metrics = compute_classification_metrics(
            val_split.y, val_preds, subjects=val_split.subjects
        )
        trial = {
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
            },
            "fit_seconds": fit_sec,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
        }
        rf_trials.append(trial)
        if best_rf is None or val_metrics["macro_f1"] > best_rf["val_metrics"]["macro_f1"]:
            best_rf = {
                "params": trial["params"],
                "fit_seconds": fit_sec,
                "train_metrics": compute_classification_metrics(
                    train_split.y,
                    rf.predict(x_tr_feat),
                    subjects=train_split.subjects,
                ),
                "val_metrics": val_metrics,
                "top_features": _top_features_from_rf(rf, feature_names),
            }

    assert best_rf is not None
    rf_cv = _run_group_cv(
        x_all_tr_feat,
        full_train.y,
        full_train.subjects,
        n_folds=int(split_cfg["cv_folds"]),
        seed=seed,
        estimator_factory=lambda: _build_rf_model(
            n_estimators,
            best_rf["params"]["max_depth"],
            int(best_rf["params"]["min_samples_leaf"]),
            seed,
        ),
    )

    elapsed = round(time.perf_counter() - t0, 3)
    report: dict[str, Any] = {
        "phase": "phase_2_classical_baselines",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "config_path": str(config_path.as_posix()),
        "seed": seed,
        "test_set_status": "LOCKED_UNTOUCHED",
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "cpu_logical_count": os.cpu_count(),
            "packages": {
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
            "locked_test_subject_ids": sorted(locked_test_subjects),
            "strictly_disjoint": True,
        },
        "features": {
            "num_features": len(feature_names),
            "stats": stats,
            "feature_names": feature_names,
        },
        "activity_names": list(ACTIVITY_NAMES),
        "models": {
            "logistic_regression": {
                "val_tuning_trials": lr_trials,
                "best_on_val": best_lr,
                "group_cv_5fold": lr_cv,
            },
            "random_forest": {
                "val_tuning_trials": rf_trials,
                "best_on_val": best_rf,
                "group_cv_5fold": rf_cv,
            },
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 subject-disjoint classical ML baselines on UCI HAR."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines.yaml"),
        help="Path to YAML baseline configuration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    report = run_baselines(cfg, args.config)
    output_path = Path(cfg["output_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    lr_best = report["models"]["logistic_regression"]["best_on_val"]
    rf_best = report["models"]["random_forest"]["best_on_val"]
    lr_cv = report["models"]["logistic_regression"]["group_cv_5fold"]
    rf_cv = report["models"]["random_forest"]["group_cv_5fold"]

    print("=== Phase 2: classical baselines ===")
    print(
        f"Split: train={report['splits']['train']['num_windows']} ({report['splits']['train']['num_subjects']} subj) | "
        f"val={report['splits']['val']['num_windows']} ({report['splits']['val']['num_subjects']} subj) | "
        f"test={report['test_set_status']}"
    )
    print(f"Features: {report['features']['num_features']} leak-free per-window summary stats")
    print(
        f"LogReg (C={lr_best['params']['C']}): "
        f"val Macro-F1={lr_best['val_metrics']['macro_f1']:.4f}, "
        f"val Acc={lr_best['val_metrics']['accuracy']:.4f} | "
        f"5-fold GroupCV Macro-F1={lr_cv['fold_macro_f1_mean']:.4f} +/- {lr_cv['fold_macro_f1_std']:.4f}"
    )
    print(
        f"RandomForest (depth={rf_best['params']['max_depth']}, leaf={rf_best['params']['min_samples_leaf']}): "
        f"val Macro-F1={rf_best['val_metrics']['macro_f1']:.4f}, "
        f"val Acc={rf_best['val_metrics']['accuracy']:.4f} | "
        f"5-fold GroupCV Macro-F1={rf_cv['fold_macro_f1_mean']:.4f} +/- {rf_cv['fold_macro_f1_std']:.4f}"
    )
    print(f"Saved report: {output_path} | runtime={report['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
