"""Phase 4 systematic HPO, 5-fold GroupCV verification, and controlled ablations.

Evaluates 1D-CNN candidates strictly on the subject-disjoint validation split
and 5-fold StratifiedGroupKFold cross-validation across the 21 official training
subjects, runs channel ablations (9ch vs 6ch dynamic vs 6ch raw IMU) and
multi-seed stability checks on the selected winner, and locks the final model
checkpoint for Phase 5. The 9 official test subjects remain untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from src.data import (
    DatasetSplit,
    assert_subject_disjointness,
    load_raw_split,
    load_subject_ids,
    make_subject_disjoint_split,
    resolve_channel_indices,
)
from src.metrics import (
    compute_classification_metrics,
    save_ablation_figure,
    save_confusion_matrices_figure,
    save_training_curves_figure,
    save_tuning_comparison_figure,
)
from src.train_cnn import run_single_training


def _merge_training_cfg(default_cfg: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(default_cfg)
    merged.update(override)
    return merged


def _run_group_cv_for_candidate(
    full_train: DatasetSplit,
    cand: dict[str, Any],
    default_train_cfg: dict[str, Any],
    ch_indices: list[int],
    n_folds: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a CNN configuration across all 5 subject-disjoint folds of official train."""
    train_cfg = _merge_training_cfg(default_train_cfg, cand.get("training", {}))
    aug_cfg = cand.get("augmentation")
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_rows: list[dict[str, Any]] = []
    oof_preds = np.empty_like(full_train.y)

    for fold_idx, (tr_idx, va_idx) in enumerate(
        sgkf.split(np.zeros(len(full_train.y)), full_train.y, groups=full_train.subjects)
    ):
        tr_sub = full_train.subset(tr_idx, name=f"cv_train_f{fold_idx}")
        va_sub = full_train.subset(va_idx, name=f"cv_val_f{fold_idx}")
        assert_subject_disjointness(
            set(tr_sub.unique_subjects), set(va_sub.unique_subjects)
        )

        res = run_single_training(
            train_split=tr_sub,
            val_split=va_sub,
            model_cfg=cand["model"],
            train_cfg=train_cfg,
            ch_indices=ch_indices,
            seed=seed + fold_idx,
            device=device,
            checkpoint_path=None,
            aug_cfg=aug_cfg,
            compute_latency=False,
            return_val_preds=True,
        )
        va_m = res["val_metrics"]
        oof_preds[va_idx] = va_m.pop("y_pred")
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_windows": int(len(tr_idx)),
                "val_windows": int(len(va_idx)),
                "val_subjects": va_sub.unique_subjects,
                "best_epoch": res["best_epoch"],
                "val_macro_f1": va_m["macro_f1"],
                "val_accuracy": va_m["accuracy"],
            }
        )

    f1s = [r["val_macro_f1"] for r in fold_rows]
    accs = [r["val_accuracy"] for r in fold_rows]
    oof_metrics = compute_classification_metrics(
        full_train.y, oof_preds, subjects=full_train.subjects
    )

    return {
        "num_folds": n_folds,
        "fold_macro_f1_mean": round(float(np.mean(f1s)), 6),
        "fold_macro_f1_std": round(float(np.std(f1s)), 6),
        "fold_accuracy_mean": round(float(np.mean(accs)), 6),
        "fold_accuracy_std": round(float(np.std(accs)), 6),
        "oof_metrics": oof_metrics,
        "folds": fold_rows,
    }


def run_tuning_pipeline(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Execute Phase 4 HPO, 5-fold GroupCV verification, and channel ablations."""
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

    all_9_indices = resolve_channel_indices(None)
    default_train_cfg = cfg["default_training"]
    candidates: list[dict[str, Any]] = cfg["candidates"]
    cand_by_id = {c["id"]: c for c in candidates}

    # Stage 1: Evaluate all candidates on primary subject-disjoint val split (16 train / 5 val)
    val_trials: list[dict[str, Any]] = []
    for cand in candidates:
        t_cfg = _merge_training_cfg(default_train_cfg, cand.get("training", {}))
        res = run_single_training(
            train_split=train_split,
            val_split=val_split,
            model_cfg=cand["model"],
            train_cfg=t_cfg,
            ch_indices=all_9_indices,
            seed=seed,
            device=device,
            checkpoint_path=None,
            aug_cfg=cand.get("augmentation"),
            compute_latency=False,
        )
        val_trials.append(
            {
                "id": cand["id"],
                "label": cand["label"],
                "model": cand["model"],
                "training": t_cfg,
                "augmentation": cand.get("augmentation", {"enabled": False}),
                "trainable_params": res["param_counts"]["trainable"],
                "best_epoch": res["best_epoch"],
                "epochs_completed": res["epochs_completed"],
                "runtime_seconds": res["runtime_seconds"],
                "train_macro_f1": res["train_metrics"]["macro_f1"],
                "val_macro_f1": res["val_metrics"]["macro_f1"],
                "val_accuracy": res["val_metrics"]["accuracy"],
                "val_per_class_f1": res["val_metrics"]["per_class_f1"],
                "val_per_subject_summary": res["val_metrics"]["per_subject_summary"],
            }
        )

    # Stage 2: Build 5-fold GroupCV shortlist (always include ref + rescnn + top_k on val)
    sorted_trials = sorted(val_trials, key=lambda r: r["val_macro_f1"], reverse=True)
    shortlist_ids: list[str] = []
    for cid in cfg.get("cv_always_include", []):
        if cid in cand_by_id and cid not in shortlist_ids:
            shortlist_ids.append(cid)
    top_k = int(cfg.get("cv_shortlist_top_k", 3))
    for trial in sorted_trials[:top_k]:
        if trial["id"] not in shortlist_ids:
            shortlist_ids.append(trial["id"])

    val0_lookup = {t["id"]: t for t in val_trials}
    cv_folds = int(split_cfg["cv_folds"])
    cv_evaluations: list[dict[str, Any]] = []
    for cid in shortlist_ids:
        cand = cand_by_id[cid]
        cv_res = _run_group_cv_for_candidate(
            full_train=full_train,
            cand=cand,
            default_train_cfg=default_train_cfg,
            ch_indices=all_9_indices,
            n_folds=cv_folds,
            seed=seed,
            device=device,
        )
        cv_evaluations.append(
            {
                "id": cid,
                "label": cand["label"],
                "trainable_params": val0_lookup[cid]["trainable_params"],
                "val0_macro_f1": val0_lookup[cid]["val_macro_f1"],
                "val0_accuracy": val0_lookup[cid]["val_accuracy"],
                "cv_macro_f1_mean": cv_res["fold_macro_f1_mean"],
                "cv_macro_f1_std": cv_res["fold_macro_f1_std"],
                "cv_accuracy_mean": cv_res["fold_accuracy_mean"],
                "cv_accuracy_std": cv_res["fold_accuracy_std"],
                "oof_macro_f1": cv_res["oof_metrics"]["macro_f1"],
                "oof_per_class_f1": cv_res["oof_metrics"]["per_class_f1"],
                "folds": cv_res["folds"],
            }
        )

    # Select winner strictly by 5-fold GroupCV mean Macro-F1 (tie-break: val0 Macro-F1)
    winner_cv = max(
        cv_evaluations,
        key=lambda r: (r["cv_macro_f1_mean"], r["val0_macro_f1"]),
    )
    winner_cand = cand_by_id[winner_cv["id"]]
    winner_train_cfg = _merge_training_cfg(
        default_train_cfg, winner_cand.get("training", {})
    )

    # Stage 3: Train & save final locked Phase 4 checkpoint on primary split + multi-seed check
    out_cfg = cfg["outputs"]
    winner_ckpt_path = Path(out_cfg["winner_checkpoint"])
    winner_primary = run_single_training(
        train_split=train_split,
        val_split=val_split,
        model_cfg=winner_cand["model"],
        train_cfg=winner_train_cfg,
        ch_indices=all_9_indices,
        seed=seed,
        device=device,
        checkpoint_path=winner_ckpt_path,
        aug_cfg=winner_cand.get("augmentation"),
        compute_latency=True,
    )

    multi_seed_runs: list[dict[str, Any]] = []
    for s in cfg.get("multi_seeds", [42, 123, 2025]):
        s_int = int(s)
        s_res = run_single_training(
            train_split=train_split,
            val_split=val_split,
            model_cfg=winner_cand["model"],
            train_cfg=winner_train_cfg,
            ch_indices=all_9_indices,
            seed=s_int,
            device=device,
            checkpoint_path=None,
            aug_cfg=winner_cand.get("augmentation"),
            compute_latency=False,
        )
        multi_seed_runs.append(
            {
                "seed": s_int,
                "best_epoch": s_res["best_epoch"],
                "val_macro_f1": s_res["val_metrics"]["macro_f1"],
                "val_accuracy": s_res["val_metrics"]["accuracy"],
            }
        )
    ms_f1s = [r["val_macro_f1"] for r in multi_seed_runs]

    # Stage 4: Controlled Channel Ablations (9ch vs 6ch dynamic vs 6ch raw IMU) via 5-fold GroupCV
    ablation_rows: list[dict[str, Any]] = []
    for abl in cfg.get("channel_ablations", []):
        abl_indices = resolve_channel_indices(abl.get("channels"))
        if abl["id"] == "all_9ch":
            # Reuse already-computed 5-fold GroupCV for the 9-channel winner.
            ablation_rows.append(
                {
                    "id": abl["id"],
                    "label": abl["label"],
                    "num_channels": len(abl_indices),
                    "cv_macro_f1_mean": winner_cv["cv_macro_f1_mean"],
                    "cv_macro_f1_std": winner_cv["cv_macro_f1_std"],
                    "cv_accuracy_mean": winner_cv["cv_accuracy_mean"],
                    "oof_macro_f1": winner_cv["oof_macro_f1"],
                    "oof_per_class_f1": winner_cv["oof_per_class_f1"],
                }
            )
            continue

        abl_cv = _run_group_cv_for_candidate(
            full_train=full_train,
            cand=winner_cand,
            default_train_cfg=default_train_cfg,
            ch_indices=abl_indices,
            n_folds=cv_folds,
            seed=seed,
            device=device,
        )
        ablation_rows.append(
            {
                "id": abl["id"],
                "label": abl["label"],
                "num_channels": len(abl_indices),
                "cv_macro_f1_mean": abl_cv["fold_macro_f1_mean"],
                "cv_macro_f1_std": abl_cv["fold_macro_f1_std"],
                "cv_accuracy_mean": abl_cv["fold_accuracy_mean"],
                "oof_macro_f1": abl_cv["oof_metrics"]["macro_f1"],
                "oof_per_class_f1": abl_cv["oof_metrics"]["per_class_f1"],
            }
        )

    # Build comparison chart combining Phase 2 baselines + Phase 4 CV shortlist
    chart_rows: list[dict[str, Any]] = []
    baselines_path = Path(cfg.get("baselines_json", ""))
    if baselines_path.is_file():
        with baselines_path.open("r", encoding="utf-8") as handle:
            b_data = json.load(handle)
        lr_m = b_data["models"]["logistic_regression"]
        rf_m = b_data["models"]["random_forest"]
        chart_rows.append(
            {
                "label": f"LogReg (C={lr_m['best_on_val']['params']['C']})",
                "val0_macro_f1": lr_m["best_on_val"]["val_metrics"]["macro_f1"],
                "cv_macro_f1_mean": lr_m["group_cv_5fold"]["fold_macro_f1_mean"],
                "cv_macro_f1_std": lr_m["group_cv_5fold"]["fold_macro_f1_std"],
            }
        )
        chart_rows.append(
            {
                "label": "RandomForest (300 trees)",
                "val0_macro_f1": rf_m["best_on_val"]["val_metrics"]["macro_f1"],
                "cv_macro_f1_mean": rf_m["group_cv_5fold"]["fold_macro_f1_mean"],
                "cv_macro_f1_std": rf_m["group_cv_5fold"]["fold_macro_f1_std"],
            }
        )
    chart_rows.extend(cv_evaluations)

    figures_dir = Path(out_cfg["figures_dir"])
    comp_fig = figures_dir / "phase4_hpo_comparison.png"
    abl_fig = figures_dir / "phase4_channel_ablation.png"
    curves_fig = figures_dir / "phase4_tuned_cnn_curves.png"
    cm_fig = figures_dir / "phase4_tuned_cnn_val_cm.png"

    save_tuning_comparison_figure(chart_rows, comp_fig)
    save_ablation_figure(
        ablation_rows,
        title=f"Phase 4 Channel Ablation (5-Fold GroupCV — {winner_cand['label']})",
        output_path=abl_fig,
    )
    save_training_curves_figure(
        winner_primary["history"],
        best_epoch=winner_primary["best_epoch"],
        title=f"Locked Tuned Winner: {winner_cand['label']} (Val Macro-F1={winner_primary['val_metrics']['macro_f1']:.4f} @ Ep {winner_primary['best_epoch']})",
        output_path=curves_fig,
    )
    save_confusion_matrices_figure(
        [
            (
                f"{winner_cand['label']} (val, ep={winner_primary['best_epoch']})\n"
                f"Macro-F1={winner_primary['val_metrics']['macro_f1']:.4f}",
                np.asarray(winner_primary["val_metrics"]["confusion_matrix"], dtype=np.int64),
            )
        ],
        output_path=cm_fig,
    )

    elapsed = round(time.perf_counter() - t_start, 3)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None

    return {
        "status": "VERIFIED",
        "phase": "phase_4_tuning_and_ablations",
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
            "primary_train_windows": int(len(train_split.y)),
            "primary_train_subjects": train_split.unique_subjects,
            "primary_val_windows": int(len(val_split.y)),
            "primary_val_subjects": val_split.unique_subjects,
            "group_cv_folds": cv_folds,
            "group_cv_total_windows": int(len(full_train.y)),
            "group_cv_subjects": full_train.unique_subjects,
            "locked_test_subject_ids": sorted(locked_test_subjects),
            "strictly_disjoint": True,
        },
        "stage1_val_trials": val_trials,
        "stage2_group_cv_shortlist": cv_evaluations,
        "locked_winner": {
            "id": winner_cand["id"],
            "label": winner_cand["label"],
            "selection_criterion": "highest 5-fold GroupCV mean Macro-F1 (all 21 train subjects)",
            "checkpoint_path": str(winner_ckpt_path.as_posix()),
            "model_config": winner_cand["model"],
            "training_config": winner_train_cfg,
            "augmentation_config": winner_cand.get("augmentation", {"enabled": False}),
            "parameters": winner_primary["param_counts"],
            "inference_latency_ms": winner_primary["inference_latency_ms"],
            "group_cv_5fold": winner_cv,
            "primary_split_best_epoch": winner_primary["best_epoch"],
            "primary_split_train_metrics": winner_primary["train_metrics"],
            "primary_split_val_metrics": winner_primary["val_metrics"],
            "multi_seed_val_stability": {
                "runs": multi_seed_runs,
                "val_macro_f1_mean": round(float(np.mean(ms_f1s)), 6),
                "val_macro_f1_std": round(float(np.std(ms_f1s)), 6),
            },
        },
        "stage4_channel_ablations": ablation_rows,
        "figures": [
            str(comp_fig.as_posix()),
            str(abl_fig.as_posix()),
            str(curves_fig.as_posix()),
            str(cm_fig.as_posix()),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 systematic HPO, 5-fold GroupCV, and ablations."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cnn_tuning.yaml"),
        help="Path to Phase 4 YAML configuration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    report = run_tuning_pipeline(cfg, args.config)
    output_path = Path(cfg["outputs"]["results_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    winner = report["locked_winner"]
    w_cv = winner["group_cv_5fold"]
    w_ms = winner["multi_seed_val_stability"]
    print("=== Phase 4 Tuning, 5-Fold GroupCV & Ablations [VERIFIED] ===")
    print(f"Stage 1 trials evaluated on primary val: {len(report['stage1_val_trials'])}")
    for row in report["stage2_group_cv_shortlist"]:
        print(
            f"  [GroupCV] {row['label']:34s} | "
            f"CV={row['cv_macro_f1_mean']:.4f} +/- {row['cv_macro_f1_std']:.4f} | "
            f"val0={row['val0_macro_f1']:.4f} | params={row['trainable_params']:,}"
        )
    print(
        f"Locked Winner: {winner['label']} | "
        f"5-fold CV Macro-F1={w_cv['cv_macro_f1_mean']:.4f} +/- {w_cv['cv_macro_f1_std']:.4f} | "
        f"Multi-seed val0={w_ms['val_macro_f1_mean']:.4f} +/- {w_ms['val_macro_f1_std']:.4f}"
    )
    print("Channel Ablations (5-fold GroupCV):")
    for abl in report["stage4_channel_ablations"]:
        print(
            f"  {abl['label']:32s} | CV={abl['cv_macro_f1_mean']:.4f} +/- {abl['cv_macro_f1_std']:.4f} | "
            f"SITTING={abl['oof_per_class_f1']['SITTING']:.4f}, "
            f"STANDING={abl['oof_per_class_f1']['STANDING']:.4f}, "
            f"LAYING={abl['oof_per_class_f1']['LAYING']:.4f}"
        )
    print(f"Saved report: {output_path} | runtime={report['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
