"""Safety baseline check for the HAR pipeline. Run from the project root.

    python tools/har_check.py before   # save baseline summary
    python tools/har_check.py after    # compare against "before"

All runs use CPU only (deterministic) and write only inside outputs/_check/<tag>/.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

CHECK_DIR = Path("outputs/_check")
DATA_DIR = "data/raw/UCI HAR Dataset"


def write_configs(run_dir: Path) -> dict[str, Path]:
    cfg_dir = run_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir.as_posix()
    configs = {
        "baselines": f"""seed: 42
data_dir: "{DATA_DIR}"
output_json: "{out}/baselines_results.json"
split: {{n_splits: 4, val_fold: 0, cv_folds: 5}}
features:
  stats: [mean, std, min, max, median, iqr, rms, mean_abs_diff]
models:
  logistic_regression: {{max_iter: 2000, c_candidates: [0.05, 0.1, 1.0, 10.0]}}
  random_forest:
    n_estimators: 300
    candidates:
      - {{max_depth: null, min_samples_leaf: 1}}
      - {{max_depth: 16, min_samples_leaf: 1}}
      - {{max_depth: 12, min_samples_leaf: 2}}
""",
        "cnn_smoke": f"""seed: 42
data_dir: "{DATA_DIR}"
baselines_json: "outputs/baselines_results.json"
split: {{n_splits: 4, val_fold: 0}}
channels: null
model:
  name: "CompactCNN1D"
  conv_channels: [32, 64, 128]
  kernel_sizes: [7, 5, 3]
  dropout: 0.3
  use_batchnorm: true
  head_hidden: null
training:
  epochs: 1
  batch_size: 64
  optimizer: "adamw"
  lr: 0.001
  weight_decay: 0.0001
  scheduler: "cosine"
  early_stopping_patience: 1
  num_workers: 0
outputs:
  checkpoint_path: "{out}/cnn_smoke.pt"
  results_json: "{out}/cnn_smoke_results.json"
""",
        "tune_smoke": f"""seed: 42
multi_seeds: [42]
data_dir: "{DATA_DIR}"
split: {{n_splits: 4, val_fold: 0, cv_folds: 2}}
default_training:
  epochs: 1
  batch_size: 64
  optimizer: "adamw"
  lr: 0.001
  weight_decay: 0.0001
  label_smoothing: 0.0
  scheduler: "cosine"
  early_stopping_patience: 1
  num_workers: 0
candidates:
  - id: "cnn_baseline_ref"
    label: "CompactCNN1D (Phase 3 ref)"
    model: {{name: "CompactCNN1D", conv_channels: [32, 64, 128], kernel_sizes: [7, 5, 3],
             dropout: 0.3, use_batchnorm: true, head_hidden: null}}
    training: {{}}
    augmentation: {{enabled: false}}
  - id: "rescnn_aug"
    label: "CompactResCNN1D (+ train aug)"
    model: {{name: "CompactResCNN1D", conv_channels: [32, 64, 128], kernel_sizes: [7, 5, 3],
             dropout: 0.35, block_dropout: 0.1, use_batchnorm: true, head_hidden: null}}
    training: {{weight_decay: 0.001, label_smoothing: 0.05}}
    augmentation: {{enabled: true, jitter_std: 0.03, scale_std: 0.08}}
cv_shortlist_top_k: 1
cv_always_include: ["cnn_baseline_ref"]
channel_ablations:
  - {{id: "all_9ch", label: "All 9 channels (Full)", channels: null}}
  - id: "no_gravity_6ch"
    label: "6 ch (body_acc + body_gyro)"
    channels: ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]
outputs:
  winner_checkpoint: "{out}/tuned_smoke.pt"
  results_json: "{out}/tune_smoke_results.json"
""",
        "final_eval": f"""seed: 42
data_dir: "{DATA_DIR}"
baselines_results: "outputs/baselines_results.json"
split: {{n_splits: 4, val_fold: 0}}
locked_models:
  cnn_baseline_checkpoint: "outputs/checkpoints/cnn_baseline_best.pt"
  cnn_tuned_checkpoint: "outputs/checkpoints/cnn_tuned_best.pt"
outputs:
  results_json: "{out}/final_test_results.json"
""",
    }
    paths = {}
    for name, text in configs.items():
        paths[name] = cfg_dir / f"{name}.yaml"
        paths[name].write_text(text, encoding="utf-8")
    return paths


def run(args: list[str]) -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")  # CPU only: deterministic results
    t0 = time.perf_counter()
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, env=env)
    print(f"   done in {time.perf_counter() - t0:.1f}s", flush=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(run_dir: Path) -> dict[str, float]:
    s: dict[str, float] = {}
    audit = load(run_dir / "dataset_summary.json")
    s["audit.total_windows"] = audit["total_windows"]
    s["audit.cross_split_duplicates"] = audit["subject_disjointness"][
        "cross_split_exact_duplicate_windows"
    ]
    s["audit.train_total_acc_x_mean"] = audit["splits"]["train"]["per_channel_stats"][
        "total_acc_x"
    ]["mean"]
    base = load(run_dir / "baselines_results.json")
    for m in ("logistic_regression", "random_forest"):
        s[f"baselines.{m}.val_macro_f1"] = base["models"][m]["best_on_val"]["val_metrics"]["macro_f1"]
        s[f"baselines.{m}.cv_mean"] = base["models"][m]["group_cv_5fold"]["fold_macro_f1_mean"]
    cnn = load(run_dir / "cnn_smoke_results.json")
    s["cnn_smoke.val_macro_f1"] = cnn["best_checkpoint_metrics"]["val"]["macro_f1"]
    s["cnn_smoke.train_loss_ep1"] = cnn["history"][0]["train_loss"]
    tune = load(run_dir / "tune_smoke_results.json")
    for t in tune["stage1_val_trials"]:
        s[f"tune_smoke.val.{t['id']}"] = t["val_macro_f1"]
    for r in tune["stage2_group_cv_shortlist"]:
        s[f"tune_smoke.cv.{r['id']}"] = r["cv_macro_f1_mean"]
    for a in tune["stage4_channel_ablations"]:
        s[f"tune_smoke.ablation.{a['id']}"] = a["cv_macro_f1_mean"]
    final = load(run_dir / "final_test_results.json")
    for m, v in final["models"].items():
        s[f"test.{m}.macro_f1"] = v["test_metrics"]["macro_f1"]
        s[f"test.{m}.errors"] = v["total_test_errors"]
    return s


def compare_with_saved_results(summary: dict[str, float]) -> None:
    """Compare with the protected result files (tolerance: other machines differ slightly)."""
    base = load(Path("outputs/baselines_results.json"))
    final = load(Path("outputs/final_test_results.json"))
    saved = {
        "baselines.logistic_regression.val_macro_f1": base["models"]["logistic_regression"][
            "best_on_val"
        ]["val_metrics"]["macro_f1"],
        "baselines.random_forest.val_macro_f1": base["models"]["random_forest"]["best_on_val"][
            "val_metrics"
        ]["macro_f1"],
    }
    for m, v in final["models"].items():
        saved[f"test.{m}.macro_f1"] = v["test_metrics"]["macro_f1"]
    print("\nComparison with the saved results in outputs/ (tolerance 0.002):")
    for key, value in saved.items():
        status = "OK  " if abs(summary[key] - value) <= 0.002 else "WARN"
        print(f"  {status} {key}: now={summary[key]} saved={value}")


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "before"
    run_dir = CHECK_DIR / tag
    cfg = write_configs(run_dir)
    run(["-m", "src.phases.inspect_data", "--output", f"{run_dir.as_posix()}/dataset_summary.json"])
    run(["-m", "src.phases.train_baselines", "--config", str(cfg["baselines"])])
    run(["-m", "src.phases.train_cnn", "--config", str(cfg["cnn_smoke"])])
    run(["-m", "src.phases.tune_cnn", "--config", str(cfg["tune_smoke"])])
    run(["-m", "src.phases.eval_test", "--config", str(cfg["final_eval"])])
    summary = summarize(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    compare_with_saved_results(summary)

    before_path = CHECK_DIR / "before" / "summary.json"
    if tag != "before" and before_path.is_file():
        before = load(before_path)
        diffs = [k for k in before if before[k] != summary.get(k)]
        print("\nBASELINE MATCH" if not diffs else f"\nBASELINE DIFF in: {diffs}")
        return 0 if not diffs else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
