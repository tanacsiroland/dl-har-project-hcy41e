# Smartphone IMU Human Activity Recognition (`NSXMG1HMLF`)

Course project for *Mély gépi tanulás* (`NSXMG1HMLF`), Óbuda University — Institute of Cyber-Physical Systems (Dr. habil. Dineva Adrienn, 2026-27-1).  
Author: Roland Tanácsi (`HCY41E`).

This repository compares compact PyTorch **1D-CNN** architectures (`CompactCNN1D`, `CompactResCNN1D`) trained from scratch on raw 9-channel smartphone inertial sensor windows `(N, 9, 128)` against classical ML baselines (**Logistic Regression**, **Random Forest**) trained on `72` per-channel summary statistics under **strictly subject-disjoint** evaluation and zero-leakage preprocessing.

---

## Verified Final Results Summary

| Model | Val Macro-F1 (5 subj) | 5-Fold GroupCV Macro-F1 (21 subj) | Locked Test Macro-F1 (9 subj) | Test Accuracy |
|---|---:|---:|---:|---:|
| **Logistic Regression** (`C=10.0`, 72 stats) | 0.8969 | 0.8939 ± 0.0291 | 0.9024 | 90.36% |
| **Random Forest** (`n=300`, 72 stats) | 0.8908 | 0.8774 ± 0.0410 | 0.8428 | 84.56% |
| **CompactCNN1D** (`32-64-128`, `38,054` params) | 0.9517 | 0.9461 ± 0.0356 | 0.9264 | 92.50% |
| **CompactCNN1D** (`48-96-160, k=9-5-3`, `74,582` params) | **0.9725** | **0.9563 ± 0.0332** | **0.9283** | **92.70%** |

---

## Repository Structure

```text
.
├── CONTEXT.md                              # Living project state & verified phase log
├── GEMINI.md                               # Engineering & scientific discipline rules
├── README.md                               # Setup, pipeline commands, and Moodle packaging guide
├── requirements.txt                        # Minimal Python dependencies
├── configs/
│   ├── baselines.yaml                      # Phase 2 classical ML baselines config
│   ├── cnn_baseline.yaml                   # Phase 3 CompactCNN1D MVP config
│   ├── cnn_tuning.yaml                     # Phase 4 HPO, 5-fold GroupCV & ablation config
│   └── final_eval.yaml                     # Phase 5 single-pass locked test evaluation config
├── data/
│   └── raw/UCI HAR Dataset/                # Extracted UCI HAR raw dataset (gitignored)
├── docs/
│   ├── NIK-KRI-TTK-Mely_gepi_tanulas-NSXMG1HMLF-26_27_1-DA.pdf
│   ├── project-plan.md                     # Requirements-to-evidence, RQ, MVP phases, risks
│   ├── dokumentacio.md                     # Full Hungarian project documentation (Markdown)
│   ├── dokumentacio.docx                   # Editable Hungarian documentation (Word, 10-15 pages)
│   └── prezentacio.pptx                    # Editable Hungarian defense slides (PowerPoint)
├── outputs/
│   ├── dataset_summary.json                # Phase 1 audit output
│   ├── baselines_results.json              # Phase 2 classical baselines output
│   ├── cnn_baseline_results.json           # Phase 3 1D-CNN baseline output
│   ├── tuning_results.json                 # Phase 4 HPO, GroupCV & ablation output
│   ├── final_test_results.json             # Phase 5 locked test evaluation output
│   ├── checkpoints/                        # Saved model checkpoints + train-fit scalers
│   └── figures/                            # Generated confusion matrices, curves & comparison plots
└── src/
    ├── __init__.py
    ├── inspect_data.py                     # Phase 1 environment & dataset integrity audit
    ├── data.py                             # Raw loader, StratifiedGroupKFold splitter, disjoint assertions
    ├── features.py                         # Leak-free per-window 72 summary feature extractor
    ├── metrics.py                          # Metrics, per-subject stats & Matplotlib figure helpers
    ├── torch_data.py                       # PyTorch Dataset wrapper
    ├── train_baselines.py                  # Phase 2 classical baselines & 5-fold GroupCV
    ├── train_cnn.py                        # Phase 3 1D-CNN training & validation pipeline
    ├── tune_cnn.py                         # Phase 4 HPO, 5-fold GroupCV, multi-seed & ablations
    ├── eval_test.py                        # Phase 5 single-pass locked test evaluator
    └── models/
        ├── __init__.py
        └── cnn1d.py                        # CompactCNN1D & CompactResCNN1D architectures
```

---

## Environment Setup

Requires **Python 3.10+** (verified locally on Python 3.11.9 / Windows 11 Pro + RTX 5090).

Create and activate an isolated `.venv` (already gitignored):

```powershell
# Option A: uv (recommended; verified on Windows 11 + Python 3.11.9)
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1

# Option B: standard library venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install minimal dependencies:

```powershell
# NVIDIA GPU with uv (verified on RTX 5090 / CUDA 12.8; install torch from cu128 index first)
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
uv pip install -r requirements.txt

# Or CPU / default PyPI via pip
pip install -r requirements.txt
```

---

## Dataset Acquisition & Verification

**Dataset:** *Human Activity Recognition Using Smartphones* (UCI ML Repository — Anguita et al., ESANN 2013).  
Expected local location: `data/raw/UCI HAR Dataset/`.

If not present locally, download `UCI HAR Dataset.zip` from the [UCI Machine Learning Repository (ID 240)](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and extract it so `data/raw/UCI HAR Dataset/activity_labels.txt`, `train/Inertial Signals/`, and `test/Inertial Signals/` exist.

---

## Reproducible End-to-End Pipeline

Run every phase cleanly from the repository root with `.venv` active:

```powershell
# Phase 1: Environment & dataset integrity audit -> outputs/dataset_summary.json
python -m src.inspect_data --strict-env

# Phase 2: Subject-disjoint classical baselines + 5-fold GroupCV -> outputs/baselines_results.json
python -m src.train_baselines --config configs/baselines.yaml

# Phase 3: Compact 1D-CNN MVP -> outputs/cnn_baseline_results.json
python -m src.train_cnn --config configs/cnn_baseline.yaml

# Phase 4: HPO, 5-fold GroupCV, multi-seed & channel ablations -> outputs/tuning_results.json
python -m src.tune_cnn --config configs/cnn_tuning.yaml

# Phase 5: Single-pass locked test evaluation -> outputs/final_test_results.json
python -m src.eval_test --config configs/final_eval.yaml
```

---

## Scientific Discipline Rules

- **Subject-disjoint splits only:** No `subject_id` ever appears in more than one split (`train`, `val`, `test`); asserted in `src/data.py`.
- **Zero leakage:** Scalers, channel means/stds, and summary statistics are computed strictly per window or fit strictly on `train`; augmentation is applied strictly inside `train` batches.
- **Locked test set:** All hyperparameter tuning and checkpoint selection occur strictly on `val` and 5-fold `StratifiedGroupKFold` over the 21 training subjects; `test` is evaluated once via `src/eval_test.py`.
