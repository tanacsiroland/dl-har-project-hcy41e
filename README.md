# Smartphone activity recognition with a small 1D-CNN

Course project for *Mély gépi tanulás* (NSXMG1HMLF), Óbuda University, Institute of Cyber-Physical Systems, 2026/27 autumn semester.
Author: Roland Tanácsi (HCY41E).

The project compares a small 1D convolutional network, trained from scratch on raw smartphone sensor windows (9 channels × 128 samples), with Logistic Regression and Random Forest trained on 72 summary statistics per window. The data is split by person, so every model is tested on people it has never seen. The test set is used once, at the end.

## Results

Test set: 9 people, 2 947 windows. Validation: 5 people. Cross-validation: 5-fold StratifiedGroupKFold over the 21 training people.

| Model | Validation Macro-F1 | 5-fold CV Macro-F1 | Test Macro-F1 | Test accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression (C=10, 72 features) | 0.897 | 0.894 ± 0.029 | 0.902 | 90.4% |
| Random Forest (300 trees, 72 features) | 0.891 | 0.877 ± 0.041 | 0.843 | 84.6% |
| CompactCNN1D, Phase 3 (38 054 parameters) | 0.952 | 0.946 ± 0.036 | 0.926 | 92.5% |
| CompactCNN1D, tuned (74 582 parameters) | 0.973 | 0.956 ± 0.033 | 0.928 | 92.7% |

## Setup

Python 3.11 (tested with 3.11.9 on Windows 11 and an NVIDIA RTX 5090).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# NVIDIA GPU only: install the CUDA 12.8 build of PyTorch first
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Without a GPU, skip the first `pip install` line. Everything also runs on the CPU, only slower.

## Data

Download *Human Activity Recognition Using Smartphones* from the [UCI Machine Learning Repository (ID 240)](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). Extract `UCI HAR Dataset.zip` so that `data/raw/UCI HAR Dataset/train/Inertial Signals/` and `data/raw/UCI HAR Dataset/test/Inertial Signals/` exist.

## Running the pipeline

Run from the repository root with the virtual environment active. To execute the entire pipeline (Phases 1–5 and figure generation) with a single command:

```powershell
python -m src.run_all --strict-env
```

Use `python -m src.run_all --figures-only` to regenerate the figures in `outputs/figures/` from existing `outputs/*.json` files without retraining. You can also run each phase individually:

```powershell
python -m src.phases.inspect_data --strict-env                        # Phase 1: data check       -> outputs/dataset_summary.json
python -m src.phases.train_baselines --config configs/baselines.yaml  # Phase 2: classical models -> outputs/baselines_results.json
python -m src.phases.train_cnn --config configs/cnn_baseline.yaml     # Phase 3: first 1D-CNN     -> outputs/cnn_baseline_results.json
python -m src.phases.tune_cnn --config configs/cnn_tuning.yaml        # Phase 4: tuning, ablation -> outputs/tuning_results.json (~10 min on GPU)
python -m src.phases.eval_test --config configs/final_eval.yaml       # Phase 5: test set, once   -> outputs/final_test_results.json
python -m src.phases.make_figures                                     # figures from JSON files   -> outputs/figures/
```

Phases 3 and 4 save model checkpoints to `outputs/checkpoints/` and overwrite previous results. GPU training is not bit-exact, so a rerun can give slightly different numbers.

## Folder structure

```text
configs/       YAML settings, one file per phase
data/raw/      UCI HAR dataset (not in git)
docs/          report (dokumentacio.docx), slides (prezentacio.pptx), course syllabus (PDF)
outputs/       result JSON files, model checkpoints, and generated figures (not in git)
src/
  run_all.py   single-command end-to-end pipeline runner
  core/        reusable data splitting, datasets, features, metrics, env, and trainer modules
  models/      1D-CNN architectures (CompactCNN1D, CompactResCNN1D)
  phases/      one CLI script per pipeline phase
```

## Evaluation rules

- No person appears in more than one of the train, validation and test sets. The code checks this (`src/core/data.py`).
- Scalers and channel statistics are fitted on the training people only.
- Hyperparameters and checkpoints are chosen on the validation people and with 5-fold cross-validation over the 21 training people. The test set is evaluated once, in `src/phases/eval_test.py`.

## License

MIT, see `LICENSE`.
