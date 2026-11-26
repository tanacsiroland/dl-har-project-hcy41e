# Project Plan — Smartphone IMU Human Activity Recognition (`NSXMG1HMLF`)

**Course:** *Mély gépi tanulás* (`NSXMG1HMLF`), Óbuda University / Institute of Cyber-Physical Systems (Dr. habil. Dineva Adrienn, 2026-27-1)
**Status:** Phase 1 in progress (dataset & environment audit; no model training executed yet).

---

## 1. Research Question & Hypotheses

- **Research Question (RQ):** Under strict subject-disjoint evaluation on the 9-channel raw smartphone inertial time-series windows `(N, 9, 128)` at 50 Hz, does a compact 1D-CNN trained from scratch in PyTorch achieve higher Macro-F1 than classical ML baselines (Logistic Regression, Random Forest) trained on per-channel summary statistics computed from the same 9 channels—and which activities or subjects drive remaining errors?
- **Primary Hypothesis ($H_1$):** Local temporal convolutions across the 2.56 s inertial window capture waveform shape, step periodicity, and cross-axis phase relationships lost in global window summary statistics, improving unseen-subject Macro-F1 (especially between dynamic locomotion classes `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`) while staying compact enough for low-latency mobile inference.
- **Null Hypothesis ($H_0$):** Global per-channel summary statistics capture all discriminative information available in the 9 channels, leaving no meaningful Macro-F1 advantage for the 1D-CNN on unseen subjects (or causing the CNN to overfit training-subject idiosyncrasies).
- **Scoped Ablation Hypothesis ($H_2$):** Retaining `total_acc_xyz` alongside `body_acc_xyz` and `body_gyro_xyz` (9 channels vs. 6 gravity-removed channels) is necessary to separate static postures (`SITTING`, `STANDING`, `LAYING`) because gravity projection encodes torso/phone orientation.

---

## 2. Requirements-to-Evidence Table

| Course / Project Requirement | Source | Planned Evidence & Artifact | Status |
|---|---|---|---|
| **PyTorch DL model trained from scratch** (no LLM/foundation fine-tuning/apps) | `GEMINI.md` | Compact 1D-CNN (`src/models/cnn1d.py`) trained end-to-end on `(9, 128)` windows | `PLANNED` (Phase 3) |
| **Classical ML comparison baselines** | `GEMINI.md` | Logistic Regression & Random Forest on train-fit summary stats (`src/baselines.py`) | `PLANNED` (Phase 2) |
| **Subject-disjoint splits & zero leakage** | `GEMINI.md` | `assert` disjoint `subject_id` across `train`/`val`/`test`; scalers fit strictly on `train`; audit in `outputs/dataset_summary.json` | `IMPLEMENTED` (audit) / `PLANNED` (Phase 2 split) |
| **Test set lock & val-driven HPO** | `GEMINI.md`, Syllabus | All tuning & checkpointing on `val` Macro-F1; single final `test` pass (`outputs/`) | `PLANNED` (Phases 4–5) |
| **Honest multi-metric & per-subject evaluation** | `GEMINI.md` | Macro-F1, Accuracy, per-class F1, confusion matrices, per-subject Macro-F1 (`outputs/`) | `PLANNED` (Phases 2–5) |
| **Rubric: Results quality (10 pts)** | `CONTEXT.md` | Verified baseline vs. tuned 1D-CNN comparisons + error analysis | `PLANNED` (Phases 2–5) |
| **Rubric: Creativity & uniqueness (10 pts)** | `CONTEXT.md` | Raw-sensor temporal modeling vs. summary baselines, channel ablation (9 vs. 6 ch), per-subject generalization diagnostics | `PLANNED` (Phases 4–5) |
| **Rubric: Complexity & technical implementation (15 pts)** | `CONTEXT.md` | Config-driven YAML pipeline, GroupKFold/val tuning, regularization/scheduler ablations, parameter/latency profile | `PLANNED` (Phases 2–4) |
| **Rubric: Code quality & documentation (15 pts)** | `CONTEXT.md`, Syllabus | Clean `src/` modules, `configs/`, `README.md`, 10–15p report (1p lit review, <=1p problem statement, provenance, pipeline, HPO, eval, lessons), PPT slides, Moodle `.zip` | `IN PROGRESS` (Phase 1) |
| **Passing / Exam-exemption target** | Syllabus PDF (75%), Assignment text (66%) | Target `>= 75%` (>= 37.5 / 50 rubric pts) until 66% vs. 75% conflict is clarified | `PLANNED` |

---

## 3. MVP Execution Phases

1. **Phase 1 — Environment & Data Audit (`CURRENT`):** Verify hardware/Python/CUDA, audit raw `UCI HAR Dataset` shapes/NaNs/labels/subject disjointness via `src/inspect_data.py`, lock project plan and foundation files.
2. **Phase 2 — Subject-Disjoint Split & Classical Baselines (`PLANNED`):** Partition the 21 training subjects into subject-disjoint `train` / `val` (or GroupKFold CV) while keeping the 9 test subjects locked; compute leak-free summary statistics and benchmark Logistic Regression & Random Forest on `val`.
3. **Phase 3 — Compact 1D-CNN MVP (`PLANNED`):** Implement config-driven PyTorch DataLoader + compact 1D-CNN trained from scratch on standardized `(9, 128)` windows, checkpoint by `val` Macro-F1, log learning curves and runtime metadata.
4. **Phase 4 — Systematic Tuning & Ablations (`PLANNED`):** Tune architecture/regularization/optimization hyperparameters on `val` (e.g., kernel sizes, channel width, dropout/weight decay, scheduler, and 9-channel vs. 6-channel ablation) without touching `test`.
5. **Phase 5 — Locked Final Evaluation & Diagnostics (`PLANNED`):** Single-pass evaluation of locked models on the 9 unseen `test` subjects; generate confusion matrices, per-class F1, per-subject Macro-F1 distributions, and failure-mode analysis.
6. **Phase 6 — Documentation, Slides & Packaging (`PLANNED`):** Write 10–15 page report covering all required sections, build PPT presentation, run full reproducibility check, and prepare Moodle `.zip` for December 5 deadline.

---

## 4. Key Technical & Project Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Subject domain shift across only 21 training subjects** | High variance in `val`/`test` Macro-F1; CNN memorizing subject gait signatures | Enforce subject-disjoint `val` (or GroupKFold), use BatchNorm + Dropout + weight decay, and report per-subject Macro-F1 spread |
| **Window-overlap / scaling leakage** | Inflated `val` scores due to 50% overlapping adjacent windows or global normalization | Never split by window; split exclusively by `subject_id` with assertions; fit `StandardScaler`/channel stats strictly on `train` |
| **Static posture confusion (`SITTING` vs. `STANDING`)** | Depresses Macro-F1 even if dynamic activities separate cleanly | Track per-class F1 and confusion matrices from Phase 2 onward; test sensitivity to `total_acc` gravity channels |
| **Test-set peeking during iteration** | Invalidates generalization claim and violates course discipline | Keep `test` evaluation isolated in a dedicated Phase 5 entrypoint; log all `outputs/` runs with timestamps |
| **CUDA / driver compatibility on RTX 5090** | PyTorch runtime failure if wheel lacks `sm_120` support | Verify PyTorch CUDA tensor execution during Phase 1 (`src/inspect_data.py --strict-env`) |
| **Unresolved course rules** | Mismatch in deliverable format or grading target | Track open questions in `CONTEXT.md` (66% vs. 75% exemption, group size / solo rule, Hungarian vs. English, talk duration, AI disclosure) and target `>= 75%` |
