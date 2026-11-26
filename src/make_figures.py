"""Draw the report figures (Hungarian labels) from the saved result JSON files.

Nothing is trained or evaluated here: every number comes from outputs/*.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from src.data import ACTIVITY_NAMES  # noqa: E402

CLASS_LABELS = {
    "WALKING": "Séta",
    "WALKING_UPSTAIRS": "Lépcsőn fel",
    "WALKING_DOWNSTAIRS": "Lépcsőn le",
    "SITTING": "Ülés",
    "STANDING": "Állás",
    "LAYING": "Fekvés",
}
# Phase 4 candidate ids -> labels used in the report.
CANDIDATE_LABELS = {
    "cnn_baseline_ref": "Alap 1D-CNN",
    "rescnn_base": "Reziduális CNN",
    "rescnn_aug": "Reziduális CNN + adatbővítés",
    "cnn_wider_k9": "Hangolt 1D-CNN",
}
# One fixed color per model, the same in every figure.
MODEL_COLORS = {
    "Logisztikus regresszió": "#f47230",
    "Véletlen erdő": "#6a4ce2",
    "Alap 1D-CNN": "#00b894",
    "Hangolt 1D-CNN": "#257de0",
}
GRID_STYLE = {"color": "#d9d9d9", "linewidth": 0.6}

plt.rcParams.update(
    {
        # Times New Roman like the report; the others are fallbacks with the same letters.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "savefig.dpi": 200,
    }
)


def hu_number(value: float, digits: int = 2) -> str:
    """Format a number with a Hungarian decimal comma."""
    return f"{value:.{digits}f}".replace(".", ",")


def comma_axis(digits: int = 1) -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: hu_number(value, digits))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_training_curves(history: list[dict], best_epoch: int, path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(6.5, 2.8))
    for ax, key, ylabel in ((ax_loss, "loss", "Veszteség"), (ax_f1, "macro_f1", "Makro-F1")):
        ax.plot(epochs, [r[f"train_{key}"] for r in history], color="#257de0", label="Tanító")
        ax.plot(epochs, [r[f"val_{key}"] for r in history], color="#f47230", label="Validációs")
        ax.axvline(best_epoch, color="#666666", ls="--", lw=1, label=f"{best_epoch}. epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", **GRID_STYLE)
        ax.yaxis.set_major_formatter(comma_axis(2))
    ax_loss.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    save(fig, path)


def plot_model_comparison(rows: list[dict], path: Path) -> None:
    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))
    height = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(rows) + 1.0))
    val0 = [r["val0"] for r in rows]
    cv_mean = [r["cv_mean"] for r in rows]
    cv_std = [r["cv_std"] for r in rows]
    ax.barh(y - height / 2, val0, height, color="#f47230", label="Validációs halmaz (5 alany)")
    ax.barh(
        y + height / 2, cv_mean, height, color="#257de0", xerr=cv_std,
        error_kw={"lw": 1, "capsize": 3}, label="5-szörös keresztvalidáció (átlag ± szórás)",
    )
    for i in range(len(rows)):
        ax.text(val0[i] + 0.003, i - height / 2, hu_number(val0[i], 3), va="center", fontsize=9)
        ax.text(
            cv_mean[i] + cv_std[i] + 0.003, i + height / 2, hu_number(cv_mean[i], 3),
            va="center", fontsize=9,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.80, 1.02)
    ax.set_xlabel("Makro-F1")
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(comma_axis(2))
    ax.grid(axis="x", **GRID_STYLE)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, path)


def plot_channel_ablation(rows: list[dict], path: Path) -> None:
    config_labels = {
        "all_9ch": "Mind a 9 csatorna",
        "no_gravity_6ch": "Testgyorsulás +\nszögsebesség\n(gravitáció nélkül)",
        "raw_imu_6ch": "Teljes gyorsulás +\nszögsebesség",
    }
    series = [
        ("Makro-F1", [r["cv_macro_f1_mean"] for r in rows], "#257de0"),
        ("Ülés F1", [r["oof_per_class_f1"]["SITTING"] for r in rows], "#f47230"),
        ("Állás F1", [r["oof_per_class_f1"]["STANDING"] for r in rows], "#6a4ce2"),
        ("Fekvés F1", [r["oof_per_class_f1"]["LAYING"] for r in rows], "#00b894"),
    ]
    x = np.arange(len(rows))
    width = 0.2
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, (name, values, color) in enumerate(series):
        offset = (i - 1.5) * width
        ax.bar(x + offset, values, width, color=color, label=name)
        for xi, val in zip(x, values, strict=True):
            ax.text(xi + offset, val + 0.005, hu_number(val), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([config_labels.get(r["id"], r["label"]) for r in rows], fontsize=10)
    ax.set_ylim(0.75, 1.03)
    ax.set_ylabel("F1 (5-szörös keresztvalidáció)")
    ax.yaxis.set_major_formatter(comma_axis(2))
    ax.grid(axis="y", **GRID_STYLE)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=9.5)
    fig.tight_layout()
    save(fig, path)


def plot_confusion_matrices(named_cms: list[tuple[str, np.ndarray]], path: Path) -> None:
    class_labels = [CLASS_LABELS[name] for name in ACTIVITY_NAMES]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.4))
    for ax, (title, cm) in zip(axes.flat, named_cms, strict=True):
        ax.spines[:].set_visible(False)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=9.5)
        ticks = np.arange(len(class_labels))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_labels, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(class_labels, fontsize=9)
        ax.set_xlabel("Becsült osztály", fontsize=9.5)
        ax.set_ylabel("Valódi osztály", fontsize=9.5)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm_norm[i, j] > 0.55 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8.5)
    fig.tight_layout()
    save(fig, path)


def plot_grouped_bars(
    groups: list[str],
    model_scores: dict[str, list[float]],
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    x = np.arange(len(groups))
    width = 0.8 / len(model_scores)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, (model, scores) in enumerate(model_scores.items()):
        offset = (i - (len(model_scores) - 1) / 2) * width
        ax.bar(x + offset, scores, width, color=MODEL_COLORS[model], label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(comma_axis(1))
    ax.grid(axis="y", **GRID_STYLE)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=9.5)
    fig.tight_layout()
    save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw report figures from outputs/*.json.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    baselines = load_json(args.results_dir / "baselines_results.json")
    cnn_baseline = load_json(args.results_dir / "cnn_baseline_results.json")
    tuning = load_json(args.results_dir / "tuning_results.json")
    final = load_json(args.results_dir / "final_test_results.json")
    out = args.figures_dir

    plot_training_curves(
        cnn_baseline["history"],
        cnn_baseline["training"]["best_epoch"],
        out / "cnn_baseline_curves.png",
    )

    comparison_rows = []
    for key, label in (
        ("logistic_regression", "Logisztikus regresszió"),
        ("random_forest", "Véletlen erdő"),
    ):
        model = baselines["models"][key]
        comparison_rows.append(
            {
                "label": label,
                "val0": model["best_on_val"]["val_metrics"]["macro_f1"],
                "cv_mean": model["group_cv_5fold"]["fold_macro_f1_mean"],
                "cv_std": model["group_cv_5fold"]["fold_macro_f1_std"],
            }
        )
    for row in tuning["stage2_group_cv_shortlist"]:
        comparison_rows.append(
            {
                "label": CANDIDATE_LABELS.get(row["id"], row["label"]),
                "val0": row["val0_macro_f1"],
                "cv_mean": row["cv_macro_f1_mean"],
                "cv_std": row["cv_macro_f1_std"],
            }
        )
    plot_model_comparison(comparison_rows, out / "model_comparison.png")

    plot_channel_ablation(tuning["stage4_channel_ablations"], out / "channel_ablation.png")

    winner = tuning["locked_winner"]
    if "primary_split_history" in winner:  # only saved by runs after the code cleanup
        plot_training_curves(
            winner["primary_split_history"],
            winner["primary_split_best_epoch"],
            out / "tuned_cnn_curves.png",
        )

    test_models = {
        "Logisztikus regresszió": final["models"]["logistic_regression"]["test_metrics"],
        "Véletlen erdő": final["models"]["random_forest"]["test_metrics"],
        "Alap 1D-CNN": final["models"]["cnn_baseline"]["test_metrics"],
        "Hangolt 1D-CNN": final["models"]["cnn_tuned_winner"]["test_metrics"],
    }
    plot_confusion_matrices(
        [
            (f"{name} (makro-F1: {hu_number(m['macro_f1'], 3)})", np.array(m["confusion_matrix"]))
            for name, m in test_models.items()
        ],
        out / "test_confusion_matrices.png",
    )
    subject_ids = sorted(test_models["Hangolt 1D-CNN"]["per_subject"], key=int)
    plot_grouped_bars(
        subject_ids,
        {
            name: [m["per_subject"][sid]["macro_f1"] for sid in subject_ids]
            for name, m in test_models.items()
        },
        "Tesztalany azonosítója",
        "Makro-F1",
        out / "test_per_subject_f1.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
