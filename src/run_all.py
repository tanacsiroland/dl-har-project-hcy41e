"""Single-command orchestrator for the end-to-end HAR pipeline.

Run from the repository root:
    python -m src.run_all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _run_module(module: str, args: list[str], dry_run: bool = False) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print(">>", " ".join(cmd), flush=True)
    if dry_run:
        return
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    print(f"   finished in {time.perf_counter() - t0:.1f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the UCI HAR pipeline end-to-end (Phases 1-5 + figures)."
    )
    parser.add_argument(
        "--strict-env",
        action="store_true",
        help="Fail in Phase 1 if any required package is missing.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip Phase 1 dataset integrity audit.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only regenerate figures from existing outputs/*.json without training.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Target directory for generated figures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the module commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t_total = time.perf_counter()

    steps: list[tuple[str, list[str]]] = []
    if not args.figures_only:
        if not args.skip_audit:
            audit_args = ["--strict-env"] if args.strict_env else []
            steps.append(("src.phases.inspect_data", audit_args))
        steps.extend(
            [
                ("src.phases.train_baselines", ["--config", "configs/baselines.yaml"]),
                ("src.phases.train_cnn", ["--config", "configs/cnn_baseline.yaml"]),
                ("src.phases.tune_cnn", ["--config", "configs/cnn_tuning.yaml"]),
                ("src.phases.eval_test", ["--config", "configs/final_eval.yaml"]),
            ]
        )
    steps.append(("src.phases.make_figures", ["--figures-dir", str(args.figures_dir)]))

    for module, mod_args in steps:
        _run_module(module, mod_args, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"Pipeline finished in {time.perf_counter() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
