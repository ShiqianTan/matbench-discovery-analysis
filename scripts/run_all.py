#!/usr/bin/env python3
"""Run the numbered Matbench-Discovery analysis scripts in order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_STEPS = [f"{number:02d}" for number in range(1, 10)]


def script_for(step: str) -> Path:
    matches = sorted(SCRIPT_DIR.glob(f"{step}_*.py"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one script for step {step!r}; found {len(matches)}."
        )
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Matbench-Discovery figures and tables."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEP",
        choices=DEFAULT_STEPS,
        help="Run selected steps (for example: --only 01 04 09).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override the directory containing the five input CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the output directory for generated figures and tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = args.only or DEFAULT_STEPS
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    if args.data_dir:
        env["MATBENCH_DATA_DIR"] = str(args.data_dir.resolve())
    if args.output_dir:
        env["MATBENCH_OUTPUT_DIR"] = str(args.output_dir.resolve())

    for index, step in enumerate(steps, start=1):
        script = script_for(step)
        print(f"[{index}/{len(steps)}] {script.name}", flush=True)
        subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
        )

    output_dir = Path(env.get("MATBENCH_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
    print(f"Completed {len(steps)} step(s). Outputs: {output_dir}")


if __name__ == "__main__":
    main()
