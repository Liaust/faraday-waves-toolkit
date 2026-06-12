#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "publication_figures"


def parse_args() -> argparse.Namespace:
    # small orchestration cli
    # this does not compute new physics quantities
    # it only dispatches the individual figure scripts against curated paper_data
    parser = argparse.ArgumentParser(
        description="Regenerate all migrated publication figures from curated paper_data CSVs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output folder for all publication figures.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_command(command: list[str]) -> None:
    # use subprocesses so each figure script keeps its own argparse defaults,
    # manifest writing, and console output
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    # child scripts handle their own manifests
    # this wrapper just prints commands, which works as a simple run log
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "plot_frequency_half_integer_bands.py"),
            "--output-dir",
            str(output_dir / "frequency"),
            "--dpi",
            str(args.dpi),
        ]
    )
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "plot_onset_publication_figures.py"),
            "--output-dir",
            str(output_dir / "onset"),
            "--dpi",
            str(args.dpi),
        ]
    )


if __name__ == "__main__":
    main()
