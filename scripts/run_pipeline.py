#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# map the short pipeline name to the real batch script
# this file stays thin; each pipeline owns its own arguments and implementation
PIPELINE_SCRIPTS = {
    "frequency": PROJECT_ROOT / "scripts" / "frequency_analysis" / "batch_frequency_analysis.py",
    "onset": PROJECT_ROOT / "scripts" / "onset_estimation" / "batch_onset_estimation.py",
    "full-fsss": PROJECT_ROOT / "scripts" / "full_fsss" / "batch_full_fsss.py",
}


# keep this help short
# real options live in each pipeline's --help
def print_help() -> None:
    names = ", ".join(sorted(PIPELINE_SCRIPTS))
    print("Usage: python scripts/run_pipeline.py <pipeline> [pipeline options]")
    print()
    print(f"Pipelines: {names}")
    print()
    print("Examples:")
    print("  python scripts/run_pipeline.py frequency --metadata inputs/batch_metadata.yaml")
    print("  python scripts/run_pipeline.py onset --metadata inputs/batch_metadata.yaml")
    print("  python scripts/run_pipeline.py full-fsss --metadata inputs/batch_metadata.yaml")
    print()
    print("Use `python scripts/run_pipeline.py <pipeline> --help` for pipeline-specific options.")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print_help()
        return 0

    pipeline = args[0]
    script = PIPELINE_SCRIPTS.get(pipeline)
    if script is None:
        print(f"error: unknown pipeline {pipeline!r}", file=sys.stderr)
        print(f"valid pipelines: {', '.join(sorted(PIPELINE_SCRIPTS))}", file=sys.stderr)
        return 2

    # dispatch to the selected script as a subprocess
    # this preserves that script's argparse behavior and runs from the repo root
    cmd = [sys.executable, str(script), *args[1:]]
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
