#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOT_TRACKING_DIR = PROJECT_ROOT / "scripts" / "dot_tracking"
if str(DOT_TRACKING_DIR) not in sys.path:
    sys.path.insert(0, str(DOT_TRACKING_DIR))

from common import (  # noqa: E402
    frequency_tracking_output_dir,
    frequency_tracking_run_output_dir,
    project_path,
    relpath,
)


BUILD_REFERENCE_SCRIPT = PROJECT_ROOT / "scripts" / "dot_tracking" / "build_frequency_reference.py"


# one small result object used by the batch runners
# kept json-friendly because the batch summary writes these directly
@dataclass(frozen=True)
class StepResult:
    name: str
    status: str
    returncode: int | None = None
    duration_s: float = 0.0
    log_path: str | None = None
    message: str = ""


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


# small file writers used when batch scripts create run metadata and summaries
def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# subprocess wrapper for pipeline stages
# runs from the repo root, writes stdout/stderr to a log,
# and returns StepResult instead of throwing immediately
def run_command(name: str, cmd: list[str], log_path: Path, *, dry_run: bool) -> StepResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return StepResult(name=name, status="dry-run", duration_s=0.0, log_path=relpath(log_path))

    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        returncode = process.wait()

    return StepResult(
        name=name,
        status="ok" if returncode == 0 else "failed",
        returncode=returncode,
        duration_s=time.monotonic() - started,
        log_path=relpath(log_path),
    )


# shared calibration metadata path
# reference building, frequency tracking, onset tracking, and fsss calibration use this
def calibration_metadata_path(batch: dict[str, Any]) -> Path:
    calibration = batch.get("calibration", {})
    raw = calibration.get("metadata_path", "inputs/calibration_metadata.yaml")
    path = project_path(raw)
    if path is None:
        raise ValueError("Missing calibration.metadata_path")
    return path


# frequency-style reference output folder
# one flat reference is built per calibration and reused by frequency/onset batches
def reference_output_dir(batch: dict[str, Any], args: Any, calibration_meta: dict[str, Any]) -> Path:
    configured = project_path(args.reference_dir)
    if configured is not None:
        return configured
    calibration = batch.get("calibration", {})
    calibration_id = calibration.get("calibration_id") or calibration_meta.get("calibration_id", "default")
    return frequency_tracking_output_dir() / "reference" / str(calibration_id)


# reference is ready only when these three files exist
# both frequency tracking and onset run-up tracking need them
def reference_ready(reference_dir: Path) -> bool:
    required = [
        reference_dir / "flat_reference_dots.csv",
        reference_dir / "flat_reference_lattice.json",
        reference_dir / "flat_reference_frame_undistorted.png",
    ]
    return all(path.exists() for path in required)


# build or reuse the flat frequency-tracking reference
# this keeps --skip-reference, --resume, roi overrides, and reference-video
# overrides consistent between frequency and onset batches
def build_reference(batch: dict[str, Any], args: Any, batch_dir: Path, reference_dir: Path) -> StepResult:
    if args.skip_reference:
        return StepResult(name="build_frequency_reference", status="skipped", message="--skip-reference")
    if args.resume and reference_ready(reference_dir):
        return StepResult(name="build_frequency_reference", status="skipped", message="Reference outputs already exist.")

    calibration = batch.get("calibration", {})
    cmd = [
        sys.executable,
        str(BUILD_REFERENCE_SCRIPT),
        "--metadata",
        str(calibration_metadata_path(batch)),
        "--output-dir",
        str(reference_dir),
    ]
    if calibration.get("flat_reference_video"):
        cmd.extend(["--reference-video", str(project_path(calibration["flat_reference_video"]))])
    if args.roi is not None:
        cmd.extend(["--roi", *[str(value) for value in args.roi]])
    if args.rotated_roi is not None:
        cmd.extend(["--rotated-roi", *[str(value) for value in args.rotated_roi]])
    return run_command(
        "build_frequency_reference",
        cmd,
        batch_dir / "logs" / "build_frequency_reference.log",
        dry_run=args.dry_run,
    )


# normalize one raw batch run into the metadata schema used by child scripts
# batch-level fluid/geometry values get merged with run-level overrides
# expected subharmonic defaults to f_d/2
def normalized_run_metadata(batch: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    project = batch.get("project", {})
    fluid = dict(batch.get("fluid", {}))
    fluid.update(run.get("fluid", {}))
    geometry = dict(batch.get("geometry", {}))
    geometry.update(run.get("geometry", {}))

    drive_meta = dict(run.get("drive", {}))
    nominal = run.get("nominal_drive_hz", drive_meta.get("nominal_frequency_hz"))
    measured = run.get("measured_drive_hz", drive_meta.get("measured_frequency_hz"))
    drive_hz = measured if measured is not None else nominal
    expected_subharmonic = drive_meta.get("expected_subharmonic_hz")
    if expected_subharmonic is None and drive_hz is not None:
        expected_subharmonic = float(drive_hz) / 2.0

    output = {
        "run_id": str(run["run_id"]),
        "kind": run.get("kind", "stable"),
        "experiment_group": project.get("name", "runs"),
        "expected_iphone_video_id": str(run.get("video_id", run.get("run_id"))),
        "video_path": str(run.get("video_path", "")),
        "accelerometer_csv": run.get("accelerometer_csv"),
        "drive": {
            "nominal_frequency_hz": nominal,
            "measured_frequency_hz": measured,
            "expected_subharmonic_hz": expected_subharmonic,
        },
        "experiment": {
            "drive_frequency_hz": drive_hz,
            "expected_subharmonic_hz": expected_subharmonic,
            "glycerol_wt_percent": fluid.get("glycerol_wt_percent"),
            "liquid_depth_mm": fluid.get("bath_height_mm", geometry.get("bath_height_mm")),
            "density_kg_per_m3": fluid.get("density_kg_m3", fluid.get("density_kg_per_m3")),
            "surface_tension_N_per_m": fluid.get("surface_tension_N_m", fluid.get("surface_tension_N_per_m")),
            "amplitude_label": run.get("amplitude_label", ""),
        },
        "fluid": {
            "glycerol_wt_percent": fluid.get("glycerol_wt_percent"),
            "temperature_C": fluid.get("temperature_C"),
            "density_kg_m3": fluid.get("density_kg_m3", fluid.get("density_kg_per_m3")),
            "surface_tension_N_m": fluid.get("surface_tension_N_m", fluid.get("surface_tension_N_per_m")),
            "bath_height_mm": fluid.get("bath_height_mm", geometry.get("bath_height_mm")),
        },
        "notes": run.get("notes", ""),
    }
    if run.get("run_output_dir"):
        output["run_output_dir"] = run["run_output_dir"]
    return output


# frequency-style tracking npz path
# frequency analysis and onset metrics both read this after tracking runs
def tracking_npz(run_output_dir: Path) -> Path:
    return run_output_dir / "tracking" / "tracked_dots_frequency.npz"
