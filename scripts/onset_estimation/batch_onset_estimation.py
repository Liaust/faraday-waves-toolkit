#!/usr/bin/env python3
from __future__ import annotations

# this is the batch runner for onset estimation
# it does not do the onset math itself
# for every run-up video it builds/reuses reference, tracks dots, then calls
# compute_runup_onset_metrics.py to make the manual review outputs

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    # let this script import pipeline_utils when launched directly
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_utils import (  # noqa: E402
    StepResult,
    build_reference,
    calibration_metadata_path,
    frequency_tracking_run_output_dir,
    normalized_run_metadata,
    project_path,
    reference_output_dir,
    reference_ready,
    relpath,
    run_command,
    safe_name,
    timestamp,
    tracking_npz,
    write_json,
    write_yaml,
)
from common import load_yaml  # noqa: E402


TRACK_SCRIPT = PROJECT_ROOT / "scripts" / "dot_tracking" / "track_frequency_dots.py"
COMPUTE_METRICS_SCRIPT = PROJECT_ROOT / "scripts" / "onset_estimation" / "compute_runup_onset_metrics.py"

# fixed match threshold for run-up tracking
# these videos can be noisy early on, so this is a bit permissive
ONSET_TRACKING_MIN_MATCH_SCORE = 0.35


# cli for the onset batch
# intentionally not too many tracking knobs here, this is more of a review workflow
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the public manual-onset review metric workflow for run-up videos listed in "
            "inputs/batch_metadata.yaml."
        )
    )
    # batch metadata lists videos, accel csvs, groups, fluids and calibration metadata
    parser.add_argument("--metadata", default="inputs/batch_metadata.yaml")
    parser.add_argument("--jobs", type=int, default=1, help="Reserved for future parallel execution.")
    parser.add_argument("--only", action="append", default=[], help="Substring filter for run ids.")
    # resume skips tracking if tracking npz already exists
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    # reference is shared with frequency tracking, skip only if it already exists
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--baseline-video-seconds", type=float, default=2.0)
    parser.add_argument("--subharmonic-half-width-hz", type=float, default=1.5)
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--lockin-window-s", type=float, default=1.0)
    parser.add_argument("--accel-lockin-window-s", type=float, default=1.0)
    parser.add_argument("--video-frequency-source", choices=("accelerometer", "nominal"), default="accelerometer")
    args = parser.parse_args()

    # compatibility fields for shared reference-building helpers
    # roi/reference choices should come from calibration_metadata.yaml
    args.reference_dir = None
    args.roi = None
    args.rotated_roi = None
    return args


# decide if a metadata row belongs to onset
def run_enabled_for_onset(run: dict[str, Any]) -> bool:
    pipelines = run.get("pipelines")
    if pipelines is None:
        # default convention: runup means onset
        return str(run.get("kind", "")).lower() == "runup"
    # explicit pipeline list overrides kind
    return "onset" in {str(item).lower() for item in pipelines}


# simple substring filter so we can rerun only a few runs
def run_matches(run: dict[str, Any], filters: list[str]) -> bool:
    if not filters:
        return True
    haystack = " ".join(
        str(run.get(key, ""))
        for key in ("run_id", "kind", "video_path", "nominal_drive_hz", "measured_drive_hz")
    ).lower()
    return any(item.lower() in haystack for item in filters)


# output folder for one onset run
def onset_analysis_output_dir(run_metadata: dict[str, Any]) -> Path:
    group, freq_label, video_id = run_path_parts(run_metadata)
    return PROJECT_ROOT / "outputs" / "onset_estimation" / "runs" / group / freq_label / video_id


def run_path_parts(run_metadata: dict[str, Any]) -> tuple[str, str, str]:
    # folders are based on group, frequency and video/run id
    # cleaner than using raw video paths
    group = str(run_metadata.get("experiment_group", "runs"))
    experiment = run_metadata.get("experiment", {})
    drive_meta = run_metadata.get("drive", {})
    drive = (
        experiment.get("drive_frequency_hz")
        or drive_meta.get("measured_frequency_hz")
        or drive_meta.get("nominal_frequency_hz")
        or "unknown"
    )
    try:
        freq_label = f"{float(drive):g}Hz"
    except (TypeError, ValueError):
        freq_label = str(drive)
    video_id = str(run_metadata.get("expected_iphone_video_id") or run_metadata.get("run_id"))
    return group, freq_label, video_id


def read_json(path: Path) -> dict[str, Any]:
    # defensive read, useful when a previous run failed or summary doesn't exist yet
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# accept a few possible field names for the calibration accel csv
# old metadata versions used different names
def accelerometer_calibration_csv(batch: dict[str, Any], run: dict[str, Any]) -> Path | None:
    for source in (
        run.get("accelerometer_calibration_csv"),
        run.get("calibration_accelerometer_csv"),
        (run.get("accelerometer") or {}).get("calibration_csv_path") if isinstance(run.get("accelerometer"), dict) else None,
        (batch.get("calibration") or {}).get("accelerometer_csv"),
        (batch.get("calibration") or {}).get("accelerometer_calibration_csv"),
    ):
        path = project_path(source)
        if path is not None:
            return path
    return None


# find the run-up accelerometer csv for this run
def runup_accelerometer_csv(run: dict[str, Any]) -> Path | None:
    for source in (
        run.get("accelerometer_csv"),
        run.get("runup_accelerometer_csv"),
        (run.get("accelerometer") or {}).get("run_csv_path") if isinstance(run.get("accelerometer"), dict) else None,
    ):
        path = project_path(source)
        if path is not None:
            return path
    return None


def drive_frequency(run_metadata: dict[str, Any]) -> float | None:
    # summary value here is nominal, metrics script measures the real peak later
    value = (run_metadata.get("experiment") or {}).get("drive_frequency_hz")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def expected_subharmonic(run_metadata: dict[str, Any]) -> float | None:
    # if expected f/2 is not explicitly stored, compute it from nominal fd
    value = (run_metadata.get("experiment") or {}).get("expected_subharmonic_hz")
    try:
        return float(value)
    except (TypeError, ValueError):
        drive_hz = drive_frequency(run_metadata)
    return 0.5 * drive_hz if drive_hz is not None else None


# append one child-process result into the run record
def append_step(record: dict[str, Any], step: StepResult) -> None:
    record.setdefault("steps", []).append(
        {
            "name": step.name,
            "status": step.status,
            "returncode": step.returncode,
            "duration_s": step.duration_s,
            "log_path": step.log_path,
            "message": step.message,
        }
    )


# process one run-up video
# write run metadata, track dots, then compute onset metrics
def run_one(
    *,
    batch: dict[str, Any],
    run: dict[str, Any],
    args: argparse.Namespace,
    batch_dir: Path,
    reference_dir: Path,
) -> dict[str, Any]:
    run_meta = normalized_run_metadata(batch, run)
    # force kind to runup because this runner only handles onset jobs
    run_meta["kind"] = "runup"
    run_id = str(run_meta["run_id"])
    run_batch_dir = batch_dir / "runs" / safe_name(run_id)
    run_metadata_path = run_batch_dir / "run_metadata.yaml"
    # each run gets its own metadata snapshot for child scripts
    write_yaml(run_metadata_path, run_meta)

    tracking_dir = frequency_tracking_run_output_dir(run_meta)
    onset_dir = onset_analysis_output_dir(run_meta)
    summary_path = onset_dir / "onset_review_summary.json"
    video_metrics_path = onset_dir / "video_onset_metrics.csv"
    accel_metrics_path = onset_dir / "accelerometer_gamma_metrics.csv"

    # record expected output paths even if the run is skipped or dry-run
    record: dict[str, Any] = {
        "run_id": run_id,
        "metadata_path": relpath(run_metadata_path),
        "video_path": relpath(project_path(run_meta.get("video_path"))),
        "drive_frequency_hz": drive_frequency(run_meta),
        "expected_subharmonic_hz": expected_subharmonic(run_meta),
        "tracking_npz": relpath(tracking_npz(tracking_dir)),
        "onset_review_summary_json": relpath(summary_path),
        "video_metrics_csv": relpath(video_metrics_path),
        "accelerometer_metrics_csv": relpath(accel_metrics_path),
        "status": "pending",
        "steps": [],
    }

    video_path = project_path(run_meta.get("video_path"))
    run_accel = runup_accelerometer_csv(run)
    calibration_accel = accelerometer_calibration_csv(batch, run)
    # check inputs before launching child scripts
    if video_path is None or not video_path.exists():
        record["status"] = "skipped"
        record["message"] = "Missing run-up video file."
        return record
    if run_accel is None or not run_accel.exists():
        record["status"] = "skipped"
        record["message"] = "Missing run-up accelerometer CSV."
        return record
    if calibration_accel is None or not calibration_accel.exists():
        record["status"] = "skipped"
        record["message"] = "Missing accelerometer calibration CSV."
        return record

    # tracking step
    # run-up videos use the same frequency dot tracker as stable videos
    if args.resume and tracking_npz(tracking_dir).exists():
        append_step(record, StepResult(name="track_frequency_dots", status="skipped", message="Tracking output exists."))
    else:
        cmd = [
            # child process 1, create tracked_dots_frequency.npz for this run-up
            sys.executable,
            str(TRACK_SCRIPT),
            str(run_metadata_path),
            "--calibration-metadata",
            str(calibration_metadata_path(batch)),
            "--reference-dir",
            str(reference_dir),
            "--output-dir",
            str(tracking_dir),
            "--frame-step",
            str(args.frame_step),
            "--min-match-score",
            str(ONSET_TRACKING_MIN_MATCH_SCORE),
        ]
        if args.max_frames is not None:
            cmd.extend(["--max-frames", str(args.max_frames)])
        step = run_command(
            "track_frequency_dots",
            cmd,
            run_batch_dir / "logs" / "track_frequency_dots.log",
            dry_run=args.dry_run,
        )
        append_step(record, step)
        if step.status not in {"ok", "dry-run"}:
            record["status"] = "failed"
            return record

    # metric step
    # combine tracked video displacement with accelerometer csvs
    cmd = [
        # child process 2, build the manual review metrics
        sys.executable,
        str(COMPUTE_METRICS_SCRIPT),
        "--run-metadata",
        str(run_metadata_path),
        "--tracking-npz",
        str(tracking_npz(tracking_dir)),
        "--runup-accel-csv",
        str(run_accel),
        "--calibration-accel-csv",
        str(calibration_accel),
        "--output-dir",
        str(onset_dir),
        "--video-frequency-source",
        str(args.video_frequency_source),
        "--baseline-video-seconds",
        str(args.baseline_video_seconds),
        "--subharmonic-half-width-hz",
        str(args.subharmonic_half_width_hz),
        "--frequency-step-hz",
        str(args.frequency_step_hz),
        "--lockin-window-s",
        str(args.lockin_window_s),
        "--accel-lockin-window-s",
        str(args.accel_lockin_window_s),
    ]
    step = run_command(
        "compute_runup_onset_metrics",
        cmd,
        run_batch_dir / "logs" / "compute_runup_onset_metrics.log",
        dry_run=args.dry_run,
    )
    append_step(record, step)
    if step.status not in {"ok", "dry-run"}:
        record["status"] = "failed"
        return record

    if args.dry_run:
        record["status"] = "dry-run"
        return record

    record["status"] = "ok"
    return record


# small batch csv, mainly points to per-run files loaded by the manual selector
def write_batch_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        # one row per run, this is a navigation table not the full summary
        rows.append(
            {
                "run_id": record.get("run_id"),
                "status": record.get("status"),
                "drive_frequency_hz": record.get("drive_frequency_hz"),
                "expected_subharmonic_hz": record.get("expected_subharmonic_hz"),
                "tracking_npz": record.get("tracking_npz"),
                "onset_review_summary_json": record.get("onset_review_summary_json"),
                "video_metrics_csv": record.get("video_metrics_csv"),
                "accelerometer_metrics_csv": record.get("accelerometer_metrics_csv"),
                "message": record.get("message", ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# full batch orchestration
# build/reuse reference, select onset runs, process each, write summaries
def main() -> int:
    args = parse_args()
    metadata_path = project_path(args.metadata)
    if metadata_path is None or not metadata_path.exists():
        raise SystemExit(f"Missing batch metadata: {args.metadata}")

    batch = load_yaml(metadata_path)
    calibration_meta = load_yaml(calibration_metadata_path(batch))
    # same reference logic as frequency pipeline
    reference_dir = reference_output_dir(batch, args, calibration_meta)
    batch_dir = PROJECT_ROOT / "outputs" / "onset_estimation" / "batch" / f"onset_{timestamp()}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    reference_step = build_reference(batch, args, batch_dir, reference_dir)
    if reference_step.status not in {"ok", "skipped", "dry-run"}:
        write_json(batch_dir / "batch_summary.json", {"reference_step": reference_step.__dict__, "runs": []})
        return 1
    if not args.dry_run and not args.skip_reference and not reference_ready(reference_dir):
        raise SystemExit(f"Reference outputs are missing: {reference_dir}")

    runs = [
        # only onset-enabled runs, then optional --only filters
        run
        for run in batch.get("runs", [])
        if run_enabled_for_onset(run) and run_matches(run, args.only)
    ]

    records: list[dict[str, Any]] = []
    for run in runs:
        records.append(
            run_one(
                batch=batch,
                run=run,
                args=args,
                batch_dir=batch_dir,
                reference_dir=reference_dir,
            )
        )

    summary = {
        "schema": "onset_batch_v1",
        "metadata_path": relpath(metadata_path),
        "reference_dir": relpath(reference_dir),
        "reference_step": reference_step.__dict__,
        "runs": records,
    }
    json_path = batch_dir / "batch_summary.json"
    csv_path = batch_dir / "batch_summary.csv"
    write_json(json_path, summary)
    write_batch_csv(csv_path, records)
    ok = sum(1 for record in records if record.get("status") in {"ok", "skipped", "dry-run"})
    failed = sum(1 for record in records if record.get("status") == "failed")
    print(f"Onset pipeline finished: ok/skipped={ok}, failed={failed}")
    print(f"CSV summary: {relpath(csv_path)}")
    print(f"JSON summary: {relpath(json_path)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
