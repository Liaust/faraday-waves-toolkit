#!/usr/bin/env python3
from __future__ import annotations

# this is the batch runner for full fsss
# it can run calibration, fsss tracking, reconstruction, wavenumber analysis
# and then writes the batch summaries

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# repo root and imports so this can run directly from the command line
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
DOT_TRACKING_DIR = SCRIPTS_DIR / "dot_tracking"
if str(DOT_TRACKING_DIR) not in sys.path:
    sys.path.insert(0, str(DOT_TRACKING_DIR))

from common import load_yaml, project_path, relpath  # noqa: E402
from pipeline_utils import (  # noqa: E402
    StepResult,
    calibration_metadata_path,
    normalized_run_metadata,
    run_command,
    safe_name,
    timestamp,
    write_json,
    write_yaml,
)


TRACK_FSSS_SCRIPT = PROJECT_ROOT / "scripts" / "dot_tracking" / "track_fsss_dots.py"
RECONSTRUCT_SCRIPT = PROJECT_ROOT / "scripts" / "full_fsss" / "reconstruct_surface.py"
DOT_GRID_POSE_SCRIPT = PROJECT_ROOT / "scripts" / "full_fsss" / "calibrate_dot_grid_pose.py"
FLAT_REFERENCE_SCRIPT = PROJECT_ROOT / "scripts" / "full_fsss" / "calibrate_flat_liquid_reference.py"
RAYTRACE_CALIBRATION_SCRIPT = PROJECT_ROOT / "scripts" / "full_fsss" / "calibrate_fsss_raytrace.py"
MEASURE_WAVENUMBER_SCRIPT = PROJECT_ROOT / "scripts" / "full_fsss" / "measure_wavenumber.py"


# cli for the public full fsss workflow
# optional calibration first, then tracking, reconstruction and wavenumbers
def parse_args() -> argparse.Namespace:
    # the batch cli exposes high-level workflow choices, not internal
    # calibration/tracking tuning.
    parser = argparse.ArgumentParser(description="Run the public full-FSSS reconstruction workflow.")
    parser.add_argument("--metadata", default="inputs/batch_metadata.yaml")
    parser.add_argument("--only", action="append", default=[], help="Substring filter for run ids.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-calibration",
        action="store_true",
        help="Run dry-grid pose, flat-liquid reference, and ray-trace calibration before processing runs.",
    )
    parser.add_argument("--calibration-output-dir", default=None)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--adaptive-tracking", action="store_true")
    return parser.parse_args()


# full-fsss runs must opt in explicitly with pipelines: [full-fsss]
# reconstruction is heavier than frequency/onset
def run_enabled_for_full_fsss(run: dict[str, Any]) -> bool:
    # full fsss is opt-in because it is computationally heavier and requires
    # calibration files from the earlier fsss setup steps
    pipelines = run.get("pipelines")
    if pipelines is None:
        return False
    return "full-fsss" in {str(item).lower() for item in pipelines}


# simple substring filter for rerunning a few runs
def run_matches(run: dict[str, Any], filters: list[str]) -> bool:
    # multiple --only filters are ORed together.
    if not filters:
        return True
    haystack = " ".join(
        str(run.get(key, ""))
        for key in ("run_id", "kind", "video_path", "nominal_drive_hz", "measured_drive_hz")
    ).lower()
    return any(item.lower() in haystack for item in filters)


def run_path_parts(run_metadata: dict[str, Any]) -> tuple[str, str, str]:
    # folder structure mirrors frequency outputs: group / drive / video id.
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


# get calibration id from batch metadata or calibration metadata
def full_fsss_calibration_id(batch: dict[str, Any]) -> str:
    # prefer calibration id from batch metadata, otherwise read the calibration
    # metadata YAML.
    calibration = batch.get("calibration", {})
    calibration_id = calibration.get("calibration_id")
    if calibration_id:
        return str(calibration_id)
    try:
        return str(load_yaml(calibration_metadata_path(batch)).get("calibration_id", "default"))
    except Exception:
        return "default"


def fsss_run_output_dir(run_metadata: dict[str, Any]) -> Path:
    # per-run full-fsss output root.
    group, freq_label, video_id = run_path_parts(run_metadata)
    return PROJECT_ROOT / "outputs" / "full_fsss" / "runs" / group / freq_label / video_id


# resolve calibration output folder from cli, batch metadata, or the default path
def full_fsss_calibration_output_dir(batch: dict[str, Any], args: argparse.Namespace) -> Path:
    configured = project_path(args.calibration_output_dir)
    if configured is not None:
        return configured
    calibration = batch.get("calibration", {})
    configured = project_path(
        calibration.get("full_fsss_output_dir")
        or calibration.get("fsss_output_dir")
        or calibration.get("calibration_output_dir")
    )
    if configured is not None:
        return configured
    calibration_id = full_fsss_calibration_id(batch)
    return PROJECT_ROOT / "outputs" / "full_fsss" / "calibration" / str(calibration_id)


def tracking_npz(run_dir: Path) -> Path:
    # main output from track_fsss_dots.py.
    return run_dir / "tracking" / "tracked_dots_fsss.npz"


def reconstruction_npz(run_dir: Path) -> Path:
    # main output from reconstruct_surface.py.
    return run_dir / "reconstruction" / "surface_height_m.npz"


def raytrace_ready(calibration_output_dir: Path) -> bool:
    # minimum artifacts required before run tracking/reconstruction can proceed.
    required = [
        calibration_output_dir / "flat_liquid_dots.csv",
        calibration_output_dir / "flat_liquid_reference_undistorted.png",
        calibration_output_dir / "raytrace_fsss_calibration.json",
    ]
    return all(path.exists() for path in required)


# the reference calibration is ready only if all key calibration files exist
def reference_calibration_ready(calibration_output_dir: Path) -> bool:
    required = [
        calibration_output_dir / "camera_pose_grid.json",
        calibration_output_dir / "dot_grid_reference.csv",
        calibration_output_dir / "flat_liquid_dots.csv",
        calibration_output_dir / "flat_liquid_reference_undistorted.png",
        calibration_output_dir / "optical_geometry_refined.json",
    ]
    return all(path.exists() for path in required)


def append_step(record: dict[str, Any], step: StepResult) -> None:
    # store subprocess status/log info in a json-friendly dict.
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


# raytrace stage
# fit displacement -> gradient calibration only if --run-calibration is set
def run_raytrace_calibration(
    *,
    batch: dict[str, Any],
    args: argparse.Namespace,
    batch_dir: Path,
    calibration_output_dir: Path,
) -> StepResult:
    if not args.run_calibration:
        # public batch run does not regenerate calibration unless explicitly
        # requested.
        return StepResult(name="calibrate_fsss_raytrace", status="skipped", message="not requested")
    if args.resume and raytrace_ready(calibration_output_dir):
        # resume avoids repeating expensive ray tracing.
        return StepResult(name="calibrate_fsss_raytrace", status="skipped", message="FSSS calibration exists.")
    cmd = [
        sys.executable,
        str(RAYTRACE_CALIBRATION_SCRIPT),
        "--metadata",
        str(calibration_metadata_path(batch)),
        "--output-dir",
        str(calibration_output_dir),
    ]
    return run_command(
        "calibrate_fsss_raytrace",
        cmd,
        batch_dir / "logs" / "calibrate_fsss_raytrace.log",
        dry_run=args.dry_run,
    )


# reference calibration stage
# dry-grid pose first, then flat-liquid reference / optical geometry
def run_reference_calibration(
    *,
    batch: dict[str, Any],
    args: argparse.Namespace,
    batch_dir: Path,
    calibration_output_dir: Path,
) -> list[StepResult]:
    if not args.run_calibration:
        return [StepResult(name="calibrate_fsss_reference", status="skipped", message="not requested")]
    if args.resume and reference_calibration_ready(calibration_output_dir):
        return [StepResult(name="calibrate_fsss_reference", status="skipped", message="FSSS reference calibration exists.")]

    metadata_path = calibration_metadata_path(batch)
    # dry-grid pose must run before flat-liquid reference because the latter
    # needs dot_grid_reference.csv and camera_pose_grid.json.
    dot_grid_step = run_command(
        "calibrate_dot_grid_pose",
        [
            sys.executable,
            str(DOT_GRID_POSE_SCRIPT),
            "--metadata",
            str(metadata_path),
            "--output-dir",
            str(calibration_output_dir),
        ],
        batch_dir / "logs" / "calibrate_dot_grid_pose.log",
        dry_run=args.dry_run,
    )
    if dot_grid_step.status not in {"ok", "dry-run"}:
        return [dot_grid_step]

    flat_reference_step = run_command(
        "calibrate_flat_liquid_reference",
        [
            sys.executable,
            str(FLAT_REFERENCE_SCRIPT),
            "--metadata",
            str(metadata_path),
            "--output-dir",
            str(calibration_output_dir),
        ],
        batch_dir / "logs" / "calibrate_flat_liquid_reference.log",
        dry_run=args.dry_run,
    )
    return [dot_grid_step, flat_reference_step]


# process one full-fsss run
# write run metadata, track fsss dots, reconstruct height maps
def run_one(
    *,
    batch: dict[str, Any],
    run: dict[str, Any],
    args: argparse.Namespace,
    batch_dir: Path,
    calibration_output_dir: Path,
) -> dict[str, Any]:
    # build one normalized run metadata file with explicit calibration/output
    # paths, then pass that file to every per-run subprocess.
    run_meta = normalized_run_metadata(batch, run)
    run_meta["kind"] = run.get("kind", "stable")
    run_meta["calibration_id"] = full_fsss_calibration_id(batch)
    run_meta["calibration_metadata_path"] = relpath(calibration_metadata_path(batch))
    run_meta["calibration_output_dir"] = relpath(calibration_output_dir)
    run_dir = fsss_run_output_dir(run_meta)
    run_meta["run_output_dir"] = relpath(run_dir)

    run_id = str(run_meta["run_id"])
    run_batch_dir = batch_dir / "runs" / safe_name(run_id)
    run_metadata_path = run_batch_dir / "run_metadata.yaml"
    write_yaml(run_metadata_path, run_meta)

    # record the expected public outputs before running steps so skipped/dry-run
    # records remain useful.
    record: dict[str, Any] = {
        "run_id": run_id,
        "metadata_path": relpath(run_metadata_path),
        "video_path": relpath(project_path(run_meta.get("video_path"))),
        "output_dir": relpath(run_dir),
        "tracking_npz": relpath(tracking_npz(run_dir)),
        "surface_npz": relpath(reconstruction_npz(run_dir)),
        "status": "pending",
        "steps": [],
    }

    video_path = project_path(run_meta.get("video_path"))
    if video_path is None or not video_path.exists():
        # missing videos skip this run but do not abort the batch.
        record["status"] = "skipped"
        record["message"] = "Missing video file."
        return record
    if not args.dry_run and not raytrace_ready(calibration_output_dir):
        # tracking requires flat_liquid_dots.csv and raytrace calibration.
        record["status"] = "skipped"
        record["message"] = f"Missing full-FSSS calibration outputs in {relpath(calibration_output_dir)}."
        return record

    # tracking step consumes flat_liquid_dots.csv and raytrace calibration to
    # save metric object-plane displacements for each dot/frame.
    if args.resume and tracking_npz(run_dir).exists():
        # resume mode trusts existing tracking npz.
        append_step(record, StepResult(name="track_fsss_dots", status="skipped", message="Tracking output exists."))
    else:
        # build the tracking subprocess command.
        cmd = [
            sys.executable,
            str(TRACK_FSSS_SCRIPT),
            str(run_metadata_path),
            "--calibration-metadata",
            str(calibration_metadata_path(batch)),
            "--calibration-output-dir",
            str(calibration_output_dir),
            "--output-dir",
            str(run_dir),
            "--frame-start",
            str(args.frame_start),
            "--frame-step",
            str(args.frame_step),
        ]
        if args.frame_stop is not None:
            cmd.extend(["--frame-stop", str(args.frame_stop)])
        if args.max_frames is not None:
            cmd.extend(["--max-frames", str(args.max_frames)])
        if args.adaptive_tracking:
            cmd.append("--adaptive-tracking")
        step = run_command(
            "track_fsss_dots",
            cmd,
            run_batch_dir / "logs" / "track_fsss_dots.log",
            dry_run=args.dry_run,
        )
        append_step(record, step)
        if step.status not in {"ok", "dry-run"}:
            record["status"] = "failed"
            return record

    # reconstruction step converts tracked object-plane displacement into
    # gradients and then signed height maps.
    if args.resume and reconstruction_npz(run_dir).exists():
        # resume mode trusts existing reconstructed surface npz.
        append_step(record, StepResult(name="reconstruct_surface", status="skipped", message="Surface output exists."))
    else:
        # reconstruction consumes the tracking npz referenced by run metadata.
        cmd = [sys.executable, str(RECONSTRUCT_SCRIPT), str(run_metadata_path)]
        step = run_command(
            "reconstruct_surface",
            cmd,
            run_batch_dir / "logs" / "reconstruct_surface.log",
            dry_run=args.dry_run,
        )
        append_step(record, step)
        if step.status not in {"ok", "dry-run"}:
            record["status"] = "failed"
            return record

    record["status"] = "dry-run" if args.dry_run else "ok"
    return record


# batch csv points to per-run fsss artifacts
# detailed quality stays in each run output folder
def write_batch_csv(path: Path, records: list[dict[str, Any]]) -> None:
    # small table for humans and figure scripts
    # detailed quality numbers stay in each run output folder
    rows = []
    for record in records:
        rows.append(
            {
                "run_id": record.get("run_id"),
                "status": record.get("status"),
                "message": record.get("message", ""),
                "metadata_path": record.get("metadata_path"),
                "video_path": record.get("video_path"),
                "output_dir": record.get("output_dir"),
                "tracking_npz": record.get("tracking_npz"),
                "surface_npz": record.get("surface_npz"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# batch-level wavenumber measurement
# collect completed surface npzs and run the 2d fourier script once
def run_wavenumber(args: argparse.Namespace, batch_dir: Path, records: list[dict[str, Any]]) -> StepResult:
    if args.dry_run or not any(record.get("status") == "ok" for record in records):
        # no surfaces to analyze in dry-run or failed-only batches.
        return StepResult(name="measure_wavenumber", status="skipped", message="no completed reconstructions")
    surface_paths = []
    for record in records:
        if record.get("status") != "ok":
            continue
        surface_path = project_path(record.get("surface_npz"))
        if surface_path is not None and surface_path.exists():
            surface_paths.append(surface_path)
    if not surface_paths:
        return StepResult(name="measure_wavenumber", status="skipped", message="no surface_height_m.npz outputs")
    cmd = [
        # one batch-level wavenumber command processes all completed surfaces.
        sys.executable,
        str(MEASURE_WAVENUMBER_SCRIPT),
        "--output-dir",
        str(batch_dir / "wavenumber"),
    ]
    for path in surface_paths:
        cmd.extend(["--surface-npz", str(path)])
    return run_command(
        "measure_wavenumber",
        cmd,
        batch_dir / "logs" / "measure_wavenumber.log",
        dry_run=False,
    )


# full batch order
# optional calibration -> per-run tracking/reconstruction -> wavenumbers -> summaries
def main() -> int:
    args = parse_args()
    # load batch metadata and resolve calibration output directory.
    metadata_path = project_path(args.metadata)
    if metadata_path is None or not metadata_path.exists():
        raise SystemExit(f"Missing batch metadata: {args.metadata}")
    batch = load_yaml(metadata_path)
    calibration_output_dir = full_fsss_calibration_output_dir(batch, args)
    batch_dir = PROJECT_ROOT / "outputs" / "full_fsss" / "batch" / f"full_fsss_{timestamp()}"
    # timestamped batch folder preserves logs/summaries for each execution.
    batch_dir.mkdir(parents=True, exist_ok=True)

    reference_calibration_steps = run_reference_calibration(
        batch=batch,
        args=args,
        batch_dir=batch_dir,
        calibration_output_dir=calibration_output_dir,
    )
    if any(step.status not in {"ok", "skipped", "dry-run"} for step in reference_calibration_steps):
        write_json(
            batch_dir / "batch_summary.json",
            {"reference_calibration_steps": [step.__dict__ for step in reference_calibration_steps], "runs": []},
        )
        return 1

    calibration_step = run_raytrace_calibration(
        batch=batch,
        args=args,
        batch_dir=batch_dir,
        calibration_output_dir=calibration_output_dir,
    )
    if calibration_step.status not in {"ok", "skipped", "dry-run"}:
        write_json(batch_dir / "batch_summary.json", {"calibration_step": calibration_step.__dict__, "runs": []})
        return 1

    runs = [
        # only explicitly enabled full-fsss runs are processed.
        run
        for run in batch.get("runs", [])
        if run_enabled_for_full_fsss(run) and run_matches(run, args.only)
    ]
    records = [
        run_one(
            batch=batch,
            run=run,
            args=args,
            batch_dir=batch_dir,
            calibration_output_dir=calibration_output_dir,
        )
        for run in runs
    ]
    wavenumber_step = run_wavenumber(args, batch_dir, records)
    csv_path = batch_dir / "batch_summary.csv"
    json_path = batch_dir / "batch_summary.json"
    write_batch_csv(csv_path, records)
    write_json(
        json_path,
        {
            # json summary keeps full step information; csv is compact.
            "schema": "full_fsss_batch_v1",
            "metadata_path": relpath(metadata_path),
            "calibration_output_dir": relpath(calibration_output_dir),
            "reference_calibration_steps": [step.__dict__ for step in reference_calibration_steps],
            "calibration_step": calibration_step.__dict__,
            "wavenumber_step": wavenumber_step.__dict__,
            "runs": records,
        },
    )
    ok = sum(1 for record in records if record.get("status") in {"ok", "skipped", "dry-run"})
    failed = sum(1 for record in records if record.get("status") == "failed")
    print(f"Full-FSSS pipeline finished: ok/skipped={ok}, failed={failed}")
    print(f"csv summary: {relpath(csv_path)}")
    print(f"json summary: {relpath(json_path)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
