#!/usr/bin/env python3
from __future__ import annotations

# this is the batch runner for frequency analysis
# for each stable video it builds/reuses reference, tracks dots, computes spectrum
# and then writes csv/json batch summaries

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


# repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# add repo/script folders so direct execution still imports local helpers
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
DOT_TRACKING_DIR = PROJECT_ROOT / "scripts" / "dot_tracking"
if str(DOT_TRACKING_DIR) not in sys.path:
    sys.path.insert(0, str(DOT_TRACKING_DIR))

from common import (  # noqa: E402
    frequency_run_path_parts,
    load_yaml,
)
from pipeline_utils import (  # noqa: E402
    StepResult,
    build_reference,
    calibration_metadata_path,
    frequency_tracking_run_output_dir,
    normalized_run_metadata,
    project_path,
    reference_output_dir,
    relpath,
    run_command,
    safe_name,
    timestamp,
    tracking_npz,
    write_json,
    write_yaml,
)


# child scripts launched by this batch runner
# we run them as subprocesses so logs/failures stay isolated per run
TRACK_SCRIPT = PROJECT_ROOT / "scripts" / "dot_tracking" / "track_frequency_dots.py"
ANALYZE_SCRIPT = PROJECT_ROOT / "scripts" / "frequency_analysis" / "compute_frequency_spectrum.py"


# frequency outputs mirror the tracking run layout
def frequency_analysis_output_dir() -> Path:
    # root folder for all frequency analysis products
    return PROJECT_ROOT / "outputs" / "frequency_analysis"


def frequency_analysis_run_output_dir(run_metadata: dict[str, Any]) -> Path:
    # group / frequency label / video id
    group, freq_label, video_id = frequency_run_path_parts(run_metadata)
    return frequency_analysis_output_dir() / "runs" / group / freq_label / video_id


def parse_args() -> argparse.Namespace:
    # batch cli, selects runs and forwards a few options to child scripts
    parser = argparse.ArgumentParser(description="Run frequency dot tracking and temporal spectral analysis.")
    parser.add_argument("--metadata", default="inputs/batch_metadata.yaml")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--only", action="append", default=[], help="Substring filter for run ids.")
    parser.add_argument("--resume", action="store_true")
    # dry-run shows/writes what would happen without running child scripts
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X", "Y", "W", "H"))
    parser.add_argument(
        "--rotated-roi",
        nargs=5,
        type=float,
        default=None,
        metavar=("CX", "CY", "W", "H", "ANGLE_DEG"),
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-match-score", type=float, default=0.35)
    parser.add_argument("--min-valid-dot-fraction", type=float, default=0.85)
    parser.add_argument("--adaptive-tracking", action="store_true")
    parser.add_argument("--preflight-frames", type=int, default=180)
    parser.add_argument("--adaptive-min-frame-valid", type=float, default=0.65)
    parser.add_argument("--adaptive-min-dots", type=int, default=300)
    return parser.parse_args()


# decide whether one metadata row should run through frequency analysis
def run_enabled_for_frequency(run: dict[str, Any]) -> bool:
    # default: stable videos are frequency videos
    # explicit pipeline lists override this
    pipelines = run.get("pipelines")
    if pipelines is None:
        return str(run.get("kind", "stable")).lower() == "stable"
    return "frequency" in {str(item).lower() for item in pipelines}


# simple substring filter for rerunning a small subset
def run_matches(run: dict[str, Any], filters: list[str]) -> bool:
    # multiple --only values are ORed together
    if not filters:
        return True
    haystack = " ".join(
        str(run.get(key, ""))
        for key in ("run_id", "kind", "video_path", "nominal_drive_hz")
    ).lower()
    return any(item.lower() in haystack for item in filters)


def tracking_summary(run_output_dir: Path) -> Path:
    # json emitted by track_frequency_dots.py
    return run_output_dir / "tracking" / "tracking_summary.json"


def frequency_summary(run_output_dir: Path) -> Path:
    # json emitted by compute_frequency_spectrum.py
    return run_output_dir / "spectrum" / "frequency_summary.json"


# summary readers, only copy the high-level fields into the batch csv
def read_json(path: Path) -> dict[str, Any]:
    # best effort read, missing/corrupt summaries become {}
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_tracking_metrics(path: Path) -> dict[str, Any]:
    # only high-level tracking health metrics
    data = read_json(path)
    keys = [
        "frames_tracked",
        "dots_tracked",
        "median_valid_fraction",
        "median_match_score",
        "adaptive_tracking",
        "tracking_profile",
    ]
    return {key: data.get(key) for key in keys}


def load_frequency_metrics(path: Path) -> dict[str, Any]:
    # flatten analysis metrics and output paths from the run summary
    data = read_json(path)
    metrics = dict(data.get("analysis", {}))
    metrics.update(data.get("outputs", {}))
    return metrics


# process one stable frequency run
# write run metadata, track dots, compute spectrum, return one batch record
def run_one(
    *,
    batch: dict[str, Any],
    run: dict[str, Any],
    args: argparse.Namespace,
    batch_dir: Path,
    reference_dir: Path,
) -> dict[str, Any]:
    # combine batch defaults and run fields into one run metadata file
    run_meta = normalized_run_metadata(batch, run)
    run_id = str(run_meta["run_id"])
    run_batch_dir = batch_dir / "runs" / safe_name(run_id)
    run_metadata_path = run_batch_dir / "run_metadata.yaml"
    # save the metadata snapshot used by child scripts
    write_yaml(run_metadata_path, run_meta)

    # resolve output paths before work starts
    # then skipped/dry-run records still show expected paths
    tracking_output_dir = frequency_tracking_run_output_dir(run_meta)
    spectrum_output_dir = frequency_analysis_run_output_dir(run_meta)
    # this record becomes both json data and one row in batch_summary.csv
    record: dict[str, Any] = {
        "run_id": run_id,
        "metadata_path": relpath(run_metadata_path),
        "video_path": relpath(project_path(run_meta.get("video_path"))),
        "drive_frequency_hz": run_meta["experiment"].get("drive_frequency_hz"),
        "expected_subharmonic_hz": run_meta["experiment"].get("expected_subharmonic_hz"),
        "tracking_output_dir": relpath(tracking_output_dir),
        "frequency_output_dir": relpath(spectrum_output_dir),
        "tracking_npz": relpath(tracking_npz(tracking_output_dir)),
        "summary_json": relpath(frequency_summary(spectrum_output_dir)),
        "status": "pending",
        "steps": [],
    }

    video_path = project_path(run_meta.get("video_path"))
    if video_path is None or not video_path.exists():
        # missing video should skip this run, not crash the full batch
        record["status"] = "skipped"
        record["message"] = "Missing video file."
        return record

    steps: list[StepResult] = []
    # tracking step
    # shared flat reference -> tracked_dots_frequency.npz
    if args.resume and tracking_npz(tracking_output_dir).exists() and tracking_summary(tracking_output_dir).exists():
        # resume trusts existing tracking npz + summary
        steps.append(StepResult(name="track_frequency_dots", status="skipped", message="Tracking outputs already exist."))
    else:
        # build command for tracking subprocess
        cmd = [
            sys.executable,
            str(TRACK_SCRIPT),
            str(run_metadata_path),
            "--calibration-metadata",
            str(calibration_metadata_path(batch)),
            "--reference-dir",
            str(reference_dir),
            "--output-dir",
            str(tracking_output_dir),
            "--frame-step",
            str(args.frame_step),
            "--min-match-score",
            str(args.min_match_score),
        ]
        if args.max_frames is not None:
            # optional quick-run cap
            cmd.extend(["--max-frames", str(args.max_frames)])
        if args.adaptive_tracking:
            # forward adaptive tracking settings to track_frequency_dots.py
            cmd.extend(
                [
                    "--adaptive-tracking",
                    "--preflight-frames",
                    str(args.preflight_frames),
                    "--adaptive-min-frame-valid",
                    str(args.adaptive_min_frame_valid),
                    "--adaptive-min-dots",
                    str(args.adaptive_min_dots),
                ]
            )
        steps.append(run_command(
            # run_command captures stdout/stderr into this run's log file
            "track_frequency_dots",
            cmd,
            run_batch_dir / "logs" / "track_frequency_dots.log",
            dry_run=args.dry_run,
        ))

    if steps[-1].status not in {"ok", "dry-run", "skipped"}:
        # no tracking npz means no spectrum, so stop this run here
        record["status"] = "failed"
        record["steps"] = [step.__dict__ for step in steps]
        record["message"] = "Tracking failed."
        return record

    # spectrum step
    # tracking npz -> full spectrum csv + half-integer csv + summary json
    if args.resume and frequency_summary(spectrum_output_dir).exists():
        # resume can skip spectrum if summary already exists
        steps.append(StepResult(name="compute_frequency_spectrum", status="skipped", message="Spectral output already exists."))
    else:
        # build command for spectrum subprocess
        cmd = [
            sys.executable,
            str(ANALYZE_SCRIPT),
            str(run_metadata_path),
            "--tracking-npz",
            str(tracking_npz(tracking_output_dir)),
            "--output-dir",
            str(spectrum_output_dir / "spectrum"),
            "--min-valid-dot-fraction",
            str(args.min_valid_dot_fraction),
        ]
        steps.append(run_command(
            "compute_frequency_spectrum",
            cmd,
            run_batch_dir / "logs" / "compute_frequency_spectrum.log",
            dry_run=args.dry_run,
        ))

    # copy step status and selected metrics into the batch record
    record["steps"] = [step.__dict__ for step in steps]
    record["status"] = "dry-run" if args.dry_run else ("ok" if all(step.status in {"ok", "skipped", "dry-run"} for step in steps) else "failed")
    record["tracking_metrics"] = load_tracking_metrics(tracking_summary(tracking_output_dir))
    record["frequency_metrics"] = load_frequency_metrics(frequency_summary(spectrum_output_dir))
    return record


# batch csv, one row per run
# enough info for quick inspection and dataset-level figures
def write_batch_csv(batch_dir: Path, records: list[dict[str, Any]]) -> Path:
    # keep it flat so it opens cleanly in spreadsheets
    path = batch_dir / "batch_summary.csv"
    fieldnames = [
        "run_id",
        "drive_frequency_hz",
        "expected_subharmonic_hz",
        "status",
        "message",
        "frames_tracked",
        "dots_tracked",
        "median_valid_fraction",
        "median_match_score",
        "full_spectrum_peak_hz",
        "full_spectrum_peak_power_px2",
        "expected_subharmonic_nearest_hz",
        "subharmonic_power_px2",
        "drive_nearest_hz",
        "drive_power_px2",
        "subharmonic_to_drive_power_ratio",
        "full_spectrum_csv",
        "half_integer_peaks_csv",
        "tracking_npz",
        "summary_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            # flatten tracking/frequency nested metrics into csv columns
            tracking = record.get("tracking_metrics", {})
            frequency = record.get("frequency_metrics", {})
            writer.writerow(
                {
                    "run_id": record.get("run_id"),
                    "drive_frequency_hz": record.get("drive_frequency_hz"),
                    "expected_subharmonic_hz": record.get("expected_subharmonic_hz"),
                    "status": record.get("status"),
                    "message": record.get("message", ""),
                    "frames_tracked": tracking.get("frames_tracked"),
                    "dots_tracked": tracking.get("dots_tracked"),
                    "median_valid_fraction": tracking.get("median_valid_fraction"),
                    "median_match_score": tracking.get("median_match_score"),
                    "full_spectrum_peak_hz": frequency.get("full_spectrum_peak_hz"),
                    "full_spectrum_peak_power_px2": frequency.get("full_spectrum_peak_power_px2"),
                    "expected_subharmonic_nearest_hz": frequency.get("expected_subharmonic_nearest_hz"),
                    "subharmonic_power_px2": frequency.get("subharmonic_power_px2"),
                    "drive_nearest_hz": frequency.get("drive_nearest_hz"),
                    "drive_power_px2": frequency.get("drive_power_px2"),
                    "subharmonic_to_drive_power_ratio": frequency.get("subharmonic_to_drive_power_ratio"),
                    "full_spectrum_csv": frequency.get("full_spectrum_csv"),
                    "half_integer_peaks_csv": frequency.get("half_integer_peaks_csv"),
                    "tracking_npz": record.get("tracking_npz"),
                    "summary_json": record.get("summary_json"),
                }
            )
    return path


# full frequency batch orchestration
# build/reuse reference, select runs, process them, write summaries
def run_frequency(args: argparse.Namespace) -> None:
    # load batch metadata with runs and defaults
    metadata_path = project_path(args.metadata)
    if metadata_path is None or not metadata_path.exists():
        raise FileNotFoundError(f"Missing batch metadata: {metadata_path}")
    batch = load_yaml(metadata_path)
    calibration_meta = load_yaml(calibration_metadata_path(batch))

    # one timestamped folder per batch execution
    batch_dir = frequency_analysis_output_dir() / "batch" / f"frequency_{timestamp()}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    reference_dir = reference_output_dir(batch, args, calibration_meta)
    # build/reuse the flat reference once for the whole batch
    reference_result = build_reference(batch, args, batch_dir, reference_dir)
    if reference_result.status == "failed":
        raise SystemExit(f"Reference build failed. See {reference_result.log_path}")

    runs = [
        # only frequency-enabled runs, then optional --only filter
        run for run in batch.get("runs", [])
        if run_enabled_for_frequency(run) and run_matches(run, args.only)
    ]
    records: list[dict[str, Any]] = []
    if args.jobs <= 1:
        # serial mode, simpler and ordered
        for run in runs:
            records.append(run_one(batch=batch, run=run, args=args, batch_dir=batch_dir, reference_dir=reference_dir))
    else:
        # parallel mode, each run has its own logs
        with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
            futures = {
                executor.submit(
                    run_one,
                    batch=batch,
                    run=run,
                    args=args,
                    batch_dir=batch_dir,
                    reference_dir=reference_dir,
                ): run
                for run in runs
            }
            for future in as_completed(futures):
                run = futures[future]
                try:
                    records.append(future.result())
                except Exception as exc:
                    # one bad run should not delete the rest of the batch summary
                    records.append(
                        {
                            "run_id": run.get("run_id", "unknown"),
                            "status": "failed",
                            "message": repr(exc),
                        }
                    )

    # stable sort so summaries are deterministic
    records = sorted(records, key=lambda item: str(item.get("run_id", "")))
    csv_path = write_batch_csv(batch_dir, records)
    json_path = batch_dir / "batch_summary.json"
    write_json(
        json_path,
        {
            # json keeps full nested records, csv is the compact table
            "pipeline": "frequency",
            "metadata_path": relpath(metadata_path),
            "batch_dir": relpath(batch_dir),
            "reference_dir": relpath(reference_dir),
            "reference_step": reference_result.__dict__,
            "runs": records,
        },
    )

    ok = sum(1 for record in records if record.get("status") in {"ok", "skipped", "dry-run"})
    failed = sum(1 for record in records if record.get("status") == "failed")
    # short final console summary for logs
    print(f"Frequency pipeline finished: ok/skipped={ok}, failed={failed}")
    print(f"CSV summary: {relpath(csv_path)}")
    print(f"JSON summary: {relpath(json_path)}")


def main() -> None:
    run_frequency(parse_args())


if __name__ == "__main__":
    main()
