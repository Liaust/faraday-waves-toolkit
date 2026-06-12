#!/usr/bin/env python3
from __future__ import annotations

# this script takes tracked dot motion from one stable video
# and turns it into a temporal frequency spectrum
# dot motion over time -> power vs frequency

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# repo root, used for default input/output paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# add repo root so direct script execution still imports local modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# fixed frequency scan grid
# using the same grid for all runs makes heatmaps/comparisons easier
FULL_SPECTRUM_MIN_HZ = 0.5
FULL_SPECTRUM_MAX_HZ = 80.0
FULL_SPECTRUM_STEP_HZ = 0.05


# local path/json/yaml helpers for this single-run script
def project_path(value: str | Path | None) -> Path | None:
    # accept absolute paths or paths relative to the repo
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relpath(path: str | Path | None) -> str | None:
    # summaries should avoid absolute local paths when possible
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p)


def load_yaml(path: str | Path) -> dict[str, Any]:
    # run metadata is yaml and should be a dict
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    # make sure parent folder exists before writing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# summarize the frame timebase used for the spectrum
def summarize_timebase(times_s: np.ndarray) -> dict[str, float | int]:
    # adjacent frame times give the effective frame spacing
    times = np.asarray(times_s, dtype=float)
    dt = np.diff(times)
    # ignore bad intervals
    positive = dt[np.isfinite(dt) & (dt > 0)]
    if len(positive) == 0:
        return {
            "frames": int(len(times)),
            "duration_s": float("nan"),
            "median_dt_s": float("nan"),
            "mean_dt_s": float("nan"),
            "fps_from_median_dt": float("nan"),
            "fps_from_mean_dt": float("nan"),
            "irregular_percentage_1pct": float("nan"),
        }
    median_dt = float(np.median(positive))
    mean_dt = float(np.mean(positive))
    # count intervals more than 1 percent away from the median as irregular
    irregular = positive[np.abs(positive - median_dt) > 0.01 * median_dt]
    return {
        "frames": int(len(times)),
        "duration_s": float(times[-1] - times[0]) if len(times) else float("nan"),
        "min_dt_s": float(np.min(positive)),
        "median_dt_s": median_dt,
        "mean_dt_s": mean_dt,
        "max_dt_s": float(np.max(positive)),
        "fps_from_median_dt": float(1.0 / median_dt),
        "fps_from_mean_dt": float(1.0 / mean_dt),
        "irregular_percentage_1pct": float(100.0 * len(irregular) / len(positive)),
    }


# same folder naming convention as tracking
# group / drive frequency / video id
def frequency_run_path_parts(run_metadata: dict[str, Any]) -> tuple[str, str, str]:
    # reconstruct the folder pieces used by track_frequency_dots.py
    group = str(run_metadata.get("experiment_group", "runs"))
    experiment = run_metadata.get("experiment", {})
    drive = (
        experiment.get("drive_frequency_hz")
        or run_metadata.get("drive", {}).get("measured_frequency_hz")
        or run_metadata.get("drive", {}).get("nominal_frequency_hz")
        or run_metadata.get("nominal_drive_hz")
        or "unknown"
    )
    try:
        # numeric drive values become labels like 15Hz
        freq_label = f"{float(drive):g}Hz"
    except (TypeError, ValueError):
        freq_label = str(drive)
    run_id = str(run_metadata.get("expected_iphone_video_id") or run_metadata.get("run_id"))
    return group, freq_label, run_id


def frequency_tracking_run_output_dir(run_metadata: dict[str, Any]) -> Path:
    group, freq_label, run_id = frequency_run_path_parts(run_metadata)
    return PROJECT_ROOT / "outputs" / "dot_tracking" / "frequency" / "runs" / group / freq_label / run_id


def frequency_analysis_run_output_dir(run_metadata: dict[str, Any]) -> Path:
    group, freq_label, run_id = frequency_run_path_parts(run_metadata)
    return PROJECT_ROOT / "outputs" / "frequency_analysis" / "runs" / group / freq_label / run_id


def parse_args() -> argparse.Namespace:
    # single-run cli, usually called by the batch runner
    parser = argparse.ArgumentParser(
        description="Compute the temporal frequency spectrum from frequency-analysis dot tracks."
    )
    parser.add_argument("run_metadata", help="metadata_run.yaml for one run.")
    parser.add_argument("--tracking-npz", default=None, help="Override tracked_dots_frequency.npz path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Spectrum output directory. Defaults under outputs/frequency_analysis/runs/.",
    )
    parser.add_argument("--min-valid-dot-fraction", type=float, default=0.85)
    return parser.parse_args()


# npz sometimes stores strings as zero-dimensional arrays
# this turns those back into normal python strings
def npz_string(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    # item() extracts scalar arrays
    if key not in data.files:
        return default
    value = data[key]
    if getattr(value, "shape", ()) == ():
        return str(value.item())
    return str(value)


# drive frequency used as the reference for half-integer lines
def load_drive_frequency_hz(run_meta: dict[str, Any]) -> float:
    # prefer explicit/measured values before nominal fallback
    experiment = run_meta.get("experiment", {})
    drive = run_meta.get("drive", {})
    return float(
        experiment.get(
            "drive_frequency_hz",
            drive.get(
                "measured_frequency_hz",
                drive.get("nominal_frequency_hz", run_meta.get("nominal_drive_hz", 25.0)),
            ),
        )
    )


# expected subharmonic is usually fd/2 unless metadata overrides it
def load_expected_subharmonic_hz(run_meta: dict[str, Any], drive_hz: float) -> float:
    # mostly fd/2, but allow measured/explicit target if it exists
    experiment = run_meta.get("experiment", {})
    drive = run_meta.get("drive", {})
    return float(
        experiment.get(
            "expected_subharmonic_hz",
            drive.get("expected_subharmonic_hz", float(drive_hz) / 2.0),
        )
    )


# hann window before fourier projection to reduce leakage from finite video length
def projection_window(t_count: int) -> np.ndarray:
    # zero at the ends, near one in the middle
    return np.hanning(t_count) if t_count >= 3 else np.ones(t_count)


# prepare dot motion before the spectrum
# keep good dots, subtract static offset, remove frame-mean drift
def prepare_dot_motion(
    dxy_px: np.ndarray,
    valid: np.ndarray,
    *,
    min_valid_dot_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    # dxy_px is frames x dots x xy, valid is frames x dots
    finite = np.isfinite(dxy_px).all(axis=2)
    valid = np.asarray(valid, dtype=bool) & finite
    # keep dots that were tracked in enough frames
    dot_valid_fraction = np.mean(valid, axis=0)
    keep = dot_valid_fraction >= float(min_valid_dot_fraction)
    if not np.any(keep):
        raise ValueError(
            f"No dots were valid in at least {min_valid_dot_fraction:.2f} of frames."
        )

    # select retained dots and use float64 for stable projection
    signal = dxy_px[:, keep, :].astype(np.float64, copy=True)
    valid_keep = valid[:, keep]
    # temporary nans so medians ignore invalid samples
    signal[~valid_keep] = np.nan

    # subtract each dot's median over time, removing static offsets
    median_xy = np.nanmedian(signal, axis=0, keepdims=True)
    signal = signal - median_xy
    # invalid samples become zero after centering
    signal = np.where(valid_keep[:, :, None] & np.isfinite(signal), signal, 0.0)

    # remove common motion in each frame across all retained dots
    counts = np.sum(valid_keep, axis=1).astype(float)
    counts[counts == 0] = np.nan
    frame_mean = np.nansum(signal, axis=1) / counts[:, None]
    frame_mean = np.where(np.isfinite(frame_mean), frame_mean, 0.0)
    signal = signal - frame_mean[:, None, :]
    # keep invalid samples at zero
    signal[~valid_keep] = 0.0
    return signal, keep


# direct frequency projection of dot motion
# basically inner products against exp(-i 2 pi f t)
def project_vector_power(
    signal_xy_px: np.ndarray,
    times_s: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    # direct projection instead of fft, so we can choose any frequency grid
    times = np.asarray(times_s, dtype=float)
    window = projection_window(signal_xy_px.shape[0])
    # apply hann window to x and y displacement
    uxw = signal_xy_px[:, :, 0] * window[:, None]
    uyw = signal_xy_px[:, :, 1] * window[:, None]
    # 2/sum(window) gives amplitude-like coefficients for one-sided projection
    norm = 2.0 / np.sum(window)

    power = np.full(len(freqs_hz), np.nan, dtype=float)
    for start in range(0, len(freqs_hz), chunk_size):
        stop = min(start + chunk_size, len(freqs_hz))
        # basis is freq x frames, one complex test wave per frequency
        basis = np.exp(-2j * np.pi * np.outer(freqs_hz[start:stop], times))
        # sum over time for every frequency and dot
        coeff_x = norm * (basis @ uxw)
        coeff_y = norm * (basis @ uyw)
        # combine x/y displacement power and average over dots
        power[start:stop] = np.mean(np.abs(coeff_x) ** 2 + np.abs(coeff_y) ** 2, axis=1)
    return power


# nearest scanned bin to a target frequency
def nearest_power(freqs_hz: np.ndarray, power: np.ndarray, target_hz: float) -> tuple[float, float]:
    # target may not land exactly on the scan grid
    idx = int(np.argmin(np.abs(freqs_hz - float(target_hz))))
    return float(freqs_hz[idx]), float(power[idx])


# table of power at n/2 * fd lines
# normalize everything by the main half-drive peak
def half_integer_peaks(
    freqs_hz: np.ndarray,
    power: np.ndarray,
    drive_hz: float,
    *,
    reference_power: float,
) -> list[dict[str, float | int]]:
    if drive_hz <= 0:
        return []
    min_hz = float(np.min(freqs_hz))
    max_hz = float(np.max(freqs_hz))
    # n=1 is fd/2, n=2 is fd, n=3 is 3fd/2, etc
    max_order = int(np.floor(2.0 * max_hz / float(drive_hz)))
    rows: list[dict[str, float | int]] = []
    for order_n in range(1, max_order + 1):
        expected_hz = 0.5 * float(order_n) * float(drive_hz)
        if expected_hz < min_hz or expected_hz > max_hz:
            continue
        nearest_hz, peak_power = nearest_power(freqs_hz, power, expected_hz)
        rows.append(
            {
                "order_n": int(order_n),
                "expected_frequency_hz": float(expected_hz),
                "nearest_frequency_hz": nearest_hz,
                "mean_displacement_power_px2": peak_power,
                "relative_to_half_drive_power": (
                    float(peak_power / reference_power)
                    if np.isfinite(reference_power) and reference_power > 0
                    else float("nan")
                ),
            }
        )
    return rows


# plain csv outputs so figures/users don't need to load the tracking npz
def write_full_spectrum_csv(path: Path, freqs_hz: np.ndarray, power: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # long table, one row per frequency bin
        writer.writerow(["frequency_hz", "mean_displacement_power_px2"])
        writer.writerows((float(freq), float(value)) for freq, value in zip(freqs_hz, power, strict=True))


def write_half_integer_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "order_n",
        "expected_frequency_hz",
        "nearest_frequency_hz",
        "mean_displacement_power_px2",
        "relative_to_half_drive_power",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # smaller table for expected half-integer response lines
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    # resolve and load run metadata
    run_metadata_path = project_path(args.run_metadata)
    if run_metadata_path is None:
        raise SystemExit("Missing run metadata path.")
    run_meta = load_yaml(run_metadata_path)

    # input is the affine-corrected tracking npz
    tracking_path = project_path(args.tracking_npz)
    if tracking_path is None:
        # default location mirrors track_frequency_dots.py
        tracking_path = frequency_tracking_run_output_dir(run_meta) / "tracking" / "tracked_dots_frequency.npz"
    if not tracking_path.exists():
        raise FileNotFoundError(f"Missing tracking NPZ: {tracking_path}")

    out_dir = project_path(args.output_dir)
    if out_dir is None:
        # default spectrum output for this run
        out_dir = frequency_analysis_run_output_dir(run_meta) / "spectrum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load corrected dot displacements, validity mask and timebase
    # tracking and affine correction already happened before this script
    with np.load(tracking_path, allow_pickle=False) as data:
        # normalize time again just in case future tracking changes
        time_s = data["time_s"].astype(np.float64)
        time_s = time_s - float(time_s[0])
        # actual motion signal used for the spectrum
        dxy_px = data["corrected_dxy_px"].astype(np.float64)
        valid = data["valid"].astype(bool)
        tracking_timebase_summary = json.loads(npz_string(data, "timebase_summary_json", "{}"))
        timebase_source = npz_string(data, "timebase_source", "")
        total_dots = int(dxy_px.shape[1])

    # clean dot motion and record which dots survived the validity cut
    signal_xy, kept_dot_mask = prepare_dot_motion(
        dxy_px,
        valid,
        min_valid_dot_fraction=float(args.min_valid_dot_fraction),
    )
    drive_hz = load_drive_frequency_hz(run_meta)
    expected_half_hz = load_expected_subharmonic_hz(run_meta, drive_hz)

    # fixed full-spectrum scan grid for all runs
    full_freqs = np.arange(
        FULL_SPECTRUM_MIN_HZ,
        FULL_SPECTRUM_MAX_HZ + 0.5 * FULL_SPECTRUM_STEP_HZ,
        FULL_SPECTRUM_STEP_HZ,
    )
    # project dot motion onto all test frequencies
    full_power = project_vector_power(signal_xy, time_s, full_freqs)
    # global max is mostly a sanity check
    peak_i = int(np.nanargmax(full_power))
    full_peak_hz = float(full_freqs[peak_i])
    full_peak_power = float(full_power[peak_i])

    # pull out the key physical lines: fd/2, fd, and n/2 * fd
    subharmonic_nearest_hz, subharmonic_power = nearest_power(full_freqs, full_power, expected_half_hz)
    drive_nearest_hz, drive_power = nearest_power(full_freqs, full_power, drive_hz)
    # ratio > 1 means half-drive has more dot-motion power than drive
    ratio = (
        float(subharmonic_power / drive_power)
        if np.isfinite(drive_power) and drive_power > 0
        else float("nan")
    )
    half_rows = half_integer_peaks(
        full_freqs,
        full_power,
        drive_hz,
        reference_power=subharmonic_power,
    )

    full_csv = out_dir / "full_spectrum_frequency_tracks.csv"
    half_integer_csv = out_dir / "half_integer_frequency_peaks.csv"
    summary_path = out_dir / "frequency_summary.json"
    write_full_spectrum_csv(full_csv, full_freqs, full_power)
    write_half_integer_csv(half_integer_csv, half_rows)

    # compact run-level summary used by batch summaries and figure scripts
    analysis = {
        # numerical analysis values, so batch scripts don't need to parse csvs
        "full_spectrum_scan_min_hz": float(FULL_SPECTRUM_MIN_HZ),
        "full_spectrum_scan_max_hz": float(FULL_SPECTRUM_MAX_HZ),
        "full_spectrum_scan_step_hz": float(FULL_SPECTRUM_STEP_HZ),
        "full_spectrum_peak_hz": full_peak_hz,
        "full_spectrum_peak_power_px2": full_peak_power,
        "expected_subharmonic_nearest_hz": subharmonic_nearest_hz,
        "subharmonic_power_px2": subharmonic_power,
        "drive_nearest_hz": drive_nearest_hz,
        "drive_power_px2": drive_power,
        "subharmonic_to_drive_power_ratio": ratio,
        "half_integer_peaks": half_rows,
    }
    summary = {
        # provenance, quality metrics, target frequencies and output paths
        "schema": "frequency_temporal_spectrum_v2",
        "run_id": str(run_meta["run_id"]),
        "run_metadata_path": relpath(run_metadata_path),
        "tracking_npz": relpath(tracking_path),
        "frames": int(len(time_s)),
        "dots_total": total_dots,
        "dots_used": int(np.sum(kept_dot_mask)),
        "min_valid_dot_fraction": float(args.min_valid_dot_fraction),
        "timebase_source": timebase_source,
        "timebase_summary": summarize_timebase(time_s),
        "tracking_timebase_summary": tracking_timebase_summary,
        "drive_frequency_hz": drive_hz,
        "expected_subharmonic_hz": expected_half_hz,
        "analysis": analysis,
        "outputs": {
            "full_spectrum_csv": relpath(full_csv),
            "half_integer_peaks_csv": relpath(half_integer_csv),
        },
    }
    write_json(summary_path, summary)

    print(f"Saved {relpath(full_csv)}")
    print(f"Saved {relpath(half_integer_csv)}")
    print(f"Saved {relpath(summary_path)}")
    print(f"Full-spectrum peak: {full_peak_hz:.3f} Hz")


if __name__ == "__main__":
    main()
