#!/usr/bin/env python3


from __future__ import annotations

# this measures the dominant spatial wave number from reconstructed height maps
# it uses a direct 2d fourier transform of the surface

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# default search/output roots for public fsss reconstruction products
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "full_fsss" / "runs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "full_fsss" / "wavenumber"

SUMMARY_COLUMNS = [
    # one row per surface npz/run. This is the compact dataset-level output.
    "run_id",
    "calibration_id",
    "surface_npz",
    "selected_output_frame_position",
    "selected_frame_index",
    "selected_time_s",
    "selected_surface_rms_m",
    "kx_rad_per_m",
    "ky_rad_per_m",
    "k_rad_per_m",
    "kx_cycles_per_m",
    "ky_cycles_per_m",
    "q_cycles_per_m",
    "wavelength_m",
    "wavelength_mm",
    "fft_peak_power",
    "fft_peak_power_fraction",
    "grid_dx_m",
    "grid_dy_m",
    "grid_nx",
    "grid_ny",
    "frame_count",
]

FRAME_COLUMNS = [
    # one row per reconstructed frame. This is useful when the selected max-RMS
    # frame needs to be checked.
    "run_id",
    "surface_npz",
    "output_frame_position",
    "frame_index",
    "time_s",
    "surface_rms_m",
    "kx_rad_per_m",
    "ky_rad_per_m",
    "k_rad_per_m",
    "kx_cycles_per_m",
    "ky_cycles_per_m",
    "q_cycles_per_m",
    "wavelength_m",
    "wavelength_mm",
    "fft_peak_power",
    "fft_peak_power_fraction",
]


# cli can either take explicit surface_height_m.npz files
# or discover all reconstructions under the full-fsss run output root
def parse_args() -> argparse.Namespace:
    # users can pass specific surface npz files or let the script recursively
    # discover all reconstructions under outputs/full_fsss/runs.
    parser = argparse.ArgumentParser(
        description="Measure dominant kx, ky, scalar k, and wavelength from FSSS height maps."
    )
    parser.add_argument(
        "--surface-npz",
        action="append",
        type=Path,
        default=[],
        help="Path to a surface_height_m.npz file. May be repeated.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root searched for surface_height_m.npz files when --surface-npz is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults under outputs/full_fsss/wavenumber/.",
    )
    parser.add_argument("--min-wavelength-mm", type=float, default=5.0)
    parser.add_argument("--max-wavelength-mm", type=float, default=100.0)
    return parser.parse_args()


def project_path(path: Path | str) -> Path:
    # resolve cli paths relative to repo root unless already absolute.
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relpath(path: Path) -> str:
    # store repo-relative paths in csv/json outputs when possible.
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def scalar(value: Any) -> Any:
    # npz scalar strings are often zero-dimensional numpy arrays. convert them
    # into python scalars for csv/json.
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return item
    return value


# surface preprocessing before the spatial fft: fill missing values, remove the
# spatial mean and best-fit plane so the dominant peak is wave structure rather
# than DC offset or tilt.
def finite_rms(values: np.ndarray) -> float:
    # rms over finite values only. Used to pick the frame with largest wave
    # amplitude.
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


def fill_missing_with_mean(field: np.ndarray) -> np.ndarray:
    # fft cannot handle nans, so fill missing points with the spatial mean.
    finite = np.isfinite(field)
    if not np.any(finite):
        return np.zeros_like(field, dtype=float)
    mean = float(np.nanmean(field))
    return np.where(finite, field, mean).astype(float)


def remove_mean_and_plane(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Remove DC and tilt terms before the spatial fft."""
    filled = fill_missing_with_mean(field)
    xx, yy = np.meshgrid(x, y)
    # fit plane z = a*x + b*y + c by least squares.
    design = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    values = filled.ravel()
    coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
    plane = (design @ coeff).reshape(filled.shape)
    # remove tilt plane and residual mean so the zero/low-k content does not
    # dominate the Fourier peak.
    detrended = filled - plane
    return detrended - float(np.mean(detrended))


# 2D spatial Fourier analysis for one height map. A separable Hann window
# reduces edge leakage, fftshift centers k=0, and the wavelength mask excludes
# the zero mode and implausible wavelengths.
def fft_peak(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    min_wavelength_m: float,
    max_wavelength_m: float,
) -> dict[str, float]:
    # preprocess one height map before spatial fft.
    processed = remove_mean_and_plane(field, x, y)
    ny, nx = processed.shape
    if nx < 2 or ny < 2:
        return empty_peak()

    dx = float(np.nanmedian(np.diff(x)))
    dy = float(np.nanmedian(np.diff(y)))
    if not (np.isfinite(dx) and np.isfinite(dy) and dx > 0 and dy > 0):
        return empty_peak()

    # separable 2D Hann window reduces edge leakage from the finite rectangular
    # field of view.
    window = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    # fft2 returns complex spatial Fourier coefficients. fftshift moves zero
    # spatial frequency to the center of the array.
    spectrum = np.fft.fftshift(np.fft.fft2(processed * window))
    power = np.abs(spectrum) ** 2

    # fftfreq returns spatial frequencies in cycles per meter. Later we convert
    # to radians per meter by multiplying by 2*pi.
    qx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    qy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    qx_grid, qy_grid = np.meshgrid(qx, qy)
    q = np.hypot(qx_grid, qy_grid)
    wavelength_m = np.divide(1.0, q, out=np.full_like(q, np.nan), where=q > 0)

    # mask keeps only physically plausible wavelengths and excludes the zero
    # mode.
    mask = (
        np.isfinite(wavelength_m)
        & (wavelength_m >= min_wavelength_m)
        & (wavelength_m <= max_wavelength_m)
        & (q > 0)
    )
    if not np.any(mask):
        return empty_peak(dx=dx, dy=dy, nx=nx, ny=ny)

    masked_power = np.where(mask, power, -np.inf)
    # strongest allowed spatial Fourier peak.
    iy, ix = np.unravel_index(int(np.argmax(masked_power)), masked_power.shape)

    qx_peak = float(qx_grid[iy, ix])
    qy_peak = float(qy_grid[iy, ix])
    # the Fourier peak has a conjugate partner at (-kx, -ky), so the sign is
    # arbitrary. use a stable convention for repeatable csv values.
    if qx_peak < 0 or (math.isclose(qx_peak, 0.0, abs_tol=1e-12) and qy_peak < 0):
        qx_peak *= -1.0
        qy_peak *= -1.0

    q_peak = float(math.hypot(qx_peak, qy_peak))
    kx_rad = 2.0 * math.pi * qx_peak
    ky_rad = 2.0 * math.pi * qy_peak
    k_rad = 2.0 * math.pi * q_peak
    peak_power = float(power[iy, ix])
    total_power = float(np.sum(power[mask]))

    return {
        "kx_cycles_per_m": qx_peak,
        "ky_cycles_per_m": qy_peak,
        "q_cycles_per_m": q_peak,
        "kx_rad_per_m": kx_rad,
        "ky_rad_per_m": ky_rad,
        "k_rad_per_m": k_rad,
        "wavelength_m": 1.0 / q_peak if q_peak > 0 else float("nan"),
        "wavelength_mm": 1000.0 / q_peak if q_peak > 0 else float("nan"),
        "fft_peak_power": peak_power,
        "fft_peak_power_fraction": peak_power / total_power if total_power > 0 else float("nan"),
        "grid_dx_m": dx,
        "grid_dy_m": dy,
        "grid_nx": float(nx),
        "grid_ny": float(ny),
    }


# empty peak row keeps csv schemas stable when a frame cannot produce a valid
# fourier peak.
def empty_peak(*, dx: float = float("nan"), dy: float = float("nan"), nx: int = 0, ny: int = 0) -> dict[str, float]:
    # return all expected columns with nan values so csv schemas remain stable.
    return {
        "kx_cycles_per_m": float("nan"),
        "ky_cycles_per_m": float("nan"),
        "q_cycles_per_m": float("nan"),
        "kx_rad_per_m": float("nan"),
        "ky_rad_per_m": float("nan"),
        "k_rad_per_m": float("nan"),
        "wavelength_m": float("nan"),
        "wavelength_mm": float("nan"),
        "fft_peak_power": float("nan"),
        "fft_peak_power_fraction": float("nan"),
        "grid_dx_m": dx,
        "grid_dy_m": dy,
        "grid_nx": float(nx),
        "grid_ny": float(ny),
    }


# load reconstructed fsss surface data and normalize older 2D/3D npz shapes into
# a frame-first zeta array plus x/y metric grid vectors.
def load_surface(path: Path) -> dict[str, Any]:
    # load one reconstructed surface npz and normalize arrays to frame-first
    # shape: zeta[frame, y, x].
    with np.load(path) as data:
        zeta = np.asarray(data["zeta_m"], dtype=float)
        if zeta.ndim == 2:
            # older/single-frame outputs may store only one 2D array.
            zeta = zeta[None, :, :]

        if "x_grid_m" in data.files and "y_grid_m" in data.files:
            # preferred compact coordinate vectors.
            x = np.asarray(data["x_grid_m"], dtype=float)
            y = np.asarray(data["y_grid_m"], dtype=float)
        elif "X_m" in data.files and "Y_m" in data.files:
            # compatibility with outputs that store full meshgrids.
            x = np.asarray(data["X_m"], dtype=float)[0, :]
            y = np.asarray(data["Y_m"], dtype=float)[:, 0]
        else:
            raise KeyError(f"{path} does not contain x/y grid coordinates.")

        frame_count = zeta.shape[0]
        time_s = np.asarray(data["time_s"], dtype=float) if "time_s" in data.files else np.arange(frame_count, dtype=float)
        frame_indices = (
            np.asarray(data["frame_indices"], dtype=int)
            if "frame_indices" in data.files
            else np.arange(frame_count, dtype=int)
        )
        run_id = str(scalar(data["run_id"])) if "run_id" in data.files else path.parents[1].name
        calibration_id = str(scalar(data["calibration_id"])) if "calibration_id" in data.files else ""

    return {
        "zeta": zeta,
        "x": x,
        "y": y,
        "time_s": time_s,
        "frame_indices": frame_indices,
        "run_id": run_id,
        "calibration_id": calibration_id,
    }


# analyze every reconstructed frame, then choose the max-RMS frame as the single
# run-level summary while keeping per-frame wavenumber rows for inspection.
def analyze_surface(path: Path, min_wavelength_m: float, max_wavelength_m: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # analyze each reconstructed frame independently.
    data = load_surface(path)
    zeta = data["zeta"]
    frame_rows: list[dict[str, Any]] = []
    for pos, frame in enumerate(zeta):
        # surface RMS is used as a proxy for which frame has strongest waves.
        rms = finite_rms(frame)
        peak = fft_peak(frame, data["x"], data["y"], min_wavelength_m, max_wavelength_m)
        frame_rows.append(
            {
                "run_id": data["run_id"],
                "surface_npz": relpath(path),
                "output_frame_position": pos,
                "frame_index": int(data["frame_indices"][pos]) if pos < len(data["frame_indices"]) else pos,
                "time_s": float(data["time_s"][pos]) if pos < len(data["time_s"]) else float(pos),
                "surface_rms_m": rms,
                **peak,
            }
        )

    valid_rms = np.asarray([row["surface_rms_m"] for row in frame_rows], dtype=float)
    if valid_rms.size and np.any(np.isfinite(valid_rms)):
        # run-level summary chooses the max-RMS reconstructed frame.
        selected_pos = int(np.nanargmax(valid_rms))
        selected = frame_rows[selected_pos]
    else:
        selected_pos = 0
        selected = frame_rows[0] if frame_rows else {}

    summary = {
        "run_id": data["run_id"],
        "calibration_id": data["calibration_id"],
        "surface_npz": relpath(path),
        "selected_output_frame_position": selected.get("output_frame_position", selected_pos),
        "selected_frame_index": selected.get("frame_index", selected_pos),
        "selected_time_s": selected.get("time_s", float("nan")),
        "selected_surface_rms_m": selected.get("surface_rms_m", float("nan")),
        "frame_count": int(zeta.shape[0]),
    }
    for key in (
        "kx_rad_per_m",
        "ky_rad_per_m",
        "k_rad_per_m",
        "kx_cycles_per_m",
        "ky_cycles_per_m",
        "q_cycles_per_m",
        "wavelength_m",
        "wavelength_mm",
        "fft_peak_power",
        "fft_peak_power_fraction",
        "grid_dx_m",
        "grid_dy_m",
        "grid_nx",
        "grid_ny",
    ):
        summary[key] = selected.get(key, float("nan"))
    return summary, frame_rows


def find_surface_npzs(root: Path) -> list[Path]:
    # recursive discovery for batch-level use.
    return sorted(project_path(root).rglob("surface_height_m.npz"))


# plain csv/json writers make the wavenumber output easy to use outside python.
def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    # timestamped default output folder prevents overwriting earlier analyses.
    output_dir = project_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    min_wavelength_m = float(args.min_wavelength_mm) / 1000.0
    max_wavelength_m = float(args.max_wavelength_mm) / 1000.0
    if min_wavelength_m <= 0 or max_wavelength_m <= min_wavelength_m:
        raise SystemExit("Expected 0 < --min-wavelength-mm < --max-wavelength-mm.")

    surface_paths = [project_path(path) for path in args.surface_npz] or find_surface_npzs(args.root)
    # drop paths that do not exist before processing.
    surface_paths = [path for path in surface_paths if path.exists()]
    if not surface_paths:
        raise SystemExit("No surface_height_m.npz files found.")

    # process each surface file independently; each contributes one summary row
    # and one row per reconstructed frame.
    summary_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for path in surface_paths:
        summary, frames = analyze_surface(path, min_wavelength_m, max_wavelength_m)
        summary_rows.append(summary)
        frame_rows.extend(frames)

    summary_path = output_dir / "full_fsss_wavenumber_summary.csv"
    frames_path = output_dir / "full_fsss_wavenumber_frames.csv"
    manifest_path = output_dir / "full_fsss_wavenumber_manifest.json"
    write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    write_csv(frames_path, frame_rows, FRAME_COLUMNS)
    # manifest records the fft method, wavelength bounds, source surfaces, and
    # output paths for reproducibility.
    write_json(
        manifest_path,
        {
            "schema": "full_fsss_2d_fft_wavenumber_v1",
            "method": "instantaneous 2D fft of reconstructed surface height; summary uses max-RMS frame",
            "preprocessing": "subtract spatial mean and best-fit plane; apply separable Hann window",
            "min_wavelength_mm": float(args.min_wavelength_mm),
            "max_wavelength_mm": float(args.max_wavelength_mm),
            "surface_npz": [relpath(path) for path in surface_paths],
            "outputs": {
                "summary_csv": relpath(summary_path),
                "frames_csv": relpath(frames_path),
            },
        },
    )

    print(f"Wrote {len(summary_rows)} summary row(s) to {relpath(summary_path)}")
    print(f"Wrote {len(frame_rows)} frame row(s) to {relpath(frames_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
