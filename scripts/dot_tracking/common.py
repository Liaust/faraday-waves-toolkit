from __future__ import annotations

# shared utilities for path handling, video io, camera undistortion,
# timebase extraction, and output folder naming

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


# repo root from scripts/dot_tracking/common.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# default calibration metadata location used by cli defaults
CALIBRATION_METADATA_PATH = PROJECT_ROOT / "inputs" / "calibration_metadata.yaml"


# path helpers let scripts run from the repo root while still accepting
# absolute paths to local data outside the repo
def project_path(value: str | Path | None) -> Path | None:
    # accept both absolute paths and paths relative to the public repo root
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relpath(path: str | Path | None) -> str | None:
    # store repo-relative paths when possible so json does not expose local
    # absolute machine paths
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p)


# small yaml/json helpers used by the tracking entry points
def load_yaml(path: str | Path) -> dict[str, Any]:
    # safe_load turns yaml into python objects
    # here we also require the top level to be a dict
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    # create parent folders automatically before writing json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def calibration_output_dir(calibration_metadata: dict[str, Any]) -> Path:
    # legacy/default calibration output location keyed by calibration_id
    return PROJECT_ROOT / "outputs" / "calibration" / str(calibration_metadata["calibration_id"])


def frequency_output_dir() -> Path:
    return frequency_tracking_output_dir()


def frequency_tracking_output_dir() -> Path:
    # all frequency dot-tracking outputs live under this tree
    return PROJECT_ROOT / "outputs" / "dot_tracking" / "frequency"


# camera intrinsics can either be inline in metadata or loaded from the
# checkerboard calibration json
# returned values are: raw camera matrix, distortion coefficients,
# and the undistorted camera matrix used by cv2.undistort
def load_intrinsics(calibration_metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = calibration_metadata.get("camera", {})
    inline_matrix = camera.get("camera_matrix")
    if inline_matrix not in (None, ""):
        # inline mode: camera matrix and distortion are written directly in yaml
        k_raw = np.array(inline_matrix, dtype=float)
        dist_raw = camera.get("distortion_coefficients", camera.get("dist_coeffs", [])) or []
        dist = np.array(dist_raw, dtype=float).reshape(-1, 1)
        if dist.size == 0:
            # opencv undistort accepts a zero distortion vector
            dist = np.zeros((5, 1), dtype=float)
        k_undist = np.array(
            camera.get("new_camera_matrix", camera.get("new_camera_matrix_alpha0", inline_matrix)),
            dtype=float,
        )
        return k_raw, dist, k_undist

    # file mode: read camera_intrinsics.json from configured path or default folder
    configured_path = project_path(camera.get("intrinsics_path"))
    path = configured_path if configured_path is not None else calibration_output_dir(calibration_metadata) / "camera_intrinsics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing camera intrinsics: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    k_raw = np.array(data["camera_matrix"], dtype=float)
    dist = np.array(
        data.get("dist_coeffs", data.get("distortion_coefficients", [])),
        dtype=float,
    ).reshape(-1, 1)
    k_undist = np.array(data.get("new_camera_matrix_alpha0", data["camera_matrix"]), dtype=float)
    return k_raw, dist, k_undist


# lens undistortion is optional
# some batches intentionally skip it when the checkerboard calibration is unreliable
# if metadata says it is required, missing intrinsics becomes an error
def undistort_gray(gray: np.ndarray, calibration_metadata: dict[str, Any]) -> np.ndarray:
    image_processing = calibration_metadata.get("image_processing", {})
    camera = calibration_metadata.get("camera", {})
    # some runs skip undistortion because calibration was poor or because
    # frequency analysis only needs relative motion
    if (
        image_processing.get("skip_lens_undistortion_for_roi_and_frequency", False)
        or camera.get("skip_lens_undistortion", False)
    ):
        return gray
    try:
        k_raw, dist, k_undist = load_intrinsics(calibration_metadata)
    except FileNotFoundError:
        # if metadata says undistortion is required, raise the error
        # otherwise return the raw image
        if (
            image_processing.get("require_lens_undistortion_for_frequency", False)
            or camera.get("require_lens_undistortion", False)
        ):
            raise
        return gray
    # cv2.undistort maps the grayscale image through the camera model
    return cv2.undistort(gray, k_raw, dist, None, k_undist)


# basic opencv video metadata used for frame selection and fallback timing
def video_metadata(video_path: Path) -> dict[str, Any]:
    # VideoCapture gives container-level metadata without decoding every frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    data = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return data


# direct single-frame reader for utilities that need a specific frame
def read_frame_at(video_path: Path, frame_index: int) -> np.ndarray:
    # returns the original BGR frame because callers decide how to process color
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def bgr_to_gray(frame: np.ndarray) -> np.ndarray:
    # opencv decodes color video as BGR, not RGB
    # dot tracking only uses intensity, so convert to grayscale
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# median reference frame construction
# sample grayscale frames and take the per-pixel median
# fixed dots stay sharp while transient noise/waves mostly disappear
def median_reference_frame(
    video_path: Path,
    *,
    max_frames: int = 100,
    stride: int = 5,
    time_window_s: tuple[float, float] | None = None,
) -> np.ndarray:
    info = video_metadata(video_path)
    frame_count = int(info["frame_count"])
    if frame_count <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    fps = float(info["fps"]) if float(info["fps"]) > 0 else 30.0
    if time_window_s is None:
        # no time window means sample the whole video
        start_i = 0
        stop_i = frame_count
    else:
        # convert requested seconds into frame indices using nominal fps
        start_i = max(0, int(round(time_window_s[0] * fps)))
        stop_i = min(frame_count, max(start_i + 1, int(round(time_window_s[1] * fps))))

    indices = np.arange(start_i, stop_i, max(1, int(stride)), dtype=int)
    if len(indices) > max_frames:
        # downsample uniformly if stride would load too many frames
        indices = np.linspace(start_i, stop_i - 1, max_frames, dtype=int)

    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            # store only grayscale frames to reduce memory use
            frames.append(bgr_to_gray(frame))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from {video_path}")
    # per-pixel median suppresses transient waves/noise while preserving fixed dots
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


# timestamp extraction
# prefer ffprobe frame timestamps because iphone videos can have small timing irregularities
# fallback is uniform opencv fps timing
def ffprobe_frame_timestamps(video_path: Path) -> tuple[np.ndarray | None, str]:
    # ffprobe is external to python/opencv
    # if missing, return None so callers can fall back to uniform fps timing
    if shutil.which("ffprobe") is None:
        return None, "ffprobe_unavailable"
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-show_frames",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        # ffprobe emits json with per-frame best_effort_timestamp_time values
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
    except Exception:
        return None, "ffprobe_failed"

    times: list[float] = []
    for frame in data.get("frames", []):
        # some frames may not have a usable timestamp
        value = frame.get("best_effort_timestamp_time")
        if value in (None, "N/A"):
            continue
        try:
            times.append(float(value))
        except ValueError:
            continue
    if not times:
        return None, "ffprobe_no_frame_timestamps"
    return np.asarray(times, dtype=float), "ffprobe_best_effort_timestamp_time"


# public timebase helper
# return frame times starting at zero and label where timing came from
def video_times(video_path: Path, frame_count: int) -> tuple[np.ndarray, str]:
    # prefer real frame timestamps because phone videos can be slightly variable frame-rate
    times, source = ffprobe_frame_timestamps(video_path)
    if times is not None and len(times) >= frame_count:
        t = np.asarray(times[:frame_count], dtype=float)
        return t - t[0], source
    if times is not None and len(times) >= 2:
        t = np.asarray(times, dtype=float)
        mismatch_source = f"{source}_count_{len(t)}_opencv_count_{frame_count}"
        return t - t[0], mismatch_source

    info = video_metadata(video_path)
    # fallback: assume uniform spacing at opencv-reported fps
    fps = float(info["fps"]) if float(info["fps"]) > 0 else 30.0
    t = np.arange(frame_count, dtype=float) / fps
    return t, f"opencv_uniform_{fps:g}_fps"


# compact timebase summary saved into tracking npz/json outputs
def summarize_timebase(times_s: np.ndarray) -> dict[str, float | int]:
    # this makes timing quality visible without storing a separate plot
    times = np.asarray(times_s, dtype=float)
    dt = np.diff(times)
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


# output-directory helpers encode run metadata into predictable public paths
def run_output_dir_from_metadata(run_metadata: dict[str, Any]) -> Path:
    # explicit run_output_dir in metadata wins, otherwise use public defaults
    configured = project_path(run_metadata.get("run_output_dir"))
    if configured is not None:
        return configured
    return frequency_output_dir() / "runs" / str(run_metadata["run_id"])


def frequency_run_path_parts(run_metadata: dict[str, Any]) -> tuple[str, str, str]:
    # frequency outputs are grouped by experiment group, drive frequency, and video/run id
    group = str(run_metadata.get("experiment_group", "runs"))
    experiment = run_metadata.get("experiment", {})
    drive_meta = run_metadata.get("drive", {})
    drive = (
        experiment.get("drive_frequency_hz")
        or drive_meta.get("measured_frequency_hz")
        or drive_meta.get("nominal_frequency_hz")
        or run_metadata.get("nominal_drive_hz")
        or "unknown"
    )
    try:
        # use compact labels like 15Hz instead of 15.0Hz
        freq_label = f"{float(drive):g}Hz"
    except (TypeError, ValueError):
        freq_label = str(drive)
    video_id = str(run_metadata.get("expected_iphone_video_id") or run_metadata.get("run_id"))
    return group, freq_label, video_id


def frequency_tracking_run_output_dir(run_metadata: dict[str, Any]) -> Path:
    group, freq_label, video_id = frequency_run_path_parts(run_metadata)
    return frequency_tracking_output_dir() / "runs" / group / freq_label / video_id


def frequency_run_output_dir(run_metadata: dict[str, Any]) -> Path:
    return frequency_tracking_run_output_dir(run_metadata)
