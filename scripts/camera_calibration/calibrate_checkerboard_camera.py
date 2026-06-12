#!/usr/bin/env python3

'''
this file is used to calibrate the camera using a checkerboard
'''

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import yaml
except ImportError:  #yaml sometimes does not import
    yaml = None


# yaml/cli helpers
# command line values can override what is in the metadata file
def load_yaml(path: Path) -> dict[str, Any]:
    # load checkerboard settings from yaml
    if yaml is None:
        raise RuntimeError("PyYAML is required when using --metadata.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Metadata file must contain a YAML mapping: {path}")
    return data


def resolve_path(path_value: str | Path, root: Path) -> Path:
    # expand the path if it is relative
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    # jump to a frame in the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    if not ok:
        return None
    return frame_bgr


def bgr_to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    # checkerboard detection only needs intensity
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


# known 3d checkerboard corner coordinates, in the board coordinate system
# z=0 because the checkerboard is flat
def checkerboard_object_points(
    inner_corners_x: int,
    inner_corners_y: int,
    square_size_mm: float,
) -> np.ndarray:
    """Known checkerboard corner positions in the board coordinate system."""
    # one row per inner corner: x,y,z
    objp = np.zeros((inner_corners_x * inner_corners_y, 3), np.float32)
    # integer checkerboard grid, then multiply by square size to get meters
    objp[:, :2] = np.mgrid[0:inner_corners_x, 0:inner_corners_y].T.reshape(-1, 2)
    objp *= float(square_size_mm) * 1e-3
    return objp


def find_checkerboard(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None, str]:
    # try the newer SB detector first, it tends to handle blur/perspective better
    if hasattr(cv2, "findChessboardCornersSB"):
        flags_sb = 0
        for name in ("CALIB_CB_EXHAUSTIVE", "CALIB_CB_ACCURACY", "CALIB_CB_NORMALIZE_IMAGE"):
            if hasattr(cv2, name):
                flags_sb |= getattr(cv2, name)
        try:
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags_sb)
            if found:
                # corners are subpixel image coordinates in pixels
                return True, corners.astype(np.float32), "findChessboardCornersSB"
        except cv2.error:
            pass

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None, "none"

    # classic detector gives approximate corners, cornerSubPix refines them
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return True, corners.astype(np.float32), "findChessboardCorners+cornerSubPix"


def reprojection_errors(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    used_frame_indices: list[int],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    # reprojection error is the main quality check
    # project the known 3d corners back into pixels and compare with detections
    rows: list[dict[str, Any]] = []
    all_errors: list[float] = []

    for view_index, (objp, imgp, rvec, tvec) in enumerate(
        zip(objpoints, imgpoints, rvecs, tvecs)
    ):
        # rvec/tvec are the checkerboard pose for this one frame
        # projectPoints predicts where the corners should land in pixels
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        observed = imgp.reshape(-1, 2)
        projected = projected.reshape(-1, 2)
        error_px = np.linalg.norm(observed - projected, axis=1)
        rows.append(
            {
                "calibration_view_index": view_index,
                "frame_index": int(used_frame_indices[view_index]),
                "mean_error_px": float(np.mean(error_px)),
                "median_error_px": float(np.median(error_px)),
                "rms_error_px": float(math.sqrt(float(np.mean(error_px**2)))),
                "max_error_px": float(np.max(error_px)),
            }
        )
        all_errors.extend(float(x) for x in error_px)

    return rows, np.asarray(all_errors, dtype=float)


def video_info(video_path: Path) -> dict[str, float | int]:
    # basic video info, mainly image size and number of frames to sample
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open checkerboard video: {video_path}")
    try:
        return {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        }
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate OpenCV camera intrinsics from a checkerboard video."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Calibration metadata YAML. If provided, checkerboard settings are read from it.",
    )
    parser.add_argument("--video", type=Path, help="Checkerboard calibration video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for camera_intrinsics.json.",
    )
    parser.add_argument("--calibration-id", help="Identifier stored in camera_intrinsics.json.")
    # important: these are inner corners, not number of printed squares
    parser.add_argument("--inner-corners-x", type=int, help="Checkerboard inner corners along x.")
    parser.add_argument("--inner-corners-y", type=int, help="Checkerboard inner corners along y.")
    parser.add_argument("--square-size-mm", type=float, help="Actual checkerboard square size.")
    parser.add_argument("--max-frames-to-scan", type=int, help="Maximum sampled frames to inspect.")
    parser.add_argument("--max-accepted-frames", type=int, help="Stop after this many detections.")
    parser.add_argument(
        "--min-accepted-frames",
        type=int,
        default=8,
        help="Minimum detected views required for calibration.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        help="Sample every Nth frame. Overrides evenly spaced sampling.",
    )
    parser.add_argument(
        "--use-rational-model",
        action="store_true",
        help="Use OpenCV CALIB_RATIONAL_MODEL distortion coefficients.",
    )
    return parser.parse_args()


# merge cli values and metadata values into one config
def merged_config(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path.cwd().resolve()
    metadata: dict[str, Any] = {}
    if args.metadata:
        metadata_path = args.metadata.resolve()
        metadata = load_yaml(metadata_path)
        # if metadata is in inputs/, repo root is one folder above that
        project_root = metadata_path.parent.parent if metadata_path.parent.name == "inputs" else Path.cwd().resolve()

    checkerboard = metadata.get("checkerboard", {}) or {}
    camera = metadata.get("camera", {}) or {}

    video_value = args.video or checkerboard.get("video_path")
    if video_value is None:
        raise ValueError("Provide --video or checkerboard.video_path in --metadata.")

    output_dir_value = args.output_dir
    # default output folder is based on calibration_id
    calibration_id = args.calibration_id or metadata.get("calibration_id") or "camera_calibration"
    if output_dir_value is None:
        output_dir_value = Path("outputs") / "camera_calibration" / str(calibration_id)

    inner_x = args.inner_corners_x or checkerboard.get("inner_corners_x")
    inner_y = args.inner_corners_y or checkerboard.get("inner_corners_y")
    square_size = args.square_size_mm or checkerboard.get("square_size_mm_actual") or checkerboard.get(
        "square_size_mm"
    )
    missing = [
        # the physical facts we must know: grid size and square size
        name
        for name, value in (
            ("inner corners x", inner_x),
            ("inner corners y", inner_y),
            ("square size mm", square_size),
        )
        if value is None
    ]
    if missing:
        raise ValueError("Missing checkerboard setting(s): " + ", ".join(missing))

    return {
        "project_root": project_root,
        "metadata": metadata,
        "camera": camera,
        "calibration_id": str(calibration_id),
        "video_path": resolve_path(video_value, project_root),
        "output_dir": resolve_path(output_dir_value, project_root),
        "inner_corners_x": int(inner_x),
        "inner_corners_y": int(inner_y),
        "square_size_mm": float(square_size),
        "max_frames_to_scan": int(
            args.max_frames_to_scan or checkerboard.get("max_frames_to_scan", 300)
        ),
        "max_accepted_frames": int(
            args.max_accepted_frames or checkerboard.get("max_accepted_frames", 80)
        ),
        "min_accepted_frames": int(args.min_accepted_frames),
        "frame_stride": args.frame_stride,
        "use_rational_model": bool(
            args.use_rational_model or checkerboard.get("use_rational_distortion_model", False)
        ),
    }


# scan frames from the checkerboard video and keep the ones where corners are found
def scan_checkerboard_views(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    video_path = config["video_path"]
    info = video_info(video_path)
    frame_count = int(info["frame_count"])
    if frame_count <= 0:
        raise ValueError(f"Video reports no frames: {video_path}")

    max_scan = max(1, min(int(config["max_frames_to_scan"]), frame_count))
    frame_stride = config["frame_stride"]
    if frame_stride and frame_stride > 0:
        # either sample every Nth frame
        frame_indices = list(range(0, frame_count, int(frame_stride)))[:max_scan]
    else:
        # or sample evenly across the whole video
        frame_indices = np.linspace(0, frame_count - 1, num=max_scan, dtype=int).tolist()

    pattern_size = (int(config["inner_corners_x"]), int(config["inner_corners_y"]))
    accepted_frames: list[dict[str, Any]] = []
    detection_methods: dict[str, int] = {}
    read_failures = 0

    cap = cv2.VideoCapture(str(video_path))
    start = time.time()
    try:
        for frame_index in frame_indices:
            frame_bgr = read_frame_at(cap, int(frame_index))
            if frame_bgr is None:
                read_failures += 1
                continue

            gray = bgr_to_gray(frame_bgr)
            found, corners, method = find_checkerboard(gray, pattern_size)
            detection_methods[method] = detection_methods.get(method, 0) + 1

            if found and corners is not None:
                # store the 2d pixel corners for this frame
                # the matching 3d checkerboard points are reused later
                accepted_frames.append(
                    {
                        "frame_index": int(frame_index),
                        "corners": corners,
                        "method": method,
                    }
                )

            if len(accepted_frames) >= int(config["max_accepted_frames"]):
                break
    finally:
        cap.release()

    info["scan_elapsed_s"] = time.time() - start
    info["frames_scanned"] = len(frame_indices)
    info["read_failures"] = int(read_failures)
    info["detection_methods"] = detection_methods
    return accepted_frames, info


# run the actual opencv camera calibration
# known 3d board points + detected 2d pixel points -> K, distortion, poses
def calibrate(config: dict[str, Any], accepted_frames: list[dict[str, Any]], info: dict[str, Any]) -> dict[str, Any]:
    if len(accepted_frames) < int(config["min_accepted_frames"]):
        raise RuntimeError(
            f"Only {len(accepted_frames)} checkerboard detections were found; "
            f"minimum required is {config['min_accepted_frames']}."
        )

    objp = checkerboard_object_points(
        int(config["inner_corners_x"]),
        int(config["inner_corners_y"]),
        float(config["square_size_mm"]),
    )

    # same physical checkerboard points for every frame
    # only the detected image positions change between views
    objpoints = [objp.copy() for _ in accepted_frames]
    imgpoints = [item["corners"].copy() for item in accepted_frames]
    used_frame_indices = [int(item["frame_index"]) for item in accepted_frames]
    image_size = (int(info["width"]), int(info["height"]))

    flags = 0
    if bool(config["use_rational_model"]) and hasattr(cv2, "CALIB_RATIONAL_MODEL"):
        # rational model adds more radial distortion terms
        flags |= cv2.CALIB_RATIONAL_MODEL

    # core solve
    # camera_matrix K gives fx, fy, cx, cy
    # dist_coeffs gives lens distortion
    # rvecs/tvecs are the pose of the checkerboard in each accepted frame
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
        flags=flags,
    )

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        # alpha=0 crops to valid pixels after undistortion
        # this avoids weird undefined borders later
        camera_matrix,
        dist_coeffs,
        image_size,
        alpha=0,
        newImgSize=image_size,
    )

    error_rows, all_errors = reprojection_errors(
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
        used_frame_indices,
    )

    per_view_mean_errors = [float(row["mean_error_px"]) for row in error_rows]
    per_view_rms_errors = [float(row["rms_error_px"]) for row in error_rows]
    per_view_max_errors = [float(row["max_error_px"]) for row in error_rows]

    intrinsics = {
        # camera_intrinsics.json is the output used by later fsss scripts
        # keep the model, quality metrics and source information together
        "calibration_id": config["calibration_id"],
        "source_video": relpath(config["video_path"], config["project_root"]),
        "camera": config.get("camera", {}),
        "checkerboard": {
            "square_size_mm_actual": float(config["square_size_mm"]),
            "inner_corners_x": int(config["inner_corners_x"]),
            "inner_corners_y": int(config["inner_corners_y"]),
            "accepted_frames": int(len(accepted_frames)),
            "used_frame_indices": used_frame_indices,
            "detection_methods": dict(info.get("detection_methods", {})),
        },
        "image_size_px": {
            "width": int(image_size[0]),
            "height": int(image_size[1]),
        },
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.ravel().tolist(),
        "distortion_model": (
            "opencv_standard_rational" if bool(config["use_rational_model"]) else "opencv_standard"
        ),
        "new_camera_matrix_alpha0": new_camera_matrix.tolist(),
        "valid_roi_alpha0": [int(x) for x in roi],
        "calibration_quality": {
            "opencv_calibration_rms_px": float(rms),
            "mean_reprojection_error_px": float(np.mean(all_errors)),
            "median_reprojection_error_px": float(np.median(all_errors)),
            "max_reprojection_error_px": float(np.max(all_errors)),
            "mean_view_mean_reprojection_error_px": float(np.mean(per_view_mean_errors)),
            "max_view_mean_reprojection_error_px": float(np.max(per_view_mean_errors)),
            "mean_view_rms_reprojection_error_px": float(np.mean(per_view_rms_errors)),
            "max_view_rms_reprojection_error_px": float(np.max(per_view_rms_errors)),
            "max_view_max_reprojection_error_px": float(np.max(per_view_max_errors)),
            "frames_scanned": int(info.get("frames_scanned", 0)),
            "read_failures": int(info.get("read_failures", 0)),
            "scan_elapsed_s": float(info.get("scan_elapsed_s", 0.0)),
        },
        "opencv_calibration_rms": float(rms),
        "mean_reprojection_error_px": float(np.mean(all_errors)),
        "median_reprojection_error_px": float(np.median(all_errors)),
        "max_reprojection_error_px": float(np.max(all_errors)),
        "notes": (
            "Use these intrinsics only with the same camera, lens, video mode, focus state, "
            "orientation, and image size. Recalibrate if any of those change."
        ),
    }

    return {"intrinsics": intrinsics}


def main() -> int:
    args = parse_args()
    try:
        config = merged_config(args)
        output_dir = config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Checkerboard video: {config['video_path']}")
        print(f"Output directory: {output_dir}")
        print(
            "Checkerboard inner corners: "
            f"{config['inner_corners_x']} x {config['inner_corners_y']}"
        )
        print(f"Square size: {config['square_size_mm']} mm")

        # first find usable checkerboard views
        # then call calibrateCamera on the 2d/3d correspondences
        accepted_frames, info = scan_checkerboard_views(config)
        print(
            f"Detected checkerboard in {len(accepted_frames)} of "
            f"{info['frames_scanned']} scanned frames."
        )

        result = calibrate(config, accepted_frames, info)
        intrinsics_path = output_dir / "camera_intrinsics.json"
        # this json is what the fsss calibration scripts read later
        with intrinsics_path.open("w", encoding="utf-8") as f:
            json.dump(result["intrinsics"], f, indent=2)

        intrinsics = result["intrinsics"]
        print(f"OpenCV calibration RMS: {intrinsics['opencv_calibration_rms']:.4f} px")
        print(f"Mean reprojection error: {intrinsics['mean_reprojection_error_px']:.4f} px")
        print(f"Median reprojection error: {intrinsics['median_reprojection_error_px']:.4f} px")
        print(f"Max reprojection error: {intrinsics['max_reprojection_error_px']:.4f} px")
        print(f"Saved intrinsics: {intrinsics_path}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
