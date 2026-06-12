#!/usr/bin/env python3
from __future__ import annotations

# this builds the dry dot-grid reference
# then gives dots metric coordinates and estimates camera pose relative to the grid

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


# repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# add dot-tracking imports so this can run directly from the command line
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DOT_TRACKING_DIR = PROJECT_ROOT / "scripts" / "dot_tracking"
if str(DOT_TRACKING_DIR) not in sys.path:
    sys.path.insert(0, str(DOT_TRACKING_DIR))

from common import (  # noqa: E402
    CALIBRATION_METADATA_PATH,
    load_yaml,
    median_reference_frame,
    project_path,
    relpath,
    video_metadata,
    write_json,
)
from dot_lattice import (  # noqa: E402
    Roi,
    RotatedRoi,
    build_flat_lattice,
)

DEFAULT_MAX_PNP_POINTS = 2500
DEFAULT_RANDOM_SEED = 42


# cli for dry-grid pose calibration
# this is where the printed dot plane becomes a metric camera-pose reference
def parse_args() -> argparse.Namespace:
    # keep the public cli small
    # pnp point limits and random seed are fixed so the calibration is repeatable
    parser = argparse.ArgumentParser(
        description=(
            "Build the dry dot-grid reference and estimate the camera pose used by full FSSS."
        )
    )
    parser.add_argument("--metadata", default=str(CALIBRATION_METADATA_PATH), help="Calibration metadata YAML.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to outputs/full_fsss/calibration/<calibration_id>.",
    )
    return parser.parse_args()


def calibration_output_dir(metadata: dict[str, Any], override: str | None) -> Path:
    # explicit cli output wins; otherwise write into the standard calibration-id
    # folder used by later fsss scripts.
    configured = project_path(override)
    if configured is not None:
        return configured
    return PROJECT_ROOT / "outputs" / "full_fsss" / "calibration" / str(metadata["calibration_id"])


def fsss_intrinsics_path(metadata: dict[str, Any], output_dir: Path) -> Path:
    # prefer an explicit camera.intrinsics_path. Otherwise use a copy in the
    # fsss output folder, then the standard camera_calibration output.
    camera = metadata.get("camera", {}) or {}
    configured = project_path(camera.get("intrinsics_path"))
    if configured is not None:
        return configured
    output_copy = output_dir / "camera_intrinsics.json"
    if output_copy.exists():
        return output_copy
    return PROJECT_ROOT / "outputs" / "camera_calibration" / str(metadata["calibration_id"]) / "camera_intrinsics.json"


# load camera intrinsics from metadata or camera_intrinsics.json
def load_fsss_intrinsics(metadata: dict[str, Any], output_dir: Path) -> tuple[Path, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    camera = metadata.get("camera", {}) or {}
    inline_matrix = camera.get("camera_matrix")
    if inline_matrix not in (None, ""):
        # inline mode: metadata itself contains the camera matrix and distortion.
        intrinsics = {
            "camera_matrix": inline_matrix,
            "distortion_coefficients": camera.get("distortion_coefficients", camera.get("dist_coeffs", [])) or [],
            "new_camera_matrix_alpha0": camera.get("new_camera_matrix_alpha0", camera.get("new_camera_matrix", inline_matrix)),
        }
        path = Path("<metadata camera matrix>")
    else:
        path = fsss_intrinsics_path(metadata, output_dir)
        if not path.exists():
            raise FileNotFoundError(f"Missing camera intrinsics: {path}")
        intrinsics = json.loads(path.read_text(encoding="utf-8"))

    k_raw = np.array(intrinsics["camera_matrix"], dtype=float)
    # distortion coefficients are column-shaped because opencv calibration
    # functions commonly expect that shape.
    dist = np.array(
        intrinsics.get("distortion_coefficients", intrinsics.get("dist_coeffs", [])),
        dtype=float,
    ).reshape(-1, 1)
    if dist.size == 0:
        dist = np.zeros((5, 1), dtype=float)
    k_undist = np.array(intrinsics.get("new_camera_matrix_alpha0", intrinsics["camera_matrix"]), dtype=float)
    return path, intrinsics, k_raw, dist, k_undist


# fsss calibration can intentionally skip lens undistortion for newer runs where
# the checkerboard calibration was unreliable. This function centralizes that
# metadata decision.
def use_full_fsss_lens_undistortion(metadata: dict[str, Any]) -> bool:
    # several metadata flags can disable full-fsss undistortion. When disabled,
    # every calibration/tracking product must remain in the same raw pixel space.
    image_processing = metadata.get("image_processing", {}) or {}
    camera = metadata.get("camera", {}) or {}
    raytrace = metadata.get("raytrace", {}) or {}
    return not (
        bool(image_processing.get("skip_lens_undistortion_for_full_fsss", False))
        or bool(camera.get("skip_lens_undistortion", False))
        or raytrace.get("include_lens_distortion") is False
    )


def undistort_for_full_fsss(
    gray: np.ndarray,
    metadata: dict[str, Any],
    k_raw: np.ndarray,
    dist: np.ndarray,
    k_undist: np.ndarray,
) -> np.ndarray:
    if not use_full_fsss_lens_undistortion(metadata):
        # keep raw grayscale pixels when undistortion was disabled in metadata.
        return gray
    # opencv remaps through camera matrix + distortion into the chosen undistorted
    # camera matrix.
    return cv2.undistort(gray, k_raw, dist, None, k_undist)


def camera_matrix_for_processed_pixels(metadata: dict[str, Any], k_raw: np.ndarray, k_undist: np.ndarray) -> np.ndarray:
    # solvePnP must use the camera matrix corresponding to the actual pixel
    # images used for dot detection.
    return k_undist if use_full_fsss_lens_undistortion(metadata) else k_raw


# dry reference video selection: this is the video of the printed dot grid
# without the flat liquid reference distortion.
def dry_video_path(metadata: dict[str, Any]) -> Path:
    dot_grid = metadata.get("dot_grid", {}) or {}
    videos = metadata.get("videos", {}) or {}
    path = project_path(dot_grid.get("dry_video_path") or videos.get("dry_grid"))
    if path is None:
        raise ValueError("Missing dry dot-grid video. Set dot_grid.dry_video_path or videos.dry_grid.")
    if not path.exists():
        raise FileNotFoundError(f"Missing dry dot-grid video: {path}")
    return path


# roi comes from calibration metadata
# streamlit roi selector is the normal way to edit this first
def roi_from_metadata(metadata: dict[str, Any], image_shape: tuple[int, int]) -> Roi | RotatedRoi:
    # the dry-grid reference uses the ROI selected by the Streamlit ROI app.
    geometry_initial = metadata.get("geometry_initial_estimates", {}) or {}
    geometry = metadata.get("geometry", {}) or {}
    rotated = geometry_initial.get("usable_roi_rotated_px", geometry.get("rotated_roi_px"))
    if isinstance(rotated, dict):
        return RotatedRoi.from_dict(rotated).clamped(image_shape)
    if isinstance(rotated, list) and len(rotated) == 5:
        return RotatedRoi.from_list(rotated).clamped(image_shape)

    axis = geometry_initial.get("usable_roi_px", geometry.get("roi_px"))
    if isinstance(axis, list) and len(axis) >= 4:
        return Roi.from_list(axis[:4]).clamped(image_shape)

    height, width = image_shape[:2]
    # if no ROI exists, fall back to the full image so the user gets an output
    # or a later explicit detection error.
    return Roi(0, 0, width, height)


def dot_spacing_mm(metadata: dict[str, Any]) -> float:
    # physical spacing between printed dot centers. This turns integer indices
    # into known object coordinates.
    dot_grid = metadata.get("dot_grid", {}) or {}
    geometry = metadata.get("geometry", {}) or {}
    return float(dot_grid.get("dot_spacing_mm_actual", dot_grid.get("dot_spacing_mm", geometry.get("dot_spacing_mm", 1.0))))


# center the integer lattice coordinates and convert them to metres. These
# object points are what solvePnP uses as the known 3D grid coordinates.
def add_centered_metric_coordinates(dots: pd.DataFrame, spacing_mm: float) -> pd.DataFrame:
    out = dots.copy()
    # the graph indexing gives arbitrary i,j origin. Centering puts the world
    # coordinate origin near the middle of the selected grid.
    i_center = 0.5 * (float(out["i"].min()) + float(out["i"].max()))
    j_center = 0.5 * (float(out["j"].min()) + float(out["j"].max()))
    out["x_grid_mm"] = (out["i"].astype(float) - i_center) * float(spacing_mm)
    out["y_grid_mm"] = (out["j"].astype(float) - j_center) * float(spacing_mm)
    out["x_grid_m"] = out["x_grid_mm"] * 1e-3
    out["y_grid_m"] = out["y_grid_mm"] * 1e-3
    out["z_grid_m"] = 0.0
    residual = out.get("lattice_residual_px")
    if residual is not None:
        # quality weight is not used by solvePnP here, but it records which dots
        # fit the lattice better.
        out["quality"] = 1.0 / (1.0 + residual.astype(float))
    return out


# camera pose solve: map known dot-plane object points (x, y, z=0) onto their
# detected image pixels using cv2.solvePnP, then compute reprojection errors for
# every indexed dot.
def solve_camera_pose(
    dots: pd.DataFrame,
    k_image: np.ndarray,
    *,
    flip_y: bool,
    max_points: int,
    random_seed: int,
) -> dict[str, Any]:
    # object points are known 3D positions on the printed dot plane.
    obj = dots[["x_grid_m", "y_grid_m", "z_grid_m"]].to_numpy(np.float64)
    if flip_y:
        # the image-based lattice can have either y orientation; try both.
        obj = obj.copy()
        obj[:, 1] *= -1.0
    img = dots[["x_px", "y_px"]].to_numpy(np.float64)

    # use a reproducible random subset if there are too many dots for PnP.
    rng = np.random.default_rng(int(random_seed))
    if len(dots) > int(max_points):
        idx = rng.choice(len(dots), size=int(max_points), replace=False)
        obj_fit = obj[idx]
        img_fit = img[idx]
    else:
        obj_fit = obj
        img_fit = img

    dist_zero = np.zeros((5, 1), dtype=np.float64)
    # solvePnP estimates rvec/tvec such that object points project to image
    # pixels under the camera matrix.
    ok, rvec, tvec = cv2.solvePnP(
        obj_fit,
        img_fit,
        k_image.astype(np.float64),
        dist_zero,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed.")
    if hasattr(cv2, "solvePnPRefineLM"):
        # Levenberg-Marquardt refinement usually improves the reprojection error.
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_fit,
            img_fit,
            k_image.astype(np.float64),
            dist_zero,
            rvec,
            tvec,
        )

    # Rodrigues converts rotation vector to a 3x3 rotation matrix.
    rotation, _ = cv2.Rodrigues(rvec)
    # camera center in world coordinates is -R^T t.
    camera_center = (-rotation.T @ tvec.reshape(3, 1)).reshape(3)
    # reproject all dots, not only the subset used for fitting, to evaluate pose
    # quality over the full grid.
    projected, _ = cv2.projectPoints(obj, rvec, tvec, k_image.astype(np.float64), dist_zero)
    projected = projected.reshape(-1, 2)
    error = np.linalg.norm(projected - img, axis=1)
    return {
        "flip_y": bool(flip_y),
        "rvec": rvec.reshape(3),
        "tvec": tvec.reshape(3),
        "R": rotation,
        "camera_center_world_m": camera_center,
        "projected_px": projected,
        "reprojection_error_px": error,
    }


# the indexed grid has an arbitrary y-axis orientation, so solve both possible
# orientations and keep the pose with the camera on the positive-z side and low
# reprojection error.
def choose_pose(dots: pd.DataFrame, k_image: np.ndarray, max_points: int, random_seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    # because the lattice y-axis can be flipped without changing the printed
    # pattern, solve both and choose the physically plausible camera-above-plane
    # pose.
    pose_a = solve_camera_pose(dots, k_image, flip_y=False, max_points=max_points, random_seed=random_seed)
    pose_b = solve_camera_pose(dots, k_image, flip_y=True, max_points=max_points, random_seed=random_seed)
    candidates = [pose_a, pose_b]
    valid = [pose for pose in candidates if float(pose["camera_center_world_m"][2]) > 0.0]
    if valid:
        chosen = min(valid, key=lambda pose: float(np.median(pose["reprojection_error_px"])))
    else:
        warnings.warn("Neither PnP orientation put the camera above z=0; choosing the lower reprojection error.")
        chosen = min(candidates, key=lambda pose: float(np.median(pose["reprojection_error_px"])))

    if chosen["flip_y"]:
        # if flipped orientation wins, permanently flip y coordinates and solve
        # again so the saved pose has flip_y=False.
        dots = dots.copy()
        dots["y_grid_m"] *= -1.0
        dots["y_grid_mm"] *= -1.0
        chosen = solve_camera_pose(dots, k_image, flip_y=False, max_points=max_points, random_seed=random_seed)
    return dots, chosen


def main() -> int:
    args = parse_args()
    # load calibration metadata and create output directory for this calibration
    # id.
    metadata_path = project_path(args.metadata)
    if metadata_path is None or not metadata_path.exists():
        raise FileNotFoundError(f"Missing calibration metadata: {args.metadata}")
    metadata = load_yaml(metadata_path)
    output_dir = calibration_output_dir(metadata, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsics_path, _, k_raw, dist, k_undist = load_fsss_intrinsics(metadata, output_dir)
    k_image = camera_matrix_for_processed_pixels(metadata, k_raw, k_undist)
    video_path = dry_video_path(metadata)
    tracking_cfg = metadata.get("tracking", {}) or {}
    max_frames = int(tracking_cfg.get("max_reference_frames", 80))
    stride = int(tracking_cfg.get("reference_stride", 5))

    # median dry-reference image construction suppresses transient image noise
    # while preserving the stationary printed dot grid used for indexing.
    raw_gray = median_reference_frame(video_path, max_frames=max_frames, stride=stride)
    gray = undistort_for_full_fsss(raw_gray, metadata, k_raw, dist, k_undist)
    # roi is clamped against the processed image dimensions.
    roi = roi_from_metadata(metadata, gray.shape)
    spacing_mm = dot_spacing_mm(metadata)

    # reuse the public dark-dot detector and local graph lattice indexing, then
    # convert integer grid indices into centered metric object coordinates.
    dots, lattice, _binary, _signal_roi = build_flat_lattice(gray, roi, dot_spacing_mm=spacing_mm)
    dots = add_centered_metric_coordinates(dots, spacing_mm)
    dots = dots.sort_values(["j", "i"]).reset_index(drop=True)
    dots["dot_id"] = np.arange(len(dots), dtype=int)

    # estimate the camera pose from object-plane dot coordinates to detected
    # pixels. The resulting pose anchors all later fsss ray tracing.
    dots, pose = choose_pose(dots, k_image, DEFAULT_MAX_PNP_POINTS, DEFAULT_RANDOM_SEED)

    dots.to_csv(output_dir / "dot_grid_reference.csv", index=False)

    errors = np.asarray(pose["reprojection_error_px"], dtype=float)

    # what this writes:
    # dot_grid_reference.csv stores dot identities and object coordinates
    # camera_pose_grid.json stores the camera pose and basic fit quality
    pose_out = {
        "calibration_id": str(metadata["calibration_id"]),
        "coordinate_system": {
            "description": (
                "World/grid coordinates are attached to the dot plane. z=0 is the printed dot plane. "
                "The origin is the center of the indexed dot grid. z is chosen so the camera center has positive z."
            ),
            "units": "metres",
        },
        "camera_matrix_used_for_undistorted_pixels": k_image.tolist(),
        "lens_undistortion_applied": bool(use_full_fsss_lens_undistortion(metadata)),
        "distortion_assumed_after_undistortion": [0, 0, 0, 0, 0],
        "rvec_world_to_camera": np.asarray(pose["rvec"], dtype=float).tolist(),
        "tvec_world_to_camera_m": np.asarray(pose["tvec"], dtype=float).tolist(),
        "R_world_to_camera": np.asarray(pose["R"], dtype=float).tolist(),
        "camera_center_world_m": np.asarray(pose["camera_center_world_m"], dtype=float).tolist(),
        "median_reprojection_error_px": float(np.median(errors)),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "max_reprojection_error_px": float(np.max(errors)),
        "n_points": int(len(dots)),
        "dot_spacing_mm_actual": float(spacing_mm),
        "estimated_dot_spacing_px": float(lattice["dot_spacing_px_median_nn"]),
        "roi_px": roi.as_list(),
        "source_files": {
            "metadata": relpath(metadata_path),
            "intrinsics": relpath(intrinsics_path) if str(intrinsics_path) != "<metadata camera matrix>" else str(intrinsics_path),
            "dry_video": relpath(video_path),
        },
        "dry_reference_state": str((metadata.get("dot_grid", {}) or {}).get("dry_reference_state", "dot_grid_reference_as_recorded")),
        "dry_reference_note": str((metadata.get("dot_grid", {}) or {}).get("dry_reference_note", "")),
    }
    write_json(output_dir / "camera_pose_grid.json", pose_out)

    print(f"Indexed {len(dots)} dry-grid dots.")
    print(f"Median PnP reprojection error: {np.median(errors):.4f} px")
    print(f"Saved {relpath(output_dir / 'dot_grid_reference.csv')}")
    print(f"Saved {relpath(output_dir / 'camera_pose_grid.json')}")
    print("Dry video metadata:", video_metadata(video_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
