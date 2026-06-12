#!/usr/bin/env python3
from __future__ import annotations

# this tracks calibrated fsss dots in run videos
# then converts pixel tracks to object-plane metric displacement for reconstruction

import argparse
import json
import math
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
# this imports scripts/path_utils.py and scripts/dot_tracking modules
# so add the scripts directory first
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from path_utils import PROJECT_ROOT, add_script_imports, project_path, relpath  # noqa: E402

add_script_imports("dot_tracking")

from common import (  # noqa: E402
    bgr_to_gray,
    load_yaml,
    summarize_timebase,
    video_metadata,
    video_times,
    write_json,
)
from dot_lattice import (  # noqa: E402
    Roi,
    RotatedRoi,
    dot_signal,
    track_frame_templates,
)
from track_frequency_dots import (  # noqa: E402
    TrackingProfile,
    dot_survival_counts,
    estimate_phase_translation,
    indexed_neighbor_pairs,
    profile_pixels,
    quality_summary_value,
    sampled_frame_indices,
    tracked_neighbor_geometry_metrics,
)


# fsss tracking keeps the public cli narrow
# these constants are the chosen recipe, not user-facing tuning knobs
DEFAULT_TEMPLATE_RADIUS_FRACTION = 0.25
DEFAULT_SEARCH_RADIUS_FRACTION = 0.42
DEFAULT_MAX_DISPLACEMENT_FRACTION = 0.48
DEFAULT_MIN_MATCH_SCORE = 0.35
DEFAULT_PREFLIGHT_FRAMES = 180
DEFAULT_MAX_OBJECT_DISPLACEMENT_MM = 1.5
DEFAULT_NEIGHBOR_K = 8
DEFAULT_NEIGHBOR_RESIDUAL_THRESHOLD_MM = 0.30


def parse_args() -> argparse.Namespace:
    # public cli stays narrow:
    # metadata paths, frame selection, output overrides, optional adaptive selection
    parser = argparse.ArgumentParser(
        description=(
            "Robust metric FSSS dot tracking. This keeps calibrated FSSS dot identities "
            "from flat_liquid_dots.csv, but uses the adaptive frequency-style template tracker."
        )
    )
    parser.add_argument("run_metadata", help="metadata_run.yaml for the stable run.")
    parser.add_argument(
        "--calibration-metadata",
        default=None,
        help="Calibration metadata YAML. Defaults to run_metadata.calibration_metadata_path.",
    )
    parser.add_argument(
        "--calibration-output-dir",
        default=None,
        help="Calibration output dir. Defaults to run metadata or outputs/full_fsss/calibration/<calibration_id>.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run output dir. Defaults to run_metadata.run_output_dir.",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Override run_metadata.video_path. Useful when large videos were archived elsewhere.",
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument(
        "--adaptive-tracking",
        action="store_true",
        help="Preflight several search profiles and use the strictest acceptable one.",
    )
    return parser.parse_args()


# manual and adaptive profiles reuse the frequency tracker machinery
# fsss keeps the chosen radii/thresholds fixed so the pipeline stays repeatable
def manual_profile() -> TrackingProfile:
    # default fsss profile uses fixed constants above, not cli tuning
    return TrackingProfile(
        name="manual",
        template_radius_fraction=DEFAULT_TEMPLATE_RADIUS_FRACTION,
        search_radius_fraction=DEFAULT_SEARCH_RADIUS_FRACTION,
        max_displacement_fraction=DEFAULT_MAX_DISPLACEMENT_FRACTION,
        min_match_score=DEFAULT_MIN_MATCH_SCORE,
        global_prereg=False,
        predict_from_previous=False,
        keep_affine_outliers=False,
    )


def fsss_adaptive_profiles() -> list[TrackingProfile]:
    # same profile names as frequency tracking, but fsss-specific defaults
    # ordered from stricter to wider search
    return [
        TrackingProfile("strict_prereg", DEFAULT_TEMPLATE_RADIUS_FRACTION, 0.42, 0.48, 0.35, True, True, True),
        TrackingProfile("medium_prereg", DEFAULT_TEMPLATE_RADIUS_FRACTION, 0.65, 0.75, 0.25, True, True, True),
        TrackingProfile("wide_prereg", DEFAULT_TEMPLATE_RADIUS_FRACTION, 0.90, 1.20, 0.15, True, True, True),
        TrackingProfile("very_wide_prereg", DEFAULT_TEMPLATE_RADIUS_FRACTION, 1.15, 1.50, 0.10, True, True, True),
    ]


# camera intrinsics are only used if lens undistortion is enabled
# the saved fsss arrays always stay in the same image space used by flat-reference calibration
def load_intrinsics(
    calibration_meta: dict[str, Any],
    calibration_output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_meta = calibration_meta.get("camera", {})
    configured_path = project_path(camera_meta.get("intrinsics_path"))
    # prefer explicit intrinsics_path, otherwise use the calibration output dir
    path = configured_path if configured_path is not None else calibration_output_dir / "camera_intrinsics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing camera intrinsics: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    # K, distortion coefficients, and undistorted K mirror opencv calibration output
    k_raw = np.array(data["camera_matrix"], dtype=float)
    dist = np.array(
        data.get("dist_coeffs", data.get("distortion_coefficients", [])),
        dtype=float,
    ).reshape(-1, 1)
    k_undist = np.array(data.get("new_camera_matrix_alpha0", data["camera_matrix"]), dtype=float)
    return k_raw, dist, k_undist


def undistort_gray_metric(
    gray: np.ndarray,
    calibration_meta: dict[str, Any],
    calibration_output_dir: Path,
) -> np.ndarray:
    image_processing = calibration_meta.get("image_processing", {})
    if image_processing.get("skip_lens_undistortion_for_full_fsss", False):
        # latest runs may deliberately skip undistortion if calibration was unstable
        # in that case all calibration products must stay in the same raw image space
        return gray
    k_raw, dist, k_undist = load_intrinsics(calibration_meta, calibration_output_dir)
    return cv2.undistort(gray, k_raw, dist, None, k_undist)


# pixel-to-object homography
# ray-trace calibration gives a flat reference mapping from pixels to object-plane mm
# tracking is done in pixels, then this converts dot positions into meters
def apply_homography(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    # homographies operate on homogeneous coordinates [x,y,1]
    # this helper preserves whatever leading shape the input had
    xy = np.asarray(xy, dtype=float)
    original_shape = xy.shape
    xy2 = xy.reshape(-1, 2)
    hom = np.column_stack([xy2, np.ones(len(xy2))])
    out = (H @ hom.T).T
    # divide by homogeneous scale coordinate to return cartesian x,y
    out = out[:, :2] / out[:, 2:3]
    return out.reshape(original_shape)


def pixel_to_object_xy_m(H_pixel_to_object_mm: np.ndarray, uv_px: np.ndarray) -> np.ndarray:
    # calibration homography returns millimetres
    # fsss reconstruction uses SI units, so multiply by 1e-3
    return apply_homography(H_pixel_to_object_mm, np.asarray(uv_px, dtype=float)) * 1e-3


def roi_from_calibration_metadata(
    calibration_meta: dict[str, Any],
    reference_dots: pd.DataFrame,
    image_shape: tuple[int, int],
) -> Roi | RotatedRoi:
    # prefer roi saved in calibration metadata
    # if missing, build a fallback around the calibrated dot table
    geom = calibration_meta.get("geometry_initial_estimates", {})
    rotated = geom.get("usable_roi_rotated_px")
    if isinstance(rotated, dict):
        # rotated roi form
        return RotatedRoi.from_dict(rotated).clamped(image_shape)
    if isinstance(rotated, (list, tuple)) and len(rotated) == 5:
        return RotatedRoi.from_list(list(rotated)).clamped(image_shape)
    axis = geom.get("usable_roi_px")
    if isinstance(axis, (list, tuple)) and len(axis) >= 4:
        # axis-aligned roi form
        return Roi.from_list(list(axis[:4])).clamped(image_shape)

    # fallback roi: bounding box of reference dots plus two grid spacings
    x = reference_dots["x_px"].to_numpy(float)
    y = reference_dots["y_px"].to_numpy(float)
    margin = 2.0 * estimate_spacing_px(reference_dots[["x_px", "y_px"]].to_numpy(float))
    if not np.isfinite(margin):
        margin = 20.0
    x0 = int(math.floor(np.nanmin(x) - margin))
    y0 = int(math.floor(np.nanmin(y) - margin))
    x1 = int(math.ceil(np.nanmax(x) + margin))
    y1 = int(math.ceil(np.nanmax(y) + margin))
    return Roi(x0, y0, x1 - x0, y1 - y0).clamped(image_shape)


def estimate_spacing_px(points: np.ndarray) -> float:
    # same nearest-neighbor median spacing estimate used elsewhere
    # kept local here to avoid importing extra lattice internals
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 4:
        return float("nan")
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    nearest = dists[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    return float(np.median(nearest)) if len(nearest) else float("nan")


# fsss calibrated dot identity loading
# keep dot ids and lattice coordinates from calibration so every tracked displacement
# maps back to a known object-plane grid point
def select_dot_subset(
    flat_dots: pd.DataFrame,
    *,
    central_fraction: float,
    max_dots: int | None,
    random_seed: int,
) -> pd.DataFrame:
    # validate calibration dot table schema
    # these columns define identity, lattice coordinates, metric grid coordinates,
    # and reference pixels
    required = {"dot_id", "i", "j", "x_grid_m", "y_grid_m", "x_px", "y_px"}
    missing = required - set(flat_dots.columns)
    if missing:
        raise ValueError(f"flat_liquid_dots.csv is missing required columns: {missing}")

    dots = flat_dots.copy()
    if float(central_fraction) < 1.0:
        # optional central crop in object coordinates
        # public defaults keep all dots, but this is useful for controlled tests
        x = dots["x_grid_m"].to_numpy(float)
        y = dots["y_grid_m"].to_numpy(float)
        x0 = float(np.nanmedian(x))
        y0 = float(np.nanmedian(y))
        rx = (float(np.nanmax(x)) - float(np.nanmin(x))) * float(central_fraction) / 2.0
        ry = (float(np.nanmax(y)) - float(np.nanmin(y))) * float(central_fraction) / 2.0
        dots = dots[
            (np.abs(dots["x_grid_m"] - x0) <= rx)
            & (np.abs(dots["y_grid_m"] - y0) <= ry)
        ].copy()
    if max_dots is not None and len(dots) > int(max_dots):
        # optional reproducible random subset
        dots = dots.sample(int(max_dots), random_state=int(random_seed)).copy()
    return dots.sort_values("dot_id").reset_index(drop=True)


# affine subtraction in metric coordinates
# after converting pixels to object-plane meters, remove frame-wide affine motion
# before using residual displacement as surface-deflection information
def fit_affine_displacement_metric(
    xy_ref_m: np.ndarray,
    obj_tracked_m: np.ndarray,
    dxy_raw_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    ransac_threshold_m: float,
    reject_outliers: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # corrected stores residual metric displacement after global affine motion is removed
    corrected = np.full_like(dxy_raw_m, np.nan, dtype=float)
    # good selects dots valid and finite in reference coordinates, tracked object
    # coordinates, and raw displacement
    good = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(xy_ref_m).all(axis=1)
        & np.isfinite(obj_tracked_m).all(axis=1)
        & np.isfinite(dxy_raw_m).all(axis=1)
    )
    final_valid = np.asarray(valid_mask, dtype=bool).copy()
    if int(np.sum(good)) < 6:
        # not enough dots for a stable affine fit, so keep raw metric displacement
        corrected[good] = dxy_raw_m[good]
        return corrected, np.full((2, 3), np.nan, dtype=float), final_valid

    # fit an affine mapping from reference object coordinates to tracked object
    # coordinates using RANSAC
    affine, inliers = cv2.estimateAffinePartial2D(
        xy_ref_m[good].astype(np.float32),
        obj_tracked_m[good].astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_m),
        maxIters=2000,
        confidence=0.995,
    )
    if affine is None:
        # fallback: least-squares model for the displacement field directly:
        # dx = a0*x + a1*y + a2, dy = b0*x + b1*y + b2.
        X = np.column_stack([xy_ref_m[good, 0], xy_ref_m[good, 1], np.ones(int(np.sum(good)))])
        beta_x, *_ = np.linalg.lstsq(X, dxy_raw_m[good, 0], rcond=None)
        beta_y, *_ = np.linalg.lstsq(X, dxy_raw_m[good, 1], rcond=None)
        coeff = np.vstack([beta_x, beta_y])
        pred_dxy = np.column_stack([xy_ref_m[:, 0], xy_ref_m[:, 1], np.ones(len(xy_ref_m))]) @ coeff.T
        corrected[final_valid] = dxy_raw_m[final_valid] - pred_dxy[final_valid]
        return corrected, coeff, final_valid

    # predict where every reference dot would land under frame-wide affine motion
    # then subtract that from the measured tracked object coordinate
    predicted_obj = np.column_stack([xy_ref_m, np.ones(len(xy_ref_m))]) @ affine.T
    corrected[final_valid] = obj_tracked_m[final_valid] - predicted_obj[final_valid]
    if reject_outliers and inliers is not None:
        # inliers are returned only for the good subset
        # map rejected entries back into full dot index space
        good_indices = np.flatnonzero(good)
        rejected = good_indices[inliers.ravel().astype(bool) == 0]
        final_valid[rejected] = False
        corrected[rejected] = np.nan
    return corrected, affine.astype(float), final_valid


# neighbor residual filtering
# remove isolated dots whose metric displacement disagrees with nearby calibrated dots
# this is a local consistency check after template matching and affine subtraction
def neighbor_vector_filter(
    xy_ref_m: np.ndarray,
    dxy_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    k: int,
    residual_threshold_m: float,
) -> np.ndarray:
    # start from current validity mask
    # this only invalidates more dots, never revalidates a dot
    valid = np.asarray(valid_mask, dtype=bool).copy()
    good_idx = np.where(valid & np.isfinite(dxy_m).all(axis=1))[0]
    if len(good_idx) < int(k) + 2:
        return valid

    # build a kd-tree in reference-object coordinates, not image pixels
    tree = cKDTree(xy_ref_m[good_idx])
    _, neigh = tree.query(xy_ref_m[good_idx], k=min(int(k) + 1, len(good_idx)))
    bad: list[int] = []
    for local_i, neigh_local in enumerate(neigh):
        this_global = int(good_idx[local_i])
        neigh_global = good_idx[neigh_local]
        neigh_global = neigh_global[neigh_global != this_global]
        if len(neigh_global) < 3:
            continue
        # compare this dot's displacement vector to the median vector nearby
        # a large residual usually means a bad template match
        med = np.nanmedian(dxy_m[neigh_global], axis=0)
        resid = float(np.linalg.norm(dxy_m[this_global] - med))
        if resid > float(residual_threshold_m):
            bad.append(this_global)
    if bad:
        valid[np.asarray(bad, dtype=int)] = False
    return valid


def tracking_quality_row(
    *,
    frame_index: int,
    time_s: float,
    valid_initial: np.ndarray,
    valid_after_affine: np.ndarray,
    valid_after_object: np.ndarray,
    valid_final: np.ndarray,
    score: np.ndarray,
    dxy_raw_m: np.ndarray,
    dxy_corr_m: np.ndarray,
    xy_px: np.ndarray,
    reference_dots: pd.DataFrame,
    neighbor_pairs: np.ndarray,
    spacing_px: float,
) -> dict[str, float | int]:
    # one row per frame for tracking_quality.csv
    # records how many dots survived each filtering stage and displacement size
    finite_scores = score[np.isfinite(score)]
    raw = dxy_raw_m[valid_final]
    corr = dxy_corr_m[valid_final]
    row: dict[str, float | int] = {
        "frame_index": int(frame_index),
        "time_s": float(time_s),
        "n_dots": int(len(valid_final)),
        "n_valid_initial": int(np.sum(valid_initial)),
        "n_valid_after_affine": int(np.sum(valid_after_affine)),
        "n_valid_after_object": int(np.sum(valid_after_object)),
        "n_valid": int(np.sum(valid_final)),
        "valid_fraction": float(np.mean(valid_final)) if len(valid_final) else 0.0,
        "median_match_score": float(np.median(finite_scores)) if len(finite_scores) else float("nan"),
        "p10_match_score": float(np.percentile(finite_scores, 10)) if len(finite_scores) else float("nan"),
        "rms_displacement_raw_m": float(np.sqrt(np.nanmean(np.sum(raw**2, axis=1)))) if len(raw) else float("nan"),
        "rms_displacement_corrected_m": float(np.sqrt(np.nanmean(np.sum(corr**2, axis=1)))) if len(corr) else float("nan"),
    }
    row["rms_displacement_raw_mm"] = (
        # duplicate displacement rms in mm for human-readable summaries
        float(row["rms_displacement_raw_m"]) * 1000.0
        if np.isfinite(float(row["rms_displacement_raw_m"]))
        else float("nan")
    )
    row["rms_displacement_corrected_mm"] = (
        float(row["rms_displacement_corrected_m"]) * 1000.0
        if np.isfinite(float(row["rms_displacement_corrected_m"]))
        else float("nan")
    )
    row.update(
        # reuse frequency tracker's neighbor geometry check in pixel space
        tracked_neighbor_geometry_metrics(
            xy_px,
            valid_final,
            neighbor_pairs,
            expected_spacing_px=spacing_px,
        )
    )
    return row


def run_tracking_pass(
    *,
    video_path: Path,
    frame_indices: np.ndarray,
    all_times: np.ndarray,
    calibration_meta: dict[str, Any],
    calibration_output_dir: Path,
    H_pixel_to_object_mm: np.ndarray,
    reference_signal: np.ndarray,
    reference_dots: pd.DataFrame,
    reference_xy_px: np.ndarray,
    reference_obj_m: np.ndarray,
    roi: Roi | RotatedRoi,
    spacing_px: float,
    profile: TrackingProfile,
    max_object_displacement_m: float,
    affine_threshold_m: float,
    remove_affine: bool,
    neighbor_filter_enabled: bool,
    neighbor_k: int,
    neighbor_residual_threshold_m: float,
    progress: bool,
) -> dict[str, Any]:
    # main fsss tracking loop
    # template matching happens in pixels
    # saved physical displacements are converted to the calibrated object plane
    pixels = profile_pixels(profile, spacing_px)
    template_radius = int(pixels["template_radius_px"])
    search_radius = int(pixels["search_radius_px"])
    max_displacement_px = float(pixels["max_displacement_px"])
    n_frames = len(frame_indices)
    n_dots = len(reference_dots)
    neighbor_pairs = indexed_neighbor_pairs(reference_dots)

    # pixel position/displacement arrays
    x_px_all = np.full((n_frames, n_dots), np.nan, dtype=np.float32)
    y_px_all = np.full((n_frames, n_dots), np.nan, dtype=np.float32)
    dx_px_all = np.full((n_frames, n_dots), np.nan, dtype=np.float32)
    dy_px_all = np.full((n_frames, n_dots), np.nan, dtype=np.float32)
    scores = np.full((n_frames, n_dots), np.nan, dtype=np.float32)
    valid_initial = np.zeros((n_frames, n_dots), dtype=bool)
    valid_final = np.zeros((n_frames, n_dots), dtype=bool)
    # metric displacement arrays in object-plane meters
    dx_obj_raw_all = np.full((n_frames, n_dots), np.nan, dtype=np.float32)
    dy_obj_raw_all = np.full((n_frames, n_dots), np.nan, dtype=np.float32)
    dx_obj_corr_all = np.full((n_frames, n_dots), np.nan, dtype=np.float32)
    dy_obj_corr_all = np.full((n_frames, n_dots), np.nan, dtype=np.float32)
    affine_coeff_all = np.full((n_frames, 2, 3), np.nan, dtype=np.float64)
    global_shifts = np.zeros((n_frames, 2), dtype=np.float32)
    global_shift_response = np.zeros(n_frames, dtype=np.float32)
    quality_rows: list[dict[str, float | int]] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    previous_xy: np.ndarray | None = None
    previous_valid: np.ndarray | None = None
    sequential_read = (
        # dense frame ranges are faster to decode sequentially
        len(frame_indices) > 1
        and np.all(np.diff(frame_indices) > 0)
        and int(np.max(np.diff(frame_indices))) <= 5
    )
    last_decoded_index: int | None = None
    max_phase_shift_px = max(3.0, 1.75 * float(spacing_px))
    t0 = time.time()

    for frame_pos, frame_index in enumerate(frame_indices):
        if sequential_read:
            # sequential decoder path
            if last_decoded_index is None or int(frame_index) <= last_decoded_index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                last_decoded_index = int(frame_index) - 1
            ok = False
            frame = None
            while last_decoded_index < int(frame_index):
                ok, frame = cap.read()
                last_decoded_index += 1
                if not ok:
                    break
        else:
            # sparse/random-access decoder path
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # match run frame processing to calibration image processing
        gray = undistort_gray_metric(bgr_to_gray(frame), calibration_meta, calibration_output_dir)
        frame_signal = dot_signal(gray)

        # adaptive profiles may use global pre-registration and previous-frame
        # prediction to keep pixel search windows centered
        shift_xy = np.zeros(2, dtype=float)
        shift_response = 0.0
        if profile.global_prereg:
            # coarse translation estimated from the whole roi signal
            shift_xy, shift_response = estimate_phase_translation(
                reference_signal,
                frame_signal,
                roi,
                max_shift_px=max_phase_shift_px,
            )
        predicted_xy = reference_xy_px + shift_xy
        if profile.predict_from_previous and previous_xy is not None and previous_valid is not None:
            # previous-frame prediction is only used for dots valid and finite
            # in the previous frame
            use_previous = previous_valid & np.isfinite(previous_xy).all(axis=1)
            predicted_xy = predicted_xy.copy()
            predicted_xy[use_previous] = previous_xy[use_previous]

        # template tracking in pixels uses the calibrated flat reference image
        # and returns the measured image-plane location of every known dot
        xy_px, score, valid0 = track_frame_templates(
            reference_signal,
            frame_signal,
            reference_dots,
            template_radius_px=template_radius,
            search_radius_px=search_radius,
            min_match_score=profile.min_match_score,
            max_displacement_px=max_displacement_px,
            search_centers_xy=predicted_xy,
        )
        dxy_px = xy_px - reference_xy_px

        # object-plane displacement conversion maps measured pixels back onto
        # the flat calibrated grid, giving raw metric dot displacement
        obj_tracked_m = pixel_to_object_xy_m(H_pixel_to_object_mm, xy_px)
        dxy_raw_m = obj_tracked_m - reference_obj_m

        # only finite template matches can be considered for metric filtering
        valid_for_affine = (
            valid0.copy()
            & np.isfinite(dxy_raw_m).all(axis=1)
            & np.isfinite(obj_tracked_m).all(axis=1)
        )

        # affine subtraction happens after metric conversion
        # so removed frame-wide motion has physical units and matches fsss geometry
        if remove_affine:
            dxy_corr_m, coeff, valid_corr = fit_affine_displacement_metric(
                reference_obj_m,
                obj_tracked_m,
                dxy_raw_m,
                valid_for_affine,
                ransac_threshold_m=float(affine_threshold_m),
                reject_outliers=not bool(profile.keep_affine_outliers),
            )
        else:
            dxy_corr_m = dxy_raw_m.copy()
            coeff = np.zeros((2, 3), dtype=float)
            valid_corr = valid_for_affine & np.isfinite(dxy_corr_m).all(axis=1)

        valid_after_affine = valid_corr.copy()
        mag_corr_m = np.linalg.norm(dxy_corr_m, axis=1)
        valid_object = valid_after_affine.copy()
        valid_object &= np.isfinite(mag_corr_m)
        # reject unrealistically large metric residual motion
        valid_object &= mag_corr_m <= float(max_object_displacement_m)

        valid_final_frame = valid_object.copy()
        if neighbor_filter_enabled:
            # local neighbor residual filtering suppresses isolated tracking mistakes
            valid_final_frame = neighbor_vector_filter(
                reference_obj_m,
                dxy_corr_m,
                valid_final_frame,
                k=int(neighbor_k),
                residual_threshold_m=float(neighbor_residual_threshold_m),
            )

        x_px_all[frame_pos] = xy_px[:, 0].astype(np.float32)
        # save this frame's arrays in frame-major format
        y_px_all[frame_pos] = xy_px[:, 1].astype(np.float32)
        dx_px_all[frame_pos] = dxy_px[:, 0].astype(np.float32)
        dy_px_all[frame_pos] = dxy_px[:, 1].astype(np.float32)
        scores[frame_pos] = score.astype(np.float32)
        valid_initial[frame_pos] = valid0
        valid_final[frame_pos] = valid_final_frame
        dx_obj_raw_all[frame_pos] = dxy_raw_m[:, 0].astype(np.float32)
        dy_obj_raw_all[frame_pos] = dxy_raw_m[:, 1].astype(np.float32)
        dx_obj_corr_all[frame_pos] = dxy_corr_m[:, 0].astype(np.float32)
        dy_obj_corr_all[frame_pos] = dxy_corr_m[:, 1].astype(np.float32)
        affine_coeff_all[frame_pos] = coeff
        global_shifts[frame_pos] = shift_xy.astype(np.float32)
        global_shift_response[frame_pos] = float(shift_response)

        time_s = float(all_times[int(frame_index)] - all_times[int(frame_indices[0])])
        # per-frame quality row mirrors the filtering stages above
        quality_rows.append(
            tracking_quality_row(
                frame_index=int(frame_index),
                time_s=time_s,
                valid_initial=valid0,
                valid_after_affine=valid_after_affine,
                valid_after_object=valid_object,
                valid_final=valid_final_frame,
                score=score,
                dxy_raw_m=dxy_raw_m,
                dxy_corr_m=dxy_corr_m,
                xy_px=xy_px,
                reference_dots=reference_dots,
                neighbor_pairs=neighbor_pairs,
                spacing_px=spacing_px,
            )
        )
        previous_xy = xy_px
        # previous prediction uses initial template validity, not final neighbor-filter validity
        # this avoids over-pruning the predictor
        previous_valid = valid0 & np.isfinite(xy_px).all(axis=1)

        if progress and ((frame_pos + 1) % 100 == 0 or frame_pos == n_frames - 1):
            elapsed = time.time() - t0
            print(f"Tracked {frame_pos + 1}/{n_frames} frames in {elapsed:.1f} s")

    cap.release()
    return {
        "x_px": x_px_all,
        "y_px": y_px_all,
        "dx_px": dx_px_all,
        "dy_px": dy_px_all,
        "scores": scores,
        "valid_initial": valid_initial,
        "valid_final": valid_final,
        "dx_obj_raw_m": dx_obj_raw_all,
        "dy_obj_raw_m": dy_obj_raw_all,
        "dx_obj_corrected_m": dx_obj_corr_all,
        "dy_obj_corrected_m": dy_obj_corr_all,
        "affine_coefficients": affine_coeff_all,
        "quality": pd.DataFrame(quality_rows),
        "global_shifts": global_shifts,
        "global_shift_response": global_shift_response,
        "profile_pixels": pixels,
        "neighbor_pairs": neighbor_pairs,
    }


def preflight_profile(
    *,
    video_path: Path,
    frame_indices: np.ndarray,
    all_times: np.ndarray,
    calibration_meta: dict[str, Any],
    calibration_output_dir: Path,
    H_pixel_to_object_mm: np.ndarray,
    reference_signal: np.ndarray,
    reference_dots: pd.DataFrame,
    reference_xy_px: np.ndarray,
    reference_obj_m: np.ndarray,
    roi: Roi | RotatedRoi,
    spacing_px: float,
    profile: TrackingProfile,
    max_object_displacement_m: float,
    affine_threshold_m: float,
    remove_affine: bool,
    neighbor_filter_enabled: bool,
    neighbor_k: int,
    neighbor_residual_threshold_m: float,
) -> dict[str, Any]:
    # preflight runs the same tracker on sampled frames
    # only summary stats are kept; arrays are discarded
    result = run_tracking_pass(
        video_path=video_path,
        frame_indices=frame_indices,
        all_times=all_times,
        calibration_meta=calibration_meta,
        calibration_output_dir=calibration_output_dir,
        H_pixel_to_object_mm=H_pixel_to_object_mm,
        reference_signal=reference_signal,
        reference_dots=reference_dots,
        reference_xy_px=reference_xy_px,
        reference_obj_m=reference_obj_m,
        roi=roi,
        spacing_px=spacing_px,
        profile=profile,
        max_object_displacement_m=max_object_displacement_m,
        affine_threshold_m=affine_threshold_m,
        remove_affine=remove_affine,
        neighbor_filter_enabled=neighbor_filter_enabled,
        neighbor_k=neighbor_k,
        neighbor_residual_threshold_m=neighbor_residual_threshold_m,
        progress=False,
    )
    quality = result["quality"]
    valid = result["valid_final"]
    return {
        # include pixel profile settings so preflight summaries explain pass/fail
        **result["profile_pixels"],
        "frames": int(len(frame_indices)),
        "median_valid_fraction": float(quality["valid_fraction"].median()) if len(quality) else 0.0,
        "p05_valid_fraction": float(quality["valid_fraction"].quantile(0.05)) if len(quality) else 0.0,
        "median_match_score": float(quality["median_match_score"].median()) if len(quality) else float("nan"),
        **dot_survival_counts(valid),
    }


def select_adaptive_profile(preflight_rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    # fsss reconstruction needs broad, stable spatial coverage
    # prefer profiles that keep many dots alive across the run
    # then use match quality and smaller search radius as tie-breakers
    best = max(
        preflight_rows,
        key=lambda row: (
            int(row["dots_ge_95"]),
            int(row["dots_ge_85"]),
            float(row["median_valid_fraction"]),
            float(row["median_match_score"]) if np.isfinite(float(row["median_match_score"])) else -1.0,
            -float(row["search_radius_px"]),
        ),
    )
    return str(best["name"])


def main() -> None:
    args = parse_args()
    # load run metadata first because it can point to calibration metadata and outputs
    run_metadata_path = project_path(args.run_metadata)
    if run_metadata_path is None or not run_metadata_path.exists():
        raise FileNotFoundError(f"Missing run metadata: {args.run_metadata}")
    run_meta = load_yaml(run_metadata_path)

    calibration_metadata_path = project_path(args.calibration_metadata or run_meta.get("calibration_metadata_path"))
    if calibration_metadata_path is None or not calibration_metadata_path.exists():
        raise FileNotFoundError(
            "Missing calibration metadata. Provide --calibration-metadata or set calibration_metadata_path in run metadata."
        )
    calibration_meta = load_yaml(calibration_metadata_path)

    # resolve calibration id from run metadata first, then calibration metadata
    calibration_id = str(run_meta.get("calibration_id", calibration_meta.get("calibration_id")))
    calibration_output_dir = project_path(
        args.calibration_output_dir
        or run_meta.get("calibration_output_dir")
        or f"outputs/full_fsss/calibration/{calibration_id}"
    )
    if calibration_output_dir is None:
        raise RuntimeError("Could not resolve calibration output dir.")

    run_id = str(run_meta["run_id"])
    # video path can be overridden if raw videos are archived outside the repo
    video_path = project_path(args.video_path or run_meta.get("video_path"))
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(f"Missing run video: {video_path}")

    output_dir = project_path(args.output_dir or run_meta.get("run_output_dir") or f"outputs/full_fsss/runs/{run_id}")
    if output_dir is None:
        raise RuntimeError("Could not resolve run output dir.")
    tracking_dir = output_dir / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    # what fsss tracking needs from calibration:
    # flat-liquid dot identities, flat reference image, and raytrace calibration
    # with the pixel-to-object homography
    flat_dots_path = calibration_output_dir / "flat_liquid_dots.csv"
    flat_ref_img_path = calibration_output_dir / "flat_liquid_reference_undistorted.png"
    raytrace_path = calibration_output_dir / "raytrace_fsss_calibration.json"
    for path in [flat_dots_path, flat_ref_img_path, raytrace_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required FSSS calibration output: {path}")

    flat_dots = pd.read_csv(flat_dots_path)
    # public defaults keep all calibrated dots in stable dot_id order
    dots = select_dot_subset(
        flat_dots,
        central_fraction=1.0,
        max_dots=None,
        random_seed=12,
    )
    reference_gray = cv2.imread(str(flat_ref_img_path), cv2.IMREAD_GRAYSCALE)
    if reference_gray is None:
        raise RuntimeError(f"Could not read flat reference image: {flat_ref_img_path}")
    reference_signal = dot_signal(reference_gray)

    # this flat-reference mapping converts tracked image coordinates into
    # metric object-plane coordinates
    raytrace = json.loads(raytrace_path.read_text(encoding="utf-8"))
    flat_ref_map = raytrace["flat_reference_mapping"]
    H_pixel_to_object_mm = np.array(flat_ref_map["homography_pixel_to_object_mm"], dtype=float)
    reference_xy_px = dots[["x_px", "y_px"]].to_numpy(float)
    # convert reference pixels to object-plane meters
    # raw displacement is measured relative to calibrated physical positions
    reference_obj_m = pixel_to_object_xy_m(H_pixel_to_object_mm, reference_xy_px)
    dots["x_ref_obj_m"] = reference_obj_m[:, 0]
    dots["y_ref_obj_m"] = reference_obj_m[:, 1]

    spacing_px = estimate_spacing_px(reference_xy_px)
    if not np.isfinite(spacing_px) or spacing_px <= 0:
        raise RuntimeError("Could not estimate flat-liquid dot spacing.")
    roi = roi_from_calibration_metadata(calibration_meta, dots, reference_gray.shape)

    # timebase follows the same ffprobe-preferred helper as frequency tracking
    info = video_metadata(video_path)
    opencv_frame_count = int(info["frame_count"])
    all_times, timebase_source = video_times(video_path, opencv_frame_count)
    frame_count = min(opencv_frame_count, len(all_times))
    start = max(0, int(args.frame_start))
    # frame_stop is exclusive, matching numpy.arange convention
    stop = frame_count if args.frame_stop is None else min(frame_count, int(args.frame_stop))
    frame_indices = np.arange(start, stop, max(1, int(args.frame_step)), dtype=int)
    if args.max_frames is not None:
        frame_indices = frame_indices[: int(args.max_frames)]
    if len(frame_indices) == 0:
        raise RuntimeError("No frames selected for tracking.")

    max_object_displacement_m = DEFAULT_MAX_OBJECT_DISPLACEMENT_MM * 1e-3
    # public constants are expressed in mm for readability, then converted to SI
    neighbor_residual_threshold_m = DEFAULT_NEIGHBOR_RESIDUAL_THRESHOLD_MM * 1e-3
    affine_threshold_m = max(0.05e-3, 0.5 * neighbor_residual_threshold_m)
    remove_affine = True
    neighbor_filter_enabled = True

    # adaptive preflight only chooses the pixel tracking profile
    # final npz always comes from one full tracking pass
    if args.adaptive_tracking:
        candidates = fsss_adaptive_profiles()
        # sample frames evenly over the run for profile choice
        preflight_indices = sampled_frame_indices(frame_indices, DEFAULT_PREFLIGHT_FRAMES)
        preflight_rows = [
            preflight_profile(
                video_path=video_path,
                frame_indices=preflight_indices,
                all_times=all_times,
                calibration_meta=calibration_meta,
                calibration_output_dir=calibration_output_dir,
                H_pixel_to_object_mm=H_pixel_to_object_mm,
                reference_signal=reference_signal,
                reference_dots=dots,
                reference_xy_px=reference_xy_px,
                reference_obj_m=reference_obj_m,
                roi=roi,
                spacing_px=spacing_px,
                profile=profile,
                max_object_displacement_m=max_object_displacement_m,
                affine_threshold_m=affine_threshold_m,
                remove_affine=remove_affine,
                neighbor_filter_enabled=neighbor_filter_enabled,
                neighbor_k=DEFAULT_NEIGHBOR_K,
                neighbor_residual_threshold_m=neighbor_residual_threshold_m,
            )
            for profile in candidates
        ]
        selected_name = select_adaptive_profile(preflight_rows, args)
        profile = next(profile for profile in candidates if profile.name == selected_name)
    else:
        preflight_rows = []
        profile = manual_profile()

    # small console summary before the potentially long tracking pass
    print("Robust metric tracking")
    print("Run:", run_id)
    print("Video:", relpath(video_path))
    print("Calibration:", calibration_id)
    print("Dots:", len(dots))
    print("Spacing:", f"{spacing_px:.2f} px")
    print("Frames:", len(frame_indices))
    print("Profile:", profile.name)

    result = run_tracking_pass(
        video_path=video_path,
        frame_indices=frame_indices,
        all_times=all_times,
        calibration_meta=calibration_meta,
        calibration_output_dir=calibration_output_dir,
        H_pixel_to_object_mm=H_pixel_to_object_mm,
        reference_signal=reference_signal,
        reference_dots=dots,
        reference_xy_px=reference_xy_px,
        reference_obj_m=reference_obj_m,
        roi=roi,
        spacing_px=spacing_px,
        profile=profile,
        max_object_displacement_m=max_object_displacement_m,
        affine_threshold_m=affine_threshold_m,
        remove_affine=remove_affine,
        neighbor_filter_enabled=neighbor_filter_enabled,
        neighbor_k=DEFAULT_NEIGHBOR_K,
        neighbor_residual_threshold_m=neighbor_residual_threshold_m,
        progress=True,
    )

    quality = result["quality"]
    quality_path = tracking_dir / "tracking_quality.csv"
    # csv outputs are human-readable summaries
    # npz below is the main data file
    quality.to_csv(quality_path, index=False)
    dots_path = tracking_dir / "tracking_dot_reference.csv"
    dots.to_csv(dots_path, index=False)

    time_s = all_times[frame_indices] - all_times[int(frame_indices[0])]
    timebase_summary = summarize_timebase(time_s)
    dt = np.diff(time_s)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    # fps_used is derived from actual selected timestamps when possible
    fps_used = float(1.0 / np.mean(positive_dt)) if len(positive_dt) else float(info["fps"])
    fps_median = float(1.0 / np.median(positive_dt)) if len(positive_dt) else float("nan")

    # what this fsss tracking npz saves:
    # calibrated dot identities, timestamps, pixel tracks, raw metric displacement,
    # affine-corrected metric displacement, validity masks, and profile metadata
    tracking_npz_path = tracking_dir / "tracked_dots_fsss.npz"
    np.savez_compressed(
        tracking_npz_path,
        # scalars are saved as zero-dimensional numpy arrays inside npz
        run_id=np.array(run_id),
        calibration_id=np.array(calibration_id),
        frame_indices=frame_indices.astype(np.int64),
        time_s=time_s.astype(float),
        fps=np.array(fps_used, dtype=float),
        fps_fallback=np.array(float(info["fps"]), dtype=float),
        fps_from_video=np.array(float(info["fps"]), dtype=float),
        fps_from_median_dt=np.array(fps_median, dtype=float),
        timestamp_source=np.array(timebase_source),
        timebase_summary_json=np.array(json.dumps(timebase_summary)),
        raw_frame_timestamps_s=all_times[frame_indices].astype(float),
        # dot identity and calibrated grid coordinates
        dot_id=dots["dot_id"].to_numpy(int),
        i=dots["i"].to_numpy(int),
        j=dots["j"].to_numpy(int),
        x_grid_m=dots["x_grid_m"].to_numpy(float),
        y_grid_m=dots["y_grid_m"].to_numpy(float),
        x_ref_px=dots["x_px"].to_numpy(float),
        y_ref_px=dots["y_px"].to_numpy(float),
        x_ref_obj_m=dots["x_ref_obj_m"].to_numpy(float),
        y_ref_obj_m=dots["y_ref_obj_m"].to_numpy(float),
        # pixel tracks and pixel displacements
        x_px=result["x_px"],
        y_px=result["y_px"],
        dx_px=result["dx_px"],
        dy_px=result["dy_px"],
        match_score=result["scores"],
        valid_initial=result["valid_initial"],
        valid_final=result["valid_final"],
        # metric raw and affine-corrected displacements in meters
        dx_obj_raw_m=result["dx_obj_raw_m"],
        dy_obj_raw_m=result["dy_obj_raw_m"],
        dx_obj_corrected_m=result["dx_obj_corrected_m"],
        dy_obj_corrected_m=result["dy_obj_corrected_m"],
        affine_coefficients=result["affine_coefficients"],
        global_prereg_shift_xy_px=result["global_shifts"],
        global_prereg_shift_response=result["global_shift_response"],
        robust_tracking_profile_json=np.array(json.dumps(result["profile_pixels"])),
    )

    # json summary is for quick inspection and batch logs
    # npz is the actual data product consumed by surface reconstruction
    profile_px = result["profile_pixels"]
    summary = {
        # this repeats some npz metadata so batch logs can be inspected without loading npz
        "schema": "robust_metric_dot_tracking_v1",
        "run_id": run_id,
        "calibration_id": calibration_id,
        "run_metadata": relpath(run_metadata_path),
        "calibration_metadata": relpath(calibration_metadata_path),
        "calibration_output_dir": relpath(calibration_output_dir),
        "video_path": relpath(video_path),
        "tracking_npz": relpath(tracking_npz_path),
        "quality_csv": relpath(quality_path),
        "dot_reference_csv": relpath(dots_path),
        "frames_tracked": int(len(frame_indices)),
        "dots_tracked": int(len(dots)),
        "dot_spacing_px": float(spacing_px),
        "roi": roi.as_dict(),
        "template_radius_px": int(profile_px["template_radius_px"]),
        "search_radius_px": int(profile_px["search_radius_px"]),
        "max_displacement_px": float(profile_px["max_displacement_px"]),
        "max_object_displacement_mm": DEFAULT_MAX_OBJECT_DISPLACEMENT_MM,
        "max_object_displacement_applied_after_affine": True,
        "min_match_score": float(profile_px["min_match_score"]),
        "adaptive_tracking": bool(args.adaptive_tracking),
        "adaptive_preflight": preflight_rows,
        "tracking_profile": str(profile.name),
        "tracking_profile_settings": profile_px,
        "remove_global_affine_motion": bool(remove_affine),
        "neighbor_filter_enabled": bool(neighbor_filter_enabled),
        "neighbor_residual_threshold_mm": DEFAULT_NEIGHBOR_RESIDUAL_THRESHOLD_MM,
        "timebase_source": timebase_source,
        "timebase_summary": timebase_summary,
        "median_valid_fraction": float(quality["valid_fraction"].median()) if len(quality) else float("nan"),
        "p05_valid_fraction": float(quality["valid_fraction"].quantile(0.05)) if len(quality) else float("nan"),
        "median_n_valid_initial": quality_summary_value(quality, "n_valid_initial"),
        "median_n_valid_after_affine": quality_summary_value(quality, "n_valid_after_affine"),
        "median_n_valid_after_object": quality_summary_value(quality, "n_valid_after_object"),
        "median_n_valid_final": quality_summary_value(quality, "n_valid"),
        "median_match_score": float(quality["median_match_score"].median()) if len(quality) else float("nan"),
        "median_tracked_neighbor_valid_pair_fraction": quality_summary_value(
            quality, "tracked_neighbor_valid_pair_fraction"
        ),
        "median_tracked_neighbor_normal_length_fraction": quality_summary_value(
            quality, "tracked_neighbor_normal_length_fraction"
        ),
        "median_tracked_neighbor_diagonal_length_fraction": quality_summary_value(
            quality, "tracked_neighbor_diagonal_length_fraction"
        ),
        "p95_tracked_neighbor_diagonal_length_fraction": quality_summary_value(
            quality, "tracked_neighbor_diagonal_length_fraction", method="p95"
        ),
        "median_tracked_neighbor_bad_length_fraction": quality_summary_value(
            quality, "tracked_neighbor_bad_length_fraction"
        ),
        "p95_tracked_neighbor_bad_length_fraction": quality_summary_value(
            quality, "tracked_neighbor_bad_length_fraction", method="p95"
        ),
        **dot_survival_counts(result["valid_final"]),
    }
    write_json(tracking_dir / "robust_metric_tracking_summary.json", summary)

    print("Saved:", relpath(tracking_npz_path))
    print(f"Median valid fraction: {summary['median_valid_fraction']:.3f}")
    print(f"P05 valid fraction: {summary['p05_valid_fraction']:.3f}")
    print(f"Median match score: {summary['median_match_score']:.3f}")


if __name__ == "__main__":
    main()
