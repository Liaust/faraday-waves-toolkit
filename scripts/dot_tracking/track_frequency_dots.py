#!/usr/bin/env python3
from __future__ import annotations

# this tracks the flat-liquid reference dots through a run video
# and saves the pixel displacement arrays used by frequency and onset analysis

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# add repo root so common/dot_lattice imports work when this is run by path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (  # noqa: E402
    CALIBRATION_METADATA_PATH,
    bgr_to_gray,
    frequency_tracking_output_dir,
    frequency_tracking_run_output_dir,
    load_yaml,
    project_path,
    relpath,
    summarize_timebase,
    undistort_gray,
    video_metadata,
    video_times,
    write_json,
)
from dot_lattice import (  # noqa: E402
    Roi,
    RotatedRoi,
    dot_signal,
    roi_from_lattice_data,
    subtract_global_affine,
    track_frame_templates,
)


# tracking profile definitions
# one profile is one template-matching recipe for one pass
@dataclass(frozen=True)
class TrackingProfile:
    # fractions are later multiplied by measured dot spacing to get pixel radii
    name: str
    template_radius_fraction: float
    search_radius_fraction: float
    max_displacement_fraction: float
    min_match_score: float
    global_prereg: bool
    predict_from_previous: bool
    keep_affine_outliers: bool


def parse_args() -> argparse.Namespace:
    # this cli consumes one run metadata file plus the reference made by
    # build_frequency_reference.py
    parser = argparse.ArgumentParser(
        description="Track flat-liquid reference dots through a run video for frequency-only analysis."
    )
    parser.add_argument("run_metadata", help="metadata_run.yaml for one run.")
    parser.add_argument(
        "--calibration-metadata",
        default=str(CALIBRATION_METADATA_PATH),
        help="Calibration metadata YAML.",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Flat reference output directory from build_frequency_reference.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Per-run dot-tracking output directory. Defaults under outputs/dot_tracking/frequency/runs/.",
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Track every Nth frame.")
    # max-frames is useful for quick checks without changing metadata
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--min-match-score", type=float, default=0.35)
    parser.add_argument("--template-radius-fraction", type=float, default=0.25)
    parser.add_argument("--search-radius-fraction", type=float, default=0.42)
    parser.add_argument("--max-displacement-fraction", type=float, default=0.48)
    parser.add_argument("--no-affine-correction", action="store_true")
    parser.add_argument(
        "--adaptive-tracking",
        action="store_true",
        help="Preflight multiple tracking profiles and use the strictest profile that tracks well.",
    )
    parser.add_argument(
        "--preflight-frames",
        type=int,
        default=180,
        help="Number of evenly spaced frames to use for adaptive profile selection.",
    )
    parser.add_argument(
        "--adaptive-min-frame-valid",
        type=float,
        default=0.65,
        help="Median per-frame valid fraction required to accept an adaptive profile.",
    )
    parser.add_argument(
        "--adaptive-min-dots",
        type=int,
        default=300,
        help="Dots that must be valid in >=85%% of preflight frames to accept an adaptive profile.",
    )
    parser.add_argument(
        "--global-prereg",
        action="store_true",
        help="Estimate frame-level translation before local template tracking.",
    )
    parser.add_argument(
        "--predict-from-previous",
        action="store_true",
        help="Use previous-frame dot positions as local search centers when available.",
    )
    parser.add_argument(
        "--keep-affine-outliers",
        action="store_true",
        help="Subtract global affine motion but keep non-affine residual dots as valid matches.",
    )
    return parser.parse_args()


# per-frame quality metrics
# enough to check tracking health without turning output into a huge debug dump
def quality_row(
    frame_index: int,
    time_s: float,
    valid_raw: np.ndarray,
    valid_final: np.ndarray,
    score: np.ndarray,
    raw_dxy: np.ndarray,
    corrected_dxy: np.ndarray,
) -> dict[str, float | int]:
    # score has one template correlation value per dot
    # filter nans from dots that could not be evaluated
    valid_scores = score[np.isfinite(score)]
    # displacement summary uses final-valid dots only
    raw = raw_dxy[valid_final]
    corr = corrected_dxy[valid_final]
    return {
        "frame_index": int(frame_index),
        "time_s": float(time_s),
        "n_dots": int(len(valid_final)),
        "n_valid_raw": int(np.sum(valid_raw)),
        "n_valid": int(np.sum(valid_final)),
        "valid_fraction": float(np.mean(valid_final)) if len(valid_final) else 0.0,
        "median_match_score": float(np.median(valid_scores)) if len(valid_scores) else float("nan"),
        "p10_match_score": float(np.percentile(valid_scores, 10)) if len(valid_scores) else float("nan"),
        "rms_displacement_raw_px": float(np.sqrt(np.nanmean(np.sum(raw**2, axis=1)))) if len(raw) else float("nan"),
        "rms_displacement_corrected_px": float(np.sqrt(np.nanmean(np.sum(corr**2, axis=1)))) if len(corr) else float("nan"),
    }


# manual profile selection uses the cli values directly
# this is the predictable path when the run is already well behaved
def manual_profile(args: argparse.Namespace) -> TrackingProfile:
    # wrap cli tuning values into a TrackingProfile object
    return TrackingProfile(
        name="manual",
        template_radius_fraction=float(args.template_radius_fraction),
        search_radius_fraction=float(args.search_radius_fraction),
        max_displacement_fraction=float(args.max_displacement_fraction),
        min_match_score=float(args.min_match_score),
        global_prereg=bool(args.global_prereg),
        predict_from_previous=bool(args.predict_from_previous),
        keep_affine_outliers=bool(args.keep_affine_outliers),
    )


# adaptive profile selection
# try increasingly permissive recipes on a small frame sample,
# then run the full video with the chosen profile
def adaptive_profiles(args: argparse.Namespace) -> list[TrackingProfile]:
    # ordered from strict to permissive
    # selection keeps the first profile that satisfies coverage requirements
    base_template = float(args.template_radius_fraction)
    return [
        TrackingProfile("strict_prereg", base_template, 0.42, 0.48, 0.35, True, True, True),
        TrackingProfile("medium_prereg", base_template, 0.65, 0.75, 0.25, True, True, True),
        TrackingProfile("wide_prereg", base_template, 0.90, 1.20, 0.15, True, True, True),
        TrackingProfile("very_wide_prereg", base_template, 1.15, 1.50, 0.10, True, True, True),
    ]


def profile_pixels(profile: TrackingProfile, spacing_px: float) -> dict[str, float | int | str | bool]:
    # convert fractions of grid spacing into pixel radii
    template_radius = max(2, int(round(profile.template_radius_fraction * spacing_px)))
    search_radius = max(template_radius + 1, int(round(profile.search_radius_fraction * spacing_px)))
    max_displacement_px = max(1.0, float(profile.max_displacement_fraction * spacing_px))
    return {
        "name": profile.name,
        "template_radius_px": int(template_radius),
        "search_radius_px": int(search_radius),
        "max_displacement_px": float(max_displacement_px),
        "min_match_score": float(profile.min_match_score),
        "global_prereg": bool(profile.global_prereg),
        "predict_from_previous": bool(profile.predict_from_previous),
        "keep_affine_outliers": bool(profile.keep_affine_outliers),
    }


def roi_bounds(roi: Any, image_shape: tuple[int, int]) -> Roi:
    # pre-registration crop must be rectangular even when the logical roi is rotated
    if isinstance(roi, (Roi, RotatedRoi)):
        return roi.bounding_roi(image_shape)
    return Roi.from_list(roi.as_list()).bounding_roi(image_shape)


# global translation pre-registration
# estimate one frame-wide x/y shift before local template matching
# this keeps local search windows centered if the whole image has moved slightly
def estimate_phase_translation(
    reference_signal: np.ndarray,
    frame_signal: np.ndarray,
    roi: Any,
    *,
    max_shift_px: float,
) -> tuple[np.ndarray, float]:
    bounds = roi_bounds(roi, reference_signal.shape)
    # crop both reference and current frame to the roi bounding rectangle
    ref = reference_signal[bounds.y : bounds.y + bounds.h, bounds.x : bounds.x + bounds.w].astype(np.float32)
    frame = frame_signal[bounds.y : bounds.y + bounds.h, bounds.x : bounds.x + bounds.w].astype(np.float32)
    if ref.size == 0 or frame.size == 0 or ref.shape != frame.shape:
        return np.zeros(2, dtype=float), 0.0
    ref = ref - float(np.mean(ref))
    frame = frame - float(np.mean(frame))
    # phase correlation is unreliable when either image has almost no contrast
    if float(np.std(ref)) < 1e-6 or float(np.std(frame)) < 1e-6:
        return np.zeros(2, dtype=float), 0.0
    # Hanning window reduces edge discontinuities before phase correlation
    window = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    # phaseCorrelate estimates translation between two images in Fourier space
    shift, response = cv2.phaseCorrelate(ref, frame, window)
    shift_xy = np.array([float(shift[0]), float(shift[1])], dtype=float)
    if not np.all(np.isfinite(shift_xy)) or float(np.linalg.norm(shift_xy)) > float(max_shift_px):
        # reject implausibly large shifts rather than centering all searches wrong
        return np.zeros(2, dtype=float), float(response) if np.isfinite(response) else 0.0
    return shift_xy, float(response) if np.isfinite(response) else 0.0


def sampled_frame_indices(frame_indices: np.ndarray, count: int) -> np.ndarray:
    # pick a uniform subset from selected frames for adaptive preflight
    if len(frame_indices) <= max(1, int(count)):
        return frame_indices.copy()
    positions = np.linspace(0, len(frame_indices) - 1, max(1, int(count)), dtype=int)
    return frame_indices[np.unique(positions)]


def dot_survival_counts(valid: np.ndarray) -> dict[str, int]:
    # valid has shape (frames, dots)
    # mean over frames gives the fraction of time each dot survived
    if valid.size == 0:
        return {"dots_ge_95": 0, "dots_ge_90": 0, "dots_ge_85": 0, "dots_ge_75": 0}
    per_dot = np.mean(valid, axis=0)
    return {
        "dots_ge_95": int(np.sum(per_dot >= 0.95)),
        "dots_ge_90": int(np.sum(per_dot >= 0.90)),
        "dots_ge_85": int(np.sum(per_dot >= 0.85)),
        "dots_ge_75": int(np.sum(per_dot >= 0.75)),
    }


def indexed_neighbor_pairs(reference_dots: pd.DataFrame) -> np.ndarray:
    # create dot-index pairs for adjacent lattice neighbors
    # used to check whether tracked dots still have plausible spacing
    by_index = {
        (int(row.i), int(row.j)): pos
        for pos, row in enumerate(reference_dots.itertuples(index=False))
    }
    pairs: list[tuple[int, int]] = []
    for (i, j), pos in by_index.items():
        for neighbor in ((i + 1, j), (i, j + 1)):
            neighbor_pos = by_index.get(neighbor)
            if neighbor_pos is not None:
                pairs.append((int(pos), int(neighbor_pos)))
    return np.asarray(pairs, dtype=np.int32)


def tracked_neighbor_geometry_metrics(
    xy: np.ndarray,
    valid: np.ndarray,
    neighbor_pairs: np.ndarray,
    *,
    expected_spacing_px: float,
) -> dict[str, float | int]:
    # defaults for frames where neighbor geometry cannot be computed
    defaults = {
        "tracked_neighbor_pair_count": int(len(neighbor_pairs)),
        "tracked_neighbor_valid_pair_count": 0,
        "tracked_neighbor_valid_pair_fraction": float("nan"),
        "tracked_neighbor_spacing_median_px": float("nan"),
        "tracked_neighbor_spacing_p95_abs_error_px": float("nan"),
        "tracked_neighbor_normal_length_fraction": float("nan"),
        "tracked_neighbor_marginal_length_fraction": float("nan"),
        "tracked_neighbor_bad_length_fraction": float("nan"),
        "tracked_neighbor_diagonal_length_fraction": float("nan"),
    }
    if len(neighbor_pairs) == 0 or not np.isfinite(expected_spacing_px) or expected_spacing_px <= 0:
        return defaults

    # a and b are integer column indices into per-dot arrays
    a = neighbor_pairs[:, 0]
    b = neighbor_pairs[:, 1]
    pair_valid = (
        # a neighbor pair is usable only if both dots are valid and finite
        valid[a]
        & valid[b]
        & np.all(np.isfinite(xy[a]), axis=1)
        & np.all(np.isfinite(xy[b]), axis=1)
    )
    valid_pair_count = int(np.sum(pair_valid))
    defaults["tracked_neighbor_valid_pair_count"] = valid_pair_count
    defaults["tracked_neighbor_valid_pair_fraction"] = float(valid_pair_count / len(neighbor_pairs))
    if valid_pair_count == 0:
        return defaults

    # compute distances between the two dots of every valid neighbor pair
    delta = xy[b[pair_valid]] - xy[a[pair_valid]]
    distances = np.hypot(delta[:, 0], delta[:, 1])
    ratios = distances / float(expected_spacing_px)
    # these bands check whether the neighbor length still looks like one grid spacing
    normal = (0.85 <= ratios) & (ratios <= 1.15)
    marginal = (~normal) & (0.70 <= ratios) & (ratios <= 1.35)
    bad = ~(normal | marginal)
    diagonal_like = (1.30 <= ratios) & (ratios <= 1.50)
    abs_error = np.abs(distances - float(expected_spacing_px))

    defaults.update(
        {
            "tracked_neighbor_spacing_median_px": float(np.median(distances)),
            "tracked_neighbor_spacing_p95_abs_error_px": float(np.percentile(abs_error, 95)),
            "tracked_neighbor_normal_length_fraction": float(np.mean(normal)),
            "tracked_neighbor_marginal_length_fraction": float(np.mean(marginal)),
            "tracked_neighbor_bad_length_fraction": float(np.mean(bad)),
            "tracked_neighbor_diagonal_length_fraction": float(np.mean(diagonal_like)),
        }
    )
    return defaults


def quality_summary_value(quality: pd.DataFrame, column: str, method: str = "median") -> float:
    # helper for summaries where a column might be absent in older outputs
    if column not in quality or quality.empty:
        return float("nan")
    values = pd.to_numeric(quality[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    if method == "p95":
        return float(values.quantile(0.95))
    if method == "max":
        return float(values.max())
    return float(values.median())


def run_tracking_pass(
    *,
    video_path: Path,
    frame_indices: np.ndarray,
    all_times: np.ndarray,
    calibration_meta: dict[str, Any],
    reference_signal: np.ndarray,
    reference_dots: pd.DataFrame,
    reference_xy: np.ndarray,
    roi: Any,
    spacing_px: float,
    profile: TrackingProfile,
    affine_threshold_px: float,
    no_affine_correction: bool,
    progress: bool,
) -> dict[str, Any]:
    # main tracking loop outputs
    # rows are selected frames, columns are fixed dot identities from the reference
    pixels = profile_pixels(profile, spacing_px)
    template_radius = int(pixels["template_radius_px"])
    search_radius = int(pixels["search_radius_px"])
    max_displacement_px = float(pixels["max_displacement_px"])

    n_frames = len(frame_indices)
    n_dots = len(reference_dots)
    neighbor_pairs = indexed_neighbor_pairs(reference_dots)
    # arrays are frame-major: (n_frames, n_dots, 2) for positions/displacements
    # dot column order is exactly reference_dots row order
    measured_xy = np.full((n_frames, n_dots, 2), np.nan, dtype=np.float32)
    raw_dxy = np.full((n_frames, n_dots, 2), np.nan, dtype=np.float32)
    corrected_dxy = np.full((n_frames, n_dots, 2), np.nan, dtype=np.float32)
    scores = np.full((n_frames, n_dots), np.nan, dtype=np.float32)
    valid = np.zeros((n_frames, n_dots), dtype=bool)
    quality_rows: list[dict[str, float | int]] = []
    global_shifts = np.zeros((n_frames, 2), dtype=np.float32)
    global_shift_response = np.zeros(n_frames, dtype=np.float32)
    previous_xy: np.ndarray | None = None
    previous_valid: np.ndarray | None = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    sequential_read = (
        # sequential reading is faster for dense frame ranges because opencv can
        # decode forward instead of seeking every frame
        len(frame_indices) > 1
        and np.all(np.diff(frame_indices) > 0)
        and int(np.max(np.diff(frame_indices))) <= 5
    )
    last_decoded_index: int | None = None
    max_phase_shift_px = max(3.0, 1.75 * float(spacing_px))
    for frame_pos, frame_index in enumerate(frame_indices):
        if sequential_read:
            # decode frames until the requested frame index is reached
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
            # sparse frame selection uses random access seeking
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
        if not ok:
            continue
        # convert current frame into the same grayscale/undistorted signal space
        # as the flat reference
        gray = undistort_gray(bgr_to_gray(frame), calibration_meta)
        frame_signal = dot_signal(gray)

        # optional global pre-registration gives one coarse frame translation
        # optional previous-frame prediction then updates per-dot search centers
        shift_xy = np.zeros(2, dtype=float)
        shift_response = 0.0
        if profile.global_prereg:
            # estimate one translation for the whole frame
            shift_xy, shift_response = estimate_phase_translation(
                reference_signal,
                frame_signal,
                roi,
                max_shift_px=max_phase_shift_px,
            )
        predicted_xy = reference_xy + shift_xy
        if profile.predict_from_previous and previous_xy is not None and previous_valid is not None:
            # for dots tracked in the previous frame, use last positions as search centers
            use_previous = previous_valid & np.all(np.isfinite(previous_xy), axis=1)
            predicted_xy = predicted_xy.copy()
            predicted_xy[use_previous] = previous_xy[use_previous]

        # template tracking is local:
        # crop a small reference patch around each dot and search nearby in this frame
        xy, score, valid_raw = track_frame_templates(
            reference_signal,
            frame_signal,
            reference_dots,
            template_radius_px=template_radius,
            search_radius_px=search_radius,
            min_match_score=profile.min_match_score,
            max_displacement_px=max_displacement_px,
            search_centers_xy=predicted_xy,
        )
        raw = xy - reference_xy

        # affine correction removes camera/container-scale translation, rotation,
        # shear, and magnification
        # the spectrum then uses residual dot motion relative to the flat grid
        if no_affine_correction:
            # rare debug mode: keep raw template displacements
            corr = raw.copy()
            valid_final = valid_raw.copy()
        else:
            # normal mode: remove global affine motion before saving spectral signal
            corr, valid_final, _ = subtract_global_affine(
                reference_xy,
                xy,
                valid_raw,
                ransac_threshold_px=affine_threshold_px,
                reject_outliers=not bool(profile.keep_affine_outliers),
            )

        measured_xy[frame_pos] = xy.astype(np.float32)
        # store all array outputs for this frame at the same frame_pos index
        raw_dxy[frame_pos] = raw.astype(np.float32)
        corrected_dxy[frame_pos] = corr.astype(np.float32)
        scores[frame_pos] = score.astype(np.float32)
        valid[frame_pos] = valid_final
        global_shifts[frame_pos] = shift_xy.astype(np.float32)
        global_shift_response[frame_pos] = float(shift_response)

        time_s = float(all_times[int(frame_index)] - all_times[int(frame_indices[0])])
        # one row per frame in tracking_quality.csv
        row = quality_row(frame_index, time_s, valid_raw, valid_final, score, raw, corr)
        # neighbor geometry catches gross tracking/indexing failures by checking
        # whether adjacent dots still have plausible spacing
        row.update(
            tracked_neighbor_geometry_metrics(
                xy,
                valid_final,
                neighbor_pairs,
                expected_spacing_px=spacing_px,
            )
        )
        row["global_shift_x_px"] = float(shift_xy[0])
        row["global_shift_y_px"] = float(shift_xy[1])
        row["global_shift_response"] = float(shift_response)
        quality_rows.append(row)
        previous_xy = xy
        # previous_valid uses raw template validity so prediction can still help
        # even when affine rejection is stricter
        previous_valid = valid_raw & np.all(np.isfinite(xy), axis=1)

        if progress and ((frame_pos + 1) % 100 == 0 or frame_pos == n_frames - 1):
            print(f"Tracked {frame_pos + 1}/{n_frames} frames from {relpath(video_path)}")

    cap.release()
    # return arrays plus quality table
    # main() decides how to serialize them
    quality = pd.DataFrame(quality_rows)
    return {
        "measured_xy": measured_xy,
        "raw_dxy": raw_dxy,
        "corrected_dxy": corrected_dxy,
        "scores": scores,
        "valid": valid,
        "quality": quality,
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
    reference_signal: np.ndarray,
    reference_dots: pd.DataFrame,
    reference_xy: np.ndarray,
    roi: Any,
    spacing_px: float,
    profile: TrackingProfile,
    affine_threshold_px: float,
    no_affine_correction: bool,
) -> dict[str, Any]:
    # preflight is just a short tracking pass
    # arrays are thrown away; only coverage/score summaries choose the final profile
    result = run_tracking_pass(
        video_path=video_path,
        frame_indices=frame_indices,
        all_times=all_times,
        calibration_meta=calibration_meta,
        reference_signal=reference_signal,
        reference_dots=reference_dots,
        reference_xy=reference_xy,
        roi=roi,
        spacing_px=spacing_px,
        profile=profile,
        affine_threshold_px=affine_threshold_px,
        no_affine_correction=no_affine_correction,
        progress=False,
    )
    quality = result["quality"]
    valid = result["valid"]
    survival = dot_survival_counts(valid)
    median_valid = float(quality["valid_fraction"].median()) if len(quality) else 0.0
    median_score = float(quality["median_match_score"].median()) if len(quality) else float("nan")
    return {
        **result["profile_pixels"],
        "frames": int(len(frame_indices)),
        "median_valid_fraction": median_valid,
        "median_match_score": median_score,
        "p05_valid_fraction": float(quality["valid_fraction"].quantile(0.05)) if len(quality) else 0.0,
        **survival,
    }


def select_adaptive_profile(preflight_rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    # accept the first profile that meets frame-level and dot-survival thresholds
    # since profiles are strict -> permissive, this chooses the narrowest reliable search
    passing = [
        row
        for row in preflight_rows
        if float(row["median_valid_fraction"]) >= float(args.adaptive_min_frame_valid)
        and int(row["dots_ge_85"]) >= int(args.adaptive_min_dots)
    ]
    if passing:
        return str(passing[0]["name"])
    best = max(
        # if none pass, choose best profile by dot survival, valid fraction, then score
        preflight_rows,
        key=lambda row: (
            int(row["dots_ge_85"]),
            float(row["median_valid_fraction"]),
            float(row["median_match_score"]) if np.isfinite(float(row["median_match_score"])) else -1.0,
        ),
    )
    return str(best["name"])


def main() -> None:
    args = parse_args()
    # resolve metadata paths before loading
    run_metadata_path = project_path(args.run_metadata)
    calibration_metadata_path = project_path(args.calibration_metadata)
    if run_metadata_path is None or calibration_metadata_path is None:
        raise SystemExit("Missing metadata path.")
    run_meta = load_yaml(run_metadata_path)
    calibration_meta = load_yaml(calibration_metadata_path)

    # reference loading
    # these three files come from build_frequency_reference.py and define dot identities
    reference_dir = project_path(args.reference_dir)
    if reference_dir is None:
        reference_dir = frequency_tracking_output_dir() / "reference" / str(calibration_meta["calibration_id"])
    dots_path = reference_dir / "flat_reference_dots.csv"
    lattice_path = reference_dir / "flat_reference_lattice.json"
    ref_frame_path = reference_dir / "flat_reference_frame_undistorted.png"
    if not dots_path.exists() or not lattice_path.exists() or not ref_frame_path.exists():
        raise FileNotFoundError(
            "Missing flat reference outputs. Run scripts/dot_tracking/build_frequency_reference.py first."
        )

    reference_dots = pd.read_csv(dots_path)
    # lattice json records roi, spacing, and physical dot grid metadata
    lattice = json.loads(lattice_path.read_text(encoding="utf-8"))
    reference_gray = cv2.imread(str(ref_frame_path), cv2.IMREAD_GRAYSCALE)
    if reference_gray is None:
        raise RuntimeError(f"Could not read {ref_frame_path}")

    video_path = project_path(run_meta.get("video_path"))
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(f"Missing run video: {video_path}")

    out_dir = project_path(args.output_dir)
    if out_dir is None:
        out_dir = frequency_tracking_run_output_dir(run_meta)
    tracking_dir = out_dir / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    spacing_px = float(lattice["dot_spacing_px_median_nn"])
    # RANSAC threshold for affine correction scales with dot spacing
    # but never goes below 1.5 px
    affine_threshold_px = max(1.5, 0.35 * spacing_px)
    roi = roi_from_lattice_data(lattice)
    # template matching uses the enhanced dark-dot signal, not raw grayscale
    reference_signal = dot_signal(reference_gray)
    reference_xy = reference_dots[["x_px", "y_px"]].to_numpy(float)

    # frame timestamp/timebase setup
    # prefer extracted timestamps, but keep opencv frame count aligned with decoding
    info = video_metadata(video_path)
    opencv_frame_count = int(info["frame_count"])
    all_times, timebase_source = video_times(video_path, opencv_frame_count)
    frame_count = min(opencv_frame_count, len(all_times))
    frame_indices = np.arange(0, frame_count, max(1, int(args.frame_step)), dtype=int)
    if args.max_frames is not None:
        frame_indices = frame_indices[: int(args.max_frames)]

    # manual vs adaptive profile selection happens before the full pass
    # adaptive only chooses settings; saved arrays still come from one final pass
    if args.adaptive_tracking:
        candidates = adaptive_profiles(args)
        preflight_indices = sampled_frame_indices(frame_indices, int(args.preflight_frames))
        preflight_rows = [
            preflight_profile(
                video_path=video_path,
                frame_indices=preflight_indices,
                all_times=all_times,
                calibration_meta=calibration_meta,
                reference_signal=reference_signal,
                reference_dots=reference_dots,
                reference_xy=reference_xy,
                roi=roi,
                spacing_px=spacing_px,
                profile=profile,
                affine_threshold_px=affine_threshold_px,
                no_affine_correction=bool(args.no_affine_correction),
            )
            for profile in candidates
        ]
        selected_name = select_adaptive_profile(preflight_rows, args)
        profile = next(profile for profile in candidates if profile.name == selected_name)
    else:
        preflight_rows = []
        profile = manual_profile(args)

    result = run_tracking_pass(
        video_path=video_path,
        frame_indices=frame_indices,
        all_times=all_times,
        calibration_meta=calibration_meta,
        reference_signal=reference_signal,
        reference_dots=reference_dots,
        reference_xy=reference_xy,
        roi=roi,
        spacing_px=spacing_px,
        profile=profile,
        affine_threshold_px=affine_threshold_px,
        no_affine_correction=bool(args.no_affine_correction),
        progress=True,
    )

    n_frames = len(frame_indices)
    n_dots = len(reference_dots)
    measured_xy = result["measured_xy"]
    raw_dxy = result["raw_dxy"]
    corrected_dxy = result["corrected_dxy"]
    scores = result["scores"]
    valid = result["valid"]
    quality = result["quality"]
    global_shifts = result["global_shifts"]
    global_shift_response = result["global_shift_response"]
    profile_px = result["profile_pixels"]
    neighbor_pairs = result["neighbor_pairs"]

    time_s = all_times[frame_indices] - all_times[int(frame_indices[0])]
    frame_indices_arr = frame_indices.astype(np.int32)
    quality_path = tracking_dir / "tracking_quality.csv"
    quality.to_csv(quality_path, index=False)

    # what this npz writes:
    # reference grid, timestamps, raw pixel displacement, affine-corrected displacement,
    # validity masks, match scores, and basic tracking metadata
    npz_path = tracking_dir / "tracked_dots_frequency.npz"
    np.savez_compressed(
        npz_path,
        frame_indices=frame_indices_arr,
        time_s=time_s.astype(np.float64),
        raw_frame_timestamps_s=all_times[frame_indices].astype(np.float64),
        x_ref_px=reference_dots["x_px"].to_numpy(float),
        y_ref_px=reference_dots["y_px"].to_numpy(float),
        i=reference_dots["i"].to_numpy(int),
        j=reference_dots["j"].to_numpy(int),
        x_grid_m=reference_dots["x_grid_m"].to_numpy(float),
        y_grid_m=reference_dots["y_grid_m"].to_numpy(float),
        measured_xy_px=measured_xy,
        raw_dxy_px=raw_dxy,
        corrected_dxy_px=corrected_dxy,
        match_score=scores,
        valid=valid,
        global_shift_xy_px=global_shifts,
        global_shift_response=global_shift_response,
        indexed_neighbor_pairs=neighbor_pairs.astype(np.int32),
        dot_spacing_mm=np.array(float(lattice["dot_spacing_mm"])),
        dot_spacing_px=np.array(spacing_px),
        roi_px=np.array(lattice.get("roi_bounding_px", roi.as_list()), dtype=float),
        roi_kind=np.array(str(lattice.get("roi_kind", "axis_aligned"))),
        roi_json=np.array(json.dumps(lattice.get("roi", roi.as_dict()))),
        tracking_profile_json=np.array(json.dumps(profile_px)),
        timebase_source=np.array(timebase_source),
        timebase_summary_json=np.array(json.dumps(summarize_timebase(time_s))),
    )

    # summary json is much smaller than the npz
    # it gives batch runners and humans a quick view of coverage and score
    summary = {
        "schema": "frequency_dot_tracking_v1",
        "run_id": str(run_meta["run_id"]),
        "run_metadata_path": relpath(run_metadata_path),
        "calibration_metadata_path": relpath(calibration_metadata_path),
        "video_path": relpath(video_path),
        "reference_dir": relpath(reference_dir),
        "tracking_npz": relpath(npz_path),
        "quality_csv": relpath(quality_path),
        "frames_tracked": int(n_frames),
        "frame_step": int(args.frame_step),
        "max_frames_requested": args.max_frames,
        "opencv_frame_count": int(opencv_frame_count),
        "timestamped_frame_count": int(len(all_times)),
        "dots_tracked": int(n_dots),
        "roi_kind": str(lattice.get("roi_kind", "axis_aligned")),
        "roi_bounding_px": lattice.get("roi_bounding_px"),
        "dot_spacing_px": spacing_px,
        "timebase_source": timebase_source,
        "timebase_summary": summarize_timebase(time_s),
        "median_valid_fraction": float(quality["valid_fraction"].median()) if len(quality) else float("nan"),
        "median_match_score": float(quality["median_match_score"].median()) if len(quality) else float("nan"),
    }
    write_json(tracking_dir / "tracking_summary.json", summary)

    print(f"Saved {relpath(npz_path)}")
    print(f"Median valid fraction: {summary['median_valid_fraction']:.3f}")
    print(f"Median match score: {summary['median_match_score']:.3f}")


if __name__ == "__main__":
    main()
