#!/usr/bin/env python3
from __future__ import annotations

# this builds the flat-liquid dot reference for frequency tracking
# the comments are intentionally more detailed so I can read through the script
# and remove them later

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# this file lives in scripts/dot_tracking, so parents[2] is the repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# when this is run directly, python does not automatically know the repo root
# so we add it to sys.path before importing shared files
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (  # noqa: E402
    CALIBRATION_METADATA_PATH,
    frequency_tracking_output_dir,
    load_yaml,
    median_reference_frame,
    project_path,
    relpath,
    undistort_gray,
    video_metadata,
    write_json,
)
from dot_lattice import (  # noqa: E402
    Roi,
    RotatedRoi,
    build_flat_lattice,
)


# cli and metadata loading
# this only needs the calibration metadata, the flat reference video,
# and the folder where the reference should be written
def parse_args() -> argparse.Namespace:
    # argparse turns command line text into python values and gives --help
    parser = argparse.ArgumentParser(
        description="Build a flat-liquid dot lattice reference for frequency-only analysis."
    )
    parser.add_argument(
        "--metadata",
        default=str(CALIBRATION_METADATA_PATH),
        # this metadata stores calibration_id, reference video paths, roi,
        # dot spacing, and camera/undistortion choices
        help="Calibration metadata YAML.",
    )
    parser.add_argument(
        "--reference-video",
        default=None,
        # this override is useful for one-off reference rebuilds without editing yaml
        help="Override the flat-liquid reference video. Defaults to dot_grid.flat_liquid_video_path.",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        default=None,
        metavar=("X", "Y", "W", "H"),
        # axis-aligned fallback
        # frequency tracking usually prefers rotated roi, but both are supported
        help="Override ROI as x y w h.",
    )
    parser.add_argument(
        "--rotated-roi",
        nargs=5,
        type=float,
        default=None,
        metavar=("CX", "CY", "W", "H", "ANGLE_DEG"),
        # rotated roi is [center x, center y, width, height, angle]
        # this lets the selected region follow the tilted dot grid
        help="Override rotated ROI as center_x center_y width height angle_deg.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to outputs/dot_tracking/frequency/reference/<calibration_id>.",
    )
    parser.add_argument("--max-reference-frames", type=int, default=100)
    parser.add_argument("--reference-stride", type=int, default=5)
    return parser.parse_args()


# roi resolution
# command-line values are for quick reruns, metadata values are the normal path
def roi_from_args_or_metadata(args: argparse.Namespace, metadata: dict) -> Roi | RotatedRoi:
    # priority order:
    # cli override, saved rotated roi, saved axis-aligned roi
    # this lets us test manually without changing the metadata file
    if args.rotated_roi is not None:
        return RotatedRoi.from_list(args.rotated_roi)
    if args.roi is not None:
        return Roi.from_list(args.roi)
    geom = metadata.get("geometry_initial_estimates", {})
    public_geom = metadata.get("geometry", {})
    # geometry_initial_estimates is the older setup field
    # geometry is the simplified public field, so read both
    rotated = geom.get("usable_roi_rotated_px")
    if rotated is None:
        rotated = public_geom.get("rotated_roi_px")
    if isinstance(rotated, dict):
        return RotatedRoi.from_dict(rotated)
    if isinstance(rotated, list) and len(rotated) == 5:
        return RotatedRoi.from_list(rotated)
    roi_values = geom.get("usable_roi_px", public_geom.get("roi_px"))
    if roi_values is None:
        raise ValueError(
            "No ROI supplied. Set geometry.roi_px, geometry.rotated_roi_px, "
            "geometry_initial_estimates.usable_roi_px, or pass --roi/--rotated-roi."
        )
    return Roi.from_list(roi_values)


def main() -> None:
    args = parse_args()
    # project_path accepts absolute paths and repo-relative paths
    metadata_path = project_path(args.metadata)
    if metadata_path is None:
        raise SystemExit("Missing metadata path.")

    # load calibration metadata and find the flat-liquid video
    # this is the reference dot position before wave motion
    metadata = load_yaml(metadata_path)
    dot_grid = metadata.get("dot_grid", {})
    videos = metadata.get("videos", {})

    # find the flat reference video from cli first, then new yaml keys,
    # then older compatibility keys
    reference_video = project_path(
        args.reference_video
        or dot_grid.get("flat_liquid_video_path")
        or videos.get("flat_reference")
        or metadata.get("flat_reference_video")
    )
    if reference_video is None or not reference_video.exists():
        raise FileNotFoundError(f"Missing flat-liquid reference video: {reference_video}")

    roi = roi_from_args_or_metadata(args, metadata)
    public_geom = metadata.get("geometry", {})
    # save physical dot spacing with the lattice so later scripts can attach
    # metric coordinates to each dot identity
    dot_spacing_mm = float(
        dot_grid.get(
            "dot_spacing_mm_actual",
            dot_grid.get("dot_spacing_mm", public_geom.get("dot_spacing_mm", 1.0)),
        )
    )

    out_dir = project_path(args.output_dir)
    if out_dir is None:
        # default location uses calibration_id because one calibration can be
        # reused by many videos
        out_dir = frequency_tracking_output_dir() / "reference" / str(metadata["calibration_id"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # build the median flat reference
    # taking a median over sampled frames removes transient noise but keeps dots sharp
    raw_gray = median_reference_frame(
        reference_video,
        max_frames=args.max_reference_frames,
        stride=args.reference_stride,
    )
    gray = undistort_gray(raw_gray, metadata)
    # clamp after undistortion in case the processed image dimensions changed
    roi = roi.clamped(gray.shape)

    # detect dark blobs inside the roi, then assign each dot an integer (i,j)
    # lattice identity
    dots, lattice, _binary, _signal_roi = build_flat_lattice(
        gray,
        roi,
        dot_spacing_mm=dot_spacing_mm,
    )

    # what this writes:
    # reference image for template matching
    # dot csv for identities/positions
    # lattice json for roi, spacing, and reference metadata
    frame_path = out_dir / "flat_reference_frame_undistorted.png"
    dots_path = out_dir / "flat_reference_dots.csv"
    lattice_path = out_dir / "flat_reference_lattice.json"

    cv2.imwrite(str(frame_path), gray)
    # one row per indexed dot
    # this is the identity table consumed by track_frequency_dots.py
    dots.to_csv(dots_path, index=False)

    lattice.update(
        {
            # schema/version fields make it clear what structure this json has
            "schema": "frequency_flat_reference_lattice_v1",
            "calibration_id": str(metadata["calibration_id"]),
            "metadata_path": relpath(metadata_path),
            "reference_video_path": relpath(reference_video),
            "reference_video_metadata": video_metadata(reference_video),
            "outputs": {
                "flat_reference_frame_undistorted_png": relpath(frame_path),
                "flat_reference_dots_csv": relpath(dots_path),
            },
        }
    )
    write_json(lattice_path, lattice)

    # print only the main result so batch logs stay readable
    print(f"Indexed {len(dots)} flat-liquid dots in ROI {roi.as_list()}.")
    print(f"Estimated dot spacing: {lattice['dot_spacing_px_median_nn']:.3f} px = {dot_spacing_mm:.3f} mm.")
    print(f"Saved {relpath(lattice_path)}")


if __name__ == "__main__":
    main()
