#!/usr/bin/env python3
from __future__ import annotations

# this tracks dry-grid dot identities into the flat-liquid reference
# then fits the flat pixel/object homography
# and writes the optical stack used by ray tracing

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


# repo root and imports so this can run directly from the command line
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
from dot_lattice import dot_signal, track_frame_templates  # noqa: E402

DEFAULT_TEMPLATE_RADIUS_FRACTION = 0.25
DEFAULT_SEARCH_RADIUS_FRACTION = 0.85
DEFAULT_MIN_MATCH_SCORE = 0.35
DEFAULT_MIN_VALID_FLAT_DOTS = 20


# cli for the flat-liquid calibration stage
# dry-grid identities -> flat-liquid dots -> optical stack metadata
def parse_args() -> argparse.Namespace:
    # keep the public cli small
    # metadata says which videos/geometry to use, output-dir says where to write
    parser = argparse.ArgumentParser(
        description=(
            "Track dry-grid dots into the flat-liquid reference and build the optical geometry used by full FSSS."
        )
    )
    parser.add_argument("--metadata", default=str(CALIBRATION_METADATA_PATH), help="Calibration metadata YAML.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Calibration output directory. Defaults to outputs/full_fsss/calibration/<calibration_id>.",
    )
    return parser.parse_args()


def calibration_output_dir(metadata: dict[str, Any], override: str | None) -> Path:
    # all fsss calibration files for this setup live in one folder
    configured = project_path(override)
    if configured is not None:
        return configured
    return PROJECT_ROOT / "outputs" / "full_fsss" / "calibration" / str(metadata["calibration_id"])


def fsss_intrinsics_path(metadata: dict[str, Any], output_dir: Path) -> Path:
    # same intrinsics resolution convention as calibrate_dot_grid_pose.py.
    camera = metadata.get("camera", {}) or {}
    configured = project_path(camera.get("intrinsics_path"))
    if configured is not None:
        return configured
    output_copy = output_dir / "camera_intrinsics.json"
    if output_copy.exists():
        return output_copy
    return PROJECT_ROOT / "outputs" / "camera_calibration" / str(metadata["calibration_id"]) / "camera_intrinsics.json"


# load camera intrinsics in the same processed-pixel convention used by the
# dry-grid pose calibration.
def load_fsss_intrinsics(metadata: dict[str, Any], output_dir: Path) -> tuple[Path, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    camera = metadata.get("camera", {}) or {}
    inline_matrix = camera.get("camera_matrix")
    if inline_matrix not in (None, ""):
        # metadata can embed intrinsics directly.
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
    dist = np.array(
        intrinsics.get("distortion_coefficients", intrinsics.get("dist_coeffs", [])),
        dtype=float,
    ).reshape(-1, 1)
    if dist.size == 0:
        dist = np.zeros((5, 1), dtype=float)
    k_undist = np.array(intrinsics.get("new_camera_matrix_alpha0", intrinsics["camera_matrix"]), dtype=float)
    return path, intrinsics, k_raw, dist, k_undist


def use_full_fsss_lens_undistortion(metadata: dict[str, Any]) -> bool:
    # one place to decide whether this run uses undistorted or raw pixels
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
        # return raw image when full-fsss undistortion was disabled.
        return gray
    return cv2.undistort(gray, k_raw, dist, None, k_undist)


def camera_matrix_for_processed_pixels(metadata: dict[str, Any], k_raw: np.ndarray, k_undist: np.ndarray) -> np.ndarray:
    # store the camera matrix that corresponds to the processed pixels in output
    # json for downstream interpretation.
    return k_undist if use_full_fsss_lens_undistortion(metadata) else k_raw


# resolve dry-grid and flat-liquid calibration videos from calibration metadata.
def calibration_video(
    metadata: dict[str, Any],
    *,
    dot_grid_key: str,
    videos_key: str,
    required: bool,
) -> Path | None:
    # metadata supports both new dot_grid keys and older videos keys.
    dot_grid = metadata.get("dot_grid", {}) or {}
    videos = metadata.get("videos", {}) or {}
    path = project_path(dot_grid.get(dot_grid_key) or videos.get(videos_key))
    if path is None:
        if required:
            raise ValueError(f"Missing calibration video for {dot_grid_key}/{videos_key}.")
        return None
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing calibration video: {path}")
        return None
    return path


# estimate pixel dot spacing from the dry-grid reference table to set template
# and search radii for dry-to-flat matching.
def estimate_spacing_px(reference_dots: pd.DataFrame) -> float:
    # estimate spacing from dry reference pixels so template/search radii scale
    # automatically with camera zoom/resolution.
    points = reference_dots[["x_px", "y_px"]].to_numpy(float)
    if len(points) < 4:
        return float("nan")
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    return float(np.median(nearest)) if len(nearest) else float("nan")


# convert template-tracking arrays into the flat_liquid_dots.csv table, keeping
# dry-grid identity, object coordinates, flat-liquid pixel position, displacement,
# match score, and validity.
def tracking_dataframe(reference_dots: pd.DataFrame, xy: np.ndarray, score: np.ndarray, valid: np.ndarray) -> pd.DataFrame:
    # convert vectorized template-tracking outputs into one csv row per dot.
    rows: list[dict[str, Any]] = []
    ref_xy = reference_dots[["x_px", "y_px"]].to_numpy(float)
    for pos, row in enumerate(reference_dots.itertuples(index=False)):
        # xy[pos] is the matched flat-liquid pixel position for this dry-grid dot
        # identity.
        measured_x = float(xy[pos, 0])
        measured_y = float(xy[pos, 1])
        rows.append(
            {
                "dot_id": int(row.dot_id),
                "i": int(row.i),
                "j": int(row.j),
                "x_grid_m": float(row.x_grid_m),
                "y_grid_m": float(row.y_grid_m),
                "z_grid_m": float(getattr(row, "z_grid_m", 0.0)),
                "x_ref_px": float(ref_xy[pos, 0]),
                "y_ref_px": float(ref_xy[pos, 1]),
                "x_px": measured_x,
                "y_px": measured_y,
                "dx_px": float(measured_x - ref_xy[pos, 0]) if np.isfinite(measured_x) else float("nan"),
                "dy_px": float(measured_y - ref_xy[pos, 1]) if np.isfinite(measured_y) else float("nan"),
                "match_score": float(score[pos]) if np.isfinite(score[pos]) else float("nan"),
                "valid": bool(valid[pos]),
            }
        )
    return pd.DataFrame(rows)


# track the already-indexed dry-grid reference dots into the flat-liquid image.
# this preserves dot identity while measuring the optical displacement caused by
# the filled bath and flat liquid surface.
def track_reference(
    *,
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    reference_dots: pd.DataFrame,
    template_radius_px: int,
    search_radius_px: int,
    min_match_score: float,
    max_displacement_px: float,
) -> pd.DataFrame:
    # reuse the shared template tracker: reference patch from dry signal, search
    # in flat-liquid signal.
    xy, score, valid = track_frame_templates(
        reference_signal,
        target_signal,
        reference_dots,
        template_radius_px=template_radius_px,
        search_radius_px=search_radius_px,
        min_match_score=min_match_score,
        max_displacement_px=max_displacement_px,
    )
    return tracking_dataframe(reference_dots, xy, score, valid)


# small set of numbers to check whether dry-to-flat tracking behaved
def tracking_summary(df: pd.DataFrame, label: str) -> dict[str, Any]:
    # summarize valid fraction, scores, and displacement size for the dry-to-flat
    # matching stage.
    valid = df[df["valid"]].copy()
    if valid.empty:
        return {
            "label": label,
            "n_total": int(len(df)),
            "n_valid": 0,
            "valid_fraction": 0.0,
            "median_match_score": float("nan"),
            "median_abs_dx_px": float("nan"),
            "median_abs_dy_px": float("nan"),
            "rms_displacement_px": float("nan"),
        }
    return {
        "label": label,
        "n_total": int(len(df)),
        "n_valid": int(len(valid)),
        "valid_fraction": float(len(valid) / len(df)) if len(df) else float("nan"),
        "median_match_score": float(np.nanmedian(valid["match_score"])),
        "median_abs_dx_px": float(np.nanmedian(np.abs(valid["dx_px"]))),
        "median_abs_dy_px": float(np.nanmedian(np.abs(valid["dy_px"]))),
        "rms_displacement_px": float(np.sqrt(np.nanmean(valid["dx_px"] ** 2 + valid["dy_px"] ** 2))),
    }


# fit the empirical flat-reference mapping. Object-plane coordinates in mm map
# to flat-liquid pixels with a homography; the inverse homography later converts
# tracked pixels back to apparent object-plane positions.
def fit_flat_reference_mapping(flat_dots: pd.DataFrame) -> dict[str, Any]:
    # object coordinates are in meters in flat_dots.csv; homography fitting here
    # uses millimeters for numerically convenient values.
    obj_mm = flat_dots[["x_grid_m", "y_grid_m"]].to_numpy(float) * 1000.0
    pix = flat_dots[["x_px", "y_px"]].to_numpy(float)

    candidates: list[dict[str, Any]] = []
    specs = [
        # try multiple opencv homography methods and keep the lowest median
        # residual. RANSAC can reject occasional bad dot matches.
        ("direct", 0, None),
        ("lmeds", cv2.LMEDS, None),
        ("ransac_5px", cv2.RANSAC, 5.0),
        ("ransac_10px", cv2.RANSAC, 10.0),
    ]
    for method_name, method, threshold in specs:
        if method == cv2.RANSAC:
            # RANSAC threshold is a reprojection threshold in pixels.
            h_candidate, mask = cv2.findHomography(
                obj_mm.astype(np.float64),
                pix.astype(np.float64),
                method=method,
                ransacReprojThreshold=float(threshold),
            )
        else:
            h_candidate, mask = cv2.findHomography(
                obj_mm.astype(np.float64),
                pix.astype(np.float64),
                method=method,
            )
        if h_candidate is None:
            continue
        # predict pixels from object coordinates and compute residuals.
        predicted_h = (h_candidate @ np.column_stack([obj_mm, np.ones(len(obj_mm))]).T).T
        predicted = predicted_h[:, :2] / predicted_h[:, 2:3]
        residual = np.linalg.norm(pix - predicted, axis=1)
        candidates.append(
            {
                "method": method_name,
                "H": h_candidate,
                "mask": mask,
                "predicted": predicted,
                "residual": residual,
                "median_residual": float(np.median(residual)),
            }
        )

    if not candidates:
        raise RuntimeError("cv2.findHomography failed for the flat-liquid reference.")
    best = min(candidates, key=lambda item: item["median_residual"])
    h_obj_to_px = np.asarray(best["H"], dtype=float)
    # inverse homography is what tracking uses to convert measured pixels back
    # into apparent object-plane coordinates.
    h_px_to_obj = np.linalg.inv(h_obj_to_px)
    residual = np.asarray(best["residual"], dtype=float)
    mask = best["mask"]
    inlier_count = int(mask.sum()) if mask is not None else int(len(flat_dots))

    design = np.column_stack([np.ones(len(obj_mm)), obj_mm[:, 0], obj_mm[:, 1]])
    # also fit a simpler affine model to estimate px/mm scale and residuals.
    beta_u, *_ = np.linalg.lstsq(design, pix[:, 0], rcond=None)
    beta_v, *_ = np.linalg.lstsq(design, pix[:, 1], rcond=None)
    pix_affine = np.column_stack([design @ beta_u, design @ beta_v])
    affine_residual = np.linalg.norm(pix - pix_affine, axis=1)
    px_per_mm_x = float(np.linalg.norm([beta_u[1], beta_v[1]]))
    px_per_mm_y = float(np.linalg.norm([beta_u[2], beta_v[2]]))

    mapping = {
        "object_xy_units": "mm",
        "homography_object_mm_to_pixel": h_obj_to_px.tolist(),
        "homography_pixel_to_object_mm": h_px_to_obj.tolist(),
        "homography_fit_method": str(best["method"]),
        "homography_inlier_count": inlier_count,
        "homography_median_residual_px": float(np.median(residual)),
        "homography_max_residual_px": float(np.max(residual)),
        "affine_u_coefficients": [float(x) for x in beta_u],
        "affine_v_coefficients": [float(x) for x in beta_v],
        "affine_model": "u=bu0+bu1*x_mm+bu2*y_mm; v=bv0+bv1*x_mm+bv2*y_mm",
        "affine_median_residual_px": float(np.median(affine_residual)),
        "affine_max_residual_px": float(np.max(affine_residual)),
        "px_per_mm_x": px_per_mm_x,
        "px_per_mm_y": px_per_mm_y,
        "mm_per_px_x": float(1.0 / px_per_mm_x) if px_per_mm_x > 0 else float("nan"),
        "mm_per_px_y": float(1.0 / px_per_mm_y) if px_per_mm_y > 0 else float("nan"),
    }
    return mapping


# liquid depth can be supplied directly or derived from total mass, density, and
# container footprint area. The raytrace optical stack needs this height.
def liquid_depth_mm(metadata: dict[str, Any]) -> float:
    # prefer explicit depth if present.
    fluid = metadata.get("fluid", {}) or {}
    for key in ("liquid_depth_mm", "bath_height_mm", "height_mm"):
        value = fluid.get(key)
        if value not in (None, ""):
            return float(value)

    total_mass_g = fluid.get("total_mass_g")
    density = fluid.get("density_kg_per_m3", fluid.get("density_kg_m3"))
    container = metadata.get("container", {}) or {}
    area_mm2 = container.get("footprint_area_mm2")
    if total_mass_g not in (None, "") and density not in (None, "") and area_mm2 not in (None, ""):
        # depth = volume / area, with mass/density giving volume.
        volume_m3 = float(total_mass_g) * 1e-3 / float(density)
        area_m2 = float(area_mm2) * 1e-6
        return float(volume_m3 / area_m2 * 1000.0)
    raise ValueError(
        "Missing liquid depth. Set fluid.liquid_depth_mm or fluid.bath_height_mm, "
        "or provide fluid.total_mass_g, fluid.density_kg_m3, and container.footprint_area_mm2."
    )


# build the layered optical stack from dot plane to liquid surface: optional air
# gaps, bath/container layers, liquid depth, and refractive indices.
def build_optical_stack(metadata: dict[str, Any]) -> dict[str, Any]:
    # the stack is a sequence of flat layers from dot plane upward to liquid
    # surface. Ray tracing later walks through this stack.
    fluid = metadata.get("fluid", {}) or {}
    container = metadata.get("container", {}) or {}
    raytrace = metadata.get("raytrace", {}) or {}
    stack_meta = metadata.get("optical_stack", {}) or {}

    n_air = float(raytrace.get("air_refractive_index", 1.0))
    n_liquid = float(fluid["refractive_index"])
    liquid_depth_m = liquid_depth_mm(metadata) * 1e-3
    layers: list[dict[str, Any]] = []
    z = 0.0

    dot_gap_m = float(stack_meta.get("dot_to_first_layer_gap_mm", 0.0)) * 1e-3
    if dot_gap_m > 0:
        # optional air gap between printed dots and first physical layer.
        layers.append(
            {
                "name": "air_gap_above_dot_grid",
                "material": "air",
                "z_bottom_m": z,
                "z_top_m": z + dot_gap_m,
                "thickness_m": dot_gap_m,
                "thickness_mm": dot_gap_m * 1000.0,
                "refractive_index": n_air,
                "present_in_dry_reference": True,
            }
        )
        z += dot_gap_m

    layer_defs = stack_meta.get("layers_dot_to_liquid")
    if layer_defs:
        # explicit stack metadata can describe arbitrary layers.
        for raw in layer_defs:
            name = str(raw.get("name", raw.get("material", f"layer_{len(layers)}")))
            gap_below_m = float(raw.get("gap_below_mm", 0.0)) * 1e-3
            if gap_below_m > 0:
                # optional air gap before this layer.
                layers.append(
                    {
                        "name": f"air_gap_below_{name}",
                        "material": "air",
                        "z_bottom_m": z,
                        "z_top_m": z + gap_below_m,
                        "thickness_m": gap_below_m,
                        "thickness_mm": gap_below_m * 1000.0,
                        "refractive_index": n_air,
                        "present_in_dry_reference": bool(raw.get("present_in_dry_reference", False)),
                    }
                )
                z += gap_below_m
            thickness_m = float(raw["thickness_mm"]) * 1e-3
            n_layer = float(raw.get("refractive_index", raw.get("n")))
            layers.append(
                {
                    "name": name,
                    "material": str(raw.get("material", name)),
                    "z_bottom_m": z,
                    "z_top_m": z + thickness_m,
                    "thickness_m": thickness_m,
                    "thickness_mm": thickness_m * 1000.0,
                    "refractive_index": n_layer,
                    "present_in_dry_reference": bool(raw.get("present_in_dry_reference", False)),
                }
            )
            z += thickness_m
    else:
        # fallback stack built from container metadata.
        gap_m = float(container.get("dot_to_outer_bottom_gap_mm", 0.0)) * 1e-3
        if gap_m > 0:
            layers.append(
                {
                    "name": "air_gap_below_container_bottom",
                    "material": "air",
                    "z_bottom_m": z,
                    "z_top_m": z + gap_m,
                    "thickness_m": gap_m,
                    "thickness_mm": gap_m * 1000.0,
                    "refractive_index": n_air,
                    "present_in_dry_reference": True,
                }
            )
            z += gap_m
        include_bottom = bool(raytrace.get("include_bath_bottom", True))
        bottom_thickness_m = float(container.get("bottom_thickness_mm", 0.0)) * 1e-3
        if include_bottom and bottom_thickness_m > 0:
            n_bottom = float(container.get("bottom_refractive_index", 1.49))
            layers.append(
                {
                    "name": "container_bottom",
                    "material": str(container.get("bottom_material", "container_bottom")),
                    "z_bottom_m": z,
                    "z_top_m": z + bottom_thickness_m,
                    "thickness_m": bottom_thickness_m,
                    "thickness_mm": bottom_thickness_m * 1000.0,
                    "refractive_index": n_bottom,
                    "present_in_dry_reference": False,
                }
            )
            z += bottom_thickness_m

    solid_top_m = max([float(layer["z_top_m"]) for layer in layers], default=0.0)
    container_layer = None
    for layer in layers:
        name = str(layer["name"]).lower()
        if name in {"container_bottom", "bath_bottom", "glass_bottom"} or "container" in name or "glass" in name:
            container_layer = layer
    if container_layer is None and layers:
        container_layer = layers[-1]

    stack = {
        # z coordinates are measured from the printed dot plane.
        "schema_version": 2,
        "z_dot_plane_m": 0.0,
        "z_solid_top_m": solid_top_m,
        "z_free_surface_m": solid_top_m + liquid_depth_m,
        "liquid_depth_m": liquid_depth_m,
        "n_air": n_air,
        "n_liquid": n_liquid,
        "layers_dot_to_liquid": layers,
        "dry_reference_state": stack_meta.get("dry_reference_state", "dot_grid_reference_as_recorded"),
        "dry_reference_note": stack_meta.get("dry_reference_note", "The pose is estimated from the recorded dry/reference grid video."),
        "include_bottom": bool(layers),
    }
    if container_layer is not None:
        stack.update(
            {
                "z_outer_bottom_m": float(container_layer["z_bottom_m"]),
                "z_inner_bottom_m": float(container_layer["z_top_m"]),
                "n_bottom": float(container_layer["refractive_index"]),
                "bottom_layer_name": str(container_layer["name"]),
            }
        )
    else:
        stack.update(
            {
                "z_outer_bottom_m": 0.0,
                "z_inner_bottom_m": 0.0,
                "n_bottom": n_air,
                "bottom_layer_name": None,
            }
        )
    return stack


def main() -> int:
    args = parse_args()
    # load the dry-grid pose calibration that ran before this step
    metadata_path = project_path(args.metadata)
    if metadata_path is None or not metadata_path.exists():
        raise FileNotFoundError(f"Missing calibration metadata: {args.metadata}")
    metadata = load_yaml(metadata_path)
    output_dir = calibration_output_dir(metadata, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dot_ref_path = output_dir / "dot_grid_reference.csv"
    pose_path = output_dir / "camera_pose_grid.json"
    if not dot_ref_path.exists():
        raise FileNotFoundError(f"Missing dry-grid reference: {dot_ref_path}")
    if not pose_path.exists():
        raise FileNotFoundError(f"Missing camera pose: {pose_path}")

    intrinsics_path, _, k_raw, dist, k_undist = load_fsss_intrinsics(metadata, output_dir)
    k_image = camera_matrix_for_processed_pixels(metadata, k_raw, k_undist)
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    dot_ref = pd.read_csv(dot_ref_path)

    dry_path = calibration_video(
        metadata,
        dot_grid_key="dry_video_path",
        videos_key="dry_grid",
        required=True,
    )
    flat_path = calibration_video(
        metadata,
        dot_grid_key="flat_liquid_video_path",
        videos_key="flat_reference",
        required=True,
    )

    tracking_cfg = metadata.get("tracking", {}) or {}
    max_frames = int(tracking_cfg.get("max_reference_frames", 80))
    stride = int(tracking_cfg.get("reference_stride", 5))
    min_match_score = float(tracking_cfg.get("min_match_score", DEFAULT_MIN_MATCH_SCORE))

    # build median reference images for both states so the dry-to-flat matching
    # uses clean, stationary dot patterns rather than individual noisy frames.
    dry_raw = median_reference_frame(dry_path, max_frames=max_frames, stride=stride)
    flat_raw = median_reference_frame(flat_path, max_frames=max_frames, stride=stride)
    # both dry and flat images must use the same processed-pixel convention.
    dry_gray = undistort_for_full_fsss(dry_raw, metadata, k_raw, dist, k_undist)
    flat_gray = undistort_for_full_fsss(flat_raw, metadata, k_raw, dist, k_undist)
    cv2.imwrite(str(output_dir / "flat_liquid_reference_undistorted.png"), flat_gray)

    dry_signal = dot_signal(dry_gray)
    flat_signal = dot_signal(flat_gray)
    spacing_px = estimate_spacing_px(dot_ref)
    if not np.isfinite(spacing_px) or spacing_px <= 0:
        raise RuntimeError("Could not estimate dry-grid dot spacing in pixels.")
    template_fraction = float(tracking_cfg.get("template_radius_fraction", DEFAULT_TEMPLATE_RADIUS_FRACTION))
    search_fraction = float(tracking_cfg.get("search_radius_fraction", DEFAULT_SEARCH_RADIUS_FRACTION))
    template_radius = max(3, int(round(template_fraction * spacing_px)))
    # search radius is larger because dots can move appreciably when liquid is
    # introduced.
    search_radius = max(template_radius + 2, int(round(search_fraction * spacing_px)))
    max_displacement_fraction = float(tracking_cfg.get("max_displacement_fraction", search_fraction))
    max_displacement = max(template_radius + 2, float(max_displacement_fraction) * spacing_px)

    # track dry-grid dot identities into the flat-liquid reference image. Valid
    # matches define the fsss flat reference used by later disturbed-run tracking.
    flat_tracked = track_reference(
        reference_signal=dry_signal,
        target_signal=flat_signal,
        reference_dots=dot_ref,
        template_radius_px=template_radius,
        search_radius_px=search_radius,
        min_match_score=min_match_score,
        max_displacement_px=max_displacement,
    )
    flat_valid = flat_tracked[flat_tracked["valid"]].copy().reset_index(drop=True)
    # only valid dry-to-flat correspondences become the fsss flat reference.
    min_valid_flat_dots = int(tracking_cfg.get("min_valid_flat_dots", DEFAULT_MIN_VALID_FLAT_DOTS))
    if len(flat_valid) < min_valid_flat_dots:
        raise RuntimeError(
            f"Too few valid flat-liquid dots: {len(flat_valid)}. "
            "Increase search radius, lower min match score, or check calibration videos."
        )

    flat_summary = tracking_summary(flat_tracked, "flat_liquid")

    flat_valid.to_csv(output_dir / "flat_liquid_dots.csv", index=False)
    mapping = fit_flat_reference_mapping(flat_valid)
    stack = build_optical_stack(metadata)

    # what this writes:
    # flat_liquid_dots.csv becomes the fsss tracking reference
    # optical_geometry_refined.json feeds the ray-traced calibration
    optical_geometry = {
        "calibration_id": str(metadata["calibration_id"]),
        "description": (
            "Empirical flat-liquid reference and optical stack metadata. "
            "This file is the input to the later curved-surface ray-tracing calibration."
        ),
        "units": {
            "length": "metres unless field name says mm",
            "pixel": "processed image pixels using camera_matrix_used_for_undistorted_pixels",
        },
        "source_files": {
            "metadata": relpath(metadata_path),
            "camera_intrinsics": relpath(intrinsics_path) if str(intrinsics_path) != "<metadata camera matrix>" else str(intrinsics_path),
            "camera_pose_grid": relpath(pose_path),
            "dot_grid_reference": relpath(dot_ref_path),
            "flat_liquid_dots": relpath(output_dir / "flat_liquid_dots.csv"),
            "flat_liquid_reference_image": relpath(output_dir / "flat_liquid_reference_undistorted.png"),
        },
        "camera": {
            "camera_matrix_used_for_undistorted_pixels": k_image.tolist(),
            "lens_undistortion_applied": bool(use_full_fsss_lens_undistortion(metadata)),
            "R_world_to_camera": pose["R_world_to_camera"],
            "tvec_world_to_camera_m": pose["tvec_world_to_camera_m"],
            "camera_center_world_m": pose["camera_center_world_m"],
        },
        "optical_stack": stack,
        "flat_liquid_empirical_mapping": mapping,
        "flat_reference_mapping": mapping,
        "flat_reference_tracking_summary": flat_summary,
        "tracking_settings": {
            "template_radius_px": int(template_radius),
            "search_radius_px": int(search_radius),
            "min_match_score": float(min_match_score),
            "max_displacement_px": float(max_displacement),
        },
        "notes": [
            "For FSSS tracking, use flat_liquid_dots.csv as the reference dot positions.",
            "The empirical homography is a flat-reference mapping, not yet the curved-surface displacement-to-gradient calibration.",
            "If flat ray-trace residuals are large later, check bottom thickness, dot-to-bottom gap, camera pose, and whether the bath bottom is flat.",
        ],
    }
    write_json(output_dir / "optical_geometry_refined.json", optical_geometry)

    print(f"Valid flat-liquid dots: {len(flat_valid)} / {len(flat_tracked)}")
    print(f"Homography median residual: {mapping['homography_median_residual_px']:.4f} px")
    print(f"Saved {relpath(output_dir / 'flat_liquid_dots.csv')}")
    print(f"Saved {relpath(output_dir / 'optical_geometry_refined.json')}")
    print("Flat video metadata:", video_metadata(flat_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
