from __future__ import annotations

'''
Just a set of tools for the roi selector app
'''

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

# set the global paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOT_TRACKING_DIR = PROJECT_ROOT / "scripts" / "dot_tracking"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DOT_TRACKING_DIR) not in sys.path:
    sys.path.insert(0, str(DOT_TRACKING_DIR))

# optional metadata override
_metadata_env = os.environ.get("FARADAY_CALIBRATION_METADATA_PATH") or os.environ.get("FSSS_METADATA_PATH")
METADATA_PATH = (
    Path(_metadata_env).expanduser()
    if _metadata_env
    else PROJECT_ROOT / "inputs" / "calibration_metadata.yaml"
)
if not METADATA_PATH.is_absolute():
    METADATA_PATH = PROJECT_ROOT / METADATA_PATH

# imported after sys.path setup
# build_flat_lattice is the same core helper used by frequency tracking
from dot_lattice import ( 
    Roi as FrequencyRoi,
    RotatedRoi as FrequencyRotatedRoi,
    build_flat_lattice,
    grid_neighbor_spacing_metrics,
    indexing_consistency_metrics,
)

RotatedRoi = FrequencyRotatedRoi

# axis aligned roi, mainly for the full fsss dry grid preview
@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_list(cls, values: list[int | float]) -> "Roi":
        # metadata can store floats but image slicing needs ints
        x, y, w, h = [int(round(float(v))) for v in values]
        return cls(x=x, y=y, w=w, h=h)

    def as_list(self) -> list[int]:
        return [int(self.x), int(self.y), int(self.w), int(self.h)]

    def clamped(self, image_shape: tuple[int, int]) -> "Roi":
        # keep the roi inside the image and avoid zero width/height
        height, width = image_shape[:2]
        x = int(np.clip(self.x, 0, max(0, width - 1)))
        y = int(np.clip(self.y, 0, max(0, height - 1)))
        w = int(np.clip(self.w, 1, max(1, width - x)))
        h = int(np.clip(self.h, 1, max(1, height - y)))
        return Roi(x, y, w, h)

def load_yaml(path: Path = METADATA_PATH) -> dict:
    # empty yaml gives None, so normalize to {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def save_yaml(data: dict, path: Path = METADATA_PATH) -> None:
    # keep yaml ordering readable instead of alphabetical
    path.write_text(yaml.safe_dump(data, sort_keys=False, width=1000), encoding="utf-8")

def project_path(path_like: str | Path | None) -> Path | None:
    # metadata paths are repo-relative unless already absolute
    if path_like in (None, ""):
        return None
    p = Path(path_like)
    return p if p.is_absolute() else PROJECT_ROOT / p

def calibration_output_dir(metadata: dict) -> Path:
    # conventional full fsss calibration output folder
    return PROJECT_ROOT / "outputs" / "full_fsss" / "calibration" / str(metadata["calibration_id"])

def finite_float(value, fallback: float = float("nan")) -> float:
    # helper for optional metrics that may be none/nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) else fallback

def finite_int(value, fallback: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return number if number > 0 else fallback

def current_roi(metadata: dict) -> Roi:
    # read current fsss rectangular roi
    # if missing, give a placeholder so the app can still open
    roi = metadata.get("geometry_initial_estimates", {}).get("usable_roi_px")
    if roi is None:
        return Roi(0, 0, 500, 500)
    return Roi.from_list(roi)

def current_frequency_roi(metadata: dict) -> FrequencyRotatedRoi:
    # read current frequency rotated roi
    # fallback is just the rectangular roi converted to zero angle
    geom = metadata.get("geometry_initial_estimates", {})
    rotated = geom.get("usable_roi_rotated_px")
    if isinstance(rotated, dict):
        return FrequencyRotatedRoi.from_dict(rotated)
    if isinstance(rotated, list) and len(rotated) == 5:
        return FrequencyRotatedRoi.from_list(rotated)
    axis = current_roi(metadata)
    return FrequencyRotatedRoi.from_axis_aligned(FrequencyRoi.from_list(axis.as_list()))

def save_roi_to_metadata(roi: Roi, metadata_path: Path = METADATA_PATH) -> None:
    # save both old and new roi fields so every script sees the same update
    metadata = load_yaml(metadata_path)
    metadata.setdefault("geometry_initial_estimates", {})["usable_roi_px"] = roi.as_list()
    metadata.setdefault("geometry", {})["roi_px"] = roi.as_list()
    save_yaml(metadata, metadata_path)

def save_frequency_roi_to_metadata(roi: FrequencyRotatedRoi, metadata_path: Path = METADATA_PATH) -> None:
    metadata = load_yaml(metadata_path)
    # save as named fields because a list like [x,y,w,h,a] is easy to misread
    roi_dict = {
        "cx": float(roi.cx),
        "cy": float(roi.cy),
        "w": float(roi.w),
        "h": float(roi.h),
        "angle_deg": float(roi.angle_deg),
    }
    metadata.setdefault("geometry_initial_estimates", {})["usable_roi_rotated_px"] = roi_dict
    metadata.setdefault("geometry", {})["rotated_roi_px"] = roi_dict
    save_yaml(metadata, metadata_path)

def load_intrinsics(metadata: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # load camera intrinsics either from metadata itself or from the calibration json
    camera = metadata.get("camera", {}) or {}
    inline_matrix = camera.get("camera_matrix")
    if inline_matrix not in (None, ""):
        # inline mode, useful for small examples without a separate json file
        k_raw = np.array(inline_matrix, dtype=float)
        dist_raw = np.array(
            camera.get("distortion_coefficients", camera.get("dist_coeffs", [])) or [],
            dtype=float,
        ).reshape(-1, 1)
        if dist_raw.size == 0:
            dist_raw = np.zeros((5, 1), dtype=float)
        k_undist = np.array(
            camera.get("new_camera_matrix_alpha0", camera.get("new_camera_matrix", inline_matrix)),
            dtype=float,
        )
        return k_raw, dist_raw, k_undist

    configured_path = project_path(camera.get("intrinsics_path"))
    path = configured_path if configured_path is not None else calibration_output_dir(metadata) / "camera_intrinsics.json"
    if not path.exists():
        # fallback for the simplified camera calibration output folder
        fallback = PROJECT_ROOT / "outputs" / "camera_calibration" / str(metadata["calibration_id"]) / "camera_intrinsics.json"
        path = fallback if fallback.exists() else path
    if not path.exists():
        raise FileNotFoundError(f"Camera intrinsics not found: {path}. Run camera calibration first or set camera.skip_lens_undistortion.")
    intrinsics = json.loads(path.read_text(encoding="utf-8"))
    k_raw = np.array(intrinsics["camera_matrix"], dtype=float)
    dist_raw = np.array(
        intrinsics.get("distortion_coefficients", intrinsics.get("dist_coeffs", [])),
        dtype=float,
    ).reshape(-1, 1)
    if dist_raw.size == 0:
        dist_raw = np.zeros((5, 1), dtype=float)
    k_undist = np.array(intrinsics.get("new_camera_matrix_alpha0", intrinsics["camera_matrix"]), dtype=float)
    return k_raw, dist_raw, k_undist

def read_video_metadata(video_path: Path) -> dict:
    # read video metadata without loading the whole video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info

def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    # jump to a frame and read it
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame

def median_reference_frame(video_path: Path, max_frames: int = 80, stride: int = 5) -> np.ndarray:
    # build a median image so waves/noise get suppressed and the dot pattern stays
    info = read_video_metadata(video_path)
    frame_count = info["frame_count"]
    if frame_count <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    indices = np.arange(0, frame_count, max(1, int(stride)), dtype=int)
    if len(indices) > max_frames:
        # if stride gives too many frames, sample uniformly across the video
        indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in indices:
        frame = read_frame_at(cap, int(idx))
        if frame is None:
            continue
        # tracking uses intensity, not color
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from: {video_path}")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

def first_reference_frame(video_path: Path) -> np.ndarray:
    # useful for seeing the real run framing before waves develop
    cap = cv2.VideoCapture(str(video_path))
    frame = read_frame_at(cap, 0)
    cap.release()
    if frame is None:
        raise RuntimeError(f"No first frame could be read from: {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def undistort_gray(gray: np.ndarray, metadata: dict) -> np.ndarray:
    # optionally undistort the image if metadata defines it
    # frequency sometimes skips this, fsss may require it
    image_processing = metadata.get("image_processing", {})
    camera = metadata.get("camera", {}) or {}
    if (
        image_processing.get("skip_lens_undistortion_for_roi_and_frequency", False)
        or camera.get("skip_lens_undistortion", False)
    ):
        return gray
    try:
        k_raw, dist_raw, k_undist = load_intrinsics(metadata)
    except FileNotFoundError:
        if image_processing.get("require_lens_undistortion_for_frequency", False) or camera.get("require_lens_undistortion", False):
            raise
        return gray
    return cv2.undistort(gray, k_raw, dist_raw, None, k_undist)

def discover_first_run_video(metadata: dict) -> Path | None:
    # find one real run video for preview if batch metadata exists
    batch_path = PROJECT_ROOT / "inputs" / "batch_metadata.yaml"
    if batch_path.exists():
        batch = yaml.safe_load(batch_path.read_text(encoding="utf-8")) or {}
        for run in batch.get("runs", []) or []:
            candidate = project_path(run.get("video_path"))
            if candidate is not None and candidate.exists():
                return candidate
    return None

def load_reference_images(metadata: dict) -> dict[str, np.ndarray]:
    # load all reference images used in the ui
    # flat liquid is required because both modes need it
    dot_grid = metadata.get("dot_grid", {}) or {}
    videos = metadata.get("videos", {}) or {}
    dry_video = project_path(dot_grid.get("dry_video_path") or videos.get("dry_grid"))
    empty_video = project_path(dot_grid.get("empty_bath_video_path") or videos.get("empty_container"))
    flat_video = project_path(dot_grid.get("flat_liquid_video_path") or videos.get("flat_reference"))
    run_video = discover_first_run_video(metadata)

    images: dict[str, np.ndarray] = {}
    if dry_video is not None and dry_video.exists():
        images["dry dot grid + acrylic"] = undistort_gray(median_reference_frame(dry_video), metadata)
    if flat_video is None or not flat_video.exists():
        raise FileNotFoundError("Missing flat-liquid reference video. Set dot_grid.flat_liquid_video_path or videos.flat_reference.")
    images["flat liquid reference"] = undistort_gray(median_reference_frame(flat_video), metadata)
    if empty_video is not None and empty_video.exists():
        images["empty container"] = undistort_gray(median_reference_frame(empty_video), metadata)
    if run_video is not None:
        images["first run first frame"] = undistort_gray(first_reference_frame(run_video), metadata)
        images["first run median"] = undistort_gray(median_reference_frame(run_video), metadata)
    return images

def draw_roi_overlay(gray: np.ndarray, roi: Roi, color: tuple[int, int, int] = (0, 255, 40)) -> Image.Image:
    # draw rectangular roi on the grayscale image
    roi = roi.clamped(gray.shape)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = roi.x, roi.y, roi.x + roi.w, roi.y + roi.h
    for offset in range(3):
        draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)
    return img

def draw_frequency_roi_overlay(
    gray: np.ndarray,
    roi: FrequencyRotatedRoi,
    color: tuple[int, int, int] = (0, 255, 40),
) -> Image.Image:
    # draw rotated roi polygon
    roi = roi.clamped(gray.shape)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    corners = np.round(roi.corners()).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        rgb,
        [corners],
        isClosed=True,
        color=color,
        thickness=5,
        lineType=cv2.LINE_AA,
    )
    return Image.fromarray(rgb)

def draw_frequency_dots_overlay(
    gray: np.ndarray,
    roi: FrequencyRoi | FrequencyRotatedRoi,
    dots: pd.DataFrame,
) -> Image.Image:
    # draw detected/indexed dots on top of the flat reference image
    roi = roi.clamped(gray.shape)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    corners = np.round(roi.corners()).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(rgb, [corners], isClosed=True, color=(0, 255, 40), thickness=4, lineType=cv2.LINE_AA)
    if not dots.empty:
        for _, dot in dots.iterrows():
            center = (int(round(float(dot["x_px"]))), int(round(float(dot["y_px"]))))
            cv2.circle(rgb, center, 4, (255, 210, 0), 1, lineType=cv2.LINE_AA)
            if "i" in dots.columns and "j" in dots.columns:
                label = f"{int(dot['i'])},{int(dot['j'])}"
                cv2.putText(
                    rgb,
                    label,
                    (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (255, 210, 0),
                    1,
                    cv2.LINE_AA,
                )
    return Image.fromarray(rgb)

def draw_tracking_overlay(
    gray: np.ndarray,
    roi: Roi,
    tracked: pd.DataFrame,
    max_vectors: int = 400,
) -> Image.Image:
    # draw valid/invalid matches and a few arrows so tracking failure is visible
    roi = roi.clamped(gray.shape)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(rgb, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 40), 4, lineType=cv2.LINE_AA)
    if tracked.empty:
        return Image.fromarray(rgb)

    valid = tracked[tracked["valid"]]
    invalid = tracked[~tracked["valid"]]
    for _, dot in invalid.iterrows():
        center = (int(round(float(dot["x_ref_px"]))), int(round(float(dot["y_ref_px"]))))
        cv2.circle(rgb, center, 4, (255, 80, 80), 1, lineType=cv2.LINE_AA)
    for _, dot in valid.iterrows():
        center = (int(round(float(dot["x_px"]))), int(round(float(dot["y_px"]))))
        cv2.circle(rgb, center, 4, (80, 255, 80), 1, lineType=cv2.LINE_AA)

    if len(valid):
        subset = valid.sample(min(len(valid), max_vectors), random_state=1)
        for _, dot in subset.iterrows():
            start = (int(round(float(dot["x_ref_px"]))), int(round(float(dot["y_ref_px"]))))
            end = (int(round(float(dot["x_px"]))), int(round(float(dot["y_px"]))))
            cv2.arrowedLine(rgb, start, end, (255, 220, 0), 1, tipLength=0.25, line_type=cv2.LINE_AA)
    return Image.fromarray(rgb)

def apply_roi(gray: np.ndarray, roi: Roi) -> tuple[np.ndarray, Roi]:
    # crop image to the roi and return the clamped roi that was actually used
    roi = roi.clamped(gray.shape)
    return gray[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w], roi

def detect_dark_dots(
    gray: np.ndarray,
    roi_offset: tuple[int, int] = (0, 0),
    threshold_mode: str = "otsu",
    blur_ksize: int = 3,
    morph_open_iter: int = 0,
    morph_close_iter: int = 0,
    min_area_px: float | None = None,
    max_area_px: float | None = None,
    min_circularity: float = 0.15,
    border_margin_px: int = 3,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # dark dot detection
    # based on DoG and Otsu
    # then normalize so dark dots become bright blobs
    img = gray.copy()
    if blur_ksize and blur_ksize > 1:
        img = cv2.GaussianBlur(img, (int(blur_ksize), int(blur_ksize)), 0)

    bg_kernel = max(31, int(min(gray.shape[:2]) // 12) | 1)
    bg = cv2.GaussianBlur(img, (bg_kernel, bg_kernel), 0)
    norm = cv2.normalize((bg.astype(np.float32) - img.astype(np.float32)), None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)

    if threshold_mode == "adaptive":
        # adaptive threshold helps when lighting changes across the roi
        binary = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -2)
    elif threshold_mode == "otsu":
        # otsu picks one threshold from the enhanced image histogram
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("threshold_mode must be 'otsu' or 'adaptive'")

    if morph_open_iter:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=int(morph_open_iter))
    if morph_close_iter:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=int(morph_close_iter))

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rows = []
    height, width = gray.shape[:2]
    for label in range(1, nlabels):
        # label 0 is background, everything else is a candidate dot blob
        x, y, bw, bh, area = stats[label]
        cx, cy = centroids[label]
        if cx < border_margin_px or cy < border_margin_px:
            continue
        if cx > width - border_margin_px or cy > height - border_margin_px:
            continue
        if min_area_px is not None and area < min_area_px:
            continue
        if max_area_px is not None and area > max_area_px:
            continue
        perimeter = 2.0 * (bw + bh)
        circ = 4.0 * math.pi * float(area) / (perimeter**2) if perimeter > 0 else 0.0
        if circ < min_circularity:
            continue
        rows.append(
            {
                "x_px": float(cx + roi_offset[0]),
                "y_px": float(cy + roi_offset[1]),
                "area_px": float(area),
                "bbox_x": int(x + roi_offset[0]),
                "bbox_y": int(y + roi_offset[1]),
                "bbox_w": int(bw),
                "bbox_h": int(bh),
                "circularity": float(circ),
            }
        )

    return pd.DataFrame(rows), binary, norm

def detect_dots_in_roi(dry_img: np.ndarray, roi: Roi) -> pd.DataFrame:
    # two pass detection
    # first get rough dots, then use median area to filter weird blobs
    roi_img, roi = apply_roi(dry_img, roi)
    first, _, _ = detect_dark_dots(roi_img, roi_offset=(roi.x, roi.y))
    if len(first) > 20:
        med_area = float(np.median(first["area_px"]))
        min_area = max(2, 0.25 * med_area)
        max_area = 4.0 * med_area
        dots, _, _ = detect_dark_dots(roi_img, roi_offset=(roi.x, roi.y), min_area_px=min_area, max_area_px=max_area)
    else:
        dots = first

    dots = dots.reset_index(drop=True)
    dots.insert(0, "dot_id", np.arange(len(dots), dtype=int))
    return dots

def refine_peak_quadratic(corr: np.ndarray, x: int, y: int) -> tuple[float, float]:
    # subpixel correction around the integer template match peak
    # using a quadratic fit in x and y
    height, width = corr.shape
    dx = 0.0
    dy = 0.0
    if 1 <= x < width - 1:
        denom = corr[y, x - 1] - 2 * corr[y, x] + corr[y, x + 1]
        if abs(float(denom)) > 1e-9:
            dx = 0.5 * (corr[y, x - 1] - corr[y, x + 1]) / denom
            dx = float(np.clip(dx, -0.5, 0.5))
    if 1 <= y < height - 1:
        denom = corr[y - 1, x] - 2 * corr[y, x] + corr[y + 1, x]
        if abs(float(denom)) > 1e-9:
            dy = 0.5 * (corr[y - 1, x] - corr[y + 1, x]) / denom
            dy = float(np.clip(dy, -0.5, 0.5))
    return dx, dy

def extract_patch(img: np.ndarray, cx: float, cy: float, radius: int) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    # square patch around a dot
    # return none if it would go outside the image
    x0 = int(round(cx)) - radius
    y0 = int(round(cy)) - radius
    x1 = int(round(cx)) + radius + 1
    y1 = int(round(cy)) + radius + 1
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        return None, None
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)

def estimate_spacing_px(dots: pd.DataFrame) -> float:
    # quick grid spacing estimate from nearest neighbors
    if len(dots) < 3:
        return float("nan")
    points = dots[["x_px", "y_px"]].to_numpy(float)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(np.median(dists[:, 1]))

def track_dots_template(
    reference_img: np.ndarray,
    target_img: np.ndarray,
    dots: pd.DataFrame,
    template_radius: int,
    search_radius: int,
    min_score: float = 0.35,
) -> pd.DataFrame:
    # simple template tracking from reference image to target image
    # each dot has a small template and a larger search window
    rows = []
    for _, r in dots.iterrows():
        cx = float(r["x_px"])
        cy = float(r["y_px"])
        template, _ = extract_patch(reference_img, cx, cy, template_radius)
        if template is None:
            continue

        search, sb = extract_patch(target_img, cx, cy, search_radius)
        if search is None or sb is None:
            continue

        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        # minMaxLoc gives the best correlation location
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        px, py = max_loc
        sub_dx, sub_dy = refine_peak_quadratic(result, px, py)

        x0, y0, _, _ = sb
        matched_cx = x0 + px + sub_dx + template_radius
        matched_cy = y0 + py + sub_dy + template_radius
        dx = matched_cx - cx
        dy = matched_cy - cy

        rows.append(
            {
                "dot_id": int(r["dot_id"]),
                "x_ref_px": cx,
                "y_ref_px": cy,
                "x_px": float(matched_cx),
                "y_px": float(matched_cy),
                "dx_px": float(dx),
                "dy_px": float(dy),
                "match_score": float(max_val),
                "valid": bool(max_val >= min_score),
            }
        )
    return pd.DataFrame(rows)

def tracking_summary(tracked: pd.DataFrame, label: str) -> dict:
    # collapse all per-dot tracking rows into a few ui metrics
    if tracked.empty:
        return {
            "label": label,
            "n_total": 0,
            "n_valid": 0,
            "valid_fraction": 0.0,
            "median_match_score": float("nan"),
            "rms_displacement_px": float("nan"),
        }
    valid = tracked[tracked["valid"]]
    return {
        "label": label,
        "n_total": int(len(tracked)),
        "n_valid": int(tracked["valid"].sum()),
        "valid_fraction": float(tracked["valid"].mean()),
        "median_match_score": float(np.median(tracked["match_score"])),
        "median_abs_dx_px": float(np.median(np.abs(valid["dx_px"]))) if len(valid) else float("nan"),
        "median_abs_dy_px": float(np.median(np.abs(valid["dy_px"]))) if len(valid) else float("nan"),
        "rms_displacement_px": float(np.sqrt(np.mean(valid["dx_px"] ** 2 + valid["dy_px"] ** 2))) if len(valid) else float("nan"),
    }

def evaluate_roi(
    metadata: dict,
    roi: Roi,
    min_match_score: float = 0.35,
    template_fraction: float = 0.25,
    search_fraction: float = 0.85,
) -> dict:
    # fsss roi check
    # detect dots in the dry image, then try matching those same dots into
    # empty container and flat liquid images
    images = load_reference_images(metadata)
    missing = [name for name in ("dry dot grid + acrylic", "empty container", "flat liquid reference") if name not in images]
    if missing:
        raise FileNotFoundError(
            "Full FSSS ROI evaluation requires dry-grid, empty-container, and flat-liquid reference videos. "
            f"Missing: {', '.join(missing)}"
        )
    dry = images["dry dot grid + acrylic"]
    empty = images["empty container"]
    flat = images["flat liquid reference"]
    roi = roi.clamped(dry.shape)

    dots = detect_dots_in_roi(dry, roi)
    spacing_px = estimate_spacing_px(dots)
    if not np.isfinite(spacing_px):
        raise RuntimeError("Too few dots were detected to estimate grid spacing.")

    template_radius = max(3, int(round(template_fraction * spacing_px)))
    search_radius = max(template_radius + 2, int(round(search_fraction * spacing_px)))

    # these two tracking checks show if refraction/lighting makes matching ambiguous
    empty_tracks = track_dots_template(dry, empty, dots, template_radius, search_radius, min_score=min_match_score)
    flat_tracks = track_dots_template(dry, flat, dots, template_radius, search_radius, min_score=min_match_score)

    return {
        "roi_px": roi.as_list(),
        "detected_dry_dots": int(len(dots)),
        "estimated_spacing_px": float(spacing_px),
        "template_radius_px": int(template_radius),
        "search_radius_px": int(search_radius),
        "min_match_score": float(min_match_score),
        "empty_bath": tracking_summary(empty_tracks, "empty_bath"),
        "flat_liquid": tracking_summary(flat_tracks, "flat_liquid"),
        "_roi": roi,
        "_dry_dots": dots,
        "_empty_tracks": empty_tracks,
        "_flat_tracks": flat_tracks,
    }

def evaluate_frequency_roi(metadata: dict, roi: FrequencyRoi | FrequencyRotatedRoi) -> dict:
    # frequency roi check
    # detect/index dots directly in the flat liquid image
    # this is the one that actually matters for frequency tracking
    images = load_reference_images(metadata)
    flat = images["flat liquid reference"]
    roi = roi.clamped(flat.shape)
    dot_spacing_mm = float(metadata.get("dot_grid", {}).get("dot_spacing_mm_actual", 1.0))

    dots, lattice, _, _ = build_flat_lattice(
        flat,
        roi,
        dot_spacing_mm=dot_spacing_mm,
    )

    # combine lattice-builder diagnostics with some extra spacing/indexing metrics
    spacing_metrics = grid_neighbor_spacing_metrics(dots)
    indexing_metrics = indexing_consistency_metrics(dots)

    lattice_neighbor_count = finite_int(lattice.get("neighbor_pair_count"))
    neighbor_pair_count = finite_int(lattice_neighbor_count, int(spacing_metrics["neighbor_pair_count"]))
    neighbor_metric_source = (
        str(lattice.get("neighbor_metric_source", "unknown"))
        if lattice_neighbor_count > 0
        else str(spacing_metrics.get("neighbor_metric_source", "unknown"))
    )
    summary = {
        # normal scalar diagnostics for display, plus private in-memory objects
        # with underscores that are only used for drawing overlays
        "mode": "frequency_estimation_flat_reference",
        "reference_builder_version": int(lattice.get("reference_builder_version", 0)),
        "roi": lattice["roi"],
        "roi_kind": lattice["roi_kind"],
        "roi_px": roi.as_list(),
        "roi_bounding_px": lattice.get("roi_bounding_px"),
        "roi_edge_exclusion_px": lattice.get("roi_edge_exclusion_px"),
        "detected_flat_dots": int(len(dots)),
        "n_candidates": int(lattice.get("n_candidates", len(dots))),
        "n_after_edge_exclusion": int(lattice.get("n_after_edge_exclusion", len(dots))),
        "n_indexed_dots": int(lattice.get("n_indexed_dots", len(dots))),
        "estimated_spacing_px": float(lattice["dot_spacing_px_median_nn"]),
        "dot_spacing_mm": dot_spacing_mm,
        "median_lattice_residual_px": float(lattice["median_lattice_residual_px"]),
        "p95_lattice_residual_px": float(lattice["p95_lattice_residual_px"]),
        "grid_slot_count": int(lattice.get("grid_slot_count", 0)),
        "grid_occupancy_fraction": finite_float(lattice.get("grid_occupancy_fraction")),
        "indexing_duplicate_cell_count": int(lattice.get("indexing_duplicate_cell_count", 0)),
        "indexing_duplicate_candidate_count": int(lattice.get("indexing_duplicate_candidate_count", 0)),
        "indexing_duplicate_removed_count": int(lattice.get("indexing_duplicate_removed_count", 0)),
        "neighbor_metric_source": neighbor_metric_source,
        "neighbor_pair_count": neighbor_pair_count,
        "local_spacing_median_px": finite_float(lattice.get("local_spacing_median_px"), float(spacing_metrics["local_spacing_median_px"])),
        "local_spacing_p05_px": finite_float(lattice.get("local_spacing_p05_px"), float(spacing_metrics["local_spacing_p05_px"])),
        "local_spacing_p95_px": finite_float(lattice.get("local_spacing_p95_px"), float(spacing_metrics["local_spacing_p95_px"])),
        "local_spacing_cv_percent": finite_float(lattice.get("local_spacing_cv_percent"), float(spacing_metrics["local_spacing_cv_percent"])),
        "local_spacing_p95_abs_error_px": finite_float(lattice.get("local_spacing_p95_abs_error_px"), float(spacing_metrics["local_spacing_p95_abs_error_px"])),
        "nearest_neighbor_pair_count": finite_int(lattice.get("nearest_neighbor_pair_count"), int(spacing_metrics["nearest_neighbor_pair_count"])),
        "nearest_neighbor_spacing_median_px": finite_float(lattice.get("nearest_neighbor_spacing_median_px"), float(spacing_metrics["nearest_neighbor_spacing_median_px"])),
        "nearest_neighbor_spacing_cv_percent": finite_float(lattice.get("nearest_neighbor_spacing_cv_percent"), float(spacing_metrics["nearest_neighbor_spacing_cv_percent"])),
        "nearest_neighbor_spacing_p95_abs_error_px": finite_float(lattice.get("nearest_neighbor_spacing_p95_abs_error_px"), float(spacing_metrics["nearest_neighbor_spacing_p95_abs_error_px"])),
        "indexed_neighbor_pair_count": finite_int(lattice.get("indexed_neighbor_pair_count"), int(spacing_metrics["indexed_neighbor_pair_count"])),
        "indexed_neighbor_spacing_median_px": finite_float(lattice.get("indexed_neighbor_spacing_median_px"), float(spacing_metrics["indexed_neighbor_spacing_median_px"])),
        "indexed_neighbor_spacing_cv_percent": finite_float(lattice.get("indexed_neighbor_spacing_cv_percent"), float(spacing_metrics["indexed_neighbor_spacing_cv_percent"])),
        "indexed_neighbor_spacing_p95_abs_error_px": finite_float(lattice.get("indexed_neighbor_spacing_p95_abs_error_px"), float(spacing_metrics["indexed_neighbor_spacing_p95_abs_error_px"])),
        "indexed_neighbor_normal_length_fraction": finite_float(lattice.get("indexed_neighbor_normal_length_fraction"), float(spacing_metrics["indexed_neighbor_normal_length_fraction"])),
        "indexed_neighbor_marginal_length_fraction": finite_float(lattice.get("indexed_neighbor_marginal_length_fraction"), float(spacing_metrics["indexed_neighbor_marginal_length_fraction"])),
        "indexed_neighbor_bad_length_fraction": finite_float(lattice.get("indexed_neighbor_bad_length_fraction"), float(spacing_metrics["indexed_neighbor_bad_length_fraction"])),
        "indexed_neighbor_diagonal_length_fraction": finite_float(lattice.get("indexed_neighbor_diagonal_length_fraction"), float(spacing_metrics["indexed_neighbor_diagonal_length_fraction"])),
        "indexing_knn_relation_count": int(lattice.get("indexing_knn_relation_count", indexing_metrics["indexing_knn_relation_count"])),
        "indexing_knn_cardinal_fraction": finite_float(lattice.get("indexing_knn_cardinal_fraction"), float(indexing_metrics["indexing_knn_cardinal_fraction"])),
        "indexing_knn_diagonal_fraction": finite_float(lattice.get("indexing_knn_diagonal_fraction"), float(indexing_metrics["indexing_knn_diagonal_fraction"])),
        "indexing_knn_nonlocal_fraction": finite_float(lattice.get("indexing_knn_nonlocal_fraction"), float(indexing_metrics["indexing_knn_nonlocal_fraction"])),
        "indexing_median_cardinal_neighbors_per_dot": finite_float(lattice.get("indexing_median_cardinal_neighbors_per_dot"), float(indexing_metrics["indexing_median_cardinal_neighbors_per_dot"])),
        "indexing_suspect_dot_count": int(lattice.get("indexing_suspect_dot_count", indexing_metrics["indexing_suspect_dot_count"])),
        "indexing_suspect_dot_fraction": finite_float(lattice.get("indexing_suspect_dot_fraction"), float(indexing_metrics["indexing_suspect_dot_fraction"])),
        "indexing_method": str(lattice.get("indexing_method", "unknown")),
        "auto_axis_angle_deg": finite_float(lattice.get("auto_axis_angle_deg")),
        "selected_axis_angle_deg": finite_float(lattice.get("selected_axis_angle_deg", lattice.get("axis_angle_deg"))),
        "grid_i_span": [
            int(lattice["grid_i_min"]),
            int(lattice["grid_i_max"]),
        ],
        "grid_j_span": [
            int(lattice["grid_j_min"]),
            int(lattice["grid_j_max"]),
        ],
        "_roi": roi,
        "_dots": dots,
    }
    return summary
