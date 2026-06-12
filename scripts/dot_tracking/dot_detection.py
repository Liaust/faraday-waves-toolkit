from __future__ import annotations

# image-processing steps that turn a grayscale dot-grid image into candidate
# dot centroids

import math
from typing import Any

import cv2
import numpy as np
import pandas as pd

from dot_geometry import Roi, RoiLike, apply_roi


# dark-dot enhancement image
# estimate the slow background with a wide gaussian blur, subtract the lightly
# blurred image, and keep only positive contrast so dark dots become bright blobs
def dot_signal(gray: np.ndarray) -> np.ndarray:
    # use float32 so subtraction does not underflow at 0 like uint8 would
    img = gray.astype(np.float32)
    # small blur removes pixel-scale noise but keeps dot-scale structure
    small = cv2.GaussianBlur(img, (3, 3), 0)
    # large odd kernel estimates illumination/background over a scale much
    # larger than one dot. `| 1` forces an odd kernel size for opencv
    bg_kernel = max(31, int(min(gray.shape[:2]) // 10) | 1)
    bg = cv2.GaussianBlur(small, (bg_kernel, bg_kernel), 0)
    # dots are dark, so background minus image is positive at dot locations
    signal = bg - small
    # negative contrast is not useful for dark-dot detection
    signal = np.maximum(signal, 0)
    if float(np.nanmax(signal)) > 0:
        # rescale to 0..255 so standard opencv thresholding behaves well
        signal = cv2.normalize(signal, None, 0, 255, cv2.NORM_MINMAX)
    return signal.astype(np.uint8)


# threshold and connected-component candidate extraction
# binarize the enhanced image, split it into connected blobs,
# and keep blobs that look dot-like
def detect_dark_dots(
    gray_roi: np.ndarray,
    *,
    roi_offset: tuple[int, int] = (0, 0),
    threshold_mode: str = "otsu",
    min_area_px: float | None = None,
    max_area_px: float | None = None,
    min_circularity: float = 0.12,
    border_margin_px: int = 3,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # first create the enhanced image where dark dots appear bright
    signal = dot_signal(gray_roi)
    if threshold_mode == "adaptive":
        # adaptive threshold uses local thresholds
        # helpful for uneven lighting, but can also admit more noise
        binary = cv2.adaptiveThreshold(
            signal,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            -2,
        )
    elif threshold_mode == "otsu":
        # Otsu chooses one global threshold from the signal histogram
        _, binary = cv2.threshold(signal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("threshold_mode must be 'otsu' or 'adaptive'")

    # opening removes tiny foreground specks before connected components
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    # connectedComponentsWithStats labels each white blob and returns its
    # bounding box, area, and centroid
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rows: list[dict[str, Any]] = []
    height, width = gray_roi.shape[:2]

    # label 0 is background, every later label is one candidate blob
    for label in range(1, nlabels):
        x, y, bw, bh, area = stats[label]
        cx, cy = centroids[label]
        # reject dots too close to the crop edge because they may be partial dots
        if cx < border_margin_px or cy < border_margin_px:
            continue
        if cx > width - border_margin_px or cy > height - border_margin_px:
            continue
        if min_area_px is not None and area < min_area_px:
            continue
        if max_area_px is not None and area > max_area_px:
            continue
        # cheap circularity estimate using the bounding-box perimeter
        # mostly filters elongated glare/streaks, not true circle geometry
        perimeter = 2.0 * (bw + bh)
        circularity = 4.0 * math.pi * float(area) / (perimeter**2) if perimeter > 0 else 0.0
        if circularity < min_circularity:
            continue
        ix = int(np.clip(round(cx), 0, width - 1))
        iy = int(np.clip(round(cy), 0, height - 1))
        rows.append(
            {
                # add roi_offset so caller receives full-image coordinates
                "x_px": float(cx + roi_offset[0]),
                "y_px": float(cy + roi_offset[1]),
                "area_px": float(area),
                "bbox_x": int(x + roi_offset[0]),
                "bbox_y": int(y + roi_offset[1]),
                "bbox_w": int(bw),
                "bbox_h": int(bh),
                "circularity": float(circularity),
                "dark_signal": float(signal[iy, ix]),
            }
        )
    return pd.DataFrame(rows), binary, signal


# roi filtering happens after blob detection
# this lets both axis-aligned and rotated rois use the same detector
def filter_candidates_to_roi(candidates: pd.DataFrame, roi: RoiLike, margin_px: float = 0.0) -> pd.DataFrame:
    # detection starts from an axis-aligned bounding crop even for rotated rois
    # this does the exact roi membership test on centroids
    if candidates.empty:
        return candidates.copy()
    x = candidates["x_px"].to_numpy(float)
    y = candidates["y_px"].to_numpy(float)
    keep = roi.contains_points(x, y, margin_px=float(margin_px))
    return candidates.loc[keep].reset_index(drop=True)


# candidate detector
# crop to roi bounds, detect dark blobs, retry with adaptive threshold if Otsu
# is too sparse, then refine area limits from the first good population
def detect_dot_candidates(gray: np.ndarray, roi: RoiLike) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, Roi, RoiLike]:
    # clamp roi to image bounds, then use its bounding box as the numpy crop
    roi = roi.clamped(gray.shape) if hasattr(roi, "clamped") else roi
    bounds = roi.bounding_roi(gray.shape)
    roi_img, bounds = apply_roi(gray, bounds)
    # first pass uses Otsu because it is simple and deterministic
    first, binary, signal = detect_dark_dots(roi_img, roi_offset=(bounds.x, bounds.y))
    first = filter_candidates_to_roi(first, roi, margin_px=0.0)
    if len(first) < 30:
        # if Otsu finds too few dots, try adaptive thresholding
        # only keep it if it actually finds more in-roi dots
        second, binary, signal = detect_dark_dots(
            roi_img,
            roi_offset=(bounds.x, bounds.y),
            threshold_mode="adaptive",
        )
        second = filter_candidates_to_roi(second, roi, margin_px=0.0)
        if len(second) > len(first):
            first = second

    if len(first) > 30:
        # use first-pass dot areas to set robust area limits
        med_area = float(np.median(first["area_px"]))
        min_area = max(2.0, 0.25 * med_area)
        max_area = 4.5 * med_area
        refined, binary, signal = detect_dark_dots(
            roi_img,
            roi_offset=(bounds.x, bounds.y),
            threshold_mode="otsu",
            min_area_px=min_area,
            max_area_px=max_area,
        )
        refined = filter_candidates_to_roi(refined, roi, margin_px=0.0)
        if len(refined) >= 0.75 * len(first):
            # keep the refined set only if it did not throw away too many dots
            first = refined

    first = first.reset_index(drop=True)
    return first, binary, signal, bounds, roi
