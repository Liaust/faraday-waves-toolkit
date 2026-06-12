from __future__ import annotations

# local template matching
# for each dot, we search a small reference patch inside each video frame

import math

import cv2
import numpy as np
import pandas as pd


# reference patch extraction
# crop a square window around a dot or search center
# return None near borders so the caller can just mark that dot invalid
def extract_patch(img: np.ndarray, cx: float, cy: float, radius: int) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    # array slicing needs integer indices, so round the center to a pixel
    # subpixel location is recovered later from the correlation peak
    x0 = int(round(cx)) - radius
    y0 = int(round(cy)) - radius
    x1 = int(round(cx)) + radius + 1
    y1 = int(round(cy)) + radius + 1
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        # do not pad edge patches
        # padding changes template statistics and makes edge dots less comparable
        return None, None
    # return the patch and its full-image top-left origin
    # origin is needed to convert matchTemplate coordinates back to image pixels
    return img[y0:y1, x0:x1], (x0, y0)


# subpixel quadratic peak refinement
# fit a tiny parabola around the best correlation pixel in x and y,
# then move the peak by at most half a pixel
def refine_peak_quadratic(corr: np.ndarray, x: int, y: int) -> tuple[float, float]:
    # corr is a 2d similarity map
    # x,y are the best integer-pixel maximum from cv2.minMaxLoc
    height, width = corr.shape
    dx = 0.0
    dy = 0.0
    if 1 <= x < width - 1:
        # fit a parabola through the three horizontal samples around the peak
        # the vertex gives the subpixel offset relative to x
        denom = corr[y, x - 1] - 2.0 * corr[y, x] + corr[y, x + 1]
        if abs(float(denom)) > 1e-9:
            dx = 0.5 * (corr[y, x - 1] - corr[y, x + 1]) / denom
            dx = float(np.clip(dx, -0.5, 0.5))
    if 1 <= y < height - 1:
        # same calculation in the vertical direction
        denom = corr[y - 1, x] - 2.0 * corr[y, x] + corr[y + 1, x]
        if abs(float(denom)) > 1e-9:
            dy = 0.5 * (corr[y - 1, x] - corr[y + 1, x]) / denom
            dy = float(np.clip(dy, -0.5, 0.5))
    return dx, dy


# template tracking for one frame
# for each reference dot, take its reference patch, search nearby in this frame,
# refine the match peak, and apply score/displacement checks
def track_frame_templates(
    reference_signal: np.ndarray,
    frame_signal: np.ndarray,
    reference_dots: pd.DataFrame,
    *,
    template_radius_px: int,
    search_radius_px: int,
    min_match_score: float,
    max_displacement_px: float,
    search_centers_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # allocate outputs for every reference dot
    # nan/False stays wherever a dot cannot be matched
    n = len(reference_dots)
    xy = np.full((n, 2), np.nan, dtype=float)
    score = np.full(n, np.nan, dtype=float)
    valid = np.zeros(n, dtype=bool)
    for pos, row in enumerate(reference_dots.itertuples(index=False)):
        # reference coordinates are the flat-reference dot location
        cx = float(row.x_px)
        cy = float(row.y_px)
        if search_centers_xy is None:
            # default search center is the reference dot itself
            sx = cx
            sy = cy
        else:
            # adaptive tracking can pass predicted centers from global shift or
            # previous frame positions
            sx = float(search_centers_xy[pos, 0])
            sy = float(search_centers_xy[pos, 1])
            if not (np.isfinite(sx) and np.isfinite(sy)):
                sx = cx
                sy = cy
        template, _ = extract_patch(reference_signal, cx, cy, template_radius_px)
        search, origin = extract_patch(frame_signal, sx, sy, search_radius_px)
        if template is None or search is None or origin is None:
            continue
        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            # matchTemplate needs the search image to be at least as large as the template
            continue

        # cv2.matchTemplate slides the reference patch over the search image
        # TM_CCOEFF_NORMED returns normalized correlation, where higher is better
        corr = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        # for TM_CCOEFF_NORMED, max value/location is the best match
        _, max_val, _, max_loc = cv2.minMaxLoc(corr)
        px, py = max_loc
        sub_dx, sub_dy = refine_peak_quadratic(corr, px, py)
        x0, y0 = origin
        # max_loc is the top-left of the best template placement in the crop
        # add template radius to recover the matched dot center
        mx = x0 + px + sub_dx + template_radius_px
        my = y0 + py + sub_dy + template_radius_px
        # displacement limit is measured relative to the search center
        # not necessarily the original reference point
        disp = math.hypot(mx - sx, my - sy)
        xy[pos] = (mx, my)
        score[pos] = float(max_val)
        # reject weak matches and large jumps relative to this frame's search center
        valid[pos] = bool(max_val >= min_match_score and disp <= max_displacement_px)
    return xy, score, valid
