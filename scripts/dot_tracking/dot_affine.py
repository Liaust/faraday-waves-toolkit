from __future__ import annotations

# removes frame-wide affine motion so later analysis sees residual dot deformation

import cv2
import numpy as np


# remove frame-wide motion from tracked dots
# the affine transform maps flat-reference coordinates to measured coordinates
# then we subtract the predicted affine motion and keep only residual deformation
def subtract_global_affine(
    reference_xy: np.ndarray,
    measured_xy: np.ndarray,
    valid: np.ndarray,
    *,
    ransac_threshold_px: float,
    reject_outliers: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    # corrected has one 2d residual vector per dot
    # it starts as nan for dots that are not usable
    corrected = np.full_like(measured_xy, np.nan, dtype=float)
    # raw displacement is simply measured minus reference position
    raw = measured_xy - reference_xy
    # only valid dots with finite coordinates can estimate the global affine transform
    enough = valid & np.all(np.isfinite(measured_xy), axis=1)
    if int(np.sum(enough)) < 6:
        # a 2d affine-like transform needs enough points to be stable
        # if not, fall back to raw displacement for valid dots
        corrected[enough] = raw[enough]
        return corrected, valid.copy(), None

    # estimateAffinePartial2D fits translation + rotation + uniform scale +
    # shear-like partial affine parameters with RANSAC outlier rejection
    affine, inliers = cv2.estimateAffinePartial2D(
        reference_xy[enough].astype(np.float32),
        measured_xy[enough].astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_px),
        maxIters=2000,
        confidence=0.995,
    )
    final_valid = valid.copy()
    if affine is None:
        # if fitting fails, return raw displacement instead of crashing
        corrected[enough] = raw[enough]
        return corrected, final_valid, None

    # convert each reference point to homogeneous [x,y,1], multiply by the 2x3
    # affine matrix, and get predicted frame-wide motion
    predicted = np.column_stack([reference_xy, np.ones(len(reference_xy))]) @ affine.T
    # residual displacement is measured position minus affine-predicted position
    corrected[valid] = measured_xy[valid] - predicted[valid]
    if reject_outliers and inliers is not None:
        # RANSAC inliers correspond only to the "enough" subset
        # map rejected rows back to original dot indices and invalidate them
        good_indices = np.flatnonzero(enough)
        rejected = good_indices[inliers.ravel().astype(bool) == 0]
        final_valid[rejected] = False
        corrected[rejected] = np.nan
    return corrected, final_valid, affine
