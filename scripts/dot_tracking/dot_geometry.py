from __future__ import annotations

# roi geometry classes and point-in-roi tests used by detection and tracking

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


# axis-aligned roi used for simple rectangular crops
# it has the same core methods as RotatedRoi so later code can treat both similarly
@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_list(cls, values: list[int | float]) -> "Roi":
        # metadata may contain float-like values
        # rectangular numpy slices need integer pixels, so round here
        x, y, w, h = [int(round(float(v))) for v in values]
        return cls(x, y, w, h)

    def as_list(self) -> list[int]:
        # compact yaml/json representation
        return [int(self.x), int(self.y), int(self.w), int(self.h)]

    def as_dict(self) -> dict[str, float | str]:
        # dict representation keeps the roi kind, useful when reading lattice json
        return {
            "kind": "axis_aligned",
            "x": float(self.x),
            "y": float(self.y),
            "w": float(self.w),
            "h": float(self.h),
        }

    @property
    def center(self) -> tuple[float, float]:
        # center is useful when converting this into a zero-angle rotated roi
        return (float(self.x) + 0.5 * float(self.w), float(self.y) + 0.5 * float(self.h))

    def clamped(self, image_shape: tuple[int, int]) -> "Roi":
        # clamp to image dimensions so later slicing cannot go out of bounds
        # image_shape is (height, width)
        height, width = image_shape[:2]
        x = int(np.clip(self.x, 0, max(0, width - 1)))
        y = int(np.clip(self.y, 0, max(0, height - 1)))
        w = int(np.clip(self.w, 1, max(1, width - x)))
        h = int(np.clip(self.h, 1, max(1, height - y)))
        return Roi(x, y, w, h)

    def bounding_roi(self, image_shape: tuple[int, int]) -> "Roi":
        # for an axis-aligned roi, the bounding rectangle is just itself
        return self.clamped(image_shape)

    def corners(self) -> np.ndarray:
        # return corners in x,y order for drawing and rotated-roi-compatible code
        x0 = float(self.x)
        y0 = float(self.y)
        x1 = x0 + float(self.w)
        y1 = y0 + float(self.h)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)

    def contains_points(self, x: np.ndarray, y: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
        # vectorized point-in-rectangle test
        # margin_px shrinks the rectangle to reject edge dots
        return (
            (x >= self.x + margin_px)
            & (x <= self.x + self.w - margin_px)
            & (y >= self.y + margin_px)
            & (y <= self.y + self.h - margin_px)
        )


# rotated roi used when the usable dot grid is not aligned with image axes
# points are transformed into the roi's local coordinates, then checked by width/height
@dataclass(frozen=True)
class RotatedRoi:
    cx: float
    cy: float
    w: float
    h: float
    angle_deg: float

    @classmethod
    def from_list(cls, values: list[int | float]) -> "RotatedRoi":
        # rotated roi list is [center_x, center_y, width, height, angle_degrees]
        cx, cy, w, h, angle_deg = [float(v) for v in values]
        return cls(cx=cx, cy=cy, w=w, h=h, angle_deg=angle_deg)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RotatedRoi":
        return cls(
            cx=float(values["cx"]),
            cy=float(values["cy"]),
            w=float(values["w"]),
            h=float(values["h"]),
            angle_deg=float(values.get("angle_deg", 0.0)),
        )

    @classmethod
    def from_axis_aligned(cls, roi: Roi) -> "RotatedRoi":
        cx, cy = roi.center
        return cls(cx=cx, cy=cy, w=float(roi.w), h=float(roi.h), angle_deg=0.0)

    def as_list(self) -> list[float]:
        return [float(self.cx), float(self.cy), float(self.w), float(self.h), float(self.angle_deg)]

    def as_dict(self) -> dict[str, float | str]:
        return {
            "kind": "rotated",
            "cx": float(self.cx),
            "cy": float(self.cy),
            "w": float(self.w),
            "h": float(self.h),
            "angle_deg": float(self.angle_deg),
        }

    def clamped(self, image_shape: tuple[int, int]) -> "RotatedRoi":
        # clamp center and size, but keep the angle unchanged
        height, width = image_shape[:2]
        return RotatedRoi(
            cx=float(np.clip(self.cx, 0, max(0, width - 1))),
            cy=float(np.clip(self.cy, 0, max(0, height - 1))),
            w=max(1.0, min(float(self.w), float(width))),
            h=max(1.0, min(float(self.h), float(height))),
            angle_deg=float(self.angle_deg),
        )

    def corners(self) -> np.ndarray:
        # build local rectangle corners, rotate them, then translate to the roi center
        theta = math.radians(float(self.angle_deg))
        c = math.cos(theta)
        s = math.sin(theta)
        local = np.array(
            [
                [-0.5 * self.w, -0.5 * self.h],
                [0.5 * self.w, -0.5 * self.h],
                [0.5 * self.w, 0.5 * self.h],
                [-0.5 * self.w, 0.5 * self.h],
            ],
            dtype=float,
        )
        rot = np.array([[c, -s], [s, c]], dtype=float)
        return local @ rot.T + np.array([self.cx, self.cy], dtype=float)

    def bounding_roi(self, image_shape: tuple[int, int]) -> Roi:
        # numpy cannot slice a rotated polygon directly
        # so use the axis-aligned bounding box for image crops
        corners = self.corners()
        x0 = int(math.floor(float(np.min(corners[:, 0]))))
        y0 = int(math.floor(float(np.min(corners[:, 1]))))
        x1 = int(math.ceil(float(np.max(corners[:, 0]))))
        y1 = int(math.ceil(float(np.max(corners[:, 1]))))
        return Roi(x0, y0, max(1, x1 - x0), max(1, y1 - y0)).clamped(image_shape)

    def local_coordinates(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # convert image coordinates into the rotated roi's local coordinate system
        # this applies the inverse rotation
        theta = math.radians(float(self.angle_deg))
        c = math.cos(theta)
        s = math.sin(theta)
        dx = x - float(self.cx)
        dy = y - float(self.cy)
        xp = c * dx + s * dy
        yp = -s * dx + c * dy
        return xp, yp

    def contains_points(self, x: np.ndarray, y: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
        # once in local coordinates, containment is just |x'| <= w/2 and |y'| <= h/2
        xp, yp = self.local_coordinates(x, y)
        return (
            (np.abs(xp) <= 0.5 * float(self.w) - margin_px)
            & (np.abs(yp) <= 0.5 * float(self.h) - margin_px)
        )


RoiLike = Roi | RotatedRoi


# recreate the roi object stored in a lattice json file
# older and newer metadata use slightly different names, so accept both
def roi_from_lattice_data(data: dict[str, Any]) -> RoiLike:
    # newer lattice json stores a full roi dict
    # older json may store roi_kind/roi_rotated_px or just roi_px
    if isinstance(data.get("roi"), dict) and data["roi"].get("kind") == "rotated":
        return RotatedRoi.from_dict(data["roi"])
    if data.get("roi_kind") == "rotated" and "roi_rotated_px" in data:
        return RotatedRoi.from_list(data["roi_rotated_px"])
    return Roi.from_list(data["roi_px"])


# crop image data to an axis-aligned bounding roi
# rotated roi filtering happens later at the point level
def apply_roi(gray: np.ndarray, roi: Roi) -> tuple[np.ndarray, Roi]:
    # image arrays are indexed [row, col] = [y, x], so y range comes first
    roi = roi.clamped(gray.shape)
    return gray[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w], roi
