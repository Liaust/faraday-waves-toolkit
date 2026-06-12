from __future__ import annotations

# compatibility facade
# this does not implement tracking itself, it just re-exports names from the
# split tracking modules so older imports keep working

# the tracking code was split into focused modules, but some scripts still import
# these names from dot_lattice
from dot_affine import subtract_global_affine
from dot_detection import (
    detect_dark_dots,
    detect_dot_candidates,
    dot_signal,
    filter_candidates_to_roi,
)
from dot_geometry import Roi, RoiLike, RotatedRoi, apply_roi, roi_from_lattice_data
from lattice_indexing import (
    build_flat_lattice,
    grid_neighbor_spacing_metrics,
    indexed_neighbor_segments,
    indexing_consistency_metrics,
    nearest_neighbor_spacing_metrics,
)
from template_tracking import extract_patch, refine_peak_quadratic, track_frame_templates

__all__ = [
    "Roi",
    "RoiLike",
    "RotatedRoi",
    "apply_roi",
    "build_flat_lattice",
    "detect_dark_dots",
    "detect_dot_candidates",
    "dot_signal",
    "extract_patch",
    "filter_candidates_to_roi",
    "grid_neighbor_spacing_metrics",
    "indexed_neighbor_segments",
    "indexing_consistency_metrics",
    "nearest_neighbor_spacing_metrics",
    "refine_peak_quadratic",
    "roi_from_lattice_data",
    "subtract_global_affine",
    "track_frame_templates",
]
