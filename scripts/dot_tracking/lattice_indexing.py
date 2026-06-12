from __future__ import annotations

# this turns unordered dark-dot centroids into an indexed square lattice
# with integer (i,j) coordinates

import math
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dot_detection import detect_dot_candidates, filter_candidates_to_roi
from dot_geometry import RoiLike, RotatedRoi


# spacing estimation
# use a kd-tree nearest-neighbor query and take the median first-neighbor distance
# as the dot-grid spacing in pixels
def estimate_spacing_px(points: np.ndarray) -> float:
    # points is an (N, 2) array of candidate centroids in pixel coordinates
    if len(points) < 4:
        return float("nan")
    # cKDTree makes nearest-neighbor lookup fast for hundreds/thousands of dots
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=min(5, len(points)))
    # dists[:, 0] is each point's distance to itself
    # dists[:, 1] is the nearest actual neighbor
    nearest = dists[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    return float(np.median(nearest)) if len(nearest) else float("nan")


# angle estimation
# collect vectors to near-neighbor dots, fold angles into the square-lattice
# 90 degree symmetry, and use a weighted circular mean for the grid direction
def estimate_lattice_angle(points: np.ndarray, spacing_px: float) -> float:
    # infer the lattice axis orientation from near-neighbor vectors
    # the grid is square, so angles separated by 90 degrees are equivalent
    tree = cKDTree(points)
    k = min(9, len(points))
    dists, idx = tree.query(points, k=k)
    angles: list[float] = []
    weights: list[float] = []
    for row_i in range(len(points)):
        for n in range(1, k):
            dist = float(dists[row_i, n])
            # keep only neighbors at roughly one grid spacing
            if not (0.6 * spacing_px <= dist <= 1.45 * spacing_px):
                continue
            vec = points[int(idx[row_i, n])] - points[row_i]
            # atan2 gives vector angle in radians
            # mod pi folds opposite directions together
            angle = math.atan2(float(vec[1]), float(vec[0])) % math.pi
            # mod pi/2 folds the square lattice's two cardinal axes together
            angle_mod = angle % (0.5 * math.pi)
            angles.append(angle_mod)
            # neighbors closer to the expected spacing receive more weight
            weights.append(max(0.0, 1.0 - abs(dist - spacing_px) / spacing_px))
    if not angles:
        return 0.0
    angles_arr = np.asarray(angles)
    weights_arr = np.asarray(weights)
    doubled = np.exp(2j * angles_arr)
    # circular mean in doubled-angle space handles the pi/2 periodicity
    mean = np.sum(weights_arr * doubled)
    theta = 0.5 * math.atan2(float(mean.imag), float(mean.real))
    return theta % (0.5 * math.pi)


def normalize_lattice_angle(theta: float) -> float:
    # store lattice angles in the canonical [0, pi/2) interval
    return float(theta % (0.5 * math.pi))


# angle trials
# graph assignment is much more stable if we try a few plausible grid angles
def lattice_angle_trials(auto_theta: float, roi: RoiLike) -> list[tuple[str, float]]:
    # instead of trusting one estimated angle, try a few plausible angles
    # and later keep the one with the largest connected indexed component
    trials: list[tuple[str, float]] = []
    seen: set[int] = set()

    def add(label: str, theta: float) -> None:
        normalized = normalize_lattice_angle(theta)
        # quantized key avoids duplicate angle guesses from floating point roundoff
        key = int(round(normalized * 1_000_000))
        if key in seen:
            return
        seen.add(key)
        trials.append((label, normalized))

    if isinstance(roi, RotatedRoi):
        # for a rotated roi, the roi angle is usually the best initial guess
        roi_theta = math.radians(float(roi.angle_deg))
        for delta_deg in (0, -5, 5, -10, 10, -15, 15, -20, 20, -25, 25):
            label = "roi_angle" if delta_deg == 0 else f"roi_angle_{delta_deg:+d}deg"
            add(label, roi_theta + math.radians(float(delta_deg)))
    else:
        # for axis-aligned roi, try image axes plus small perturbations
        for delta_deg in (0, -5, 5, -10, 10, -15, 15, -20, 20, -25, 25):
            label = "image_axis" if delta_deg == 0 else f"image_axis_{delta_deg:+d}deg"
            add(label, math.radians(float(delta_deg)))

    add("auto_estimate", auto_theta)
    for delta_deg in (-5, 5, -10, 10):
        add(f"auto_estimate_{delta_deg:+d}deg", auto_theta + math.radians(float(delta_deg)))
    return trials


def circular_phase(values: np.ndarray, period: float) -> float:
    # circular mean phase for values with a known period
    # kept for lattice phase calculations/compatibility
    z = np.exp(2j * np.pi * values / period)
    mean = np.mean(z)
    phase = math.atan2(float(mean.imag), float(mean.real)) / (2.0 * math.pi) * period
    return float(phase % period)


# centroid refinement
# re-center an accepted dot using the enhanced dark-dot signal inside a small window
def refine_dark_centroid(
    signal_roi: np.ndarray,
    expected_x: float,
    expected_y: float,
    signal_offset: tuple[int, int],
    radius_px: int,
) -> tuple[float, float, float]:
    # expected_x/y are full-image coordinates
    # signal_roi is a crop, so subtract signal_offset to work crop-locally
    cx = expected_x - signal_offset[0]
    cy = expected_y - signal_offset[1]
    x0 = max(0, int(round(cx)) - radius_px)
    y0 = max(0, int(round(cy)) - radius_px)
    x1 = min(signal_roi.shape[1], int(round(cx)) + radius_px + 1)
    y1 = min(signal_roi.shape[0], int(round(cy)) + radius_px + 1)
    if x1 <= x0 or y1 <= y0:
        return expected_x, expected_y, 0.0
    patch = signal_roi[y0:y1, x0:x1].astype(float)
    # treat enhanced dark-dot signal as mass and compute its center of mass
    mass = float(np.sum(patch))
    if mass <= 1e-9:
        return expected_x, expected_y, 0.0
    yy, xx = np.indices(patch.shape)
    rx = float(np.sum((xx + x0) * patch) / mass) + signal_offset[0]
    ry = float(np.sum((yy + y0) * patch) / mass) + signal_offset[1]
    return rx, ry, float(np.max(patch))


# candidate cardinal edge generation
# connect nearby dots that are about one grid spacing apart and aligned with
# one of the trial lattice axes
# each edge proposes an integer step such as (di,dj)=(1,0)
def candidate_cardinal_edges(
    points: np.ndarray,
    spacing_px: float,
    theta: float,
    *,
    k_neighbors: int = 9,
    min_distance_fraction: float = 0.62,
    max_distance_fraction: float = 1.35,
    min_major_fraction: float = 0.55,
    max_minor_fraction: float = 0.48,
) -> pd.DataFrame:
    # build possible grid-neighbor edges between detected dot candidates
    # each edge says "candidate b is one lattice step from candidate a"
    if len(points) < 2:
        return pd.DataFrame(columns=["a", "b", "di", "dj", "axis", "distance_px", "score"])

    # unit vectors for the two cardinal lattice directions
    u = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    v = np.array([math.cos(theta + 0.5 * math.pi), math.sin(theta + 0.5 * math.pi)], dtype=float)
    tree = cKDTree(points)
    k = min(k_neighbors, len(points))
    dists, idx = tree.query(points, k=k)
    if idx.ndim == 1:
        idx = idx[:, None]
        dists = dists[:, None]

    seen: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    for a in range(len(points)):
        for pos in range(1, k):
            b = int(idx[a, pos])
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                # avoid adding the same undirected edge twice
                continue
            seen.add(key)

            distance = float(dists[a, pos])
            # only near-one-spacing neighbors can be cardinal grid edges
            if not (min_distance_fraction * spacing_px <= distance <= max_distance_fraction * spacing_px):
                continue

            vec = points[b] - points[a]
            # project the candidate edge onto the two trial lattice axes
            du = float(vec @ u)
            dv = float(vec @ v)
            if abs(du) >= abs(dv):
                # edge is mostly aligned with the i axis
                major = abs(du)
                minor = abs(dv)
                di = 1 if du > 0 else -1
                dj = 0
                axis = "i"
            else:
                # edge is mostly aligned with the j axis
                major = abs(dv)
                minor = abs(du)
                di = 0
                dj = 1 if dv > 0 else -1
                axis = "j"

            if major < min_major_fraction * spacing_px:
                continue
            if minor > max_minor_fraction * spacing_px:
                # reject edges with too much perpendicular component
                # those are diagonal/non-cardinal relationships
                continue

            # lower score means closer to one-spacing cardinal geometry
            score = (
                abs(distance - spacing_px) / spacing_px
                + 0.7 * minor / spacing_px
                + 0.3 * abs(major - spacing_px) / spacing_px
            )
            rows.append(
                {
                    "a": int(a),
                    "b": int(b),
                    "di": int(di),
                    "dj": int(dj),
                    "axis": axis,
                    "distance_px": distance,
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows)


# for each dot and each cardinal direction, keep only the lowest-score edge
# this prevents one noisy dot from creating multiple competing neighbors
def select_cardinal_edges(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()

    selected: list[dict[str, Any]] = []
    # track which direction has already been used for each dot
    # e.g. "this dot already has a +i neighbor"
    used_direction: set[tuple[int, int, int]] = set()
    for row in edges.sort_values("score").itertuples(index=False):
        a = int(row.a)
        b = int(row.b)
        di = int(row.di)
        dj = int(row.dj)
        forward_key = (a, di, dj)
        backward_key = (b, -di, -dj)
        if forward_key in used_direction or backward_key in used_direction:
            # keep only the best edge for each directed cardinal slot
            continue
        used_direction.add(forward_key)
        used_direction.add(backward_key)
        selected.append(row._asdict())

    return pd.DataFrame(selected)


# local graph assignment
# build a graph from accepted cardinal edges, choose the largest connected part,
# seed it near the center, then propagate integer lattice coordinates through edges
def local_graph_index_assignment(
    candidates: pd.DataFrame,
    *,
    spacing_px: float,
    theta: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(candidates) < 30:
        raise RuntimeError(f"Too few candidate dots for local graph indexing: {len(candidates)}")

    # candidate table becomes a plain coordinate array for graph construction
    points = candidates[["x_px", "y_px"]].to_numpy(float)
    # generate all plausible one-step neighbor edges, then prune conflicts
    # so each dot has at most one neighbor in each cardinal direction
    candidate_edges = candidate_cardinal_edges(points, spacing_px, theta)
    accepted_edges = select_cardinal_edges(candidate_edges)
    if accepted_edges.empty:
        raise RuntimeError("Local graph indexing found no usable cardinal neighbor edges.")

    # adjacency maps candidate index -> neighboring candidates
    # together with the integer lattice step needed to get there
    adjacency: dict[int, list[tuple[int, int, int, float]]] = {idx: [] for idx in range(len(candidates))}
    for row in accepted_edges.itertuples(index=False):
        a = int(row.a)
        b = int(row.b)
        di = int(row.di)
        dj = int(row.dj)
        score = float(row.score)
        adjacency[a].append((b, di, dj, score))
        # add reverse edge with opposite lattice step
        adjacency[b].append((a, -di, -dj, score))

    for idx in adjacency:
        # use lower-score edges first during propagation
        adjacency[idx].sort(key=lambda item: item[3])

    # find connected components in the accepted-edge graph
    unvisited = set(range(len(candidates)))
    components: list[list[int]] = []
    while unvisited:
        start = unvisited.pop()
        queue = deque([start])
        component = [start]
        while queue:
            node = queue.popleft()
            for neighbor, _, _, _ in adjacency[node]:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        components.append(component)

    # tiny components are usually accidental local matches, not the usable grid
    nontrivial = [component for component in components if len(component) >= 30]
    if not nontrivial:
        raise RuntimeError("Local graph indexing found no connected component with at least 30 dots.")

    center = np.nanmedian(points, axis=0)
    component_records: list[tuple[int, int, dict[int, tuple[int, int]]]] = []
    for component in nontrivial:
        component_set = set(component)
        # choose a seed dot near the center with many in-component neighbors
        # this starts propagation from a stable local region
        seed = min(
            component,
            key=lambda idx: (
                -len([edge for edge in adjacency[idx] if edge[0] in component_set]),
                float(np.linalg.norm(points[idx] - center)),
            ),
        )
        assigned: dict[int, tuple[int, int]] = {seed: (0, 0)}
        conflicts = 0
        queue = deque([seed])
        # breadth-first propagation:
        # if node has coordinate (i0,j0), neighbor gets (i0+di,j0+dj)
        while queue:
            node = queue.popleft()
            i0, j0 = assigned[node]
            for neighbor, di, dj, _ in adjacency[node]:
                if neighbor not in component_set:
                    continue
                expected = (i0 + int(di), j0 + int(dj))
                if neighbor not in assigned:
                    assigned[neighbor] = expected
                    queue.append(neighbor)
                elif assigned[neighbor] != expected:
                    # conflicts mean a graph loop implied inconsistent integer coordinates
                    conflicts += 1
        component_records.append((len(assigned), conflicts, assigned))

    # prefer the largest propagated component; for equal sizes, fewer conflicts
    component_records.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    kept_size, _conflict_count, index_map = component_records[0]
    kept_indices = sorted(index_map)
    assigned = candidates.iloc[kept_indices].copy()
    # keep original candidate index so we can trace back to the detector output
    assigned["source_candidate_index"] = kept_indices
    assigned["i_raw"] = [index_map[idx][0] for idx in kept_indices]
    assigned["j_raw"] = [index_map[idx][1] for idx in kept_indices]
    assigned["i"] = assigned["i_raw"]
    assigned["j"] = assigned["j_raw"]

    return assigned, {"kept_component_size": int(kept_size)}


# lattice coordinate fitting
# after graph assignment, fit an affine map from integer grid coordinates
# (1,i,j) to pixel coordinates
# residuals tell us how regular the indexed lattice is
def fit_lattice(
    candidates: pd.DataFrame,
    roi: RoiLike,
    signal_roi: np.ndarray,
    signal_offset: tuple[int, int],
    *,
    dot_spacing_mm: float,
    max_residual_fraction: float = 0.75,
    edge_exclusion_fraction: float = 0.45,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # this turns raw dot candidates into final indexed reference dots
    if len(candidates) < 30:
        raise RuntimeError(f"Too few dot candidates for lattice fit: {len(candidates)}")

    points = candidates[["x_px", "y_px"]].to_numpy(float)
    # estimate spacing and orientation from unordered candidate centroids
    spacing = estimate_spacing_px(points)
    if not np.isfinite(spacing) or spacing <= 0:
        raise RuntimeError("Could not estimate dot spacing.")

    auto_theta = estimate_lattice_angle(points, spacing)
    # exclude edge dots because partial/near-boundary dots are more likely to
    # miss neighbors and destabilize graph indexing
    edge_margin_px = max(2.0, float(edge_exclusion_fraction) * float(spacing))
    candidate_pool = filter_candidates_to_roi(candidates, roi, margin_px=edge_margin_px)
    if len(candidate_pool) < 30:
        raise RuntimeError(
            f"Too few dots remain after rotated-ROI edge exclusion: {len(candidate_pool)}"
        )
    n_after_edge_exclusion = int(len(candidate_pool))

    theta_trials = lattice_angle_trials(auto_theta, roi)
    graph_errors: list[str] = []
    best_graph: tuple[int, float, pd.DataFrame] | None = None
    candidate_pool_reset = candidate_pool.reset_index(drop=True)
    for theta_source, theta_trial in theta_trials:
        # try graph indexing for this angle guess
        try:
            trial_assigned, trial_stats = local_graph_index_assignment(
                candidate_pool_reset,
                spacing_px=spacing,
                theta=theta_trial,
            )
        except RuntimeError as exc:
            graph_errors.append(f"{theta_source}: {exc}")
            continue
        kept_size = int(trial_stats.get("kept_component_size", len(trial_assigned)))
        if best_graph is None or kept_size > best_graph[0]:
            # keep the angle that preserves the most connected indexed dots
            best_graph = (kept_size, theta_trial, trial_assigned)

    if best_graph is None:
        details = "; ".join(graph_errors[:5])
        suffix = f" Tried {len(theta_trials)} angle hypotheses. {details}" if details else ""
        raise RuntimeError(f"Local graph indexing failed for every angle hypothesis.{suffix}")

    _, theta, assigned = best_graph
    # multiple candidates can occasionally receive the same lattice coordinate
    # keep the one with the strongest dark-dot signal
    duplicate_mask = assigned.duplicated(["i", "j"], keep=False)
    duplicate_cell_count = int(assigned.loc[duplicate_mask, ["i", "j"]].drop_duplicates().shape[0])
    duplicate_candidate_count = int(duplicate_mask.sum())
    n_before_dedup = int(len(assigned))
    assigned = assigned.sort_values(["dark_signal"], ascending=[False])
    assigned = assigned.drop_duplicates(["i", "j"], keep="first").copy()
    duplicate_removed_count = int(n_before_dedup - len(assigned))

    # shift raw propagated indices so the final lattice starts at i=0, j=0
    min_i = int(assigned["i"].min())
    min_j = int(assigned["j"].min())
    assigned["i"] = assigned["i"] - min_i
    assigned["j"] = assigned["j"] - min_j

    # fit affine pixel model:
    #   x_px = a0 + a1*i + a2*j
    #   y_px = b0 + b1*i + b2*j
    # this describes the best regular lattice passing through the indexed dots
    design = np.column_stack(
        [
            np.ones(len(assigned)),
            assigned["i"].to_numpy(float),
            assigned["j"].to_numpy(float),
        ]
    )
    coeff_x, *_ = np.linalg.lstsq(design, assigned["x_px"].to_numpy(float), rcond=None)
    coeff_y, *_ = np.linalg.lstsq(design, assigned["y_px"].to_numpy(float), rcond=None)
    pred_x = design @ coeff_x
    pred_y = design @ coeff_y
    # residual is geometric error between detected center and fitted lattice prediction
    residual = np.hypot(assigned["x_px"].to_numpy(float) - pred_x, assigned["y_px"].to_numpy(float) - pred_y)
    assigned["lattice_residual_px"] = residual

    radius = max(2, int(round(0.25 * spacing)))
    refined_rows = []
    for _, row in assigned.iterrows():
        # recompute each dot center as the local center-of-mass of the enhanced signal
        # this improves subpixel reference positions
        rx, ry, peak = refine_dark_centroid(
            signal_roi,
            float(row["x_px"]),
            float(row["y_px"]),
            signal_offset,
            radius,
        )
        refined_rows.append((rx, ry, peak))
    refined = np.asarray(refined_rows, dtype=float)
    assigned["x_px"] = refined[:, 0]
    assigned["y_px"] = refined[:, 1]
    assigned["refined_dark_peak"] = refined[:, 2]

    assigned = assigned.sort_values(["j", "i"]).reset_index(drop=True)
    # dot_id is the stable column index used by tracking arrays
    assigned.insert(0, "dot_id", np.arange(len(assigned), dtype=int))
    # physical grid coordinates assume the printed dot spacing is known
    assigned["x_grid_mm"] = assigned["i"].astype(float) * float(dot_spacing_mm)
    assigned["y_grid_mm"] = assigned["j"].astype(float) * float(dot_spacing_mm)
    assigned["x_grid_m"] = assigned["x_grid_mm"] / 1000.0
    assigned["y_grid_m"] = assigned["y_grid_mm"] / 1000.0
    spacing_metrics = grid_neighbor_spacing_metrics(assigned)
    indexing_metrics = indexing_consistency_metrics(assigned)
    grid_slot_count = int(
        (int(assigned["i"].max()) - int(assigned["i"].min()) + 1)
        * (int(assigned["j"].max()) - int(assigned["j"].min()) + 1)
    )

    # final flat-reference lattice output:
    # dot table carries per-dot pixel and physical grid coordinates
    # lattice metadata records roi, spacing, affine fit, occupancy, and indexing quality
    lattice = {
        "reference_builder_version": 4,
        "roi": roi.as_dict(),
        "roi_kind": str(roi.as_dict()["kind"]),
        "roi_px": roi.as_list(),
        "roi_bounding_px": [int(signal_offset[0]), int(signal_offset[1]), int(signal_roi.shape[1]), int(signal_roi.shape[0])],
        "roi_edge_exclusion_px": float(edge_margin_px),
        "n_candidates": int(len(candidates)),
        "n_after_edge_exclusion": n_after_edge_exclusion,
        "n_indexed_dots": int(len(assigned)),
        "indexing_duplicate_cell_count": duplicate_cell_count,
        "indexing_duplicate_candidate_count": duplicate_candidate_count,
        "indexing_duplicate_removed_count": duplicate_removed_count,
        "grid_slot_count": grid_slot_count,
        "grid_occupancy_fraction": float(len(assigned) / grid_slot_count) if grid_slot_count else float("nan"),
        "dot_spacing_px_median_nn": float(spacing),
        "dot_spacing_mm": float(dot_spacing_mm),
        "axis_angle_deg": float(np.degrees(theta)),
        "indexing_method": "local_graph",
        "auto_axis_angle_deg": float(np.degrees(auto_theta)),
        "selected_axis_angle_deg": float(np.degrees(theta)),
        "affine_pixel_from_ij": {
            "x": [float(v) for v in coeff_x],
            "y": [float(v) for v in coeff_y],
        },
        "grid_i_min": int(assigned["i"].min()),
        "grid_i_max": int(assigned["i"].max()),
        "grid_j_min": int(assigned["j"].min()),
        "grid_j_max": int(assigned["j"].max()),
        "median_lattice_residual_px": float(np.median(assigned["lattice_residual_px"])),
        "p95_lattice_residual_px": float(np.percentile(assigned["lattice_residual_px"], 95)),
        **spacing_metrics,
        **indexing_metrics,
    }
    return assigned, lattice


# shared distribution helper for spacing metrics
# json stores medians/spreads instead of plots
def distance_distribution_metrics(values: np.ndarray, prefix: str, source: str) -> dict[str, float | int | str]:
    # build the same metric fields for nearest-neighbor and indexed-neighbor distances
    values = np.asarray(values, dtype=float)
    # keep only positive finite distances
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return {
            f"{prefix}_metric_source": source,
            f"{prefix}_pair_count": 0,
            f"{prefix}_spacing_median_px": float("nan"),
            f"{prefix}_spacing_p05_px": float("nan"),
            f"{prefix}_spacing_p95_px": float("nan"),
            f"{prefix}_spacing_cv_percent": float("nan"),
            f"{prefix}_spacing_p95_abs_error_px": float("nan"),
        }

    median = float(np.median(values))
    # absolute deviation from median is robust to a few bad neighbor distances
    abs_error = np.abs(values - median)
    return {
        f"{prefix}_metric_source": source,
        f"{prefix}_pair_count": int(len(values)),
        f"{prefix}_spacing_median_px": median,
        f"{prefix}_spacing_p05_px": float(np.percentile(values, 5)),
        f"{prefix}_spacing_p95_px": float(np.percentile(values, 95)),
        f"{prefix}_spacing_cv_percent": float(100.0 * np.std(values) / median) if median > 0 else float("nan"),
        f"{prefix}_spacing_p95_abs_error_px": float(np.percentile(abs_error, 95)),
    }


# spacing metrics start with nearest-neighbor spacing
# this is independent of lattice indices and catches big scale problems
def nearest_neighbor_spacing_metrics(dots: pd.DataFrame) -> dict[str, float | int | str]:
    if dots.empty or len(dots) < 2:
        return distance_distribution_metrics(np.array([], dtype=float), "nearest_neighbor", "nearest_neighbor")

    points = dots[["x_px", "y_px"]].to_numpy(float)
    tree = cKDTree(points)
    # k=2 returns self and nearest other point
    dists, _ = tree.query(points, k=2)
    values = dists[:, 1]
    return distance_distribution_metrics(values, "nearest_neighbor", "nearest_neighbor")


# indexed neighbor segments use assigned (i,j) coordinates to measure only
# expected horizontal/vertical grid neighbors
def indexed_neighbor_segments(dots: pd.DataFrame) -> pd.DataFrame:
    if dots.empty:
        return pd.DataFrame(columns=["x0", "y0", "x1", "y1", "i0", "j0", "i1", "j1", "axis", "distance_px"])

    # lookup table from integer lattice coordinate to row data
    by_index = {
        (int(row.i), int(row.j)): row
        for row in dots.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for (i, j), dot in by_index.items():
        # only look forward in +i and +j directions to avoid duplicated edges
        for axis, neighbor in (("i", (i + 1, j)), ("j", (i, j + 1))):
            neighbor_dot = by_index.get(neighbor)
            if neighbor_dot is None:
                continue
            x0 = float(dot.x_px)
            y0 = float(dot.y_px)
            x1 = float(neighbor_dot.x_px)
            y1 = float(neighbor_dot.y_px)
            rows.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "i0": i,
                    "j0": j,
                    "i1": int(neighbor_dot.i),
                    "j1": int(neighbor_dot.j),
                    "axis": axis,
                    "distance_px": float(math.hypot(x1 - x0, y1 - y0)),
                }
            )
    return pd.DataFrame(rows)


# compare indexed-neighbor distances with the nearest-neighbor scale
# this flags normal, marginal, diagonal-like, or bad edge lengths
def grid_neighbor_spacing_metrics(dots: pd.DataFrame) -> dict[str, float | int | str]:
    # start with index-free nearest-neighbor spacing
    # then compare indexed grid-neighbor lengths against that expected spacing
    nearest = nearest_neighbor_spacing_metrics(dots)
    expected_spacing = float(nearest.get("nearest_neighbor_spacing_median_px", float("nan")))
    if dots.empty:
        return {
            "neighbor_metric_source": "none",
            "neighbor_pair_count": 0,
            "local_spacing_median_px": float("nan"),
            "local_spacing_p05_px": float("nan"),
            "local_spacing_p95_px": float("nan"),
            "local_spacing_cv_percent": float("nan"),
            "local_spacing_p95_abs_error_px": float("nan"),
            **nearest,
            "indexed_neighbor_normal_length_fraction": float("nan"),
            "indexed_neighbor_marginal_length_fraction": float("nan"),
            "indexed_neighbor_bad_length_fraction": float("nan"),
            "indexed_neighbor_diagonal_length_fraction": float("nan"),
        }

    segments = indexed_neighbor_segments(dots)
    distances = segments["distance_px"].to_numpy(float) if len(segments) else np.array([], dtype=float)

    if len(distances) == 0:
        # if there are no indexed neighbors, fall back to nearest-neighbor stats
        # so downstream summaries still have a scale estimate
        fallback = {
            "neighbor_metric_source": "nearest_neighbor_fallback",
            "neighbor_pair_count": int(nearest["nearest_neighbor_pair_count"]),
            "local_spacing_median_px": float(nearest["nearest_neighbor_spacing_median_px"]),
            "local_spacing_p05_px": float(nearest["nearest_neighbor_spacing_p05_px"]),
            "local_spacing_p95_px": float(nearest["nearest_neighbor_spacing_p95_px"]),
            "local_spacing_cv_percent": float(nearest["nearest_neighbor_spacing_cv_percent"]),
            "local_spacing_p95_abs_error_px": float(nearest["nearest_neighbor_spacing_p95_abs_error_px"]),
            **nearest,
            **distance_distribution_metrics(np.array([], dtype=float), "indexed_neighbor", "indexed_grid_neighbors"),
            "indexed_neighbor_normal_length_fraction": float("nan"),
            "indexed_neighbor_marginal_length_fraction": float("nan"),
            "indexed_neighbor_bad_length_fraction": float("nan"),
            "indexed_neighbor_diagonal_length_fraction": float("nan"),
        }
        return fallback

    indexed = distance_distribution_metrics(distances, "indexed_neighbor", "indexed_grid_neighbors")
    if np.isfinite(expected_spacing) and expected_spacing > 0 and len(distances):
        # ratios classify indexed neighbor lengths relative to nearest-neighbor spacing
        ratios = np.asarray(distances, dtype=float) / expected_spacing
        normal = (0.85 <= ratios) & (ratios <= 1.15)
        marginal = (~normal) & (0.70 <= ratios) & (ratios <= 1.35)
        bad = ~(normal | marginal)
        diagonal_like = (1.30 <= ratios) & (ratios <= 1.50)
        normal_fraction = float(np.mean(normal))
        marginal_fraction = float(np.mean(marginal))
        bad_fraction = float(np.mean(bad))
        diagonal_fraction = float(np.mean(diagonal_like))
    else:
        normal_fraction = float("nan")
        marginal_fraction = float("nan")
        bad_fraction = float("nan")
        diagonal_fraction = float("nan")
    return {
        "neighbor_metric_source": "indexed_grid_neighbors",
        "neighbor_pair_count": int(indexed["indexed_neighbor_pair_count"]),
        "local_spacing_median_px": float(indexed["indexed_neighbor_spacing_median_px"]),
        "local_spacing_p05_px": float(indexed["indexed_neighbor_spacing_p05_px"]),
        "local_spacing_p95_px": float(indexed["indexed_neighbor_spacing_p95_px"]),
        "local_spacing_cv_percent": float(indexed["indexed_neighbor_spacing_cv_percent"]),
        "local_spacing_p95_abs_error_px": float(indexed["indexed_neighbor_spacing_p95_abs_error_px"]),
        **nearest,
        **indexed,
        "indexed_neighbor_normal_length_fraction": normal_fraction,
        "indexed_neighbor_marginal_length_fraction": marginal_fraction,
        "indexed_neighbor_bad_length_fraction": bad_fraction,
        "indexed_neighbor_diagonal_length_fraction": diagonal_fraction,
    }


# index consistency metrics compare image-space nearest neighbors with their
# assigned integer-index relationship
# a good lattice has mostly cardinal nearest-neighbor relations
def indexing_consistency_metrics(dots: pd.DataFrame, k_neighbors: int = 4) -> dict[str, float | int]:
    if len(dots) < 2:
        return {
            "indexing_knn_relation_count": 0,
            "indexing_knn_cardinal_fraction": float("nan"),
            "indexing_knn_diagonal_fraction": float("nan"),
            "indexing_knn_nonlocal_fraction": float("nan"),
            "indexing_median_cardinal_neighbors_per_dot": float("nan"),
            "indexing_suspect_dot_count": 0,
            "indexing_suspect_dot_fraction": float("nan"),
        }

    points = dots[["x_px", "y_px"]].to_numpy(float)
    indices = dots[["i", "j"]].to_numpy(int)
    tree = cKDTree(points)
    # for each dot, inspect closest image-space neighbors and ask how they relate
    # in integer lattice space
    k = min(k_neighbors + 1, len(dots))
    _, neighbor_idx = tree.query(points, k=k)
    if neighbor_idx.ndim == 1:
        neighbor_idx = neighbor_idx[:, None]

    cardinal = 0
    diagonal = 0
    nonlocal_count = 0
    total = 0
    cardinal_per_dot: list[int] = []
    for row_idx in range(len(dots)):
        per_dot_cardinal = 0
        for neighbor in neighbor_idx[row_idx, 1:]:
            # di,dj are integer-lattice differences, not pixel distances
            di = abs(int(indices[int(neighbor), 0]) - int(indices[row_idx, 0]))
            dj = abs(int(indices[int(neighbor), 1]) - int(indices[row_idx, 1]))
            total += 1
            if (di == 1 and dj == 0) or (di == 0 and dj == 1):
                # a true cardinal neighbor differs by one index in one direction only
                cardinal += 1
                per_dot_cardinal += 1
            elif di == 1 and dj == 1:
                # diagonal neighbors are plausible but should not dominate
                diagonal += 1
            else:
                nonlocal_count += 1
        cardinal_per_dot.append(per_dot_cardinal)

    cardinal_counts = np.asarray(cardinal_per_dot, dtype=float)
    # a dot with fewer than two cardinal near-neighbors is suspicious because
    # interior square-lattice dots should normally have more
    suspect = cardinal_counts < 2
    return {
        "indexing_knn_relation_count": int(total),
        "indexing_knn_cardinal_fraction": float(cardinal / total) if total else float("nan"),
        "indexing_knn_diagonal_fraction": float(diagonal / total) if total else float("nan"),
        "indexing_knn_nonlocal_fraction": float(nonlocal_count / total) if total else float("nan"),
        "indexing_median_cardinal_neighbors_per_dot": float(np.median(cardinal_counts)) if len(cardinal_counts) else float("nan"),
        "indexing_suspect_dot_count": int(np.sum(suspect)),
        "indexing_suspect_dot_fraction": float(np.mean(suspect)) if len(suspect) else float("nan"),
    }


# end-to-end flat-reference builder
# detect candidate dots, index them into a lattice, and return dot table + metadata
def build_flat_lattice(
    gray: np.ndarray,
    roi: RoiLike,
    *,
    dot_spacing_mm: float,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray]:
    # step 1: image processing and blob detection inside the selected roi
    candidates, binary, signal_roi, bounds, roi = detect_dot_candidates(gray, roi)
    # step 2: graph-based lattice indexing and metric metadata creation
    dots, lattice = fit_lattice(
        candidates,
        roi,
        signal_roi,
        (bounds.x, bounds.y),
        dot_spacing_mm=dot_spacing_mm,
    )
    # return binary/signal images too because callers may want quick visual checks
    return dots, lattice, binary, signal_roi
