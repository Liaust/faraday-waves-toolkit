from __future__ import annotations

# this is the manual roi selector
# the ui itself mostly just edits metadata and calls roi_tools.py for the actual image stuff
# frequency and full fsss use slightly different roi logic so both modes live here

import math

import streamlit as st

from roi_tools import (
    METADATA_PATH,
    PROJECT_ROOT,
    Roi,
    RotatedRoi,
    current_frequency_roi,
    current_roi,
    draw_frequency_dots_overlay,
    draw_frequency_roi_overlay,
    draw_roi_overlay,
    draw_tracking_overlay,
    evaluate_frequency_roi,
    evaluate_roi,
    finite_float,
    load_reference_images,
    load_yaml,
    save_frequency_roi_to_metadata,
    save_roi_to_metadata,
)

# basic streamlit page setup
st.set_page_config(page_title="Faraday ROI Selector", layout="wide")

# load the reference images but cache them
# metadata_text is mostly just a cache key, if yaml changes then this reloads
@st.cache_data(show_spinner="Loading and undistorting reference images...")
def cached_reference_images(metadata_text: str):
    metadata = load_yaml()
    return load_reference_images(metadata)

# initialize normal rectangular roi controls for fsss
def initialize_state(default_roi: Roi) -> None:
    for key, value in zip(("x", "y", "w", "h"), default_roi.as_list()):
        st.session_state.setdefault(key, int(value))
        st.session_state.setdefault(f"{key}_num", int(value))

# initialize rotated roi controls for frequency analysis
def initialize_frequency_state(default_roi: RotatedRoi) -> None:
    defaults = {
        "fcx": float(default_roi.cx),
        "fcy": float(default_roi.cy),
        "fw": float(default_roi.w),
        "fh": float(default_roi.h),
        "fangle": float(default_roi.angle_deg),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
        st.session_state.setdefault(f"{key}_num", value)

# sliders and number inputs are separate widgets
# so we manually keep them synced
def sync_slider_to_number(key: str) -> None:
    st.session_state[f"{key}_num"] = int(st.session_state[key])

def sync_number_to_slider(key: str) -> None:
    st.session_state[key] = int(st.session_state[f"{key}_num"])

def sync_float_slider_to_number(key: str) -> None:
    st.session_state[f"{key}_num"] = float(st.session_state[key])

def sync_float_number_to_slider(key: str) -> None:
    st.session_state[key] = float(st.session_state[f"{key}_num"])

# read the roi values from session_state and clamp them inside the image
def roi_from_state(image_shape) -> Roi:
    return Roi(
        int(st.session_state.x),
        int(st.session_state.y),
        int(st.session_state.w),
        int(st.session_state.h),
    ).clamped(image_shape)

# same thing but for rotated frequency roi
def frequency_roi_from_state(image_shape) -> RotatedRoi:
    return RotatedRoi(
        cx=float(st.session_state.fcx),
        cy=float(st.session_state.fcy),
        w=float(st.session_state.fw),
        h=float(st.session_state.fh),
        angle_deg=float(st.session_state.fangle),
    ).clamped(image_shape)

# small metric row for the fsss dry to empty/flat tracking check
def metric_row(summary: dict) -> None:
    flat = summary["flat_liquid"]
    empty = summary["empty_bath"]
    cols = st.columns(5)
    cols[0].metric("Dry dots", summary["detected_dry_dots"])
    cols[1].metric("Flat valid", f"{flat['n_valid']} / {flat['n_total']}", f"{100 * flat['valid_fraction']:.1f}%")
    cols[2].metric("Flat score", f"{flat['median_match_score']:.3f}")
    cols[3].metric("Empty valid", f"{empty['n_valid']} / {empty['n_total']}", f"{100 * empty['valid_fraction']:.1f}%")
    cols[4].metric("Spacing", f"{summary['estimated_spacing_px']:.2f} px")

# small metric rows for the frequency flat lattice check
def frequency_metric_row(summary: dict) -> None:
    indexed_cv = finite_float(summary.get("indexed_neighbor_spacing_cv_percent", summary.get("local_spacing_cv_percent")))
    nearest_cv = finite_float(summary.get("nearest_neighbor_spacing_cv_percent"))
    cols = st.columns(5)
    cols[0].metric("Flat dots", summary["detected_flat_dots"])
    cols[1].metric("Spacing", f"{summary['estimated_spacing_px']:.2f} px")
    cols[2].metric("Edge margin", f"{summary['roi_edge_exclusion_px']:.2f} px")
    cols[3].metric("Indexed CV", f"{indexed_cv:.2f}%" if math.isfinite(indexed_cv) else "n/a")
    cols[4].metric("Nearest CV", f"{nearest_cv:.2f}%" if math.isfinite(nearest_cv) else "n/a")

    indexed_p95 = finite_float(summary.get("indexed_neighbor_spacing_p95_abs_error_px", summary.get("local_spacing_p95_abs_error_px")))
    nearest_p95 = finite_float(summary.get("nearest_neighbor_spacing_p95_abs_error_px"))
    cardinal_fraction = finite_float(summary.get("indexing_knn_cardinal_fraction"))
    suspect_fraction = finite_float(summary.get("indexing_suspect_dot_fraction"))
    cols = st.columns(5)
    cols[0].metric("Indexed pairs", summary.get("indexed_neighbor_pair_count", summary.get("neighbor_pair_count", 0)))
    cols[1].metric("Indexed P95 err", f"{indexed_p95:.2f} px" if math.isfinite(indexed_p95) else "n/a")
    cols[2].metric("Nearest P95 err", f"{nearest_p95:.2f} px" if math.isfinite(nearest_p95) else "n/a")
    cols[3].metric("kNN cardinal", f"{100 * cardinal_fraction:.1f}%" if math.isfinite(cardinal_fraction) else "n/a")
    cols[4].metric("Suspect dots", f"{summary.get('indexing_suspect_dot_count', 0)}", f"{100 * suspect_fraction:.1f}%" if math.isfinite(suspect_fraction) else None)

    normal_links = finite_float(summary.get("indexed_neighbor_normal_length_fraction"))
    marginal_links = finite_float(summary.get("indexed_neighbor_marginal_length_fraction"))
    bad_links = finite_float(summary.get("indexed_neighbor_bad_length_fraction"))
    diagonal_links = finite_float(summary.get("indexed_neighbor_diagonal_length_fraction"))
    cols = st.columns(5)
    cols[0].metric("Normal links", f"{100 * normal_links:.1f}%" if math.isfinite(normal_links) else "n/a")
    cols[1].metric("Marginal links", f"{100 * marginal_links:.1f}%" if math.isfinite(marginal_links) else "n/a")
    cols[2].metric("Bad links", f"{100 * bad_links:.1f}%" if math.isfinite(bad_links) else "n/a")
    cols[3].metric("Diagonal-like", f"{100 * diagonal_links:.1f}%" if math.isfinite(diagonal_links) else "n/a")
    cols[4].metric("Duplicates removed", summary.get("indexing_duplicate_removed_count", 0))

# main app flow
# load metadata -> choose mode -> edit roi -> preview/evaluate -> save to yaml
def main() -> None:
    # metadata has the current roi defaults
    metadata = load_yaml()
    default_roi = current_roi(metadata)
    default_frequency_roi = current_frequency_roi(metadata)
    initialize_state(default_roi)
    initialize_frequency_state(default_frequency_roi)

    st.title("Faraday ROI Selector")
    st.caption(f"Project root: `{PROJECT_ROOT}`")
    st.caption(f"Metadata: `{METADATA_PATH.relative_to(PROJECT_ROOT)}`")

    # raw yaml text is used so the cache notices when metadata changes
    metadata_text = METADATA_PATH.read_text(encoding="utf-8")
    try:
        images = cached_reference_images(metadata_text)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.info(
            "Update the calibration metadata with existing reference videos, or set "
            "`FARADAY_CALIBRATION_METADATA_PATH` before launching the app."
        )
        return
    # prefer showing an actual run frame if we have one
    # otherwise the flat reference is the best default
    preferred_preview = "first run first frame" if "first run first frame" in images else "flat liquid reference"
    image_name = st.sidebar.selectbox(
        "Preview image",
        list(images.keys()),
        index=list(images.keys()).index(preferred_preview) if preferred_preview in images else 0,
    )
    image = images[image_name]
    height, width = image.shape[:2]

    # frequency and fsss are not checking the same thing
    # frequency detects/indexes dots directly in the flat liquid image
    # fsss checks if dry grid dots can be followed into empty/flat references
    mode = st.sidebar.radio(
        "Evaluation mode",
        [
            "Frequency estimation: flat lattice only",
            "FSSS full: dry-to-empty/flat tracking",
        ],
        index=0,
    )

    st.sidebar.header("ROI")
    max_x = max(0, width - 1)
    max_y = max(0, height - 1)
    max_w = max(1, width)
    max_h = max(1, height)

    if mode.startswith("Frequency estimation"):
        # frequency uses a rotated rectangle since the dot grid can be tilted
        frequency_controls = [
            ("fcx", "center x", 0.0, float(max_x), 1.0),
            ("fcy", "center y", 0.0, float(max_y), 1.0),
            ("fw", "width", 1.0, float(max_w), 1.0),
            ("fh", "height", 1.0, float(max_h), 1.0),
            ("fangle", "angle deg", -45.0, 45.0, 0.05),
        ]
        for key, label, min_value, max_value, step in frequency_controls:
            # slider is for fast changes
            st.sidebar.slider(
                label,
                min_value=min_value,
                max_value=max_value,
                step=step,
                key=key,
                on_change=sync_float_slider_to_number,
                args=(key,),
            )
            # number input is for exact values
            st.sidebar.number_input(
                f"{label} value",
                min_value=min_value,
                max_value=max_value,
                step=step,
                key=f"{key}_num",
                on_change=sync_float_number_to_slider,
                args=(key,),
            )
        frequency_roi = frequency_roi_from_state(image.shape)
        # some drawing/evaluation helpers still want an axis aligned box
        roi = Roi.from_list(
            [
                frequency_roi.bounding_roi(image.shape).x,
                frequency_roi.bounding_roi(image.shape).y,
                frequency_roi.bounding_roi(image.shape).w,
                frequency_roi.bounding_roi(image.shape).h,
            ]
        )
    else:
        # full fsss starts from a rectangular roi on the dry dot grid
        for label, max_value in (("x", max_x), ("y", max_y), ("w", max_w), ("h", max_h)):
            st.sidebar.slider(
                label,
                min_value=0 if label in {"x", "y"} else 1,
                max_value=max_value,
                key=label,
                on_change=sync_slider_to_number,
                args=(label,),
            )
            st.sidebar.number_input(
                f"{label} value",
                min_value=0 if label in {"x", "y"} else 1,
                max_value=max_value,
                key=f"{label}_num",
                on_change=sync_number_to_slider,
                args=(label,),
            )
        roi = roi_from_state(image.shape)
        frequency_roi = RotatedRoi(
            cx=float(roi.x) + 0.5 * float(roi.w),
            cy=float(roi.y) + 0.5 * float(roi.h),
            w=float(roi.w),
            h=float(roi.h),
            angle_deg=0.0,
        )  # unused in FSSS mode
        if roi.as_list() != [st.session_state.x, st.session_state.y, st.session_state.w, st.session_state.h]:
            # warn if the typed values had to be clamped back into the image
            st.sidebar.warning(f"ROI was clamped to image bounds: `{roi.as_list()}`")

    st.sidebar.divider()
    min_match_score = 0.35
    template_fraction = 0.25
    search_fraction = 0.85
    if mode.startswith("FSSS full"):
        # these only affect the quick fsss check in the app
        # they are not saved as pipeline defaults
        min_match_score = st.sidebar.slider("Minimum match score", 0.05, 0.95, 0.35, 0.01)
        template_fraction = st.sidebar.slider("Template radius / dot spacing", 0.10, 0.45, 0.25, 0.01)
        search_fraction = st.sidebar.slider("Search radius / dot spacing", 0.20, 1.20, 0.85, 0.01)
        st.sidebar.caption(
            "This checks the full FSSS dry-reference correspondence. It is stricter than the "
            "frequency-only path and can fail even when flat-liquid dots are trackable."
        )
    else:
        st.sidebar.caption(
            "This checks the frequency-only reference: dots are detected and indexed directly "
            "in the flat-liquid image, avoiding dry-to-flat correspondence."
        )

    action_cols = st.sidebar.columns(2)
    evaluate = action_cols[0].button("Evaluate ROI", type="primary", use_container_width=True)
    save = action_cols[1].button("Save ROI", use_container_width=True)

    if save:
        # save the chosen geometry back into calibration_metadata.yaml
        if mode.startswith("Frequency estimation"):
            save_frequency_roi_to_metadata(frequency_roi)
            st.sidebar.success(f"Saved rotated frequency ROI `{frequency_roi.as_list()}` to metadata.")
        else:
            save_roi_to_metadata(roi)
            st.sidebar.success(f"Saved `{roi.as_list()}` to metadata.")
        st.cache_data.clear()

    # just draw the roi first before doing any expensive detection/tracking
    st.subheader("Live Rectangle")
    if mode.startswith("Frequency estimation"):
        st.image(
            draw_frequency_roi_overlay(image, frequency_roi),
            caption=f"{image_name} with rotated ROI {frequency_roi.as_list()}",
            use_container_width=True,
        )
    else:
        st.image(draw_roi_overlay(image, roi), caption=f"{image_name} with ROI {roi.as_list()}", use_container_width=True)

    if evaluate:
        if mode.startswith("Frequency estimation"):
            # run the same flat lattice logic used by the frequency tracker
            # but only for this roi
            with st.spinner("Detecting and indexing the flat-liquid lattice..."):
                st.session_state["last_result"] = evaluate_frequency_roi(metadata, frequency_roi)
        else:
            # test if dry grid dots can still be matched in empty/flat references
            with st.spinner("Detecting dry dots and tracking into empty-bath / flat-liquid references..."):
                st.session_state["last_result"] = evaluate_roi(
                    metadata,
                    roi,
                    min_match_score=min_match_score,
                    template_fraction=template_fraction,
                    search_fraction=search_fraction,
                )

    result = st.session_state.get("last_result")
    if result:
        st.subheader("Last Evaluation")
        if result.get("mode") == "frequency_estimation_flat_reference":
            # frequency mode is about lattice quality and indexing
            # not about dry-to-flat matching
            frequency_metric_row(result)
            p95_spacing_error = finite_float(result.get("local_spacing_p95_abs_error_px"))
            if math.isfinite(p95_spacing_error) and p95_spacing_error > 0.25 * result["estimated_spacing_px"]:
                st.warning(
                    "Local dot spacing is varying strongly inside this ROI. Try a smaller region "
                    "or move away from glare, the container edge, or visibly warped regions."
                )
            st.json(
                {
                    "roi_px": result["roi_px"],
                    "roi_kind": result["roi_kind"],
                    "reference_builder_version": result.get("reference_builder_version"),
                    "indexing_method": result.get("indexing_method"),
                    "roi_edge_exclusion_px": result["roi_edge_exclusion_px"],
                    "grid_i_span": result["grid_i_span"],
                    "grid_j_span": result["grid_j_span"],
                    "grid_slot_count": result.get("grid_slot_count"),
                    "grid_occupancy_fraction": result.get("grid_occupancy_fraction"),
                    "n_candidates": result.get("n_candidates"),
                    "n_after_edge_exclusion": result.get("n_after_edge_exclusion"),
                    "n_indexed_dots": result.get("n_indexed_dots"),
                    "indexing_duplicate_cell_count": result.get("indexing_duplicate_cell_count"),
                    "indexing_duplicate_candidate_count": result.get("indexing_duplicate_candidate_count"),
                    "indexing_duplicate_removed_count": result.get("indexing_duplicate_removed_count"),
                    "neighbor_metric_source": result.get("neighbor_metric_source"),
                    "neighbor_pair_count": result.get("neighbor_pair_count"),
                    "nearest_neighbor_pair_count": result.get("nearest_neighbor_pair_count"),
                    "nearest_neighbor_spacing_median_px": result.get("nearest_neighbor_spacing_median_px"),
                    "nearest_neighbor_spacing_cv_percent": result.get("nearest_neighbor_spacing_cv_percent"),
                    "nearest_neighbor_spacing_p95_abs_error_px": result.get("nearest_neighbor_spacing_p95_abs_error_px"),
                    "indexed_neighbor_pair_count": result.get("indexed_neighbor_pair_count"),
                    "indexed_neighbor_spacing_median_px": result.get("indexed_neighbor_spacing_median_px"),
                    "indexed_neighbor_spacing_cv_percent": result.get("indexed_neighbor_spacing_cv_percent"),
                    "indexed_neighbor_spacing_p95_abs_error_px": result.get("indexed_neighbor_spacing_p95_abs_error_px"),
                    "indexed_neighbor_normal_length_fraction": result.get("indexed_neighbor_normal_length_fraction"),
                    "indexed_neighbor_marginal_length_fraction": result.get("indexed_neighbor_marginal_length_fraction"),
                    "indexed_neighbor_bad_length_fraction": result.get("indexed_neighbor_bad_length_fraction"),
                    "indexed_neighbor_diagonal_length_fraction": result.get("indexed_neighbor_diagonal_length_fraction"),
                    "indexing_knn_cardinal_fraction": result.get("indexing_knn_cardinal_fraction"),
                    "indexing_knn_diagonal_fraction": result.get("indexing_knn_diagonal_fraction"),
                    "indexing_knn_nonlocal_fraction": result.get("indexing_knn_nonlocal_fraction"),
                    "indexing_median_cardinal_neighbors_per_dot": result.get("indexing_median_cardinal_neighbors_per_dot"),
                    "indexing_suspect_dot_count": result.get("indexing_suspect_dot_count"),
                    "indexing_suspect_dot_fraction": result.get("indexing_suspect_dot_fraction"),
                    "local_spacing_p05_px": result["local_spacing_p05_px"],
                    "local_spacing_median_px": result["local_spacing_median_px"],
                    "local_spacing_p95_px": result["local_spacing_p95_px"],
                    "affine_warp_residual_note": "The affine residual measures optical/grid warp, not dot detection accuracy.",
                    "median_affine_warp_residual_px": result["median_lattice_residual_px"],
                    "p95_affine_warp_residual_px": result["p95_lattice_residual_px"],
                },
                expanded=False,
            )
            dots = result.get("_dots")
            if dots is not None:
                # visual check that detected indices actually follow the grid
                st.image(
                    draw_frequency_dots_overlay(images["flat liquid reference"], result.get("_roi", frequency_roi), dots),
                    caption="Detected and indexed flat-liquid dots",
                    use_container_width=True,
                )
        else:
            # fsss mode is about whether the dry-reference dots track into
            # empty and flat reference images
            metric_row(result)
            if result["flat_liquid"]["valid_fraction"] < 0.8:
                st.warning(
                    "Flat-liquid tracking is failing even though the dots may be visually clear. "
                    "This usually means the dry-grid-to-flat correspondence is ambiguous. "
                    "Use the frequency-only mode for frequency extraction."
                )
            st.json(
                {
                    "roi_px": result["roi_px"],
                    "template_radius_px": result["template_radius_px"],
                    "search_radius_px": result["search_radius_px"],
                    "empty_bath": result["empty_bath"],
                    "flat_liquid": result["flat_liquid"],
                },
                expanded=False,
            )
            flat_tracks = result.get("_flat_tracks")
            empty_tracks = result.get("_empty_tracks")
            if flat_tracks is not None and empty_tracks is not None:
                # side by side overlays make bad matching obvious
                evaluated_roi = result.get("_roi", roi)
                col1, col2 = st.columns(2)
                col1.image(
                    draw_tracking_overlay(images["flat liquid reference"], evaluated_roi, flat_tracks),
                    caption="Flat-liquid tracking check",
                    use_container_width=True,
                )
                col2.image(
                    draw_tracking_overlay(images["empty container"], evaluated_roi, empty_tracks),
                    caption="Empty-container tracking check",
                    use_container_width=True,
                )

    st.divider()
    # commands we normally run after choosing/saving a roi
    st.code("python scripts/run_pipeline.py frequency --metadata inputs/batch_metadata.yaml", language="bash")
    st.code("python scripts/run_pipeline.py full-fsss --metadata inputs/batch_metadata.yaml --run-calibration", language="bash")
    st.code("python scripts/dot_tracking/build_frequency_reference.py --metadata inputs/calibration_metadata.yaml", language="bash")
    st.code("python scripts/full_fsss/calibrate_dot_grid_pose.py --metadata inputs/calibration_metadata.yaml", language="bash")

if __name__ == "__main__":
    main()
