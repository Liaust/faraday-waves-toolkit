from __future__ import annotations

'''
the point of this file is not to calculate onset automatically
since automatic selection was far too buggy, so
it just loads the onset pipeline outputs and lets us manually choose the time/gamma
where the subharmonic response actually starts looking relevant
'''

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# repo root, everything else is resolved from here
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# optional override for where manual picks are saved
# if we don't set this then it just saves inside outputs/
_manual_pick_env = os.environ.get("FARADAY_MANUAL_PICK_ROOT")
MANUAL_PICK_ROOT = (
    Path(_manual_pick_env).expanduser()
    if _manual_pick_env
    else PROJECT_ROOT / "outputs" / "onset_estimation" / "manual_onset_picks"
)
if not MANUAL_PICK_ROOT.is_absolute():
    MANUAL_PICK_ROOT = PROJECT_ROOT / MANUAL_PICK_ROOT

# basic streamlit page config
st.set_page_config(page_title="Manual Onset Review", layout="wide")

# convert strings from csv/json into actual paths
# they can either be absolute or relative to the repo
def project_path(value: str | Path | None) -> Path | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path

# show paths as repo-relative when possible, since absolute paths are ugly
def relpath(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)

# make a string safe to use as a folder name
def path_slug(value: Any) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    slug = slug.strip("._-")
    return slug or "unknown"

# find all onset batch summaries that already exist
def available_summary_paths() -> list[Path]:
    batch_root = PROJECT_ROOT / "outputs" / "onset_estimation" / "batch"
    if not batch_root.exists():
        return []
    candidates = list(batch_root.glob("*/batch_summary.csv"))
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)

# temporary-ish fallback for older run names that don't explicitly store concentration
def infer_concentration(run_id: str) -> int | None:
    match = re.search(r"(\d+)wt", str(run_id).lower())
    if match:
        return int(match.group(1))
    if str(run_id).lower().startswith("may26"):
        return 40
    return None

# format numbers for the ui, but don't crash on nan/missing values
def format_number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"

# same idea but returns a float, useful for defaults like slider position
def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default

# cached readers, otherwise moving the slider would keep re-reading files
@st.cache_data(show_spinner=False)
def load_csv(path_text: str) -> pd.DataFrame:
    return pd.read_csv(Path(path_text))

@st.cache_data(show_spinner=False)
def load_json(path_text: str) -> dict[str, Any]:
    return json.loads(Path(path_text).read_text(encoding="utf-8"))

# read the batch summary and normalize a few old/new column names
def load_batch_summary(path: Path) -> pd.DataFrame:
    df = load_csv(str(path))
    if "onset_review_summary_json" not in df.columns and "summary_json" in df.columns:
        df["onset_review_summary_json"] = df["summary_json"]
    if "concentration_wt" not in df.columns:
        df["concentration_wt"] = df["run_id"].map(infer_concentration)
    for column in ["drive_frequency_hz", "expected_subharmonic_hz", "concentration_wt"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

# one folder for the manual picks of this specific batch
def manual_batch_dir(summary_csv_path: Path) -> Path:
    return MANUAL_PICK_ROOT / path_slug(summary_csv_path.parent.name)

# all paths related to saving one manual onset pick
def manual_pick_paths(summary_csv_path: Path, run_id: str) -> dict[str, Path]:
    batch_dir = manual_batch_dir(summary_csv_path)
    run_dir = batch_dir / path_slug(run_id)
    return {
        "batch_dir": batch_dir,
        "run_dir": run_dir,
        "run_json": run_dir / "manual_onset_pick.json",
        "batch_csv": batch_dir / "manual_onset_picks.csv",
    }

# json doesn't like numpy scalars or nan, so clean everything before saving
def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value

# load previous manual picks
# the combined csv is preferred, but if it is missing we rebuild from the json files
def load_manual_picks(summary_csv_path: Path) -> pd.DataFrame:
    paths = manual_pick_paths(summary_csv_path, "export")
    batch_csv = paths["batch_csv"]
    if batch_csv.exists():
        return pd.read_csv(batch_csv)
    records: list[dict[str, Any]] = []
    for path in sorted(paths["batch_dir"].glob("*/manual_onset_pick.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(records)

# load one saved pick if this run was already reviewed
def load_manual_pick(summary_csv_path: Path, run_id: str) -> dict[str, Any] | None:
    path = manual_pick_paths(summary_csv_path, run_id)["run_json"]
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

# the slider time is continuous but the data is sampled at video frame times
# so here we interpolate any metric to the exact time we chose
def interpolate_metric(metrics: pd.DataFrame, column: str, time_s: float) -> float:
    if metrics.empty or column not in metrics.columns:
        return float("nan")
    time = pd.to_numeric(metrics["time_s"], errors="coerce").to_numpy(float)
    values = pd.to_numeric(metrics[column], errors="coerce").to_numpy(float)
    good = np.isfinite(time) & np.isfinite(values)
    if not np.any(good):
        return float("nan")
    return float(np.interp(float(time_s), time[good], values[good]))

# these are the actual numbers that get shown/saved for the chosen onset time
def selected_values(metrics: pd.DataFrame, time_s: float) -> dict[str, float]:
    return {
        "time_s": float(time_s),
        "manual_onset_gamma_g": interpolate_metric(metrics, "gamma_interpolated_g", time_s),
        "manual_envelope_px": interpolate_metric(metrics, "subharmonic_band_envelope_px", time_s),
        "manual_exact_envelope_px": interpolate_metric(metrics, "exact_subharmonic_envelope_px", time_s),
        "manual_log10_excess_px": interpolate_metric(metrics, "log10_excess_px", time_s),
        "manual_best_frequency_hz": interpolate_metric(metrics, "best_frequency_hz", time_s),
        "manual_valid_dot_fraction": interpolate_metric(metrics, "valid_dot_fraction", time_s),
    }

# load all files for one run
# batch csv -> summary json -> detailed csv/npz/png outputs
def load_selected_run(row: pd.Series) -> dict[str, Any]:
    summary_path = project_path(row.get("onset_review_summary_json"))
    if summary_path is None or not summary_path.exists():
        raise FileNotFoundError(f"Missing onset review summary: {summary_path}")
    summary = load_json(str(summary_path))
    outputs = summary.get("outputs", {})
    video_metrics_path = project_path(row.get("video_metrics_csv") or outputs.get("video_metrics_csv"))
    accel_metrics_path = project_path(row.get("accelerometer_metrics_csv") or outputs.get("accelerometer_metrics_csv"))
    lockin_path = project_path(outputs.get("lockin_power_npz"))
    review_plot_path = project_path(outputs.get("manual_review_plot"))
    if video_metrics_path is None or not video_metrics_path.exists():
        raise FileNotFoundError(f"Missing video metrics CSV: {video_metrics_path}")
    if accel_metrics_path is None or not accel_metrics_path.exists():
        raise FileNotFoundError(f"Missing accelerometer metrics CSV: {accel_metrics_path}")
    return {
        "summary_path": summary_path,
        "summary": summary,
        "video_metrics_path": video_metrics_path,
        "accel_metrics_path": accel_metrics_path,
        "lockin_path": lockin_path,
        "review_plot_path": review_plot_path,
        "video_metrics": load_csv(str(video_metrics_path)),
        "accel_metrics": load_csv(str(accel_metrics_path)),
    }

# label used in the run dropdown
def run_label(row: pd.Series, manual_saved: bool) -> str:
    gamma = "saved" if manual_saved else "no pick"
    concentration = format_number(row.get("concentration_wt"), 0)
    drive = format_number(row.get("drive_frequency_hz"), 2)
    return f"{row['run_id']} | {concentration} wt% | fd={drive} Hz | {gamma}"

# red vertical line showing the current manual pick on the plots
def add_time_marker(fig: go.Figure, time_s: float, label: str, *, row: int | None = None) -> None:
    if not math.isfinite(time_s):
        return
    if row is None:
        fig.add_vline(x=time_s, line_color="#dc2626", line_dash="dash", line_width=1.4)
    else:
        fig.add_vline(x=time_s, line_color="#dc2626", line_dash="dash", line_width=1.4, row=row, col=1)
    fig.add_annotation(
        x=time_s,
        y=1.0,
        yref="paper",
        text=label,
        showarrow=False,
        yanchor="bottom",
        textangle=-90,
        font={"color": "#dc2626", "size": 11},
    )

# main plot: video response and gamma in the same time axis
# gamma gets its own y axis since the units are different
def review_time_figure(metrics: pd.DataFrame, accel: pd.DataFrame, selected_time: float) -> go.Figure:
    fig = go.Figure()
    # the main signal from the video, strongest response inside the subharmonic band
    fig.add_trace(
        go.Scatter(
            x=metrics["time_s"],
            y=metrics["subharmonic_band_envelope_px"],
            mode="lines",
            name="max subharmonic envelope",
        )
    )
    if "exact_subharmonic_envelope_px" in metrics.columns:
        # also show exactly f_d/2 if that column exists
        fig.add_trace(
            go.Scatter(
                x=metrics["time_s"],
                y=metrics["exact_subharmonic_envelope_px"],
                mode="lines",
                name="exact half-drive envelope",
                opacity=0.75,
            )
    )
    if "gamma_vector_measured_drive_g" in accel.columns:
        # gamma envelope from the accelerometer, using the measured drive frequency
        fig.add_trace(
            go.Scatter(
                x=accel["time_s"],
                y=accel["gamma_vector_measured_drive_g"],
                mode="lines",
                name="gamma envelope",
                yaxis="y2",
            )
        )
    add_time_marker(fig, selected_time, "manual")
    fig.update_layout(
        height=430,
        xaxis_title="time [s]",
        yaxis_title="video envelope [px]",
        yaxis2={"title": "gamma [g]", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 60, "r": 60, "t": 35, "b": 50},
    )
    return fig

# second plot: envelope as a function of gamma
# this is useful because onset is really chosen in terms of gamma, not only time
def envelope_vs_gamma_figure(metrics: pd.DataFrame, selected: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    color = metrics["time_s"] if "time_s" in metrics.columns else None
    fig.add_trace(
        go.Scatter(
            x=metrics["gamma_interpolated_g"],
            y=metrics["subharmonic_band_envelope_px"],
            mode="markers",
            marker={"size": 5, "color": color, "colorscale": "Viridis", "showscale": True, "colorbar": {"title": "time [s]"}},
            name="video samples",
            text=metrics["time_s"],
            hovertemplate="t=%{text:.3f}s<br>gamma=%{x:.4f}<br>env=%{y:.4g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[selected["manual_onset_gamma_g"]],
            y=[selected["manual_envelope_px"]],
            mode="markers",
            marker={"size": 13, "color": "#dc2626", "symbol": "x"},
            name="manual pick",
        )
    )
    fig.update_layout(
        height=420,
        xaxis_title="gamma [g]",
        yaxis_title="subharmonic envelope [px]",
        margin={"l": 60, "r": 25, "t": 35, "b": 55},
    )
    return fig

# heatmap of the lock-in power around the subharmonic band
# this comes from the npz generated by compute_runup_onset_metrics.py
def lockin_heatmap(path: Path | None, selected_time: float) -> go.Figure | None:
    if path is None or not path.exists():
        return None
    with np.load(path) as data:
        time_s = data["time_s"]
        freqs = data["frequencies_hz"]
        power = data["power_px2"]
    # add a small floor otherwise log10(0) becomes -inf
    finite_power = power[np.isfinite(power)]
    eps = max(float(np.nanpercentile(finite_power, 5)), 1e-12) if finite_power.size else 1e-12
    fig = go.Figure(
        data=go.Heatmap(
            x=time_s,
            y=freqs,
            z=np.log10(power.T + eps),
            colorscale="Viridis",
            colorbar={"title": "log10 power"},
        )
    )
    add_time_marker(fig, selected_time, "manual")
    fig.update_layout(
        height=380,
        xaxis_title="time [s]",
        yaxis_title="frequency [Hz]",
        margin={"l": 60, "r": 25, "t": 35, "b": 55},
    )
    return fig

# make the actual row/json object that represents one manual pick
# we only save scalar values and paths, not the full time series
def manual_record(
    summary_csv_path: Path,
    row: pd.Series,
    loaded: dict[str, Any],
    selected: dict[str, float],
    *,
    reviewer: str,
    status: str,
    notes: str,
) -> dict[str, Any]:
    summary = loaded["summary"]
    return {
        "run_id": str(row["run_id"]),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reviewer": reviewer.strip(),
        "manual_status": status,
        "manual_notes": notes.strip(),
        "manual_onset_time_s": selected["time_s"],
        "manual_onset_gamma_g": selected["manual_onset_gamma_g"],
        "manual_envelope_px": selected["manual_envelope_px"],
        "manual_exact_envelope_px": selected["manual_exact_envelope_px"],
        "manual_log10_excess_px": selected["manual_log10_excess_px"],
        "manual_best_frequency_hz": selected["manual_best_frequency_hz"],
        "manual_valid_dot_fraction": selected["manual_valid_dot_fraction"],
        "concentration_wt": row.get("concentration_wt"),
        "drive_frequency_hz": row.get("drive_frequency_hz"),
        "expected_subharmonic_hz": row.get("expected_subharmonic_hz"),
        "measured_drive_frequency_hz": summary.get("accelerometer", {}).get("measured_drive_frequency_hz"),
        "source_batch_summary": relpath(summary_csv_path),
        "source_summary_json": relpath(loaded["summary_path"]),
        "source_video_metrics_csv": relpath(loaded["video_metrics_path"]),
        "source_accelerometer_metrics_csv": relpath(loaded["accel_metrics_path"]),
    }

# save the pick twice:
# one json per run for traceability, and one combined csv for later plotting
def save_manual_pick(
    summary_csv_path: Path,
    row: pd.Series,
    loaded: dict[str, Any],
    selected: dict[str, float],
    *,
    reviewer: str,
    status: str,
    notes: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = manual_pick_paths(summary_csv_path, str(row["run_id"]))
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["batch_dir"].mkdir(parents=True, exist_ok=True)
    record = json_safe(
        manual_record(
            summary_csv_path,
            row,
            loaded,
            selected,
            reviewer=reviewer,
            status=status,
            notes=notes,
        )
    )
    paths["run_json"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    record_df = pd.DataFrame([record])
    if paths["batch_csv"].exists():
        # if we saved this run before, replace that row instead of duplicating it
        existing = pd.read_csv(paths["batch_csv"])
        existing = existing[existing["run_id"].astype(str) != str(row["run_id"])] if "run_id" in existing.columns else existing
        combined = pd.concat([existing, record_df], ignore_index=True)
    else:
        combined = record_df
    sort_columns = [column for column in ["concentration_wt", "drive_frequency_hz", "run_id"] if column in combined.columns]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="stable")
    combined.to_csv(paths["batch_csv"], index=False)
    return record, paths

# actual streamlit app
# choose batch -> choose run -> move slider -> save the manual pick
def main() -> None:
    st.title("Manual Onset Review")
    st.caption(f"Project root: `{PROJECT_ROOT}`")
    if "manual_save_message" in st.session_state:
        st.success(st.session_state.pop("manual_save_message"))

    summary_paths = available_summary_paths()
    custom_path = st.sidebar.text_input("Custom batch summary CSV", value="")
    if custom_path.strip():
        # manual path wins if we type one
        selected_summary_path = project_path(custom_path.strip())
    elif summary_paths:
        labels = [relpath(path) for path in summary_paths]
        selected_label = st.sidebar.selectbox("Batch summary", labels, index=0)
        selected_summary_path = PROJECT_ROOT / selected_label
    else:
        selected_summary_path = None

    if selected_summary_path is None or not selected_summary_path.exists():
        st.error("No onset batch summary found. Run `python scripts/run_pipeline.py onset --metadata inputs/batch_metadata.yaml` first.")
        return

    summary_df = load_batch_summary(selected_summary_path)
    manual_df = load_manual_picks(selected_summary_path)
    # used for the saved/no pick labels and the unsaved only filter
    saved_run_ids = set(manual_df["run_id"].astype(str)) if not manual_df.empty and "run_id" in manual_df else set()

    st.caption(f"Batch summary: `{relpath(selected_summary_path)}`")
    cols = st.columns(4)
    cols[0].metric("Runs", len(summary_df))
    cols[1].metric("Manual picks", len(saved_run_ids))
    cols[2].metric("Output folder", relpath(manual_batch_dir(selected_summary_path)))
    cols[3].metric("Concentrations", len(summary_df["concentration_wt"].dropna().unique()) if "concentration_wt" in summary_df else 0)

    with st.expander("Manual pick export", expanded=bool(saved_run_ids)):
        paths = manual_pick_paths(selected_summary_path, "export")
        st.caption(f"CSV: `{relpath(paths['batch_csv'])}`")
        manual_csv = manual_df.to_csv(index=False).encode("utf-8") if not manual_df.empty else b""
        st.download_button(
            "Download manual_onset_picks.csv",
            data=manual_csv,
            file_name="manual_onset_picks.csv",
            mime="text/csv",
            disabled=manual_df.empty,
        )
        if not manual_df.empty:
            st.dataframe(manual_df, use_container_width=True, hide_index=True)

    filters = summary_df.copy()
    if "concentration_wt" in filters:
        # filter by concentration without changing the original summary table
        options = sorted(int(v) for v in filters["concentration_wt"].dropna().unique())
        selected = st.sidebar.multiselect("Concentration", options, default=options, format_func=lambda value: f"{value} wt%")
        if selected:
            filters = filters[filters["concentration_wt"].isin(selected)]
    if "drive_frequency_hz" in filters:
        freq_options = sorted(float(v) for v in filters["drive_frequency_hz"].dropna().unique())
        selected_freqs = st.sidebar.multiselect("Drive frequency", freq_options, default=freq_options, format_func=lambda value: f"{value:g} Hz")
        if selected_freqs:
            filters = filters[filters["drive_frequency_hz"].isin(selected_freqs)]
    show_unsaved_only = st.sidebar.checkbox("Unsaved only", value=False)
    if show_unsaved_only:
        filters = filters[~filters["run_id"].astype(str).isin(saved_run_ids)]
    search = st.sidebar.text_input("Run ID contains", value="")
    if search.strip():
        filters = filters[filters["run_id"].astype(str).str.contains(search.strip(), case=False, na=False)]
    if filters.empty:
        st.warning("No runs match the current filters.")
        return

    filters = filters.sort_values(["concentration_wt", "drive_frequency_hz", "run_id"], kind="stable")
    labels = [run_label(row, str(row["run_id"]) in saved_run_ids) for _, row in filters.iterrows()]
    selected_label = st.selectbox("Run", labels, index=0)
    row = filters.iloc[labels.index(selected_label)]
    loaded = load_selected_run(row)
    metrics = loaded["video_metrics"]
    accel = loaded["accel_metrics"]
    existing = load_manual_pick(selected_summary_path, str(row["run_id"]))

    st.subheader(str(row["run_id"]))
    summary = loaded["summary"]
    meta_cols = st.columns(5)
    meta_cols[0].metric("Concentration", f"{format_number(row.get('concentration_wt'), 0)} wt%")
    meta_cols[1].metric("Drive", f"{format_number(row.get('drive_frequency_hz'), 3)} Hz")
    meta_cols[2].metric("Expected f/2", f"{format_number(row.get('expected_subharmonic_hz'), 3)} Hz")
    meta_cols[3].metric("Measured drive", f"{format_number(summary.get('accelerometer', {}).get('measured_drive_frequency_hz'), 3)} Hz")
    meta_cols[4].metric("Dots used", summary.get("tracking", {}).get("dots_used", "n/a"))

    min_t = float(np.nanmin(metrics["time_s"]))
    max_t = float(np.nanmax(metrics["time_s"]))
    # if this was already reviewed, start the slider at the saved time
    # otherwise start at the first available time
    default_t = finite_float(existing.get("manual_onset_time_s") if existing else None, min_t)
    default_t = min(max(default_t, min_t), max_t)
    selected_time = st.slider("Manual onset time [s]", min_value=min_t, max_value=max_t, value=default_t, step=max((max_t - min_t) / 1000.0, 0.001))
    selected_values_now = selected_values(metrics, selected_time)

    pick_cols = st.columns(5)
    pick_cols[0].metric("Manual t", f"{selected_values_now['time_s']:.3f} s")
    pick_cols[1].metric("Manual Γ", format_number(selected_values_now["manual_onset_gamma_g"], 4))
    pick_cols[2].metric("Envelope", format_number(selected_values_now["manual_envelope_px"], 4))
    pick_cols[3].metric("Best f", f"{format_number(selected_values_now['manual_best_frequency_hz'], 3)} Hz")
    pick_cols[4].metric("Valid dots", format_number(selected_values_now["manual_valid_dot_fraction"], 3))

    st.plotly_chart(review_time_figure(metrics, accel, selected_time), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(envelope_vs_gamma_figure(metrics, selected_values_now), use_container_width=True)
    with right:
        heatmap = lockin_heatmap(loaded["lockin_path"], selected_time)
        if heatmap is not None:
            st.plotly_chart(heatmap, use_container_width=True)
        elif loaded["review_plot_path"] and loaded["review_plot_path"].exists():
            st.image(str(loaded["review_plot_path"]))

    with st.expander("Loaded files", expanded=False):
        # quick provenance table, basically "what did this app load?"
        files = pd.DataFrame(
            [
                {"file": "onset_review_summary.json", "path": relpath(loaded["summary_path"])},
                {"file": "video_onset_metrics.csv", "path": relpath(loaded["video_metrics_path"])},
                {"file": "accelerometer_gamma_metrics.csv", "path": relpath(loaded["accel_metrics_path"])},
                {"file": "subharmonic_lockin_band_power.npz", "path": relpath(loaded["lockin_path"])},
                {"file": "manual_onset_review.png", "path": relpath(loaded["review_plot_path"])},
            ]
        )
        st.dataframe(files, use_container_width=True, hide_index=True)
        st.dataframe(metrics.head(500), use_container_width=True)

    st.divider()
    save_cols = st.columns([1, 1, 2])
    reviewer = save_cols[0].text_input("Reviewer", value=(existing or {}).get("reviewer", ""))
    status = save_cols[1].selectbox(
        "Status",
        ["accepted", "needs_review", "rejected"],
        index=["accepted", "needs_review", "rejected"].index((existing or {}).get("manual_status", "accepted"))
        if (existing or {}).get("manual_status", "accepted") in ["accepted", "needs_review", "rejected"]
        else 0,
    )
    notes = save_cols[2].text_input("Notes", value=(existing or {}).get("manual_notes", ""))
    if st.button("Save Manual Pick", type="primary"):
        # save and rerun so the progress counters update immediately
        _, paths = save_manual_pick(
            selected_summary_path,
            row,
            loaded,
            selected_values_now,
            reviewer=reviewer,
            status=status,
            notes=notes,
        )
        st.session_state["manual_save_message"] = f"Saved `{relpath(paths['run_json'])}` and updated `{relpath(paths['batch_csv'])}`"
        st.rerun()

if __name__ == "__main__":
    main()
