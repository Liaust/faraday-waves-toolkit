#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "paper_data" / "processed_results" / "onset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "publication_figures" / "onset"

COLORS = {
    20: "#2563eb",
    30: "#7c3aed",
    40: "#16a34a",
    50: "#0891b2",
    60: "#d97706",
    70: "#dc2626",
    80: "#4b5563",
}
MARKERS = {
    20: "o",
    30: "s",
    40: "^",
    50: "D",
    60: "v",
    70: "P",
    80: "X",
}


# Publication figure CLI: this script intentionally consumes curated CSVs rather
# than raw pipeline outputs, so the plotted dataset exactly matches the paper.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate publication onset figures from curated manual-onset CSV files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing the curated onset CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where the figure files will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output resolution.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


# Curated input CSV contract: fail early if a required processed-results table
# is missing, instead of silently making an incomplete publication figure.
def read_required_csv(data_dir: Path, name: str) -> pd.DataFrame:
    path = project_path(data_dir) / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


# Save every publication figure as both PNG and PDF and return paths for the
# manifest JSON.
def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    basename: str,
    *,
    dpi: int,
) -> dict[str, str]:
    output_dir = project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def color_for(concentration: int) -> str:
    return COLORS.get(int(concentration), "#111827")


def marker_for(concentration: int) -> str:
    return MARKERS.get(int(concentration), "o")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column != "concentration_wt"]


# Dataset-wide onset heatmap: summarize mean manual onset gamma by glycerol
# concentration and nominal drive frequency, with run counts annotated per cell.
def plot_dataset_wide_onset_heatmap(
    values: pd.DataFrame,
    counts: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int,
) -> dict[str, str]:
    frequencies = numeric_columns(values)
    matrix = values[frequencies].to_numpy(dtype=float)
    count_matrix = counts[frequencies].to_numpy(dtype=float)
    concentrations = values["concentration_wt"].to_numpy(dtype=float)

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e7eb")

    fig, ax = plt.subplots(figsize=(8.3, 6.0), dpi=dpi)
    image = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(np.arange(len(frequencies)))
    ax.set_xticklabels([f"{float(freq):g}" for freq in frequencies], fontsize=10)
    ax.set_yticks(np.arange(len(concentrations)))
    ax.set_yticklabels([f"{int(c)} wt%" for c in concentrations], fontsize=10)
    ax.set_xticks(np.arange(len(frequencies) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(concentrations) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.15, alpha=0.72)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel(r"nominal drive frequency $f_{d,\mathrm{nom}}$ [Hz]")
    ax.set_ylabel("glycerol concentration")
    ax.set_title(r"Measured onset acceleration across the run-up dataset")

    finite_max = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    threshold = 0.52 * finite_max
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            count = count_matrix[row_index, col_index]
            if np.isfinite(value):
                label = (
                    f"{value:.2f}\n$n={int(count)}$"
                    if np.isfinite(count)
                    else f"{value:.2f}"
                )
                text_color = "white" if value >= threshold else "#111827"
            else:
                label = "NA"
                text_color = "#111827"
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8.7,
                color=text_color,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(r"mean measured onset $\Gamma_z$ [g]")
    return save_figure(
        fig,
        output_dir,
        "manual_dataset_wide_vertical_onset_heatmap",
        dpi=dpi,
    )


# Theory-collapse plot: compare grouped manual onset accelerations with the
# continuous-min-k theoretical threshold and the saved global fit parameters.
def plot_theory_collapse(
    grouped: pd.DataFrame,
    global_fits: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int,
) -> dict[str, str]:
    x = grouped["continuous_min_theory_gamma_g"].to_numpy(dtype=float)
    y = grouped["mean_measured_gamma_g"].to_numpy(dtype=float)
    yerr = grouped["plot_error_gamma_g"].to_numpy(dtype=float)
    xmax = float(max(np.nanmax(x), np.nanmax(y)) * 1.12)
    line = np.linspace(0.0, xmax, 400)

    affine = global_fits[global_fits["fit"].str.contains("affine")].iloc[0]
    zero = global_fits[global_fits["fit"].str.contains("zero_intercept")].iloc[0]

    fig, ax = plt.subplots(figsize=(8.4, 6.8), dpi=dpi)
    ax.plot(line, line, color="0.35", linestyle=":", linewidth=1.45, label="1:1")
    ax.plot(
        line,
        float(affine["slope_a"]) * line + float(affine["intercept_b_g"]),
        color="#db2777",
        linewidth=2.2,
        label=f"affine fit, $R^2={float(affine['r_squared']):.2f}$",
    )
    ax.plot(
        line,
        float(zero["slope_a"]) * line,
        color="#7c3aed",
        linestyle="--",
        linewidth=1.9,
        label=f"zero-intercept fit, $R^2={float(zero['r_squared']):.2f}$",
    )

    for concentration, sub in grouped.groupby("concentration_wt", sort=True):
        c = int(concentration)
        ax.errorbar(
            sub["continuous_min_theory_gamma_g"],
            sub["mean_measured_gamma_g"],
            yerr=sub["plot_error_gamma_g"],
            fmt=marker_for(c),
            color=color_for(c),
            markersize=7.0,
            capsize=3,
            linestyle="none",
            label=f"{c} wt%",
        )

    ax.text(
        0.04,
        0.96,
        (
            rf"$\Gamma_z={float(affine['slope_a']):.3f}"
            rf"\Gamma_{{\mathrm{{min}}\!-\!\mathrm{{k}}}}"
            rf"{float(affine['intercept_b_g']):+.3f}$"
            "\n"
            rf"$R^2={float(affine['r_squared']):.3f}$, "
            rf"RMSE={float(affine['rmse_g']):.3f} g"
            "\n"
            rf"zero-intercept: $\Gamma_z="
            rf"{float(zero['slope_a']):.3f}"
            rf"\Gamma_{{\mathrm{{min}}\!-\!\mathrm{{k}}}}$"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "alpha": 0.92,
        },
    )
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, xmax)
    ax.set_xlabel(r"continuous-min-k theory $\Gamma_{\mathrm{min}\!-\!\mathrm{k}}$ [g]")
    ax.set_ylabel(r"measured onset $\Gamma_z$ [g]")
    ax.set_title("Measured onsets versus continuous-min-k threshold")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8.0, frameon=True, loc="lower right", ncol=2)
    return save_figure(
        fig,
        output_dir,
        "manual_theory_dependency_collapse_publication",
        dpi=dpi,
    )


# Onset-versus-frequency plot: show individual manual picks faintly and grouped
# mean/error values prominently for each concentration.
def plot_onset_vs_frequency(
    individual: pd.DataFrame,
    grouped: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(9.0, 6.2), dpi=dpi)
    for concentration, sub_ind in individual.groupby("concentration_wt", sort=True):
        c = int(concentration)
        color = color_for(c)
        sub_group = grouped[grouped["concentration_wt"] == c].sort_values(
            "mean_actual_drive_frequency_hz"
        )
        ax.scatter(
            sub_ind["actual_drive_frequency_hz"],
            sub_ind["manual_onset_gamma_z_g"],
            s=22,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.errorbar(
            sub_group["mean_actual_drive_frequency_hz"],
            sub_group["mean_measured_gamma_g"],
            yerr=sub_group["plot_error_gamma_g"],
            fmt=marker_for(c),
            color=color,
            markersize=7.0,
            capsize=3,
            linewidth=2.0,
            linestyle="-",
            label=f"{c} wt%",
        )

    ax.set_xlabel(r"measured drive frequency $f_d$ [Hz]")
    ax.set_ylabel(r"measured onset $\Gamma_z$ [g]")
    ax.set_title("Manual onset acceleration versus drive frequency")
    ax.grid(True, alpha=0.28)
    ax.set_xlim(14.0, 31.0)
    ymax = float(np.nanmax(grouped["mean_measured_gamma_g"] + grouped["plot_error_gamma_g"]))
    ax.set_ylim(0.0, ymax * 1.16)
    ax.legend(fontsize=8.3, frameon=True, ncol=2)
    return save_figure(
        fig,
        output_dir,
        "manual_onset_vs_frequency_publication",
        dpi=dpi,
    )


def main() -> None:
    args = parse_args()
    data_dir = project_path(args.data_dir)
    output_dir = project_path(args.output_dir)

    # Load the curated publication tables produced after manual onset review and
    # downstream aggregation.
    individual = read_required_csv(data_dir, "manual_publication_onsets_individual.csv")
    grouped = read_required_csv(
        data_dir,
        "manual_publication_onsets_grouped_by_nominal_frequency.csv",
    )
    heatmap_values = read_required_csv(data_dir, "dataset_wide_vertical_heatmap_values.csv")
    heatmap_counts = read_required_csv(data_dir, "dataset_wide_vertical_heatmap_run_counts.csv")
    global_fits = read_required_csv(data_dir, "manual_min_k_global_fits.csv")

    outputs = {
        "dataset_wide_onset_heatmap": plot_dataset_wide_onset_heatmap(
            heatmap_values,
            heatmap_counts,
            output_dir,
            dpi=args.dpi,
        ),
        "theory_collapse": plot_theory_collapse(
            grouped,
            global_fits,
            output_dir,
            dpi=args.dpi,
        ),
        "onset_vs_frequency": plot_onset_vs_frequency(
            individual,
            grouped,
            output_dir,
            dpi=args.dpi,
        ),
    }

    # Manifest records the exact figure files and dataset sizes used for this
    # publication-figure regeneration.
    summary = {
        "figure_group": "onset_publication_figures",
        "data_dir": str(data_dir),
        "n_individual_onsets": int(len(individual)),
        "n_grouped_conditions": int(len(grouped)),
        "outputs": outputs,
    }
    summary_path = output_dir / "onset_publication_figures_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    for figure_outputs in outputs.values():
        print(f"Wrote {figure_outputs['png']}")
        print(f"Wrote {figure_outputs['pdf']}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
