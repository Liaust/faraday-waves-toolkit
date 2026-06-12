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
DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "paper_data"
    / "processed_results"
    / "frequency"
    / "dataset_wide_half_integer_bands"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "publication_figures" / "frequency"


def parse_args() -> argparse.Namespace:
    # figure cli
    # input is curated processed data, not raw videos
    # this only rebuilds the final dataset-wide heatmap figure
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the dataset-wide half-integer frequency-band figure from "
            "curated processed spectral-power CSV files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder containing heatmap_log10_power.csv and heatmap_row_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where the figure files will be written.",
    )
    parser.add_argument(
        "--image-name",
        default="dataset_wide_half_integer_frequency_bands",
        help="Output basename, without extension.",
    )
    parser.add_argument(
        "--log-floor",
        type=float,
        default=None,
        help="Lower color scale limit. Defaults to the smallest finite value in the heatmap.",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=0.26,
        help="Opacity of the half-integer guide lines.",
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


def load_frequency_heatmap(data_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    # curated csv inputs:
    # heatmap_log10_power.csv has one row-normalized spectrum per selected run
    # heatmap_row_manifest.csv has the concentration/frequency labels in the same order
    data_dir = project_path(data_dir)
    heatmap_path = data_dir / "heatmap_log10_power.csv"
    manifest_path = data_dir / "heatmap_row_manifest.csv"
    if not heatmap_path.exists():
        raise FileNotFoundError(heatmap_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    heatmap_df = pd.read_csv(heatmap_path)
    ratio_grid = np.asarray([float(column) for column in heatmap_df.columns], dtype=float)
    heatmap = heatmap_df.to_numpy(dtype=float)
    manifest = pd.read_csv(manifest_path)

    if heatmap.shape[0] != len(manifest):
        raise ValueError(
            "The heatmap row count does not match heatmap_row_manifest.csv: "
            f"{heatmap.shape[0]} != {len(manifest)}"
        )
    required = {"concentration_wt", "nominal_drive_frequency_hz"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
    return ratio_grid, heatmap, manifest


def concentration_groups(manifest: pd.DataFrame) -> list[tuple[float, int, int]]:
    # assume the manifest is already curated into contiguous concentration bands
    # these row spans are only for separators/labels, not for reordering data
    groups: list[tuple[float, int, int]] = []
    for concentration in pd.unique(manifest["concentration_wt"]):
        indices = manifest.index[manifest["concentration_wt"] == concentration].to_numpy()
        if len(indices) == 0:
            continue
        groups.append((float(concentration), int(indices.min()), int(indices.max()) + 1))
    return groups


def plot_heatmap(
    ratio_grid: np.ndarray,
    heatmap: np.ndarray,
    manifest: pd.DataFrame,
    output_dir: Path,
    image_name: str,
    *,
    log_floor: float | None,
    line_alpha: float,
    dpi: int,
) -> dict[str, str]:
    output_dir = project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # invalid heatmap entries stay as nan
    # masking keeps missing bins transparent instead of making fake low-power bands
    finite = heatmap[np.isfinite(heatmap)]
    if finite.size == 0:
        raise ValueError("The heatmap contains no finite values.")
    vmin = float(np.nanmin(finite)) if log_floor is None else float(log_floor)
    vmax = 0.0
    n_rows = heatmap.shape[0]
    extent = [float(ratio_grid[0]), float(ratio_grid[-1]), n_rows - 0.5, -0.5]

    fig_height = max(6.2, 0.28 * n_rows + 1.65)
    fig, ax = plt.subplots(figsize=(8.6, fig_height), dpi=dpi)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    image = ax.imshow(
        np.ma.masked_invalid(heatmap),
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )

    # half-integer guide lines mark the expected Faraday response sequence f/f_d = n/2
    # low opacity keeps them visible without hiding bright spectral ridges
    half_integer_ratios = np.arange(0.5, 2.6, 0.5)
    for ratio in half_integer_ratios:
        ax.axvline(
            ratio,
            color="white",
            linestyle="--",
            linewidth=0.85,
            alpha=line_alpha,
        )

    # concentration labels go top-right inside each concentration band
    # this avoids colliding with the drive-frequency labels on the left
    for concentration, start, end in concentration_groups(manifest):
        if start > 0:
            ax.axhline(start - 0.5, color="white", linewidth=1.1, alpha=0.88)
        ax.text(
            0.985,
            start + 0.25,
            f"{int(round(concentration))} wt%",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="top",
            color="white",
            fontsize=8.0,
            fontweight="bold",
            bbox={
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.38,
                "pad": 2.0,
            },
            clip_on=True,
        )

    ax.set_title("Dataset-wide half-integer frequency bands", fontsize=13)
    ax.set_xlabel(r"frequency ratio $f/f_d$")
    ax.set_ylabel(r"nominal drive frequency $f_d$ [Hz]")
    ax.set_xlim(float(ratio_grid[0]), float(ratio_grid[-1]))
    ax.set_xticks(np.arange(0.5, 2.6, 0.25))

    ax.set_yticks(np.arange(n_rows))
    # row labels come directly from the manifest
    # this preserves the curated ordering used for the figure
    ax.set_yticklabels(
        [f"{float(value):g}" for value in manifest["nominal_drive_frequency_hz"]],
        fontsize=7,
    )
    ax.tick_params(axis="x", labelsize=8)

    cbar = fig.colorbar(image, ax=ax, pad=0.018)
    cbar.set_label(r"$\log_{10}$ row-normalized spectral power")
    fig.subplots_adjust(left=0.12, right=0.88, top=0.94, bottom=0.10)

    png_path = output_dir / f"{image_name}.png"
    pdf_path = output_dir / f"{image_name}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {"png": str(png_path), "pdf": str(pdf_path)}


def main() -> None:
    args = parse_args()
    ratio_grid, heatmap, manifest = load_frequency_heatmap(args.data_dir)
    outputs = plot_heatmap(
        ratio_grid,
        heatmap,
        manifest,
        args.output_dir,
        args.image_name,
        log_floor=args.log_floor,
        line_alpha=args.line_alpha,
        dpi=args.dpi,
    )
    # manifest records data source, plotted ratio domain, and generated files
    # useful later when checking exactly what figure was made
    summary = {
        "figure": "dataset_wide_half_integer_frequency_bands",
        "data_dir": str(project_path(args.data_dir)),
        "rows": int(heatmap.shape[0]),
        "ratio_min": float(ratio_grid[0]),
        "ratio_max": float(ratio_grid[-1]),
        "outputs": outputs,
    }
    summary_path = project_path(args.output_dir) / f"{args.image_name}_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {outputs['png']}")
    print(f"Wrote {outputs['pdf']}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
