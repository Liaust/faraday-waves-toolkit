#!/usr/bin/env python3

'''
This file is used to calculate the fluid properties of glycerol-water mixtures, based on Tjseard Bron's code
'''


from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# default output goes under outputs/
# this keeps generated setup values separate from hand-written metadata in inputs/
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "fluid_properties" / "fluid_properties.csv"
# default footprint assumes the square bath used in the experiment
# cli can override the area directly or give length/width
DEFAULT_CONTAINER_LENGTH_MM = 100.0
DEFAULT_CONTAINER_WIDTH_MM = 100.0


# one row of fluid properties
# it includes some duplicate aliases because downstream scripts/report tables use
# slightly different names for the same quantities
@dataclass(frozen=True)
class GlycerolWaterProperties:
    sample_id: str | None
    temperature_C: float
    glycerol_mass_fraction: float
    glycerol_wt_percent: float
    water_mass_fraction: float
    water_wt_percent: float
    glycerol_volume_fraction: float
    density_g_per_ml: float
    density_kg_per_m3: float
    rho_kg_m3: float
    dynamic_viscosity_Pa_s: float
    mu_pa_s: float
    dynamic_viscosity_cP: float
    kinematic_viscosity_m2_per_s: float
    kinematic_viscosity_m2_s: float
    nu_m2_s: float
    surface_tension_N_per_m: float
    surface_tension_dyn_per_cm: float
    refractive_index: float
    total_mass_g: float | None
    glycerol_mass_g: float | None
    water_mass_g: float | None
    estimated_volume_ml: float | None
    container_area_mm2: float | None
    bath_height_mm: float | None
    depth_mm: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


OUTPUT_COLUMNS = [
    "sample_id",
    "temperature_C",
    "glycerol_wt_percent",
    "concentration_wt",
    "glycerol_mass_fraction",
    "water_wt_percent",
    "water_mass_fraction",
    "total_mass_g",
    "glycerol_mass_g",
    "water_mass_g",
    "density_g_per_ml",
    "density_kg_per_m3",
    "rho_kg_m3",
    "dynamic_viscosity_Pa_s",
    "mu_pa_s",
    "dynamic_viscosity_cP",
    "kinematic_viscosity_m2_per_s",
    "kinematic_viscosity_m2_s",
    "nu_m2_s",
    "surface_tension_N_per_m",
    "surface_tension_dyn_per_cm",
    "refractive_index",
    "glycerol_volume_fraction",
    "estimated_volume_ml",
    "container_area_mm2",
    "bath_height_mm",
    "depth_mm",
]


# shared parsing/interpolation helpers for tables and cli inputs
def _validate_mass_fraction(glycerol_mass_fraction: float) -> float:
    # internally concentration is always a 0-1 mass fraction
    # cli wt% values are converted before reaching most formulas
    value = float(glycerol_mass_fraction)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"glycerol_mass_fraction must be in [0, 1], got {value}")
    return value


def _optional_float(value: Any) -> float | None:
    # csv/yaml inputs often use blank strings for optional masses/dimensions
    # normalize those blanks to None
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "null", "nan"}:
            return None
        return float(stripped)
    return float(value)


def _linear_interpolate(x: float, points: dict[float, float]) -> float:
    # linear interpolation for sparse literature tables
    # raise instead of extrapolating outside the table
    keys = sorted(points)
    value = float(x)
    if value < keys[0] or value > keys[-1]:
        raise ValueError(f"Cannot interpolate {value}; valid range is [{keys[0]}, {keys[-1]}].")
    for key in keys:
        if math.isclose(value, key, rel_tol=0.0, abs_tol=1e-12):
            return float(points[key])
    for left, right in zip(keys, keys[1:]):
        if left <= value <= right:
            weight = (value - left) / (right - left)
            return float(points[left] + weight * (points[right] - points[left]))
    raise RuntimeError("unreachable interpolation state")


def pure_glycerol_density_g_per_ml(temperature_C: float) -> float:
    """Pure glycerol density in g/cm^3, numerically equal to g/mL."""
    # empirical temperature fit
    # divide by 1000 to get g/mL from the original numeric scale
    return (1273.3 - 0.6121 * float(temperature_C)) / 1000.0


# density model
# compute pure-component densities, convert mass fraction to volume fraction,
# then apply the glycerol-water contraction fit
def pure_water_density_g_per_ml(temperature_C: float) -> float:
    """Pure water density in g/cm^3, numerically equal to g/mL."""
    # compact empirical water-density fit around ordinary lab temperatures
    return 1.0 - math.pow(abs(float(temperature_C) - 4.0) / 622.0, 1.7)


def volume_fraction_from_mass_fraction(
    glycerol_mass_fraction: float,
    temperature_C: float,
) -> float:
    """Convert glycerol mass fraction to glycerol volume fraction."""
    mass_fraction = _validate_mass_fraction(glycerol_mass_fraction)
    glycerol_density = pure_glycerol_density_g_per_ml(temperature_C)
    water_density = pure_water_density_g_per_ml(temperature_C)
    # assume total mass of 1
    # volume = mass / density for each component, so volume fraction follows directly
    glycerol_volume = mass_fraction / glycerol_density
    water_volume = (1.0 - mass_fraction) / water_density
    if glycerol_volume + water_volume == 0.0:
        return 0.0
    return glycerol_volume / (glycerol_volume + water_volume)


def mixture_density_g_per_ml(
    glycerol_mass_fraction: float,
    temperature_C: float,
) -> float:
    """Glycerol-water mixture density using the original contraction fit."""
    mass_fraction = _validate_mass_fraction(glycerol_mass_fraction)
    wt_percent = mass_fraction * 100.0
    volume_fraction = volume_fraction_from_mass_fraction(mass_fraction, temperature_C)
    glycerol_density = pure_glycerol_density_g_per_ml(temperature_C)
    water_density = pure_water_density_g_per_ml(temperature_C)

    contraction_average_percent = (
        1.0
        - math.pow(3.520e-8 * wt_percent, 3.0)
        + math.pow(1.027e-6 * wt_percent, 2.0)
        + 2.5e-4 * wt_percent
        - 1.691e-4
    )
    contraction = 1.0 + contraction_average_percent / 100.0
    ideal_density = glycerol_density * volume_fraction + water_density * (1.0 - volume_fraction)
    return ideal_density * contraction


def dynamic_viscosity_pa_s(
    glycerol_mass_fraction: float,
    temperature_C: float,
) -> float:
    """Glycerol-water dynamic viscosity in Pa s."""
    mass_fraction = _validate_mass_fraction(glycerol_mass_fraction)
    temperature = float(temperature_C)

    # first compute pure-component dynamic viscosities at the requested temperature
    # leading 0.001 converts cP-like values to Pa s
    glycerol_viscosity = (
        0.001
        * 12100.0
        * math.exp((-1233.0 + temperature) * temperature / (9900.0 + 70.0 * temperature))
    )
    water_viscosity = (
        0.001
        * 1.790
        * math.exp((-1230.0 - temperature) * temperature / (36100.0 + 360.0 * temperature))
    )

    # alpha is the empirical blending exponent
    # final line interpolates in log(viscosity), which makes sense because
    # glycerol-water viscosity changes by orders of magnitude with concentration
    a = 0.705 - 0.0017 * temperature
    b = (4.9 + 0.036 * temperature) * math.pow(a, 2.5)
    alpha = (
        1.0
        - mass_fraction
        + (a * b * mass_fraction * (1.0 - mass_fraction))
        / (a * mass_fraction + b * (1.0 - mass_fraction))
    )
    return glycerol_viscosity * math.exp(math.log(water_viscosity / glycerol_viscosity) * alpha)


# surface-tension lookup table from Takamura et al.
# first interpolate over concentration at each tabulated temperature,
# then interpolate over temperature
SURFACE_TENSION_SALINITY_I_DYN_PER_CM = {
    # Takamura et al., Journal of Petroleum Science and Engineering 98-99
    20.0: {0.0: 73.2, 20.0: 71.7, 40.0: 70.0, 60.0: 68.5, 78.0: 67.4, 91.0: 66.5},
    30.0: {0.0: 71.7, 20.0: 70.7, 40.0: 68.7, 60.0: 67.3, 78.0: 66.6, 91.0: 65.4},
    40.0: {0.0: 70.3, 20.0: 69.4, 40.0: 67.6, 60.0: 66.2, 78.0: 65.6, 91.0: 64.4},
    50.0: {0.0: 68.8, 20.0: 67.7, 40.0: 66.7, 60.0: 65.3, 78.0: 64.4, 91.0: 63.4},
    65.0: {0.0: 67.3, 20.0: 65.9, 40.0: 64.5, 60.0: 63.8, 78.0: 62.9, 91.0: 61.3},
    80.0: {0.0: 65.8, 20.0: 64.4, 40.0: 63.6, 60.0: 62.4, 78.0: 61.4, 91.0: 59.6},
}


def surface_tension_dyn_per_cm(
    glycerol_mass_fraction: float,
    temperature_C: float,
) -> float:
    """Surface tension against air from Takamura et al. Table 4, Salinity I."""
    mass_fraction = _validate_mass_fraction(glycerol_mass_fraction)
    wt_percent = mass_fraction * 100.0
    # interpolate across concentration at each table temperature
    # this gives one surface-tension estimate per temperature row
    by_temperature = {
        temperature: _linear_interpolate(wt_percent, by_concentration)
        for temperature, by_concentration in SURFACE_TENSION_SALINITY_I_DYN_PER_CM.items()
    }
    # then interpolate those concentration-specific estimates across temperature
    return _linear_interpolate(float(temperature_C), by_temperature)


def surface_tension_n_per_m(
    glycerol_mass_fraction: float,
    temperature_C: float,
) -> float:
    """Surface tension against air in N/m. 1 dyn/cm = 0.001 N/m."""
    return 0.001 * surface_tension_dyn_per_cm(glycerol_mass_fraction, temperature_C)


# refractive-index fit used by calibration metadata
# this is a 20 C concentration fit, matching the report setup assumption
def refractive_index_20c(glycerol_mass_fraction: float) -> float:
    """Approximate visible-light refractive index from the concentration fits used in metadata."""
    wt_percent = _validate_mass_fraction(glycerol_mass_fraction) * 100.0
    # the data sheet uses different concentration fits below/above 44 wt%
    # this deliberately matches the 20 C report assumption
    if wt_percent <= 44.0:
        return (
            1.3303
            + 0.001124 * wt_percent
            + 0.00000605 * wt_percent**2
            - 0.0000000555 * wt_percent**3
        )
    return 0.00149 * wt_percent + 1.32359


def bath_height_mm(
    total_mass_g: float,
    density_g_per_ml: float,
    container_area_mm2: float,
) -> float:
    """Bath height from mass, density, and horizontal footprint area."""
    # density is g/mL, so mass/density gives volume in mL = cm^3
    volume_ml = float(total_mass_g) / float(density_g_per_ml)
    # 1 cm^2 = 100 mm^2
    # height in cm is volume_cm3 / area_cm2, then multiply by 10 for mm
    area_cm2 = float(container_area_mm2) / 100.0
    return 10.0 * volume_ml / area_cm2


# main property calculator for one mixture
# density, viscosity, surface tension, refractive index, masses, volume,
# and bath height are computed in one place
def properties_from_mass_fraction(
    glycerol_mass_fraction: float,
    temperature_C: float = 20.0,
    *,
    sample_id: str | None = None,
    total_mass_g: float | None = None,
    container_area_mm2: float | None = None,
) -> GlycerolWaterProperties:
    """Return physical properties for one glycerol-water mixture."""
    mass_fraction = _validate_mass_fraction(glycerol_mass_fraction)
    # compute core physical properties first
    # then derive aliases and optional preparation quantities from them
    density_g_per_ml = mixture_density_g_per_ml(mass_fraction, temperature_C)
    density_kg_per_m3 = density_g_per_ml * 1000.0
    mu = dynamic_viscosity_pa_s(mass_fraction, temperature_C)
    nu = mu / density_kg_per_m3
    sigma_dyn_per_cm = surface_tension_dyn_per_cm(mass_fraction, temperature_C)

    total_mass = _optional_float(total_mass_g)
    area = _optional_float(container_area_mm2)
    # these preparation values only exist when total mass and bath area are provided
    # the fluid properties themselves do not depend on total mass
    volume_ml = total_mass / density_g_per_ml if total_mass is not None else None
    height_mm = bath_height_mm(total_mass, density_g_per_ml, area) if total_mass is not None and area else None
    glycerol_mass = total_mass * mass_fraction if total_mass is not None else None
    water_mass = total_mass * (1.0 - mass_fraction) if total_mass is not None else None

    return GlycerolWaterProperties(
        sample_id=sample_id,
        temperature_C=float(temperature_C),
        glycerol_mass_fraction=mass_fraction,
        glycerol_wt_percent=mass_fraction * 100.0,
        water_mass_fraction=1.0 - mass_fraction,
        water_wt_percent=(1.0 - mass_fraction) * 100.0,
        glycerol_volume_fraction=volume_fraction_from_mass_fraction(mass_fraction, temperature_C),
        density_g_per_ml=density_g_per_ml,
        density_kg_per_m3=density_kg_per_m3,
        rho_kg_m3=density_kg_per_m3,
        dynamic_viscosity_Pa_s=mu,
        mu_pa_s=mu,
        dynamic_viscosity_cP=mu * 1000.0,
        kinematic_viscosity_m2_per_s=nu,
        kinematic_viscosity_m2_s=nu,
        nu_m2_s=nu,
        surface_tension_N_per_m=0.001 * sigma_dyn_per_cm,
        surface_tension_dyn_per_cm=sigma_dyn_per_cm,
        refractive_index=refractive_index_20c(mass_fraction),
        total_mass_g=total_mass,
        glycerol_mass_g=glycerol_mass,
        water_mass_g=water_mass,
        estimated_volume_ml=volume_ml,
        container_area_mm2=area,
        bath_height_mm=height_mm,
        depth_mm=height_mm,
    )


# alternate input mode
# user gives glycerol/water masses instead of concentration directly
def properties_from_masses(
    glycerol_mass_g: float,
    water_mass_g: float,
    temperature_C: float = 20.0,
    *,
    sample_id: str | None = None,
    container_area_mm2: float | None = None,
) -> GlycerolWaterProperties:
    # compute concentration from weighed glycerol/water masses,
    # then use the normal mass-fraction path
    total_mass = float(glycerol_mass_g) + float(water_mass_g)
    if total_mass <= 0.0:
        raise ValueError("total mass must be positive")
    return properties_from_mass_fraction(
        float(glycerol_mass_g) / total_mass,
        temperature_C,
        sample_id=sample_id,
        total_mass_g=total_mass,
        container_area_mm2=container_area_mm2,
    )


def _area_from_lengths(length_mm: Any, width_mm: Any) -> float | None:
    # container area can be explicit footprint_area_mm2 or length*width
    length = _optional_float(length_mm)
    width = _optional_float(width_mm)
    if length is None or width is None:
        return None
    return length * width


# metadata input mode for this repo's fluid yaml dictionaries
def properties_from_fluid_yaml_dict(data: dict[str, Any]) -> GlycerolWaterProperties:
    """Compute properties from one of this repo's fluid metadata dictionaries."""
    # yaml mode mirrors the lab metadata layout:
    # composition under composition, temperature under recommended values,
    # bath dimensions under container
    composition = data["composition"]
    analysis_values = (
        data.get("recommended_analysis_values")
        or data.get("recommended_analysis_values_if_temperature_unknown")
        or data.get("properties", {})
    )
    container = data.get("container", {})
    area = (
        _optional_float(container.get("footprint_area_mm2"))
        or _area_from_lengths(container.get("inner_length_mm"), container.get("inner_width_mm"))
    )
    return properties_from_mass_fraction(
        float(composition["glycerol_mass_fraction"]),
        float(analysis_values.get("temperature_C", data.get("temperature_C", 20.0))),
        sample_id=data.get("sample_id"),
        total_mass_g=_optional_float(composition.get("total_mass_g")),
        container_area_mm2=area,
    )


def _row_mass_fraction(row: dict[str, str]) -> float:
    # input csvs can use several names for concentration
    # normalize all of them to a 0-1 glycerol mass fraction
    for key in ("glycerol_mass_fraction", "mass_fraction"):
        value = _optional_float(row.get(key))
        if value is not None:
            return _validate_mass_fraction(value)
    for key in ("glycerol_wt_percent", "concentration_wt", "wt_percent"):
        value = _optional_float(row.get(key))
        if value is not None:
            return _validate_mass_fraction(value / 100.0)
    raise ValueError("input row needs glycerol_mass_fraction, mass_fraction, glycerol_wt_percent, or concentration_wt")


# csv input can give direct footprint area or length/width columns
# fall back to cli container dimensions if the row omits them
def _row_container_area(row: dict[str, str], args: argparse.Namespace) -> float | None:
    row_area = _optional_float(row.get("container_area_mm2"))
    if row_area is not None:
        return row_area
    row_area = _area_from_lengths(row.get("container_length_mm"), row.get("container_width_mm"))
    if row_area is not None:
        return row_area
    if args.container_area_mm2 is not None:
        return args.container_area_mm2
    return _area_from_lengths(args.container_length_mm, args.container_width_mm)


# batch csv input mode
# compute one property row per input mixture row
def rows_from_input_csv(path: Path, args: argparse.Namespace) -> list[GlycerolWaterProperties]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader, start=1):
            # each csv row can override temperature, mass, and bath area
            # missing values fall back to cli defaults
            mass_fraction = _row_mass_fraction(row)
            temperature = _optional_float(row.get("temperature_C")) or args.temperature_C
            sample_id = row.get("sample_id") or f"{mass_fraction * 100.0:g}wt_row{index}"
            total_mass = _optional_float(row.get("total_mass_g"))
            if total_mass is None:
                total_mass = args.total_mass_g
            rows.append(
                properties_from_mass_fraction(
                    mass_fraction,
                    temperature,
                    sample_id=sample_id,
                    total_mass_g=total_mass,
                    container_area_mm2=_row_container_area(row, args),
                )
            )
    return rows


# cli input modes
# read a csv table, compute one concentration, or compute several wt% values
# with shared temperature/mass/container settings
def rows_from_args(args: argparse.Namespace) -> list[GlycerolWaterProperties]:
    # dispatch between the mutually exclusive cli input modes
    if args.input_csv is not None:
        return rows_from_input_csv(args.input_csv, args)
    if args.mass_fraction is not None:
        concentrations = [args.mass_fraction]
    elif args.wt_percent is not None:
        concentrations = [args.wt_percent / 100.0]
    else:
        concentrations = [value / 100.0 for value in args.concentrations]

    area = args.container_area_mm2
    if area is None:
        area = _area_from_lengths(args.container_length_mm, args.container_width_mm)
    # when multiple concentrations are requested, they share the same settings
    return [
        properties_from_mass_fraction(
            mass_fraction,
            args.temperature_C,
            sample_id=f"{mass_fraction * 100.0:g}wt",
            total_mass_g=args.total_mass_g,
            container_area_mm2=area,
        )
        for mass_fraction in concentrations
    ]


def _output_row(properties: GlycerolWaterProperties) -> dict[str, Any]:
    row = properties.as_dict()
    # convenience alias for figure/table code that expects concentration in wt%
    row["concentration_wt"] = properties.glycerol_wt_percent
    return {column: row.get(column, "") for column in OUTPUT_COLUMNS}


# csv output
# fixed column order with descriptive names and shorthand aliases used elsewhere
def write_csv(path: Path, rows: list[GlycerolWaterProperties]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for properties in rows:
            writer.writerow(_output_row(properties))


# public cli for generating fluid_properties.csv
# input can be csv, one wt%, one mass fraction, or a list of concentrations
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate glycerol-water properties and write fluid_properties.csv."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    # exactly one concentration input mode is allowed so the generated csv is unambiguous
    group.add_argument("--input-csv", type=Path, help="CSV table of mixture rows.")
    group.add_argument("--wt-percent", type=float, help="One glycerol mass percent.")
    group.add_argument("--mass-fraction", type=float, help="One glycerol mass fraction.")
    group.add_argument("--concentrations", nargs="+", type=float, help="Glycerol mass percents.")
    parser.add_argument("--temperature-C", type=float, default=20.0)
    parser.add_argument("--total-mass-g", type=float, default=None)
    parser.add_argument("--container-length-mm", type=float, default=DEFAULT_CONTAINER_LENGTH_MM)
    parser.add_argument("--container-width-mm", type=float, default=DEFAULT_CONTAINER_WIDTH_MM)
    parser.add_argument("--container-area-mm2", type=float, default=None)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # rows_from_args builds the dataclass rows
    # write_csv serializes them with the fixed public column order
    rows = rows_from_args(args)
    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} row(s) to {args.output_csv}")


if __name__ == "__main__":
    main()
