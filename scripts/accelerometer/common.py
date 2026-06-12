from __future__ import annotations

'''
this file contains the common functions for the accelerometer analysis
'''

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# summing the values on a window so we can average them out 
def rolling_sum_centered_2d(values: np.ndarray, window: int) -> np.ndarray:
    
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError("rolling_sum_centered_2d expects a 2D array.")
    # window 1 just means no rolling needed
    window = max(1, int(window))
    if window == 1:
        return arr.copy()
    # split the window around the current sample
    # ex: window 5 = 2 before + current + 2 after
    before = window // 2
    after = window - 1 - before
    pad_width = ((before, after), (0, 0))
    # zero padding lets us keep the same output length
    # callers compute denominators separately when missing samples matter
    padded = np.pad(arr, pad_width, mode="constant", constant_values=0)
    zeros = np.zeros((1, padded.shape[1]), dtype=padded.dtype)
    # prefix sums make each rolling sum just one subtraction
    cumsum = np.concatenate([zeros, np.cumsum(padded, axis=0)], axis=0)
    return cumsum[window:] - cumsum[:-window]


def rolling_sum_centered_1d(values: np.ndarray, window: int) -> np.ndarray:
    result = rolling_sum_centered_2d(np.asarray(values)[:, None], window)
    return result[:, 0]


def rolling_mean_centered_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    window = max(1, int(window))
    # ignore nans by summing valid values and dividing by valid counts
    valid = np.isfinite(arr)
    numerator = rolling_sum_centered_1d(np.where(valid, arr, 0.0), window)
    denominator = rolling_sum_centered_1d(valid.astype(float), window)
    out = np.full_like(arr, np.nan, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


# robust sample spacing from the positive time differences
# median is used because accelerometer csvs can have small timing jitter
def median_dt_seconds(times_s: np.ndarray) -> float:
    dt = np.diff(np.asarray(times_s, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return float("nan")
    return float(np.median(dt))


# load accelerometer csv into time_s and xyz axes in g
def load_accel_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    required = {"timestamp_us", "x_g", "y_g", "z_g"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    # timestamps are microseconds, subtract first sample so this run starts at 0 s
    t = (df["timestamp_us"].astype(float).to_numpy() - float(df["timestamp_us"].iloc[0])) / 1e6
    axes = df[["x_g", "y_g", "z_g"]].astype(float).to_numpy()
    axes[~np.isfinite(axes)] = np.nan
    # interpolate missing samples so later dot products don't get poisoned by nans
    for col in range(axes.shape[1]):
        y = axes[:, col]
        good = np.isfinite(y)
        if np.any(good) and not np.all(good):
            axes[:, col] = np.interp(t, t[good], y[good])
    return {"dataframe": df, "time_s": t, "axes_g": axes}


# complex coefficient at one test frequency for x/y/z
# basically a fourier projection that keeps the phase of each axis
def complex_coeff_axes(times_s: np.ndarray, axes_g: np.ndarray, frequency_hz: float) -> np.ndarray:
    times = np.asarray(times_s, dtype=float)
    axes = np.asarray(axes_g, dtype=float)
    # hann window 
    window = np.hanning(len(times)) if len(times) >= 3 else np.ones(len(times))
    # remove static offset / gravity bias before measuring oscillation
    centered = axes - np.nanmedian(axes, axis=0, keepdims=True)
    # complex fourier coefficient at one frequency:
    basis = np.exp(-2j * np.pi * float(frequency_hz) * times)
    return (2.0 / np.sum(window)) * (basis * window) @ centered


# estimate the real drive direction from the complex 3-axis coefficient
def dominant_real_axis_from_coeff(coeff: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=complex).reshape(3)
    # each axis can have phase, so coeff is complex
    # real(c c*) gives a 3x3 energy matrix, dominant eigenvector is the drive axis
    matrix = np.real(np.outer(coeff, np.conj(coeff)))
    values, vectors = np.linalg.eigh(matrix)
    axis = np.asarray(vectors[:, int(np.argmax(values))], dtype=float)
    
    if np.dot(axis, np.real(coeff)) < 0:
        axis = -axis
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0:
        return np.array([0.0, 0.0, 1.0])
    return axis / norm


# spectrum for a scalar signal, after projecting xyz onto the drive axis
def projection_spectrum(
    times_s: np.ndarray,
    signal: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    times = np.asarray(times_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    # remove static offset before scanning frequencies
    signal = signal - np.nanmedian(signal)
    window = np.hanning(len(times)) if len(times) >= 3 else np.ones(len(times))
    y = np.where(np.isfinite(signal), signal, 0.0) * window
    norm = 2.0 / np.sum(window)
    power = np.full(len(frequencies_hz), np.nan, dtype=float)
    # chunk the frequency scan so we don't allocate a huge freq x time matrix
    for start in range(0, len(frequencies_hz), chunk_size):
        stop = min(start + chunk_size, len(frequencies_hz))
        basis = np.exp(-2j * np.pi * np.outer(frequencies_hz[start:stop], times))
        coeff = norm * (basis @ y)
        power[start:stop] = np.abs(coeff) ** 2
    return power


# vector spectrum across xyz before choosing a projection axis
# useful for finding the actual measured drive frequency
def vector_spectrum_axes(
    times_s: np.ndarray,
    axes_g: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    times = np.asarray(times_s, dtype=float)
    axes = np.asarray(axes_g, dtype=float)
    axes = axes - np.nanmedian(axes, axis=0, keepdims=True)
    window = np.hanning(len(times)) if len(times) >= 3 else np.ones(len(times))
    y = np.where(np.isfinite(axes), axes, 0.0) * window[:, None]
    norm = 2.0 / np.sum(window)
    power = np.full(len(frequencies_hz), np.nan, dtype=float)
    for start in range(0, len(frequencies_hz), chunk_size):
        stop = min(start + chunk_size, len(frequencies_hz))
        basis = np.exp(-2j * np.pi * np.outer(frequencies_hz[start:stop], times))
        coeff = norm * (basis @ y)
        # sum xyz power so we can find peaks before choosing the drive axis
        power[start:stop] = np.sum(np.abs(coeff) ** 2, axis=1)
    return power


# pick the largest peak inside a band
def peak_in_band(freqs: np.ndarray, power: np.ndarray, fmin: float, fmax: float) -> dict[str, float]:
    mask = (freqs >= float(fmin)) & (freqs <= float(fmax)) & np.isfinite(power)
    if not np.any(mask):
        return {"frequency_hz": float("nan"), "power": float("nan")}
    local_freqs = freqs[mask]
    local_power = power[mask]
    idx = int(np.argmax(local_power))
    return {"frequency_hz": float(local_freqs[idx]), "power": float(local_power[idx])}


# 1d lock-in envelope at one frequency
# multiply by complex basis, average locally, return oscillation amplitude
def lockin_envelope_1d(times_s: np.ndarray, signal: np.ndarray, frequency_hz: float, window_s: float) -> np.ndarray:
    times = np.asarray(times_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    dt = median_dt_seconds(times)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Cannot estimate acceleration sample spacing.")
    window = max(3, int(round(float(window_s) / dt)))
    if window % 2 == 0:
        window += 1
    y = np.where(np.isfinite(signal), signal, 0.0)
    # demodulate the target frequency down to dc
    # rolling mean then gives local complex amplitude
    basis = np.exp(-2j * np.pi * float(frequency_hz) * times)
    numerator = rolling_sum_centered_1d(y.astype(np.complex64) * basis.astype(np.complex64), window)
    denom = rolling_sum_centered_1d(np.ones_like(y, dtype=np.float32), window)
    coeff = np.zeros_like(numerator, dtype=np.complex64)
    usable = denom > 0
    coeff[usable] = numerator[usable] / denom[usable]
    # for A cos(2 pi f t), coeff magnitude is A/2, so multiply by 2
    return 2.0 * np.abs(coeff)


# 3-axis lock-in envelope
# compute lock-in amplitude for each axis and combine with vector norm
def lockin_envelope_axes(
    times_s: np.ndarray,
    axes_g: np.ndarray,
    frequency_hz: float,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=float)
    axes = np.asarray(axes_g, dtype=float)
    dt = median_dt_seconds(times)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Cannot estimate acceleration sample spacing.")
    window = max(3, int(round(float(window_s) / dt)))
    if window % 2 == 0:
        window += 1
    y = np.where(np.isfinite(axes), axes, 0.0)
    basis = np.exp(-2j * np.pi * float(frequency_hz) * times)
    numerator = rolling_sum_centered_2d(
        y.astype(np.complex64) * basis.astype(np.complex64)[:, None],
        window,
    )
    denom = rolling_sum_centered_1d(np.ones(len(times), dtype=np.float32), window)
    coeff = np.zeros_like(numerator, dtype=np.complex64)
    usable = denom > 0
    coeff[usable, :] = numerator[usable, :] / denom[usable, None]
    # components are per-axis amplitudes, vector_norm is the axis-independent gamma
    components = 2.0 * np.abs(coeff)
    vector_norm = np.linalg.norm(components, axis=1)
    return components, vector_norm


# full accelerometer analysis for one run-up
# subtract bias, find measured drive frequency, compute projection axes and gamma
def analyze_accelerometer(
    *,
    runup_csv: Path,
    calibration_csv: Path,
    drive_hz: float,
    window_s: float,
    auto_search_min_hz: float,
    auto_search_max_hz: float,
    accel_time_offset_s: float,
) -> dict[str, Any]:
    # load run-up and calibration/background csvs
    # calibration median becomes the static bias
    run = load_accel_csv(runup_csv)
    cal = load_accel_csv(calibration_csv)
    bias = np.nanmedian(cal["axes_g"], axis=0)
    run_axes = run["axes_g"] - bias[None, :]
    cal_axes = cal["axes_g"] - bias[None, :]

    # search xyz spectrum for the real drive peak
    # don't fully trust the signal generator frequency
    freqs = np.arange(float(auto_search_min_hz), float(auto_search_max_hz) + 0.025, 0.05)
    vector_spectrum = vector_spectrum_axes(run["time_s"], run_axes, freqs)
    vector_global_peak = peak_in_band(freqs, vector_spectrum, auto_search_min_hz, auto_search_max_hz)
    vector_drive_peak = peak_in_band(freqs, vector_spectrum, drive_hz - 2.0, drive_hz + 2.0)
    vector_double_peak = peak_in_band(freqs, vector_spectrum, 2.0 * drive_hz - 3.0, 2.0 * drive_hz + 3.0)
    measured_drive_hz = float(vector_drive_peak["frequency_hz"])
    if not np.isfinite(measured_drive_hz):
        measured_drive_hz = float(drive_hz)

    # estimate drive direction at nominal fd and measured fd
    # measured one is the one we mainly use later
    coeff_drive = complex_coeff_axes(run["time_s"], run_axes, drive_hz)
    drive_axis = dominant_real_axis_from_coeff(coeff_drive)
    coeff_measured_drive = complex_coeff_axes(run["time_s"], run_axes, measured_drive_hz)
    measured_drive_axis = dominant_real_axis_from_coeff(coeff_measured_drive)
    run_projection = run_axes @ drive_axis
    cal_projection = cal_axes @ drive_axis
    run_measured_projection = run_axes @ measured_drive_axis
    cal_measured_projection = cal_axes @ measured_drive_axis

    spectrum = projection_spectrum(run["time_s"], run_projection, freqs)
    global_peak = peak_in_band(freqs, spectrum, auto_search_min_hz, auto_search_max_hz)
    drive_peak = peak_in_band(freqs, spectrum, drive_hz - 2.0, drive_hz + 2.0)
    double_peak = peak_in_band(freqs, spectrum, 2.0 * drive_hz - 3.0, 2.0 * drive_hz + 3.0)

    # compute both projected scalar gamma and xyz vector-norm gamma
    # onset mostly uses gamma_vector_measured_drive_g
    gamma_drive = lockin_envelope_1d(run["time_s"], run_projection, drive_hz, window_s)
    cal_gamma_drive = lockin_envelope_1d(cal["time_s"], cal_projection, drive_hz, window_s)
    gamma_projected_measured = lockin_envelope_1d(
        run["time_s"],
        run_measured_projection,
        measured_drive_hz,
        window_s,
    )
    cal_gamma_projected_measured = lockin_envelope_1d(
        cal["time_s"],
        cal_measured_projection,
        measured_drive_hz,
        window_s,
    )

    gamma_vector_drive_components, gamma_vector_drive = lockin_envelope_axes(
        run["time_s"],
        run_axes,
        drive_hz,
        window_s,
    )
    cal_gamma_vector_drive_components, cal_gamma_vector_drive = lockin_envelope_axes(
        cal["time_s"],
        cal_axes,
        drive_hz,
        window_s,
    )
    gamma_vector_measured_components, gamma_vector_measured = lockin_envelope_axes(
        run["time_s"],
        run_axes,
        measured_drive_hz,
        window_s,
    )
    cal_gamma_vector_measured_components, cal_gamma_vector_measured = lockin_envelope_axes(
        cal["time_s"],
        cal_axes,
        measured_drive_hz,
        window_s,
    )

    auto_hz = float(global_peak["frequency_hz"])
    gamma_auto = (
        lockin_envelope_1d(run["time_s"], run_projection, auto_hz, window_s)
        if np.isfinite(auto_hz)
        else np.full(len(run["time_s"]), np.nan)
    )
    cal_gamma_auto = (
        lockin_envelope_1d(cal["time_s"], cal_projection, auto_hz, window_s)
        if np.isfinite(auto_hz)
        else np.full(len(cal["time_s"]), np.nan)
    )

    precomputed = run["dataframe"].get("gamma_envelope")
    # this dataframe is written directly to csv by the onset script
    # one row per accelerometer sample
    accel_df = pd.DataFrame(
        {
            "time_s": run["time_s"] + float(accel_time_offset_s),
            "projected_acceleration_g": run_projection,
            "gamma_drive_g": gamma_drive,
            "gamma_auto_g": gamma_auto,
            "projected_measured_drive_acceleration_g": run_measured_projection,
            "gamma_projected_measured_drive_g": gamma_projected_measured,
            "gamma_vector_nominal_drive_g": gamma_vector_drive,
            "gamma_vector_measured_drive_g": gamma_vector_measured,
            "gamma_vector_measured_x_g": gamma_vector_measured_components[:, 0],
            "gamma_vector_measured_y_g": gamma_vector_measured_components[:, 1],
            "gamma_vector_measured_z_g": gamma_vector_measured_components[:, 2],
        }
    )
    if precomputed is not None:
        accel_df["csv_gamma_envelope"] = precomputed.astype(float).to_numpy()

    spectrum_df = pd.DataFrame(
        {
            "frequency_hz": freqs,
            "projected_acceleration_power_g2": spectrum,
            "vector_acceleration_power_g2": vector_spectrum,
        }
    )
    return {
        # keep arrays and dataframes together so downstream code can write csvs and plots
        "time_s": run["time_s"] + float(accel_time_offset_s),
        "calibration_time_s": cal["time_s"],
        "projection_axis_xyz": drive_axis,
        "measured_drive_projection_axis_xyz": measured_drive_axis,
        "projected_acceleration_g": run_projection,
        "projected_measured_drive_acceleration_g": run_measured_projection,
        "gamma_drive_g": gamma_drive,
        "gamma_auto_g": gamma_auto,
        "gamma_projected_measured_drive_g": gamma_projected_measured,
        "gamma_vector_nominal_drive_g": gamma_vector_drive,
        "gamma_vector_measured_drive_g": gamma_vector_measured,
        "gamma_vector_nominal_drive_components_g": gamma_vector_drive_components,
        "gamma_vector_measured_drive_components_g": gamma_vector_measured_components,
        "calibration_gamma_drive_g": cal_gamma_drive,
        "calibration_gamma_auto_g": cal_gamma_auto,
        "calibration_gamma_projected_measured_drive_g": cal_gamma_projected_measured,
        "calibration_gamma_vector_nominal_drive_g": cal_gamma_vector_drive,
        "calibration_gamma_vector_measured_drive_g": cal_gamma_vector_measured,
        "calibration_gamma_vector_nominal_drive_components_g": cal_gamma_vector_drive_components,
        "calibration_gamma_vector_measured_drive_components_g": cal_gamma_vector_measured_components,
        "spectrum_freqs_hz": freqs,
        "spectrum_power_g2": spectrum,
        "vector_spectrum_power_g2": vector_spectrum,
        "global_peak": global_peak,
        "drive_band_peak": drive_peak,
        "double_drive_band_peak": double_peak,
        "vector_global_peak": vector_global_peak,
        "vector_drive_band_peak": vector_drive_peak,
        "vector_double_drive_band_peak": vector_double_peak,
        "accel_metrics": accel_df,
        "accel_spectrum": spectrum_df,
        "calibration_bias_xyz_g": bias,
        "measured_drive_frequency_hz": measured_drive_hz,
        "csv_gamma_envelope": (
            precomputed.astype(float).to_numpy()
            if precomputed is not None
            else None
        ),
    }
