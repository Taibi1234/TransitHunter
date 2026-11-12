"""Transit-aware rolling-median detrending for normalized light curves."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class DetrendReport:
    """Summary of the detrending operation."""

    rows_input: int
    rows_output: int
    rows_protected: int
    segment_count: int
    robust_scatter: float
    window_days: float
    gap_days: float
    protection_sigma: float


@dataclass(frozen=True)
class DetrendedLightCurve:
    """Detrended measurements and their diagnostic information."""

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    trend: np.ndarray
    protected: np.ndarray
    segment_id: np.ndarray
    report: DetrendReport


def load_processed_light_curve(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time, normalized flux, and normalized flux uncertainty."""
    table = np.genfromtxt(csv_path, delimiter=",", names=True)
    required = {"time", "flux", "flux_err"}
    available = set(table.dtype.names or ())
    missing = required - available
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    return (
        np.atleast_1d(np.asarray(table["time"], dtype=float)),
        np.atleast_1d(np.asarray(table["flux"], dtype=float)),
        np.atleast_1d(np.asarray(table["flux_err"], dtype=float)),
    )


def _segment_indices(time: np.ndarray, gap_days: float) -> list[np.ndarray]:
    """Split array indices wherever the time gap exceeds the threshold."""
    boundaries = np.flatnonzero(np.diff(time) > gap_days) + 1
    return list(np.split(np.arange(len(time)), boundaries))


def _rolling_median(
    time: np.ndarray,
    flux: np.ndarray,
    fit_mask: np.ndarray,
    window_days: float,
    minimum_points: int,
) -> np.ndarray:
    """Calculate a rolling median using only points approved for fitting."""
    trend = np.empty(len(time), dtype=float)
    half_window = window_days / 2.0
    fallback_flux = flux[fit_mask]

    if len(fallback_flux) == 0:
        raise ValueError("No points remain available for trend estimation.")

    for index, current_time in enumerate(time):
        left = int(np.searchsorted(time, current_time - half_window, side="left"))
        right = int(
            np.searchsorted(time, current_time + half_window, side="right")
        )
        local_mask = fit_mask[left:right]
        local_flux = flux[left:right][local_mask]

        if len(local_flux) >= minimum_points:
            trend[index] = np.median(local_flux)
        else:
            trend[index] = np.median(fallback_flux)

    return trend


def _estimate_trend(
    time: np.ndarray,
    flux: np.ndarray,
    fit_mask: np.ndarray,
    segments: list[np.ndarray],
    window_days: float,
    minimum_points: int,
) -> np.ndarray:
    """Estimate the baseline independently inside every observing segment."""
    trend = np.empty(len(time), dtype=float)
    for indices in segments:
        trend[indices] = _rolling_median(
            time[indices],
            flux[indices],
            fit_mask[indices],
            window_days,
            minimum_points,
        )
    return trend


def detrend_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    window_days: float = 1.0,
    gap_days: float = 0.5,
    protection_sigma: float = 3.0,
    minimum_points: int = 15,
    maximum_protected_fraction: float = 0.20,
) -> DetrendedLightCurve:
    """Remove slow baseline variation while protecting negative deviations.

    Protection is determined once from an initial robust trend. Repeating this
    process while expanding the mask can shift the baseline and attenuate real
    transit signals, so the final trend uses one fixed protection mask.

    Protected points are excluded only from baseline fitting. They remain in
    the returned light curve and are available to the subsequent BLS search.
    """
    if window_days <= 0 or gap_days <= 0:
        raise ValueError("window_days and gap_days must be positive.")
    if protection_sigma <= 0:
        raise ValueError("protection_sigma must be positive.")
    if minimum_points < 1:
        raise ValueError("minimum_points must be at least one.")

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    if len(time) == 0 or len({len(time), len(flux), len(flux_err)}) != 1:
        raise ValueError("Inputs must be non-empty and have equal lengths.")
    if any(array.ndim != 1 for array in (time, flux, flux_err)):
        raise ValueError("Inputs must be one-dimensional.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(flux)):
        raise ValueError("Detrending requires finite time and flux values.")
    if not np.all(np.isfinite(flux_err)) or np.any(flux_err <= 0):
        raise ValueError("Flux uncertainties must be finite and positive.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Time values must be strictly increasing.")

    segments = _segment_indices(time, gap_days)
    segment_id = np.empty(len(time), dtype=np.int64)
    for number, indices in enumerate(segments):
        segment_id[indices] = number

    all_points = np.ones(len(time), dtype=bool)
    initial_trend = _estimate_trend(
        time,
        flux,
        all_points,
        segments,
        window_days,
        minimum_points,
    )

    residual = flux - initial_trend
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    robust_scatter = max(
        1.4826 * mad,
        float(np.median(flux_err)),
        np.finfo(float).eps,
    )
    protected = residual < -(protection_sigma * robust_scatter)

    protected_fraction = float(np.mean(protected))
    if protected_fraction > maximum_protected_fraction:
        raise ValueError(
            "Too many points were protected from trend estimation. "
            "Inspect the light curve and detrending parameters."
        )

    # Use a fixed mask to avoid progressively attenuating negative signals.
    trend = _estimate_trend(
        time,
        flux,
        ~protected,
        segments,
        window_days,
        minimum_points,
    )

    if not np.all(np.isfinite(trend)) or np.any(trend <= 0):
        raise ValueError("The estimated trend must be finite and positive.")

    detrended_flux = flux / trend
    detrended_flux_err = flux_err / trend
    report = DetrendReport(
        rows_input=len(time),
        rows_output=len(time),
        rows_protected=int(np.count_nonzero(protected)),
        segment_count=len(segments),
        robust_scatter=robust_scatter,
        window_days=window_days,
        gap_days=gap_days,
        protection_sigma=protection_sigma,
    )
    return DetrendedLightCurve(
        time=time,
        flux=detrended_flux,
        flux_err=detrended_flux_err,
        trend=trend,
        protected=protected,
        segment_id=segment_id,
        report=report,
    )


def save_detrended_light_curve(
    detrended: DetrendedLightCurve,
    output_path: Path,
) -> None:
    """Save detrended data together with trend and protection diagnostics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack(
        (
            detrended.time,
            detrended.flux,
            detrended.flux_err,
            detrended.trend,
            detrended.protected.astype(np.int64),
            detrended.segment_id,
        )
    )
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header="time,flux,flux_err,trend,protected,segment_id",
        comments="",
        fmt=["%.17g", "%.17g", "%.17g", "%.17g", "%d", "%d"],
    )


def save_detrending_plot(
    input_time: np.ndarray,
    input_flux: np.ndarray,
    detrended: DetrendedLightCurve,
    output_path: Path,
) -> None:
    """Plot the fitted baseline and the resulting detrended light curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].scatter(
        input_time,
        input_flux,
        s=3,
        color="#64748b",
        alpha=0.45,
        rasterized=True,
        label="Normalized input",
    )
    axes[0].plot(
        detrended.time,
        detrended.trend,
        color="#dc2626",
        linewidth=1.2,
        label="Estimated baseline",
    )
    axes[0].scatter(
        input_time[detrended.protected],
        input_flux[detrended.protected],
        s=9,
        color="#0284c7",
        label="Protected negative deviations",
        rasterized=True,
    )
    axes[0].set_ylabel("Normalized flux")
    axes[0].set_title("Transit-aware baseline estimation")
    axes[0].legend(loc="best")

    axes[1].scatter(
        detrended.time,
        detrended.flux,
        s=3,
        color="#0f172a",
        alpha=0.6,
        rasterized=True,
    )
    axes[1].axhline(1.0, color="#dc2626", linewidth=0.8)
    axes[1].set_title("Detrended light curve — all measurements retained")
    axes[1].set_xlabel("Time (mission-relative days)")
    axes[1].set_ylabel("Detrended flux")

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def detrend_file(
    csv_path: Path,
    *,
    output_dir: Path,
    plot_dir: Path,
    window_days: float,
    gap_days: float,
    protection_sigma: float,
) -> tuple[Path, Path, Path, DetrendReport]:
    """Detrend one processed CSV and write all output artifacts."""
    time, flux, flux_err = load_processed_light_curve(csv_path)
    detrended = detrend_light_curve(
        time,
        flux,
        flux_err,
        window_days=window_days,
        gap_days=gap_days,
        protection_sigma=protection_sigma,
    )

    output_path = output_dir / f"{csv_path.stem}_detrended.csv"
    report_path = output_dir / f"{csv_path.stem}_detrended_report.json"
    plot_path = plot_dir / f"{csv_path.stem}_detrending.png"

    save_detrended_light_curve(detrended, output_path)
    save_detrending_plot(time, flux, detrended, plot_path)

    document = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path),
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "method": (
            "gap-aware rolling median with fixed negative-deviation protection"
        ),
        "protection_iterations": 1,
        "known_ephemeris_supplied": False,
        "rows_removed": False,
        "report": asdict(detrended.report),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, report_path, plot_path, detrended.report


def main() -> None:
    """Command-line entry point for transit-aware detrending."""
    parser = argparse.ArgumentParser(
        description="Detrend a normalized TransitHunter light curve."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=Path("outputs/diagnostics")
    )
    parser.add_argument("--window-days", type=float, default=1.0)
    parser.add_argument("--gap-days", type=float, default=0.5)
    parser.add_argument("--protection-sigma", type=float, default=3.0)
    args = parser.parse_args()

    try:
        output_path, report_path, plot_path, report = detrend_file(
            args.csv_path,
            output_dir=args.output_dir,
            plot_dir=args.plot_dir,
            window_days=args.window_days,
            gap_days=args.gap_days,
            protection_sigma=args.protection_sigma,
        )
    except Exception as error:
        raise SystemExit(f"Detrending failed: {error}") from error

    print(f"Detrended CSV: {output_path}")
    print(f"Report:        {report_path}")
    print(f"Plot:          {plot_path}")
    print(f"Rows:          {report.rows_output}")
    print(f"Protected:     {report.rows_protected}")
    print(f"Segments:      {report.segment_count}")
    print(f"Robust scatter: {report.robust_scatter:.8f}")


if __name__ == "__main__":
    main()