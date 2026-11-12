"""Conservative, transit-safe preprocessing for archived light curves."""

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
class PreprocessReport:
    """Auditable counts describing every preprocessing rejection stage."""

    rows_input: int
    rows_bad_quality: int
    rows_nonfinite: int
    rows_nonpositive_error: int
    rows_positive_outliers: int
    rows_output: int
    median_input_flux: float
    robust_sigma_normalized: float
    maximum_gap_days: float


@dataclass(frozen=True)
class ProcessedLightCurve:
    """Normalized measurements and their processing report."""

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    report: PreprocessReport


def load_raw_light_curve(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the four columns produced by the acquisition stage."""
    table = np.genfromtxt(csv_path, delimiter=",", names=True)
    required = {"time", "flux", "flux_err", "quality"}
    available = set(table.dtype.names or ())
    missing = required - available
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    return (
        np.atleast_1d(np.asarray(table["time"], dtype=float)),
        np.atleast_1d(np.asarray(table["flux"], dtype=float)),
        np.atleast_1d(np.asarray(table["flux_err"], dtype=float)),
        np.atleast_1d(np.asarray(table["quality"], dtype=np.int64)),
    )


def preprocess_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray,
    *,
    positive_outlier_sigma: float = 8.0,
) -> ProcessedLightCurve:
    """Clean and normalize measurements without suppressing transits.

    Only positive flux outliers are clipped. Negative deviations are retained
    because they may represent transits, eclipses, or other real variability.
    No smoothing, interpolation, or detrending is performed here.
    """
    if positive_outlier_sigma <= 0:
        raise ValueError("positive_outlier_sigma must be positive.")

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    quality = np.asarray(quality, dtype=np.int64)

    lengths = {len(time), len(flux), len(flux_err), len(quality)}
    if len(time) == 0:
        raise ValueError("The input light curve is empty.")
    if len(lengths) != 1:
        raise ValueError("Input light-curve columns have different lengths.")
    if any(array.ndim != 1 for array in (time, flux, flux_err, quality)):
        raise ValueError("Input light-curve columns must be one-dimensional.")

    rows_input = len(time)

    good_quality = quality == 0
    rows_bad_quality = int(np.count_nonzero(~good_quality))
    time, flux, flux_err = (
        time[good_quality],
        flux[good_quality],
        flux_err[good_quality],
    )

    finite = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    rows_nonfinite = int(np.count_nonzero(~finite))
    time, flux, flux_err = time[finite], flux[finite], flux_err[finite]

    valid_error = flux_err > 0
    rows_nonpositive_error = int(np.count_nonzero(~valid_error))
    time, flux, flux_err = (
        time[valid_error],
        flux[valid_error],
        flux_err[valid_error],
    )

    if len(time) == 0:
        raise ValueError("No valid measurements remain after basic filtering.")

    order = np.argsort(time, kind="mergesort")
    time, flux, flux_err = time[order], flux[order], flux_err[order]

    median_flux = float(np.median(flux))
    if not np.isfinite(median_flux) or median_flux <= 0:
        raise ValueError("Median flux must be finite and positive.")

    flux = flux / median_flux
    flux_err = flux_err / median_flux

    center = float(np.median(flux))
    median_deviation = float(np.median(np.abs(flux - center)))
    mad_sigma = 1.4826 * median_deviation
    median_error = float(np.median(flux_err))
    robust_sigma = max(mad_sigma, median_error, np.finfo(float).eps)

    upper_limit = center + positive_outlier_sigma * robust_sigma
    keep = flux <= upper_limit
    rows_positive_outliers = int(np.count_nonzero(~keep))
    time, flux, flux_err = time[keep], flux[keep], flux_err[keep]

    if len(time) < 2:
        raise ValueError("Fewer than two measurements remain after processing.")

    maximum_gap = float(np.max(np.diff(time)))
    report = PreprocessReport(
        rows_input=rows_input,
        rows_bad_quality=rows_bad_quality,
        rows_nonfinite=rows_nonfinite,
        rows_nonpositive_error=rows_nonpositive_error,
        rows_positive_outliers=rows_positive_outliers,
        rows_output=len(time),
        median_input_flux=median_flux,
        robust_sigma_normalized=robust_sigma,
        maximum_gap_days=maximum_gap,
    )
    return ProcessedLightCurve(time, flux, flux_err, report)


def save_processed_light_curve(
    processed: ProcessedLightCurve,
    output_path: Path,
) -> None:
    """Save processed measurements in a compact, documented CSV format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack(
        (processed.time, processed.flux, processed.flux_err)
    )
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header="time,flux,flux_err",
        comments="",
        fmt="%.17g",
    )


def save_diagnostic_plot(
    raw_time: np.ndarray,
    raw_flux: np.ndarray,
    raw_quality: np.ndarray,
    processed: ProcessedLightCurve,
    output_path: Path,
) -> None:
    """Plot the finite raw measurements and the processed light curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.isfinite(raw_time) & np.isfinite(raw_flux)
    good = finite & (raw_quality == 0)
    flagged = finite & (raw_quality != 0)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].scatter(
        raw_time[good],
        raw_flux[good],
        s=3,
        color="#64748b",
        alpha=0.55,
        label="Quality = 0",
        rasterized=True,
    )
    axes[0].scatter(
        raw_time[flagged],
        raw_flux[flagged],
        s=5,
        color="#f97316",
        alpha=0.65,
        label="Quality flagged",
        rasterized=True,
    )
    axes[0].set_ylabel("Raw flux")
    axes[0].set_title("Archived light curve before preprocessing")
    axes[0].legend(loc="best")

    axes[1].scatter(
        processed.time,
        processed.flux,
        s=3,
        color="#0f172a",
        alpha=0.65,
        rasterized=True,
    )
    axes[1].axhline(1.0, color="#dc2626", linewidth=0.8, alpha=0.7)
    axes[1].set_title("Quality-filtered and median-normalized light curve")
    axes[1].set_xlabel("Time (mission-relative days)")
    axes[1].set_ylabel("Normalized flux")

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def process_file(
    csv_path: Path,
    *,
    output_dir: Path,
    plot_dir: Path,
    positive_outlier_sigma: float,
) -> tuple[Path, Path, Path, PreprocessReport]:
    """Process one acquired CSV and write data, report, and diagnostic plot."""
    time, flux, flux_err, quality = load_raw_light_curve(csv_path)
    processed = preprocess_light_curve(
        time,
        flux,
        flux_err,
        quality,
        positive_outlier_sigma=positive_outlier_sigma,
    )

    output_path = output_dir / f"{csv_path.stem}_clean.csv"
    report_path = output_dir / f"{csv_path.stem}_clean_report.json"
    plot_path = plot_dir / f"{csv_path.stem}_overview.png"

    save_processed_light_curve(processed, output_path)
    save_diagnostic_plot(time, flux, quality, processed, plot_path)

    report_document = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path),
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "operations": [
            "retain quality == 0",
            "remove non-finite time, flux, and flux_err",
            "require flux_err > 0",
            "sort by time",
            "divide flux and flux_err by median flux",
            "remove extreme positive outliers only",
        ],
        "positive_outlier_sigma": positive_outlier_sigma,
        "negative_outliers_removed": False,
        "smoothing_applied": False,
        "detrending_applied": False,
        "report": asdict(processed.report),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, report_path, plot_path, processed.report


def main() -> None:
    """Command-line entry point for conservative light-curve preprocessing."""
    parser = argparse.ArgumentParser(
        description="Clean and normalize an acquired TransitHunter CSV."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=Path("outputs/diagnostics")
    )
    parser.add_argument("--positive-outlier-sigma", type=float, default=8.0)
    args = parser.parse_args()

    try:
        output_path, report_path, plot_path, report = process_file(
            args.csv_path,
            output_dir=args.output_dir,
            plot_dir=args.plot_dir,
            positive_outlier_sigma=args.positive_outlier_sigma,
        )
    except Exception as error:
        raise SystemExit(f"Preprocessing failed: {error}") from error

    print(f"Processed CSV: {output_path}")
    print(f"Report:        {report_path}")
    print(f"Plot:          {plot_path}")
    print(f"Input rows:    {report.rows_input}")
    print(f"Output rows:   {report.rows_output}")
    print(f"Bad quality:   {report.rows_bad_quality}")
    print(f"Non-finite:    {report.rows_nonfinite}")
    print(f"Invalid error: {report.rows_nonpositive_error}")
    print(f"Positive outliers removed: {report.rows_positive_outliers}")
    print(f"Maximum gap (days): {report.maximum_gap_days:.6f}")


if __name__ == "__main__":
    main()