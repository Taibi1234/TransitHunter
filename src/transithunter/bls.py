"""Blind Box Least Squares searches for periodic transit-like signals."""

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
from astropy.timeseries import BoxLeastSquares


@dataclass(frozen=True)
class TransitSignal:
    """The highest-power signal returned by a BLS search."""

    period_days: float
    duration_days: float
    transit_time: float
    depth: float
    depth_err: float
    depth_snr: float
    power: float


@dataclass(frozen=True)
class BLSearchResult:
    """Best signal and its complete periodogram."""

    signal: TransitSignal
    period_grid: np.ndarray
    power: np.ndarray
    durations: np.ndarray


def load_detrended_light_curve(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the detrended time, flux, and flux uncertainty columns."""
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


def run_bls_search(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    durations: np.ndarray,
    minimum_period: float = 0.5,
    maximum_period: float = 10.0,
    frequency_factor: float = 1.0,
    minimum_n_transit: int = 3,
) -> BLSearchResult:
    """Search period, duration, and epoch without an external ephemeris."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    durations = np.asarray(durations, dtype=float)

    if len(time) == 0 or len({len(time), len(flux), len(flux_err)}) != 1:
        raise ValueError("Inputs must be non-empty and have equal lengths.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(flux)):
        raise ValueError("BLS requires finite time and flux values.")
    if not np.all(np.isfinite(flux_err)) or np.any(flux_err <= 0):
        raise ValueError("Flux uncertainties must be finite and positive.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Time values must be strictly increasing.")
    if minimum_period <= 0 or maximum_period <= minimum_period:
        raise ValueError("The period limits are invalid.")
    if frequency_factor <= 0:
        raise ValueError("frequency_factor must be positive.")
    if minimum_n_transit < 2:
        raise ValueError("At least two transits must be required.")
    if durations.ndim != 1 or len(durations) == 0:
        raise ValueError("At least one trial duration is required.")
    if np.any(~np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("Trial durations must be finite and positive.")
    if np.any(durations >= minimum_period):
        raise ValueError("Every duration must be shorter than minimum_period.")

    model = BoxLeastSquares(time, flux, dy=flux_err)
    periodogram = model.autopower(
        durations,
        objective="likelihood",
        minimum_period=minimum_period,
        maximum_period=maximum_period,
        frequency_factor=frequency_factor,
        minimum_n_transit=minimum_n_transit,
    )

    best_index = int(np.nanargmax(periodogram.power))
    signal = TransitSignal(
        period_days=float(periodogram.period[best_index]),
        duration_days=float(periodogram.duration[best_index]),
        transit_time=float(periodogram.transit_time[best_index]),
        depth=float(periodogram.depth[best_index]),
        depth_err=float(periodogram.depth_err[best_index]),
        depth_snr=float(periodogram.depth_snr[best_index]),
        power=float(periodogram.power[best_index]),
    )
    if signal.depth <= 0:
        raise ValueError("The strongest BLS solution has a non-positive depth.")

    return BLSearchResult(
        signal=signal,
        period_grid=np.asarray(periodogram.period, dtype=float),
        power=np.asarray(periodogram.power, dtype=float),
        durations=durations,
    )


def fold_time(
    time: np.ndarray,
    period_days: float,
    transit_time: float,
) -> np.ndarray:
    """Fold timestamps around mid-transit, returning phase in days."""
    if period_days <= 0:
        raise ValueError("period_days must be positive.")
    return (
        (np.asarray(time, dtype=float) - transit_time + period_days / 2.0)
        % period_days
    ) - period_days / 2.0


def _binned_phase(
    phase_hours: np.ndarray,
    flux: np.ndarray,
    bin_minutes: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate median flux bins for visualisation only."""
    bin_width_hours = bin_minutes / 60.0
    edges = np.arange(
        float(np.min(phase_hours)),
        float(np.max(phase_hours)) + bin_width_hours,
        bin_width_hours,
    )
    bin_index = np.digitize(phase_hours, edges) - 1
    centers: list[float] = []
    medians: list[float] = []

    for index in range(len(edges) - 1):
        selected = bin_index == index
        if np.count_nonzero(selected) < 2:
            continue
        centers.append(float((edges[index] + edges[index + 1]) / 2.0))
        medians.append(float(np.median(flux[selected])))

    return np.asarray(centers), np.asarray(medians)


def save_search_plot(
    time: np.ndarray,
    flux: np.ndarray,
    result: BLSearchResult,
    output_path: Path,
) -> None:
    """Save the periodogram and full/zoomed phase-folded diagnostics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal = result.signal
    phase_days = fold_time(time, signal.period_days, signal.transit_time)
    phase_hours = phase_days * 24.0
    binned_hours, binned_flux = _binned_phase(phase_hours, flux)

    figure = plt.figure(figsize=(13, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    period_axis = figure.add_subplot(grid[0, :])
    full_axis = figure.add_subplot(grid[1, 0])
    zoom_axis = figure.add_subplot(grid[1, 1])

    period_axis.plot(
        result.period_grid,
        result.power,
        color="#0f172a",
        linewidth=0.7,
    )
    period_axis.axvline(
        signal.period_days,
        color="#dc2626",
        linewidth=1.1,
        label=f"Best period: {signal.period_days:.8f} d",
    )
    period_axis.set_title("Blind BLS periodogram")
    period_axis.set_xlabel("Period (days)")
    period_axis.set_ylabel("BLS likelihood power")
    period_axis.legend(loc="best")

    for axis in (full_axis, zoom_axis):
        axis.scatter(
            phase_hours,
            flux,
            s=4,
            color="#64748b",
            alpha=0.3,
            rasterized=True,
            label="Measurements",
        )
        axis.scatter(
            binned_hours,
            binned_flux,
            s=15,
            color="#dc2626",
            label="5-minute median bins",
        )
        axis.axhline(1.0, color="#0f172a", linewidth=0.7, alpha=0.5)
        axis.axvline(0.0, color="#0284c7", linewidth=0.8, alpha=0.7)
        axis.set_xlabel("Hours from mid-transit")
        axis.set_ylabel("Detrended flux")

    full_axis.set_title("Full orbital phase")
    full_axis.set_xlim(-signal.period_days * 12.0, signal.period_days * 12.0)
    zoom_half_width = max(2.0, 3.0 * signal.duration_days * 24.0)
    zoom_axis.set_xlim(-zoom_half_width, zoom_half_width)
    zoom_axis.set_title("Transit-phase zoom")
    zoom_axis.legend(loc="best", fontsize=8)

    figure.suptitle(
        "Unvetted BLS transit signal — not a confirmed planet",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def search_file(
    csv_path: Path,
    *,
    output_dir: Path,
    durations: np.ndarray,
    minimum_period: float,
    maximum_period: float,
    frequency_factor: float,
) -> tuple[Path, Path, Path, TransitSignal]:
    """Run BLS on one detrended CSV and save all search products."""
    time, flux, flux_err = load_detrended_light_curve(csv_path)
    result = run_bls_search(
        time,
        flux,
        flux_err,
        durations=durations,
        minimum_period=minimum_period,
        maximum_period=maximum_period,
        frequency_factor=frequency_factor,
    )

    candidate_path = output_dir / f"{csv_path.stem}_bls_candidate.json"
    periodogram_path = output_dir / f"{csv_path.stem}_bls_periodogram.csv"
    plot_path = output_dir / f"{csv_path.stem}_bls_diagnostic.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    periodogram_table = np.column_stack((result.period_grid, result.power))
    np.savetxt(
        periodogram_path,
        periodogram_table,
        delimiter=",",
        header="period_days,power",
        comments="",
        fmt="%.17g",
    )
    save_search_plot(time, flux, result, plot_path)

    document = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path),
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "method": "Astropy BoxLeastSquares.autopower",
        "objective": "likelihood",
        "known_ephemeris_supplied": False,
        "catalogue_crossmatch_performed": False,
        "candidate_status": "unvetted",
        "search": {
            "minimum_period_days": minimum_period,
            "maximum_period_days": maximum_period,
            "durations_days": result.durations.tolist(),
            "frequency_factor": frequency_factor,
            "minimum_n_transit": 3,
            "period_grid_points": len(result.period_grid),
        },
        "best_signal": {
            **asdict(result.signal),
            "duration_hours": result.signal.duration_days * 24.0,
            "depth_ppm": result.signal.depth * 1_000_000.0,
            "depth_err_ppm": result.signal.depth_err * 1_000_000.0,
        },
        "scientific_notice": (
            "This is an unvetted periodic transit-like signal. It is not a "
            "confirmed planet and requires alias, odd/even, secondary-eclipse, "
            "shape, variability, quality, and catalogue checks."
        ),
    }
    candidate_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return candidate_path, periodogram_path, plot_path, result.signal


def _parse_durations(value: str) -> np.ndarray:
    """Parse a comma-separated duration grid supplied at the command line."""
    try:
        durations = np.array(
            [float(item.strip()) for item in value.split(",")], dtype=float
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Durations must be comma-separated numbers."
        ) from error
    if len(durations) == 0:
        raise argparse.ArgumentTypeError("At least one duration is required.")
    return durations


def main() -> None:
    """Command-line entry point for a blind BLS search."""
    parser = argparse.ArgumentParser(
        description="Search a detrended light curve for periodic transit signals."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/search")
    )
    parser.add_argument("--minimum-period", type=float, default=0.5)
    parser.add_argument("--maximum-period", type=float, default=10.0)
    parser.add_argument(
        "--durations",
        type=_parse_durations,
        default=_parse_durations("0.04,0.06,0.08,0.12"),
    )
    parser.add_argument("--frequency-factor", type=float, default=1.0)
    args = parser.parse_args()

    print("Running blind BLS search; no published ephemeris is being used...")
    try:
        candidate_path, periodogram_path, plot_path, signal = search_file(
            args.csv_path,
            output_dir=args.output_dir,
            durations=args.durations,
            minimum_period=args.minimum_period,
            maximum_period=args.maximum_period,
            frequency_factor=args.frequency_factor,
        )
    except Exception as error:
        raise SystemExit(f"BLS search failed: {error}") from error

    print(f"Candidate report: {candidate_path}")
    print(f"Periodogram:      {periodogram_path}")
    print(f"Diagnostic plot:  {plot_path}")
    print(f"Period (days):    {signal.period_days:.10f}")
    print(f"Duration (hours): {signal.duration_days * 24.0:.4f}")
    print(f"Transit epoch:    {signal.transit_time:.10f}")
    print(f"Depth (ppm):      {signal.depth * 1_000_000.0:.2f}")
    print(f"Depth SNR:        {signal.depth_snr:.3f}")
    print("Status:           unvetted signal, not a confirmed planet")


if __name__ == "__main__":
    main()