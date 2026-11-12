"""Initial false-positive screening for periodic transit-like signals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from transithunter.bls import fold_time, load_detrended_light_curve


@dataclass(frozen=True)
class EventDepth:
    """Depth measured from one primary or secondary event."""

    event_number: int
    center_time: float
    depth: float
    depth_err: float
    points_in: int
    points_out: int


@dataclass(frozen=True)
class VettingResult:
    """Individual event measurements and their summary statistics."""

    primary_events: list[EventDepth]
    secondary_events: list[EventDepth]
    summary: dict


def _weighted_mean(
    values: np.ndarray,
    errors: np.ndarray,
) -> tuple[float, float]:
    """Return an inverse-variance weighted mean and formal uncertainty."""
    weights = 1.0 / np.square(errors)
    mean = float(np.sum(weights * values) / np.sum(weights))
    error = float(np.sqrt(1.0 / np.sum(weights)))
    return mean, error


def _measure_events(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    centers: np.ndarray,
    event_numbers: np.ndarray,
    duration_days: float,
) -> list[EventDepth]:
    """Measure each event against its nearby out-of-event baseline."""
    events: list[EventDepth] = []

    for event_number, center in zip(event_numbers, centers, strict=True):
        offset = time - center
        absolute_offset = np.abs(offset)
        in_event = absolute_offset <= duration_days / 2.0
        out_event = (
            (absolute_offset >= 0.75 * duration_days)
            & (absolute_offset <= 2.0 * duration_days)
        )

        if np.count_nonzero(in_event) < 2 or np.count_nonzero(out_event) < 4:
            continue

        in_flux, in_error = _weighted_mean(
            flux[in_event], flux_err[in_event]
        )
        out_flux, out_error = _weighted_mean(
            flux[out_event], flux_err[out_event]
        )
        if out_flux <= 0:
            continue

        depth = (out_flux - in_flux) / out_flux
        depth_err = np.sqrt(in_error**2 + out_error**2) / out_flux
        events.append(
            EventDepth(
                event_number=int(event_number),
                center_time=float(center),
                depth=float(depth),
                depth_err=float(depth_err),
                points_in=int(np.count_nonzero(in_event)),
                points_out=int(np.count_nonzero(out_event)),
            )
        )

    return events


def _combine_events(
    events: list[EventDepth],
) -> tuple[float, float]:
    """Combine independent event-depth estimates."""
    if not events:
        raise ValueError("No measurable events are available.")
    depths = np.array([event.depth for event in events])
    errors = np.array([event.depth_err for event in events])
    return _weighted_mean(depths, errors)


def _event_grid(
    time: np.ndarray,
    period_days: float,
    first_center: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct integer event numbers and centers covering the dataset."""
    minimum = int(np.ceil((np.min(time) - first_center) / period_days))
    maximum = int(np.floor((np.max(time) - first_center) / period_days))
    numbers = np.arange(minimum, maximum + 1, dtype=np.int64)
    centers = first_center + numbers * period_days
    return numbers, centers


def vet_transit_signal(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    period_days: float,
    duration_days: float,
    transit_time: float,
    significance_threshold: float = 3.0,
) -> VettingResult:
    """Measure odd/even depths and the phase-0.5 secondary eclipse."""
    if period_days <= 0 or duration_days <= 0:
        raise ValueError("Period and duration must be positive.")
    if duration_days >= period_days / 2.0:
        raise ValueError("Duration is too long for primary/secondary vetting.")
    if significance_threshold <= 0:
        raise ValueError("significance_threshold must be positive.")

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    if len(time) == 0 or len({len(time), len(flux), len(flux_err)}) != 1:
        raise ValueError("Inputs must be non-empty and have equal lengths.")
    if not np.all(np.isfinite(flux_err)) or np.any(flux_err <= 0):
        raise ValueError("Flux uncertainties must be finite and positive.")

    primary_numbers, primary_centers = _event_grid(
        time, period_days, transit_time
    )
    primary_events = _measure_events(
        time,
        flux,
        flux_err,
        primary_centers,
        primary_numbers,
        duration_days,
    )

    secondary_epoch = transit_time + period_days / 2.0
    secondary_numbers, secondary_centers = _event_grid(
        time, period_days, secondary_epoch
    )
    secondary_events = _measure_events(
        time,
        flux,
        flux_err,
        secondary_centers,
        secondary_numbers,
        duration_days,
    )

    odd_events = [
        event for event in primary_events if event.event_number % 2 != 0
    ]
    even_events = [
        event for event in primary_events if event.event_number % 2 == 0
    ]
    if len(odd_events) < 2 or len(even_events) < 2:
        raise ValueError("At least two odd and two even transits are required.")
    if len(secondary_events) < 2:
        raise ValueError("At least two measurable secondary phases are required.")

    odd_depth, odd_error = _combine_events(odd_events)
    even_depth, even_error = _combine_events(even_events)
    secondary_depth, secondary_error = _combine_events(secondary_events)

    odd_even_difference = odd_depth - even_depth
    odd_even_error = float(np.hypot(odd_error, even_error))
    odd_even_z = odd_even_difference / odd_even_error
    secondary_snr = secondary_depth / secondary_error

    flags: list[str] = []
    if abs(odd_even_z) >= significance_threshold:
        flags.append("significant_odd_even_depth_difference")
    if secondary_snr >= significance_threshold:
        flags.append("significant_secondary_eclipse")

    summary = {
        "primary_events_measured": len(primary_events),
        "odd_transits_measured": len(odd_events),
        "even_transits_measured": len(even_events),
        "secondary_events_measured": len(secondary_events),
        "odd_depth_ppm": odd_depth * 1_000_000.0,
        "odd_depth_err_ppm": odd_error * 1_000_000.0,
        "even_depth_ppm": even_depth * 1_000_000.0,
        "even_depth_err_ppm": even_error * 1_000_000.0,
        "odd_even_difference_ppm": odd_even_difference * 1_000_000.0,
        "odd_even_difference_err_ppm": odd_even_error * 1_000_000.0,
        "odd_even_z": float(odd_even_z),
        "secondary_depth_ppm": secondary_depth * 1_000_000.0,
        "secondary_depth_err_ppm": secondary_error * 1_000_000.0,
        "secondary_snr": float(secondary_snr),
        "significance_threshold": significance_threshold,
        "flags": flags,
        "status": (
            "flagged_for_false_positive_review"
            if flags
            else "passed_limited_automated_checks"
        ),
        "formal_uncertainty_notice": (
            "Uncertainties are local inverse-variance estimates and do not "
            "include all correlated-noise or systematic effects."
        ),
    }
    return VettingResult(primary_events, secondary_events, summary)


def _binned_phase(
    phase_hours: np.ndarray,
    flux: np.ndarray,
    bin_minutes: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return median bins for diagnostic visualisation."""
    width_hours = bin_minutes / 60.0
    edges = np.arange(
        float(np.min(phase_hours)),
        float(np.max(phase_hours)) + width_hours,
        width_hours,
    )
    indices = np.digitize(phase_hours, edges) - 1
    centers: list[float] = []
    medians: list[float] = []

    for index in range(len(edges) - 1):
        selected = indices == index
        if np.count_nonzero(selected) >= 2:
            centers.append(float((edges[index] + edges[index + 1]) / 2.0))
            medians.append(float(np.median(flux[selected])))

    return np.asarray(centers), np.asarray(medians)


def _plot_folded_event(
    axis: plt.Axes,
    time: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    epoch: float,
    duration_days: float,
    title: str,
) -> None:
    """Plot one folded event and its median bins."""
    phase_hours = fold_time(time, period_days, epoch) * 24.0
    binned_hours, binned_flux = _binned_phase(phase_hours, flux)
    axis.scatter(
        phase_hours,
        flux,
        s=4,
        color="#64748b",
        alpha=0.25,
        rasterized=True,
    )
    axis.scatter(
        binned_hours,
        binned_flux,
        s=16,
        color="#dc2626",
    )
    axis.axhline(1.0, color="#0f172a", linewidth=0.7, alpha=0.5)
    axis.axvline(0.0, color="#0284c7", linewidth=0.8)
    half_width = max(2.0, 3.0 * duration_days * 24.0)
    axis.set_xlim(-half_width, half_width)
    axis.set_title(title)
    axis.set_xlabel("Hours from event center")
    axis.set_ylabel("Detrended flux")


def save_vetting_plot(
    time: np.ndarray,
    flux: np.ndarray,
    result: VettingResult,
    *,
    period_days: float,
    duration_days: float,
    transit_time: float,
    output_path: Path,
) -> None:
    """Plot primary, secondary, and per-transit depth diagnostics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        constrained_layout=True,
    )

    _plot_folded_event(
        axes[0],
        time,
        flux,
        period_days,
        transit_time,
        duration_days,
        "Primary transit",
    )
    _plot_folded_event(
        axes[1],
        time,
        flux,
        period_days,
        transit_time + period_days / 2.0,
        duration_days,
        "Phase-0.5 secondary test",
    )

    for events, color, label in (
        (
            [e for e in result.primary_events if e.event_number % 2 != 0],
            "#0284c7",
            "Odd transits",
        ),
        (
            [e for e in result.primary_events if e.event_number % 2 == 0],
            "#f97316",
            "Even transits",
        ),
    ):
        axes[2].errorbar(
            [e.event_number for e in events],
            [e.depth * 1_000_000.0 for e in events],
            yerr=[e.depth_err * 1_000_000.0 for e in events],
            fmt="o",
            markersize=3,
            alpha=0.65,
            color=color,
            label=label,
        )

    axes[2].axhline(0.0, color="#0f172a", linewidth=0.7)
    axes[2].set_title("Individual primary-transit depths")
    axes[2].set_xlabel("Transit epoch number")
    axes[2].set_ylabel("Depth (ppm)")
    axes[2].legend(loc="best")

    figure.suptitle(
        "Automated candidate screening — not planetary confirmation",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_event_depths(
    result: VettingResult,
    output_path: Path,
) -> None:
    """Save all primary and secondary event-depth measurements."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "event_type",
                "event_number",
                "center_time",
                "depth",
                "depth_err",
                "points_in",
                "points_out",
            ]
        )
        for event_type, events in (
            ("primary", result.primary_events),
            ("secondary", result.secondary_events),
        ):
            for event in events:
                writer.writerow(
                    [
                        event_type,
                        event.event_number,
                        f"{event.center_time:.17g}",
                        f"{event.depth:.17g}",
                        f"{event.depth_err:.17g}",
                        event.points_in,
                        event.points_out,
                    ]
                )


def vet_file(
    csv_path: Path,
    candidate_path: Path,
    *,
    output_dir: Path,
    significance_threshold: float,
) -> tuple[Path, Path, Path, dict]:
    """Run limited automated vetting using a previous blind BLS result."""
    time, flux, flux_err = load_detrended_light_curve(csv_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    signal = candidate["best_signal"]
    period_days = float(signal["period_days"])
    duration_days = float(signal["duration_days"])
    transit_time = float(signal["transit_time"])

    result = vet_transit_signal(
        time,
        flux,
        flux_err,
        period_days=period_days,
        duration_days=duration_days,
        transit_time=transit_time,
        significance_threshold=significance_threshold,
    )

    report_path = output_dir / f"{csv_path.stem}_vetting_report.json"
    events_path = output_dir / f"{csv_path.stem}_event_depths.csv"
    plot_path = output_dir / f"{csv_path.stem}_vetting.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_event_depths(result, events_path)
    save_vetting_plot(
        time,
        flux,
        result,
        period_days=period_days,
        duration_days=duration_days,
        transit_time=transit_time,
        output_path=plot_path,
    )

    document = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path),
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "source_candidate": str(candidate_path),
        "candidate_sha256": hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        "candidate_status": "automated_screening_complete",
        "catalogue_crossmatch_performed": False,
        "independent_followup_performed": False,
        "vetting": result.summary,
        "scientific_notice": (
            "Passing these limited checks does not confirm a planet. "
            "Aliases, transit shape, stellar variability, data quality, "
            "catalogue matches, and independent follow-up still require review."
        ),
    }
    report_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path, events_path, plot_path, result.summary


def main() -> None:
    """Command-line entry point for initial transit-signal vetting."""
    parser = argparse.ArgumentParser(
        description="Check odd/even depths and a phase-0.5 secondary eclipse."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("candidate_json", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/vetting")
    )
    parser.add_argument("--significance-threshold", type=float, default=3.0)
    args = parser.parse_args()

    try:
        report_path, events_path, plot_path, summary = vet_file(
            args.csv_path,
            args.candidate_json,
            output_dir=args.output_dir,
            significance_threshold=args.significance_threshold,
        )
    except Exception as error:
        raise SystemExit(f"Vetting failed: {error}") from error

    print(f"Vetting report: {report_path}")
    print(f"Event depths:   {events_path}")
    print(f"Vetting plot:   {plot_path}")
    print(f"Primary events: {summary['primary_events_measured']}")
    print(f"Odd depth ppm:  {summary['odd_depth_ppm']:.2f}")
    print(f"Even depth ppm: {summary['even_depth_ppm']:.2f}")
    print(f"Odd/even z:     {summary['odd_even_z']:.3f}")
    print(f"Secondary ppm:  {summary['secondary_depth_ppm']:.2f}")
    print(f"Secondary SNR:  {summary['secondary_snr']:.3f}")
    print(f"Flags:          {summary['flags']}")
    print(f"Status:         {summary['status']}")
    print("This screening does not confirm a planet.")


if __name__ == "__main__":
    main()