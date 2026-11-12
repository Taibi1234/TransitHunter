"""Controlled BLS injection/recovery experiments on real light curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from transithunter.bls import (
    TransitSignal,
    fold_time,
    load_detrended_light_curve,
    run_bls_search,
)


@dataclass(frozen=True)
class InjectionTrial:
    """Inputs and outcome for one synthetic injection."""

    trial: int
    injected_period_days: float
    injected_epoch: float
    injected_duration_days: float
    injected_depth_ppm: float
    recovered_period_days: float
    recovered_epoch: float
    recovered_depth_ppm: float
    recovered_depth_snr: float
    recovered_power: float
    period_ratio: float
    epoch_error_days: float
    power_above_control: bool
    classification: str
    recovered: bool


def inject_box_signal(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    epoch: float,
    duration_days: float,
    depth: float,
) -> np.ndarray:
    """Return a copy of flux with a periodic box-shaped decrement."""
    if period_days <= 0 or duration_days <= 0 or depth < 0:
        raise ValueError("Period and duration must be positive; depth cannot be negative.")
    phase = fold_time(time, period_days, epoch)
    in_transit = np.abs(phase) <= duration_days / 2.0
    injected = np.array(flux, dtype=float, copy=True)
    injected[in_transit] -= depth
    return injected


def evaluate_recovery(
    signal: TransitSignal,
    *,
    injected_period: float,
    injected_epoch: float,
    injected_duration: float,
    control_power: float,
    period_tolerance: float = 0.005,
    minimum_depth_snr: float = 5.0,
) -> dict[str, float | bool | str]:
    """Score a search result without exposing injection values to the search."""
    ratio = signal.period_days / injected_period
    epoch_error = abs(
        float(fold_time(np.array([signal.transit_time]), injected_period, injected_epoch)[0])
    )

    same_period = abs(ratio - 1.0) <= period_tolerance
    double_period = abs(ratio - 2.0) <= 2.0 * period_tolerance
    half_period = abs(ratio - 0.5) <= 0.5 * period_tolerance
    epoch_matches = epoch_error <= injected_duration
    power_above_control = signal.power > control_power
    snr_passes = signal.depth_snr >= minimum_depth_snr

    recovered = (
        same_period and epoch_matches and power_above_control and snr_passes
    )
    if recovered:
        classification = "recovered"
    elif double_period:
        classification = "double_period_alias"
    elif half_period:
        classification = "half_period_alias"
    elif same_period:
        classification = "period_matched_but_failed_checks"
    else:
        classification = "not_recovered"

    return {
        "period_ratio": ratio,
        "epoch_error_days": epoch_error,
        "power_above_control": power_above_control,
        "classification": classification,
        "recovered": recovered,
    }


def run_injection_recovery(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    known_period: float,
    known_epoch: float,
    known_duration: float,
    periods: np.ndarray,
    depths_ppm: np.ndarray,
    injected_duration: float,
    search_durations: np.ndarray,
    maximum_period: float,
) -> tuple[list[InjectionTrial], TransitSignal, int]:
    """Mask the strongest known signal, then run a zero control and injections."""
    known_phase = fold_time(time, known_period, known_epoch)
    retain = np.abs(known_phase) > 0.75 * known_duration
    masked_time = time[retain]
    masked_flux = flux[retain]
    masked_flux_err = flux_err[retain]
    rows_masked = int(np.count_nonzero(~retain))

    if len(masked_time) < 100:
        raise ValueError("Too few measurements remain after known-signal masking.")

    control = run_bls_search(
        masked_time,
        masked_flux,
        masked_flux_err,
        durations=search_durations,
        minimum_period=0.5,
        maximum_period=maximum_period,
    ).signal

    trials: list[InjectionTrial] = []
    trial_number = 0
    for period in periods:
        for depth_ppm in depths_ppm:
            trial_number += 1
            epoch = float(
                np.min(masked_time)
                + (0.23 + 0.11 * ((trial_number - 1) % 5)) * period
            )
            injected_flux = inject_box_signal(
                masked_time,
                masked_flux,
                period_days=float(period),
                epoch=epoch,
                duration_days=injected_duration,
                depth=float(depth_ppm) / 1_000_000.0,
            )
            result = run_bls_search(
                masked_time,
                injected_flux,
                masked_flux_err,
                durations=search_durations,
                minimum_period=0.5,
                maximum_period=maximum_period,
            ).signal
            evaluation = evaluate_recovery(
                result,
                injected_period=float(period),
                injected_epoch=epoch,
                injected_duration=injected_duration,
                control_power=control.power,
            )
            trials.append(
                InjectionTrial(
                    trial=trial_number,
                    injected_period_days=float(period),
                    injected_epoch=epoch,
                    injected_duration_days=injected_duration,
                    injected_depth_ppm=float(depth_ppm),
                    recovered_period_days=result.period_days,
                    recovered_epoch=result.transit_time,
                    recovered_depth_ppm=result.depth * 1_000_000.0,
                    recovered_depth_snr=result.depth_snr,
                    recovered_power=result.power,
                    **evaluation,
                )
            )
            print(
                f"Trial {trial_number}: P={period:.3f} d, "
                f"depth={depth_ppm:.0f} ppm -> "
                f"{evaluation['classification']}"
            )

    return trials, control, rows_masked


def save_trials(trials: list[InjectionTrial], output_path: Path) -> None:
    """Save trial-level results in tabular form."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(trial) for trial in trials]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_trial_plot(trials: list[InjectionTrial], output_path: Path) -> None:
    """Plot the small, explicitly bounded recovery experiment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    styles = {
        "recovered": ("#16a34a", "o", "Recovered"),
        "double_period_alias": ("#f97316", "^", "Double-period alias"),
        "half_period_alias": ("#f97316", "v", "Half-period alias"),
        "period_matched_but_failed_checks": (
            "#eab308",
            "s",
            "Period matched; other check failed",
        ),
        "not_recovered": ("#dc2626", "x", "Not recovered"),
    }

    for classification, (color, marker, label) in styles.items():
        selected = [trial for trial in trials if trial.classification == classification]
        if not selected:
            continue
        axis.scatter(
            [trial.injected_period_days for trial in selected],
            [trial.injected_depth_ppm for trial in selected],
            s=75,
            c=color,
            marker=marker,
            label=label,
            edgecolors="#0f172a",
            linewidths=0.4,
        )

    axis.set_xscale("log")
    axis.set_xticks(sorted({trial.injected_period_days for trial in trials}))
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel("Injected period (days)")
    axis.set_ylabel("Injected depth (ppm)")
    axis.set_title("Limited one-quarter BLS injection/recovery benchmark")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=8)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def injection_file(
    csv_path: Path,
    candidate_path: Path,
    *,
    output_dir: Path,
) -> tuple[Path, Path, Path, dict]:
    """Run and save the bounded injection/recovery benchmark."""
    time, flux, flux_err = load_detrended_light_curve(csv_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    known = candidate["best_signal"]

    periods = np.array([1.1, 2.3, 4.1], dtype=float)
    depths_ppm = np.array([100.0, 250.0, 500.0], dtype=float)
    search_durations = np.array([0.04, 0.06, 0.08, 0.12], dtype=float)
    injected_duration = 0.08

    trials, control, rows_masked = run_injection_recovery(
        time,
        flux,
        flux_err,
        known_period=float(known["period_days"]),
        known_epoch=float(known["transit_time"]),
        known_duration=float(known["duration_days"]),
        periods=periods,
        depths_ppm=depths_ppm,
        injected_duration=injected_duration,
        search_durations=search_durations,
        maximum_period=10.0,
    )

    trials_path = output_dir / f"{csv_path.stem}_injection_trials.csv"
    summary_path = output_dir / f"{csv_path.stem}_injection_summary.json"
    plot_path = output_dir / f"{csv_path.stem}_injection_recovery.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_trials(trials, trials_path)
    save_trial_plot(trials, plot_path)

    by_depth = []
    for depth in depths_ppm:
        selected = [trial for trial in trials if trial.injected_depth_ppm == depth]
        recovered = sum(trial.recovered for trial in selected)
        by_depth.append(
            {
                "depth_ppm": float(depth),
                "trials": len(selected),
                "recovered": recovered,
                "recovery_fraction": recovered / len(selected),
            }
        )

    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path),
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "source_candidate": str(candidate_path),
        "candidate_sha256": hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        "scope": (
            "BLS-only injections into one detrended Kepler quarter after "
            "masking the strongest blindly recovered known signal"
        ),
        "general_performance_estimate": False,
        "known_signal_mask": {
            "period_days": float(known["period_days"]),
            "epoch": float(known["transit_time"]),
            "duration_days": float(known["duration_days"]),
            "half_width_days": 0.75 * float(known["duration_days"]),
            "rows_masked": rows_masked,
        },
        "zero_depth_control": {
            "period_days": control.period_days,
            "epoch": control.transit_time,
            "depth_ppm": control.depth * 1_000_000.0,
            "depth_snr": control.depth_snr,
            "power": control.power,
        },
        "recovery_criteria": {
            "relative_period_tolerance": 0.005,
            "maximum_epoch_error_days": injected_duration,
            "minimum_formal_depth_snr": 5.0,
            "power_must_exceed_zero_depth_control": True,
            "aliases_counted_as_recovered": False,
        },
        "trial_count": len(trials),
        "recovered_count": sum(trial.recovered for trial in trials),
        "by_depth": by_depth,
        "trials": [asdict(trial) for trial in trials],
        "scientific_notice": (
            "This small deterministic benchmark tests one implementation on "
            "one real light curve. It does not estimate mission-wide or "
            "real-world discovery performance."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return trials_path, summary_path, plot_path, summary


def main() -> None:
    """Command-line entry point for the limited recovery benchmark."""
    parser = argparse.ArgumentParser(
        description="Run zero-depth control and nine BLS injection trials."
    )
    parser.add_argument("detrended_csv", type=Path)
    parser.add_argument("bls_candidate_json", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/injection")
    )
    args = parser.parse_args()

    try:
        trials_path, summary_path, plot_path, summary = injection_file(
            args.detrended_csv,
            args.bls_candidate_json,
            output_dir=args.output_dir,
        )
    except Exception as error:
        raise SystemExit(f"Injection/recovery failed: {error}") from error

    print(f"Trials CSV:   {trials_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Plot:         {plot_path}")
    print(
        f"Recovered:    {summary['recovered_count']} / "
        f"{summary['trial_count']}"
    )
    for row in summary["by_depth"]:
        print(
            f"Depth {row['depth_ppm']:.0f} ppm: "
            f"{row['recovered']} / {row['trials']} recovered"
        )
    print("Scope: limited one-quarter software benchmark, not general performance.")


if __name__ == "__main__":
    main()