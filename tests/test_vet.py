"""Tests for odd/even and secondary-eclipse candidate screening."""

import numpy as np

from transithunter.bls import fold_time
from transithunter.vet import vet_transit_signal


def _screening_input(
    *,
    alternating_depths: bool = False,
    secondary_depth: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Generate controlled primary and secondary box-shaped events."""
    time = np.arange(0.0, 30.0, 0.02)
    period = 1.37
    epoch = 0.37
    duration = 0.09
    flux = np.ones(len(time))
    flux_err = np.full(len(time), 0.0002)

    primary_phase = fold_time(time, period, epoch)
    primary = np.abs(primary_phase) <= duration / 2.0
    event_number = np.rint((time - epoch) / period).astype(int)

    if alternating_depths:
        depths = np.where(event_number % 2 == 0, 0.008, 0.004)
        flux[primary] -= depths[primary]
    else:
        flux[primary] -= 0.004

    secondary_phase = fold_time(time, period, epoch + period / 2.0)
    secondary = np.abs(secondary_phase) <= duration / 2.0
    flux[secondary] -= secondary_depth
    return time, flux, flux_err, period, duration, epoch


def test_consistent_planet_like_events_pass_limited_checks() -> None:
    """Equal primary depths and no secondary produce no screening flags."""
    time, flux, flux_err, period, duration, epoch = _screening_input()

    result = vet_transit_signal(
        time,
        flux,
        flux_err,
        period_days=period,
        duration_days=duration,
        transit_time=epoch,
    )

    assert abs(result.summary["odd_even_z"]) < 3.0
    assert result.summary["secondary_snr"] < 3.0
    assert result.summary["flags"] == []


def test_alternating_depths_are_flagged() -> None:
    """Different odd/even depths indicate a possible doubled-period binary."""
    time, flux, flux_err, period, duration, epoch = _screening_input(
        alternating_depths=True
    )

    result = vet_transit_signal(
        time,
        flux,
        flux_err,
        period_days=period,
        duration_days=duration,
        transit_time=epoch,
    )

    assert abs(result.summary["odd_even_z"]) > 3.0
    assert "significant_odd_even_depth_difference" in result.summary["flags"]


def test_significant_secondary_eclipse_is_flagged() -> None:
    """A phase-0.5 flux decrement is reported as a secondary-eclipse flag."""
    time, flux, flux_err, period, duration, epoch = _screening_input(
        secondary_depth=0.003
    )

    result = vet_transit_signal(
        time,
        flux,
        flux_err,
        period_days=period,
        duration_days=duration,
        transit_time=epoch,
    )

    assert result.summary["secondary_snr"] > 3.0
    assert "significant_secondary_eclipse" in result.summary["flags"]