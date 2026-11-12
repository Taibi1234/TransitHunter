"""Tests for controlled signal injection and recovery scoring."""

import numpy as np

from transithunter.bls import TransitSignal
from transithunter.injection import evaluate_recovery, inject_box_signal


def _signal(period: float = 1.3, epoch: float = 0.2) -> TransitSignal:
    return TransitSignal(
        period_days=period,
        duration_days=0.08,
        transit_time=epoch,
        depth=0.0005,
        depth_err=0.00002,
        depth_snr=25.0,
        power=100.0,
    )


def test_injection_changes_only_in_transit_points() -> None:
    """Injection is non-mutating and produces the requested decrement."""
    time = np.arange(0.0, 3.0, 0.02)
    flux = np.ones(len(time))

    injected = inject_box_signal(
        time,
        flux,
        period_days=1.3,
        epoch=0.2,
        duration_days=0.08,
        depth=0.0005,
    )

    assert np.all(flux == 1.0)
    assert np.isclose(np.min(injected), 0.9995)
    assert np.count_nonzero(injected < 1.0) > 0


def test_recovery_requires_fundamental_period_and_control_excess() -> None:
    """Correct recovery passes, while a double-period alias does not."""
    recovered = evaluate_recovery(
        _signal(),
        injected_period=1.3,
        injected_epoch=0.2,
        injected_duration=0.08,
        control_power=10.0,
    )
    alias = evaluate_recovery(
        _signal(period=2.6),
        injected_period=1.3,
        injected_epoch=0.2,
        injected_duration=0.08,
        control_power=10.0,
    )

    assert recovered["recovered"] is True
    assert recovered["classification"] == "recovered"
    assert alias["recovered"] is False
    assert alias["classification"] == "double_period_alias"