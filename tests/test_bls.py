"""Tests for blind BLS recovery and phase folding."""

import numpy as np

from transithunter.bls import fold_time, run_bls_search


def test_blind_bls_recovers_synthetic_period() -> None:
    """BLS recovers a period without receiving the generating ephemeris."""
    rng = np.random.default_rng(42)
    time = np.arange(0.0, 30.0, 0.03)
    flux = np.ones(len(time))
    flux_err = np.full(len(time), 0.0002)

    # These values generate the test signal but are not search inputs.
    generating_period = 1.37
    generating_epoch = 0.37
    generating_duration = 0.09
    phase = fold_time(time, generating_period, generating_epoch)
    in_transit = np.abs(phase) < generating_duration / 2.0
    flux[in_transit] -= 0.004
    flux += rng.normal(0.0, 0.0002, len(time))

    result = run_bls_search(
        time,
        flux,
        flux_err,
        durations=np.array([0.04, 0.08, 0.12]),
        minimum_period=0.5,
        maximum_period=5.0,
    )

    assert abs(result.signal.period_days - generating_period) < 0.01
    assert result.signal.depth > 0.002
    assert result.signal.depth_snr > 5.0


def test_fold_time_centers_repeated_transits() -> None:
    """Equivalent mid-transit epochs fold to zero phase."""
    period = 2.0
    epoch = 0.4
    time = np.array([epoch, epoch + period, epoch + 5.0 * period])

    phase = fold_time(time, period, epoch)

    assert np.allclose(phase, 0.0)