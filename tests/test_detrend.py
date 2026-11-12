"""Tests for gap-aware, transit-protecting baseline estimation."""

import numpy as np

from transithunter.detrend import detrend_light_curve


def test_transit_is_protected_and_retained() -> None:
    """A short dip survives baseline removal without using an ephemeris."""
    time = np.arange(0.0, 10.0, 0.02)
    slowly_varying_baseline = 1.0 + 0.02 * np.sin(2.0 * np.pi * time / 5.0)
    flux = slowly_varying_baseline.copy()
    flux_err = np.full(len(time), 0.0005)

    transit = (time >= 4.8) & (time <= 4.9)
    flux[transit] -= 0.005

    result = detrend_light_curve(time, flux, flux_err)

    assert result.report.rows_output == len(time)
    assert np.all(result.protected[transit])
    assert np.all(result.flux[transit] < 0.998)
    assert np.isclose(
        np.median(result.flux[~transit]), 1.0, atol=0.002
    )


def test_large_gap_creates_independent_segments() -> None:
    """The rolling baseline never interpolates across a large gap."""
    time = np.array([0.0, 0.02, 0.04, 2.0, 2.02, 2.04])
    flux = np.ones(len(time))
    flux_err = np.full(len(time), 0.001)

    result = detrend_light_curve(
        time, flux, flux_err, gap_days=0.5
    )

    assert result.report.segment_count == 2
    assert result.report.rows_output == len(time)
    assert result.segment_id.tolist() == [0, 0, 0, 1, 1, 1]
    assert np.allclose(result.flux, 1.0)