"""Tests for conservative, transit-safe light-curve preprocessing."""

import numpy as np

from transithunter.preprocess import preprocess_light_curve


def test_transit_dip_survives_positive_outlier_filtering() -> None:
    """A negative transit-like deviation must never be sigma-clipped."""
    time = np.arange(7, dtype=float)
    flux = np.array([1000.0, 1000.0, 1000.0, 990.0, 1000.0, 1000.0, 10000.0])
    flux_err = np.ones(7)
    quality = np.zeros(7, dtype=np.int64)

    processed = preprocess_light_curve(
        time, flux, flux_err, quality, positive_outlier_sigma=8.0
    )

    assert processed.report.rows_input == 7
    assert processed.report.rows_output == 6
    assert processed.report.rows_positive_outliers == 1
    assert 3.0 in processed.time
    transit_index = int(np.flatnonzero(processed.time == 3.0)[0])
    assert np.isclose(processed.flux[transit_index], 0.99)
    assert 6.0 not in processed.time


def test_quality_finite_and_uncertainty_filters_are_applied() -> None:
    """Basic filtering is sequential and produces auditable counts."""
    time = np.arange(5, dtype=float)
    flux = np.array([100.0, 100.0, np.nan, 100.0, 100.0])
    flux_err = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
    quality = np.array([0, 8, 0, 0, 0], dtype=np.int64)

    processed = preprocess_light_curve(time, flux, flux_err, quality)

    assert processed.report.rows_bad_quality == 1
    assert processed.report.rows_nonfinite == 1
    assert processed.report.rows_nonpositive_error == 1
    assert processed.report.rows_output == 2
    assert processed.report.median_input_flux == 100.0
    assert np.allclose(processed.flux, 1.0)
    assert np.allclose(processed.flux_err, 0.01)