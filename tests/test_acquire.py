"""Tests for light-curve caching and lossless acquisition export."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.utils.masked import Masked

from transithunter.acquire import build_cache_paths, export_light_curve


def test_cache_paths_are_deterministic_and_query_specific() -> None:
    """The same request reuses a cache; another quarter does not."""
    first = build_cache_paths(
        Path("data/raw"), "KIC 11904151", "Kepler", quarter=3
    )
    repeated = build_cache_paths(
        Path("data/raw"), "KIC 11904151", "Kepler", quarter=3
    )
    other_quarter = build_cache_paths(
        Path("data/raw"), "KIC 11904151", "Kepler", quarter=4
    )

    assert first == repeated
    assert first != other_quarter


def test_export_preserves_quality_flags_and_missing_values(
    tmp_path: Path,
) -> None:
    """Acquisition must not silently clean or normalize the measurements."""
    light_curve = SimpleNamespace(
        time=SimpleNamespace(value=np.array([100.0, 101.0, 102.0])),
        flux=SimpleNamespace(value=np.array([1000.0, np.nan, 998.0])),
        flux_err=SimpleNamespace(value=np.array([2.0, 2.0, 2.0])),
        quality=np.array([0, 32, 0], dtype=np.int64),
    )
    csv_path = tmp_path / "light_curve.csv"

    row_count = export_light_curve(light_curve, csv_path)
    loaded = np.genfromtxt(csv_path, delimiter=",", names=True)

    assert row_count == 3
    assert loaded["quality"].tolist() == [0.0, 32.0, 0.0]
    assert np.isnan(loaded["flux"][1])
    assert loaded["flux"][0] == 1000.0


def test_export_handles_astropy_masked_arrays(tmp_path: Path) -> None:
    """Astropy masked values become NaN without triggering NumPy errors."""
    masked_flux = Masked(
        np.array([1000.0, 999.0, 998.0]),
        mask=np.array([False, True, False]),
    )
    light_curve = SimpleNamespace(
        time=SimpleNamespace(value=np.array([100.0, 101.0, 102.0])),
        flux=SimpleNamespace(value=masked_flux),
        flux_err=SimpleNamespace(value=np.array([2.0, 2.0, 2.0])),
        quality=np.array([0, 32, 0], dtype=np.int64),
    )
    csv_path = tmp_path / "masked_light_curve.csv"

    row_count = export_light_curve(light_curve, csv_path)
    loaded = np.genfromtxt(csv_path, delimiter=",", names=True)

    assert row_count == 3
    assert loaded["flux"].tolist()[0] == 1000.0
    assert np.isnan(loaded["flux"][1])
    assert loaded["flux"].tolist()[2] == 998.0
    assert loaded["quality"].tolist() == [0.0, 32.0, 0.0]