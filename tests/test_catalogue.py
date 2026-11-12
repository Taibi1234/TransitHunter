"""Network-independent tests for catalogue cross-matching and its CLI."""

from pathlib import Path
import sys

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from transithunter import catalogue
from transithunter.catalogue import CrossmatchResult, crossmatch_catalogues


def _coordinates() -> SkyCoord:
    return SkyCoord(ra=285.679 * u.deg, dec=50.241 * u.deg)


def _confirmed_table(period: float = 0.8375) -> Table:
    return Table(
        {
            "pl_name": ["Kepler-10 b"],
            "hostname": ["Kepler-10"],
            "pl_orbper": [f"{period} d"],
            "ra": ["285.679 deg"],
            "dec": ["50.241 deg"],
            "disc_year": [2011],
            "discoverymethod": ["Transit"],
        }
    )


def _koi_table(
    period: float = 0.8375,
    disposition: str = "CONFIRMED",
) -> Table:
    return Table(
        {
            "kepid": [11904151],
            "kepoi_name": ["K00072.01"],
            "kepler_name": ["Kepler-10 b"],
            "koi_disposition": [disposition],
            "koi_period": [f"{period} d"],
            "koi_time0bk": [131.0],
            "koi_duration": ["1.8 h"],
            "koi_depth": ["150.0 ppm"],
            "ra": ["285.679 deg"],
            "dec": ["50.241 deg"],
        }
    )


def _empty_result() -> CrossmatchResult:
    return CrossmatchResult(
        status="no_match_in_searched_catalogues",
        confirmed_matches=[],
        koi_matches=[],
        confirmed_rows=[],
        koi_rows=[],
    )


def test_confirmed_period_match_is_identified() -> None:
    """Unit-bearing archive strings must still produce a confirmed match."""
    result = crossmatch_catalogues(
        0.83743,
        _coordinates(),
        _confirmed_table(),
        _koi_table(),
    )

    assert result.status == "known_confirmed_planet_signal"
    assert result.confirmed_matches[0]["pl_name"] == "Kepler-10 b"
    assert result.confirmed_matches[0]["period_match_type"] == "same_period"
    assert result.confirmed_matches[0]["separation_arcsec"] is not None


def test_matching_false_positive_is_prioritized_over_unrelated_planets() -> None:
    """A KOI false-positive period match is retained when no planet matches."""
    result = crossmatch_catalogues(
        0.83743,
        _coordinates(),
        _confirmed_table(period=45.0),
        _koi_table(disposition="FALSE POSITIVE"),
    )

    assert result.status == "known_false_positive_signal"
    assert result.koi_matches[0]["koi_disposition"] == "FALSE POSITIVE"


def test_no_catalogue_match_is_not_called_a_discovery() -> None:
    """An empty exact-ID search receives a deliberately limited classification."""
    result = crossmatch_catalogues(
        1.234,
        _coordinates(),
        _confirmed_table()[:0],
        _koi_table()[:0],
    )

    assert result.status == "no_match_in_searched_catalogues"
    assert result.confirmed_matches == []
    assert result.koi_matches == []


def test_cli_passes_positional_paths_correctly(monkeypatch, tmp_path: Path) -> None:
    """The CLI must use the argument names declared by its parser."""
    received: dict[str, Path] = {}

    def fake_crossmatch_file(
        raw_metadata_path: Path,
        candidate_path: Path,
        **kwargs,
    ) -> tuple[Path, CrossmatchResult]:
        received["metadata"] = raw_metadata_path
        received["candidate"] = candidate_path
        return tmp_path / "report.json", _empty_result()

    monkeypatch.setattr(catalogue, "crossmatch_file", fake_crossmatch_file)
    monkeypatch.setattr(
        sys,
        "argv",
        ["transithunter.catalogue", "metadata.json", "candidate.json"],
    )

    catalogue.main()

    assert received["metadata"] == Path("metadata.json")
    assert received["candidate"] == Path("candidate.json")