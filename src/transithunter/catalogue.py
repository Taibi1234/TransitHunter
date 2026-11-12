"""Cross-match transit signals with NASA Exoplanet Archive catalogues."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive


CONFIRMED_COLUMNS = (
    "pl_name,hostname,pl_orbper,ra,dec,disc_year,discoverymethod"
)
KOI_COLUMNS = (
    "kepid,kepoi_name,kepler_name,koi_disposition,"
    "koi_period,koi_time0bk,koi_duration,koi_depth,ra,dec"
)


@dataclass(frozen=True)
class CrossmatchResult:
    """Catalogue rows and the interpretation of their period matches."""

    status: str
    confirmed_matches: list[dict[str, Any]]
    koi_matches: list[dict[str, Any]]
    confirmed_rows: list[dict[str, Any]]
    koi_rows: list[dict[str, Any]]


def _json_value(value: Any) -> Any:
    """Convert table values into strict JSON-compatible values."""
    if value is np.ma.masked:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _float_or_none(value: Any) -> float | None:
    """Parse a finite number, including archive strings with trailing units."""
    converted = _json_value(value)
    if converted is None:
        return None

    if isinstance(converted, str):
        parts = converted.strip().split(maxsplit=1)
        if not parts:
            return None
        converted = parts[0]

    try:
        result = float(converted)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _period_match_type(
    candidate_period: float,
    catalogue_period: float | None,
    tolerance: float,
) -> str | None:
    """Identify same-period, half-period, or double-period agreement."""
    if catalogue_period is None or catalogue_period <= 0:
        return None

    ratio = candidate_period / catalogue_period
    for label, expected in (
        ("same_period", 1.0),
        ("twice_catalogue_period", 2.0),
        ("half_catalogue_period", 0.5),
    ):
        if abs(ratio - expected) <= tolerance * expected:
            return label
    return None


def _table_records(
    table: Any,
    coordinates: SkyCoord,
    candidate_period: float,
    period_column: str,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Convert catalogue rows and annotate separation and period matching."""
    records: list[dict[str, Any]] = []

    for row in table:
        record = {
            name: _json_value(row[name])
            for name in row.colnames
        }
        ra = _float_or_none(row["ra"])
        dec = _float_or_none(row["dec"])
        if ra is not None and dec is not None:
            row_coordinates = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            record["separation_arcsec"] = float(
                coordinates.separation(row_coordinates).arcsec
            )
        else:
            record["separation_arcsec"] = None

        catalogue_period = _float_or_none(row[period_column])
        record["period_match_type"] = _period_match_type(
            candidate_period, catalogue_period, tolerance
        )
        records.append(record)

    return records


def crossmatch_catalogues(
    candidate_period: float,
    coordinates: SkyCoord,
    confirmed_table: Any,
    koi_table: Any,
    *,
    period_tolerance: float = 0.01,
) -> CrossmatchResult:
    """Classify catalogue agreement without making a discovery claim."""
    if candidate_period <= 0:
        raise ValueError("candidate_period must be positive.")
    if not 0 < period_tolerance < 1:
        raise ValueError("period_tolerance must be between zero and one.")

    confirmed_rows = _table_records(
        confirmed_table,
        coordinates,
        candidate_period,
        "pl_orbper",
        period_tolerance,
    )
    koi_rows = _table_records(
        koi_table,
        coordinates,
        candidate_period,
        "koi_period",
        period_tolerance,
    )
    confirmed_matches = [
        row for row in confirmed_rows if row["period_match_type"] is not None
    ]
    koi_matches = [
        row for row in koi_rows if row["period_match_type"] is not None
    ]

    false_positive_matches = [
        row
        for row in koi_matches
        if str(row.get("koi_disposition", "")).upper() == "FALSE POSITIVE"
    ]
    candidate_matches = [
        row
        for row in koi_matches
        if str(row.get("koi_disposition", "")).upper() == "CANDIDATE"
    ]

    if confirmed_matches:
        status = "known_confirmed_planet_signal"
    elif false_positive_matches:
        status = "known_false_positive_signal"
    elif candidate_matches:
        status = "known_koi_candidate_signal"
    elif koi_matches:
        status = "known_koi_signal"
    elif confirmed_rows or koi_rows:
        status = "catalogued_target_without_matching_period"
    else:
        status = "no_match_in_searched_catalogues"

    return CrossmatchResult(
        status=status,
        confirmed_matches=confirmed_matches,
        koi_matches=koi_matches,
        confirmed_rows=confirmed_rows,
        koi_rows=koi_rows,
    )


def _empty_confirmed_table() -> Table:
    """Return an empty table with the fields needed by cross-matching."""
    return Table(
        names=(
            "pl_name",
            "hostname",
            "pl_orbper",
            "ra",
            "dec",
            "disc_year",
            "discoverymethod",
        ),
        dtype=(str, str, float, float, float, int, str),
    )


def _hostname_from_kepler_name(value: Any) -> str | None:
    """Convert a name such as 'Kepler-10 b' to its host name."""
    converted = _json_value(value)
    if converted is None:
        return None
    match = re.match(r"^(.*?)\s+[a-z]$", str(converted).strip())
    return match.group(1) if match else None


def query_catalogues(target: str) -> tuple[Any, Any, dict[str, Any]]:
    """Query exact KIC matches without using the archive cone-search path."""
    match = re.fullmatch(r"\s*KIC\s*(\d+)\s*", target, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            "The current exact-ID cross-match requires a target like "
            "'KIC 11904151'. TESS TOI cross-matching will be added separately."
        )

    kepid = int(match.group(1))
    kois = NasaExoplanetArchive.query_criteria(
        table="cumulative",
        select=KOI_COLUMNS,
        where=f"kepid={kepid}",
        cache=False,
    )

    hostnames = sorted(
        {
            hostname
            for row in kois
            if (hostname := _hostname_from_kepler_name(row["kepler_name"]))
            is not None
        }
    )
    confirmed_tables = []
    for hostname in hostnames:
        escaped_hostname = hostname.replace("'", "''")
        confirmed_tables.append(
            NasaExoplanetArchive.query_criteria(
                table="pscomppars",
                select=CONFIRMED_COLUMNS,
                where=f"hostname='{escaped_hostname}'",
                cache=False,
            )
        )

    if confirmed_tables:
        confirmed = vstack(confirmed_tables)
    else:
        confirmed = _empty_confirmed_table()

    query_info = {
        "strategy": "exact KIC to cumulative KOI to confirmed hostname",
        "target": target,
        "kepid": kepid,
        "confirmed_hostnames": hostnames,
        "cone_search_used": False,
    }
    return confirmed, kois, query_info


def crossmatch_file(
    raw_metadata_path: Path,
    candidate_path: Path,
    *,
    output_dir: Path,
    period_tolerance: float,
) -> tuple[Path, CrossmatchResult]:
    """Query catalogues for one acquisition and save the complete snapshot."""
    metadata = json.loads(raw_metadata_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    archive_record = metadata["archive_record"]
    ra = _float_or_none(archive_record.get("s_ra"))
    dec = _float_or_none(archive_record.get("s_dec"))
    if ra is None or dec is None:
        raise ValueError("The acquisition metadata lacks finite sky coordinates.")

    target = str(metadata["query"]["target"])
    candidate_period = float(candidate["best_signal"]["period_days"])
    coordinates = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    confirmed_table, koi_table, query_info = query_catalogues(target)
    result = crossmatch_catalogues(
        candidate_period,
        coordinates,
        confirmed_table,
        koi_table,
        period_tolerance=period_tolerance,
    )

    output_path = output_dir / f"{candidate_path.stem}_catalogue_crossmatch.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata": str(raw_metadata_path),
        "source_metadata_sha256": hashlib.sha256(
            raw_metadata_path.read_bytes()
        ).hexdigest(),
        "source_candidate": str(candidate_path),
        "source_candidate_sha256": hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        "archive": "NASA Exoplanet Archive",
        "tables": ["pscomppars", "cumulative"],
        "query": {
            **query_info,
            "ra_deg": ra,
            "dec_deg": dec,
            "period_tolerance_fraction": period_tolerance,
        },
        "candidate_period_days": candidate_period,
        "status": result.status,
        "confirmed_period_matches": result.confirmed_matches,
        "koi_period_matches": result.koi_matches,
        "target_confirmed_planets": result.confirmed_rows,
        "target_kois": result.koi_rows,
        "scientific_notice": (
            "A catalogue period match identifies an already known signal. "
            "No catalogue match does not establish a new discovery; aliases, "
            "incomplete catalogues, and false positives still require review."
        ),
    }
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path, result


def main() -> None:
    """Command-line entry point for NASA catalogue cross-matching."""
    parser = argparse.ArgumentParser(
        description="Cross-match a BLS signal with confirmed planets and KOIs."
    )
    parser.add_argument("raw_metadata_json", type=Path)
    parser.add_argument("candidate_json", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/catalogue")
    )
    parser.add_argument("--period-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    try:
        output_path, result = crossmatch_file(
            args.raw_metadata_json,
            args.candidate_json,
            output_dir=args.output_dir,
            period_tolerance=args.period_tolerance,
        )
    except Exception as error:
        raise SystemExit(f"Catalogue cross-match failed: {error}") from error

    print(f"Cross-match report: {output_path}")
    print(f"Target confirmed:   {len(result.confirmed_rows)}")
    print(f"Target KOIs:        {len(result.koi_rows)}")
    print(f"Confirmed matches:  {len(result.confirmed_matches)}")
    print(f"KOI matches:        {len(result.koi_matches)}")
    print(f"Status:             {result.status}")

    for row in result.confirmed_matches:
        print(
            "Confirmed: "
            f"{row.get('pl_name')} | "
            f"period={row.get('pl_orbper')} | "
            f"match={row['period_match_type']}"
        )
    for row in result.koi_matches:
        print(
            "KOI: "
            f"{row.get('kepoi_name')} | "
            f"disposition={row.get('koi_disposition')} | "
            f"period={row.get('koi_period')} | "
            f"match={row['period_match_type']}"
        )

    print("No catalogue match alone would not constitute a discovery.")


if __name__ == "__main__":
    main()