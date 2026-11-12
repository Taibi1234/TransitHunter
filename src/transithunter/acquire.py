"""Acquire and cache real light curves from the MAST archive."""

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

import lightkurve as lk
import numpy as np


@dataclass(frozen=True)
class AcquisitionResult:
    """Paths and row count for a cached light curve."""

    csv_path: Path
    metadata_path: Path
    row_count: int
    cached: bool


def _slug(value: str) -> str:
    """Convert a target identifier into a safe filename component."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not slug:
        raise ValueError("The target must contain letters or numbers.")
    return slug


def _json_value(value: Any) -> Any:
    """Convert archive metadata values into strict JSON values."""
    if value is np.ma.masked:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _column_values(column: Any) -> Any:
    """Return plain values from an Astropy column or quantity."""
    return getattr(column, "value", column)


def _filled_array(column: Any, fill_value: Any, dtype: Any) -> np.ndarray:
    """Convert NumPy or Astropy masked columns into plain arrays.

    Astropy's MaskedNDArray cannot safely be handled through every NumPy
    masked-array shortcut. Its values and mask are therefore extracted
    explicitly before filling missing measurements.
    """
    values = _column_values(column)

    if hasattr(values, "unmasked") and hasattr(values, "mask"):
        data = np.asarray(values.unmasked, dtype=dtype)
        mask = np.asarray(values.mask, dtype=bool)
        return np.asarray(np.where(mask, fill_value, data), dtype=dtype)

    masked = np.ma.asarray(values)
    filled = np.ma.filled(masked, fill_value)
    return np.asarray(filled, dtype=dtype)


def build_cache_paths(
    output_dir: Path,
    target: str,
    mission: str,
    *,
    quarter: int | None = None,
    sector: int | None = None,
    author: str | None = None,
    cadence: str | None = None,
) -> tuple[Path, Path]:
    """Build stable, query-specific CSV and metadata paths."""
    query = {
        "target": target,
        "mission": mission,
        "quarter": quarter,
        "sector": sector,
        "author": author,
        "cadence": cadence,
    }
    serialized = json.dumps(query, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    stem = f"{mission.lower()}_{_slug(target)}_{digest}"
    return output_dir / f"{stem}.csv", output_dir / f"{stem}.json"


def export_light_curve(light_curve: Any, csv_path: Path) -> int:
    """Export measurements without quality masking or normalization."""
    time = _filled_array(light_curve.time, np.nan, float)
    flux = _filled_array(light_curve.flux, np.nan, float)
    flux_err = _filled_array(light_curve.flux_err, np.nan, float)
    quality = _filled_array(light_curve.quality, -1, np.int64)

    lengths = {len(time), len(flux), len(flux_err), len(quality)}
    if len(time) == 0:
        raise ValueError("The downloaded light curve contains no rows.")
    if len(lengths) != 1:
        raise ValueError("Downloaded light-curve columns have different lengths.")

    table = np.column_stack((time, flux, flux_err, quality))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="time,flux,flux_err,quality",
        comments="",
        fmt=["%.17g", "%.17g", "%.17g", "%d"],
    )
    return len(time)


def acquire_light_curve(
    target: str,
    mission: str,
    *,
    output_dir: Path = Path("data/raw"),
    quarter: int | None = None,
    sector: int | None = None,
    author: str | None = None,
    cadence: str | None = None,
    force: bool = False,
) -> AcquisitionResult:
    """Download one light curve or reuse its complete local cache entry."""
    if mission not in {"Kepler", "TESS"}:
        raise ValueError("Mission must be either 'Kepler' or 'TESS'.")
    if mission == "Kepler" and sector is not None:
        raise ValueError("Use --quarter for Kepler, not --sector.")
    if mission == "TESS" and quarter is not None:
        raise ValueError("Use --sector for TESS, not --quarter.")

    csv_path, metadata_path = build_cache_paths(
        output_dir,
        target,
        mission,
        quarter=quarter,
        sector=sector,
        author=author,
        cadence=cadence,
    )

    if not force and csv_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return AcquisitionResult(
            csv_path=csv_path,
            metadata_path=metadata_path,
            row_count=int(metadata["row_count"]),
            cached=True,
        )

    if not force and (csv_path.exists() or metadata_path.exists()):
        raise RuntimeError(
            "An incomplete cache entry exists. Repeat the command with --force."
        )

    search_arguments: dict[str, Any] = {"mission": mission}
    if quarter is not None:
        search_arguments["quarter"] = quarter
    if sector is not None:
        search_arguments["sector"] = sector
    if author is not None:
        search_arguments["author"] = author
    if cadence is not None:
        search_arguments["cadence"] = cadence

    search = lk.search_lightcurve(target, **search_arguments)
    if len(search) == 0:
        raise ValueError(f"No light curves found for {target!r}.")
    if len(search) != 1:
        raise ValueError(
            f"Expected one data product but found {len(search)}. "
            "Add a quarter, sector, author, or cadence filter."
        )

    # A zero bitmask retains every cadence and its original quality flag.
    light_curve = search[0].download(quality_bitmask=0)
    if light_curve is None:
        raise RuntimeError("MAST returned a data product but no light curve.")

    row_count = export_light_curve(light_curve, csv_path)
    archive_row = search.table[0]
    meta = getattr(light_curve, "meta", {}) or {}

    time = _filled_array(light_curve.time, np.nan, float)
    flux = _filled_array(light_curve.flux, np.nan, float)
    flux_err = _filled_array(light_curve.flux_err, np.nan, float)
    finite_rows = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "query": {
            "target": target,
            "mission": mission,
            "quarter": quarter,
            "sector": sector,
            "author": author,
            "cadence": cadence,
        },
        "source": {
            "archive": "MAST",
            "access_library": "lightkurve",
            "lightkurve_version": lk.__version__,
        },
        "archive_record": {
            name: _json_value(archive_row[name])
            for name in archive_row.colnames
        },
        "object": _json_value(meta.get("OBJECT") or meta.get("LABEL")),
        "row_count": row_count,
        "finite_row_count": int(np.count_nonzero(finite_rows)),
        "time_format": str(light_curve.time.format),
        "time_scale": str(light_curve.time.scale),
        "bjdref_days": float(
            meta.get("BJDREFI", 0.0) + meta.get("BJDREFF", 0.0)
        ),
        "flux_unit": str(getattr(light_curve.flux, "unit", "unknown")),
        "quality_bitmask_applied": 0,
        "normalized_by_transithunter": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return AcquisitionResult(
        csv_path=csv_path,
        metadata_path=metadata_path,
        row_count=row_count,
        cached=False,
    )


def main() -> None:
    """Command-line entry point for one-product acquisition."""
    parser = argparse.ArgumentParser(
        description="Download and cache one Kepler or TESS light curve."
    )
    parser.add_argument("target", help="For example: 'KIC 11904151'")
    parser.add_argument("--mission", choices=["Kepler", "TESS"], required=True)
    parser.add_argument("--quarter", type=int)
    parser.add_argument("--sector", type=int)
    parser.add_argument("--author")
    parser.add_argument("--cadence")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        result = acquire_light_curve(
            args.target,
            args.mission,
            output_dir=args.output_dir,
            quarter=args.quarter,
            sector=args.sector,
            author=args.author,
            cadence=args.cadence,
            force=args.force,
        )
    except Exception as error:
        raise SystemExit(f"Acquisition failed: {error}") from error

    print(f"CSV:      {result.csv_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Rows:     {result.row_count}")
    print(f"Cached:   {result.cached}")


if __name__ == "__main__":
    main()