"""End-to-end orchestration for the TransitHunter research pipeline."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from transithunter.acquire import acquire_light_curve
from transithunter.bls import search_file
from transithunter.catalogue import crossmatch_file
from transithunter.detrend import detrend_file
from transithunter.injection import injection_file
from transithunter.preprocess import process_file
from transithunter.vet import vet_file


def _run_name(target: str, mission: str, quarter: int | None, sector: int | None) -> str:
    """Construct a filesystem-safe run identifier."""
    target_slug = re.sub(r"[^A-Za-z0-9]+", "_", target).strip("_").lower()
    suffix = ""
    if quarter is not None:
        suffix = f"_q{quarter:02d}"
    if sector is not None:
        suffix = f"_s{sector:04d}"
    return f"{mission.lower()}_{target_slug}{suffix}"


def run_pipeline(
    target: str,
    mission: str,
    *,
    quarter: int | None = None,
    sector: int | None = None,
    author: str | None = None,
    cadence: str | None = None,
    force_download: bool = False,
    skip_catalogue: bool = False,
    run_injection: bool = False,
    manifest_dir: Path = Path("outputs/runs"),
) -> dict:
    """Run every requested stage and write a machine-readable manifest."""
    acquisition = acquire_light_curve(
        target,
        mission,
        quarter=quarter,
        sector=sector,
        author=author,
        cadence=cadence,
        force=force_download,
    )
    print(f"[1/6] Acquisition: {acquisition.row_count} rows")

    processed_path, preprocess_report_path, preprocess_plot, preprocess_report = (
        process_file(
            acquisition.csv_path,
            output_dir=Path("data/processed"),
            plot_dir=Path("outputs/diagnostics"),
            positive_outlier_sigma=8.0,
        )
    )
    print(f"[2/6] Preprocessing: {preprocess_report.rows_output} rows retained")

    detrended_path, detrend_report_path, detrend_plot, detrend_report = (
        detrend_file(
            processed_path,
            output_dir=Path("data/processed"),
            plot_dir=Path("outputs/diagnostics"),
            window_days=1.0,
            gap_days=0.5,
            protection_sigma=3.0,
        )
    )
    print(f"[3/6] Detrending: {detrend_report.segment_count} segments")

    candidate_path, periodogram_path, bls_plot, signal = search_file(
        detrended_path,
        output_dir=Path("outputs/search"),
        durations=np.array([0.04, 0.06, 0.08, 0.12]),
        minimum_period=0.5,
        maximum_period=10.0,
        frequency_factor=1.0,
    )
    print(f"[4/6] Blind BLS: period={signal.period_days:.10f} d")

    vetting_path, events_path, vetting_plot, vetting_summary = vet_file(
        detrended_path,
        candidate_path,
        output_dir=Path("outputs/vetting"),
        significance_threshold=3.0,
    )
    print(f"[5/6] Vetting: {vetting_summary['status']}")

    catalogue_path = None
    catalogue_status = "not_run"
    if not skip_catalogue:
        catalogue_path, crossmatch = crossmatch_file(
            acquisition.metadata_path,
            candidate_path,
            output_dir=Path("outputs/catalogue"),
            period_tolerance=0.01,
        )
        catalogue_status = crossmatch.status
    print(f"[6/6] Catalogue: {catalogue_status}")

    injection_summary_path = None
    injection_plot = None
    injection_summary = None
    if run_injection:
        _, injection_summary_path, injection_plot, injection_summary = (
            injection_file(
                detrended_path,
                candidate_path,
                output_dir=Path("outputs/injection"),
            )
        )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "mission": mission,
        "quarter": quarter,
        "sector": sector,
        "known_ephemeris_supplied": False,
        "best_period_days": signal.period_days,
        "vetting_status": vetting_summary["status"],
        "vetting_flags": vetting_summary["flags"],
        "catalogue_status": catalogue_status,
        "artifacts": {
            "raw_csv": str(acquisition.csv_path),
            "raw_metadata": str(acquisition.metadata_path),
            "processed_csv": str(processed_path),
            "preprocessing_report": str(preprocess_report_path),
            "preprocessing_plot": str(preprocess_plot),
            "detrended_csv": str(detrended_path),
            "detrending_report": str(detrend_report_path),
            "detrending_plot": str(detrend_plot),
            "bls_candidate": str(candidate_path),
            "bls_periodogram": str(periodogram_path),
            "bls_plot": str(bls_plot),
            "vetting_report": str(vetting_path),
            "event_depths": str(events_path),
            "vetting_plot": str(vetting_plot),
            "catalogue_report": (
                str(catalogue_path) if catalogue_path is not None else None
            ),
            "injection_summary": (
                str(injection_summary_path)
                if injection_summary_path is not None
                else None
            ),
            "injection_plot": (
                str(injection_plot) if injection_plot is not None else None
            ),
        },
        "injection_recovery": (
            {
                "trial_count": injection_summary["trial_count"],
                "recovered_count": injection_summary["recovered_count"],
                "general_performance_estimate": False,
            }
            if injection_summary is not None
            else None
        ),
        "scientific_notice": (
            "Pipeline outputs are transit-like signals requiring review. "
            "Catalogue matches identify known objects; no match does not "
            "establish discovery, and automated vetting cannot confirm planets."
        ),
    }

    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{_run_name(target, mission, quarter, sector)}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    """Command-line entry point for the complete research workflow."""
    parser = argparse.ArgumentParser(
        description="Run TransitHunter from MAST acquisition through vetting."
    )
    parser.add_argument("target")
    parser.add_argument("--mission", choices=["Kepler", "TESS"], required=True)
    parser.add_argument("--quarter", type=int)
    parser.add_argument("--sector", type=int)
    parser.add_argument("--author")
    parser.add_argument("--cadence")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-catalogue", action="store_true")
    parser.add_argument("--run-injection", action="store_true")
    parser.add_argument("--manifest-dir", type=Path, default=Path("outputs/runs"))
    args = parser.parse_args()

    try:
        manifest = run_pipeline(
            args.target,
            args.mission,
            quarter=args.quarter,
            sector=args.sector,
            author=args.author,
            cadence=args.cadence,
            force_download=args.force_download,
            skip_catalogue=args.skip_catalogue,
            run_injection=args.run_injection,
            manifest_dir=args.manifest_dir,
        )
    except Exception as error:
        raise SystemExit(f"Pipeline failed: {error}") from error

    print(f"Manifest:         {manifest['manifest_path']}")
    print(f"Period:           {manifest['best_period_days']:.10f} d")
    print(f"Vetting:          {manifest['vetting_status']}")
    print(f"Catalogue:        {manifest['catalogue_status']}")
    print("Pipeline outputs do not constitute planetary confirmation.")


if __name__ == "__main__":
    main()