"""Integration-style test of pipeline orchestration without network access."""

from pathlib import Path
from types import SimpleNamespace

from transithunter import pipeline
from transithunter.catalogue import CrossmatchResult


def test_pipeline_runs_stages_and_writes_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The orchestrator connects all six core stages in the correct order."""
    calls: list[str] = []
    signal = SimpleNamespace(period_days=0.837429)

    def acquire(*args, **kwargs):
        calls.append("acquire")
        return SimpleNamespace(
            csv_path=tmp_path / "raw.csv",
            metadata_path=tmp_path / "raw.json",
            row_count=100,
        )

    def preprocess(*args, **kwargs):
        calls.append("preprocess")
        return (
            tmp_path / "clean.csv",
            tmp_path / "preprocess.json",
            tmp_path / "preprocess.png",
            SimpleNamespace(rows_output=90),
        )

    def detrend(*args, **kwargs):
        calls.append("detrend")
        return (
            tmp_path / "detrended.csv",
            tmp_path / "detrend.json",
            tmp_path / "detrend.png",
            SimpleNamespace(segment_count=2),
        )

    def search(*args, **kwargs):
        calls.append("bls")
        return (
            tmp_path / "candidate.json",
            tmp_path / "periodogram.csv",
            tmp_path / "bls.png",
            signal,
        )

    def vet(*args, **kwargs):
        calls.append("vet")
        return (
            tmp_path / "vet.json",
            tmp_path / "events.csv",
            tmp_path / "vet.png",
            {"status": "passed_limited_automated_checks", "flags": []},
        )

    def crossmatch(*args, **kwargs):
        calls.append("catalogue")
        return tmp_path / "catalogue.json", CrossmatchResult(
            status="known_confirmed_planet_signal",
            confirmed_matches=[],
            koi_matches=[],
            confirmed_rows=[],
            koi_rows=[],
        )

    monkeypatch.setattr(pipeline, "acquire_light_curve", acquire)
    monkeypatch.setattr(pipeline, "process_file", preprocess)
    monkeypatch.setattr(pipeline, "detrend_file", detrend)
    monkeypatch.setattr(pipeline, "search_file", search)
    monkeypatch.setattr(pipeline, "vet_file", vet)
    monkeypatch.setattr(pipeline, "crossmatch_file", crossmatch)

    manifest = pipeline.run_pipeline(
        "KIC 11904151",
        "Kepler",
        quarter=3,
        manifest_dir=tmp_path / "runs",
    )

    assert calls == ["acquire", "preprocess", "detrend", "bls", "vet", "catalogue"]
    assert manifest["catalogue_status"] == "known_confirmed_planet_signal"
    assert manifest["known_ephemeris_supplied"] is False
    assert Path(manifest["manifest_path"]).exists()