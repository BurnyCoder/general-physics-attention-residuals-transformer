"""Global: added baseline tables must preserve planned seeds and distinguish zero from trained updates.

Local: all numeric fixtures stay in isolated pytest folders.
Sources: docs/untrained-baselines.md and the existing results.aggregate_estimates contract.
"""
import pandas as pd  # Local: inspect the exact CSV-compatible summary schema.
import pytest  # Local: incompatible comparisons must be rejected.
from gphyt_attnres_study import untrained_reporting as reporting  # Global: production reductions feed README and paper.
from gphyt_attnres_study.runtime import write_json  # Local: use the same safe JSON persistence as real artifacts.


def setup_results(tmp_path, monkeypatch, seeds):
    """Local: construct a two-seed planned baseline with selected available results."""
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path / "artifacts"))  # Global: fixture scores cannot reach public scientific artifacts.
    monkeypatch.setattr(reporting, "PROJECT", tmp_path)  # Local: generated reports are also isolated.
    monkeypatch.setattr(reporting, "config", lambda: dict(seeds=[0,1], control_seeds=[0], data={"fixture": {}}))  # Local: two primary seeds and one exploratory seed exercise the planned denominators.
    for seed in seeds:
        write_json(tmp_path / "artifacts/untrained/evaluation" / f"standard_seed{seed}.json", dict(status="complete", post_hoc=True, training_updates=0, datasets={"fixture": {str(h): 2. + 2*seed for h in (1,12,24)}}))  # Local: known initial means are 2 or 3, depending on completeness.


def test_initialized_summary_requires_all_planned_seeds(tmp_path, monkeypatch):
    """Global: one available initialization cannot substitute for the original seed matrix."""
    setup_results(tmp_path, monkeypatch, [0])  # Local: seed one is deliberately missing.
    frames = reporting.untrained_tables()  # Local: exercise real file discovery and aggregation.
    row = frames["accuracy_summary"].query("mode == 'standard' and horizon == 12").iloc[0]  # Local: isolate the primary horizon.
    assert row.n_finite == 1 and row.n_planned == 2 and pd.isna(row["mean"])  # Global: no survivor-only comparison against a two-seed trained mean.


def test_initialized_and_trained_table_uses_matching_denominators(tmp_path, monkeypatch):
    """Global: compare initialized/trained scores only for the same planned seed counts."""
    setup_results(tmp_path, monkeypatch, [0,1])  # Local: both planned initialized errors are available.
    initial = reporting.untrained_tables()["accuracy_summary"].query("mode == 'standard' and horizon == 12")  # Local: fixed known mean is three.
    trained = initial.copy()  # Local: preserve identifiers and planned seed count.
    trained["mean"] = 8.  # Local: fixed trained error supports an independently known table entry.
    paired = reporting.initialization_comparison(initial, trained).iloc[0]  # Global: table columns must retain which state produced each score.
    assert paired.mean_initial == 3. and paired.mean_trained == 8.  # Local: no ratio or geometric transformation replaces the requested absolute scores.
    trained["n_planned"] = 3  # Local: a seemingly reasonable but unmatched trained comparison is invalid.
    with pytest.raises(ValueError, match="seed"):
        reporting.initialization_comparison(initial, trained)  # Global: differing populations cannot be displayed as paired initialization/training.


def test_initialized_table_rejects_trained_checkpoint_results(tmp_path, monkeypatch):
    """Global: moving a trained evaluation into the new folder cannot label it as untrained."""
    setup_results(tmp_path, monkeypatch, [0])  # Local: replace one expected zero-update record with invalid metadata.
    write_json(tmp_path / "artifacts/untrained/evaluation/standard_seed0.json", dict(training_updates=3175, post_hoc=True, datasets={}))  # Local: explicit trained update count must fail before aggregation.
    with pytest.raises(ValueError, match="zero-update"):
        reporting.untrained_tables()  # Global: refuse scientific mislabeling even if the filename looks correct.


def test_markdown_table_has_no_optional_dependency_and_keeps_undefined():
    """Global: report phase must work in the frozen environment without an undeclared tabulate dependency."""
    frame = pd.DataFrame({"Model": ["A|B", "missing"], "NMSE": [1.25, float("nan")]})  # Local: literal separators and absent values exercise readable output.
    text = reporting.markdown_table(frame)  # Local: shared rendering feeds README and individual experiment reports.
    assert "| Model | NMSE |" in text and "A\\|B | 1.25" in text and "missing | undefined" in text  # Global: missing measurements are never printed as zeros.
