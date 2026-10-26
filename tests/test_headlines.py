"""Global: headlines require complete measured comparisons; local: fixtures stay outside public reports."""
import csv  # Local: use the report CSV schema without model execution.
import pytest  # Local: exercise missing, divergent, and unfavorable evidence; https://docs.pytest.org/en/stable/how-to/parametrize.html.
from gphyt_attnres_study import headlines  # Global: test the shared production headline rather than copied prose.


def write_scores(folder, initialized=False, defect=None):
    """Local: save known nine-model comparisons; global: no favorable claim may ignore a comparator."""
    path = folder / ("untrained/accuracy_summary.csv" if initialized else "accuracy_summary.csv")  # Local: preserve the separate post-hoc evidence source.
    path.parent.mkdir(parents=True, exist_ok=True)  # Local: pytest owns the entire temporary destination.
    means = dict(standard=.4, block=.2, full=.3, uniform_block=.35, persistence=.5)  # Local: known ordering makes percentage assertions independent of implementation.
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "mode", "horizon", "n_finite", "n_planned", "mean"))  # Local: only fields used by the actual public summaries are needed.
        writer.writeheader()  # Local: exact schema keeps fixture parsing realistic.
        for mode, mean in means.items():
            if initialized and mode == "persistence" or defect == "missing" and mode == "full":
                continue  # Local: persistence has no parameter initialization; a missing model must prevent the global claim.
            count = 6 if mode in ("standard", "block") else 1 if mode == "persistence" else 3  # Local: reproduce the fixed seed counts.
            value = mean + 1 if initialized else mean  # Local: initialized scores are separate observations, not trained replacements.
            value = float("nan") if defect == "nonfinite" and mode == "full" else .1 if defect == "better" and mode == "full" else value  # Local: missing or superior controls must not disappear from ranking.
            writer.writerow(dict(dataset="turbulent_radiative_layer_2D", mode=mode, horizon=12, n_finite=count - (defect == "partial" and mode == "full"), n_planned=count, mean=value))  # Global: planned seeds govern the completeness check.


def test_headline_uses_complete_means_and_keeps_inference_scope(tmp_path, monkeypatch):
    """Global: favorable descriptive reporting must retain post-hoc and preregistration distinctions."""
    monkeypatch.setattr(headlines, "PROJECT", tmp_path)  # Local: prevent any access to the real study outputs.
    write_scores(tmp_path / "reports")  # Local: supply the five full-test trained/persistence rows.
    write_scores(tmp_path / "reports", initialized=True)  # Local: supply all four requested untrained baselines.
    result = headlines.radiative_highlight()  # Global: invoke the same computation used by every publication surface.
    assert result["block_nmse"] == .2 and result["standard_reduction_percent"] == 50  # Local: exact known arithmetic, not hard-coded study outcomes.
    assert result["persistence_reduction_percent"] == 60 and result["comparators"] == 9  # Local: all nine rows contribute to the claim.
    assert "post-hoc" in result["scope"] and "preregistered" in result["scope"]  # Global: headline placement cannot change the original inferential endpoint.
    assert "without Attention Residuals" in result["summary"]  # Local: identify the standard trained baseline explicitly.


@pytest.mark.parametrize("initialized,defect", [(False, "missing"), (True, "missing"), (False, "nonfinite"), (True, "nonfinite"), (False, "partial"), (True, "partial"), (False, "better"), (True, "better")])
def test_no_winner_claim_from_incomplete_or_unfavorable_scores(tmp_path, monkeypatch, initialized, defect):
    """Global: incomplete and contradictory evidence must suppress the prewritten positive headline."""
    monkeypatch.setattr(headlines, "PROJECT", tmp_path)  # Local: isolate every negative fixture.
    write_scores(tmp_path / "reports", defect=defect if not initialized else None)  # Local: independently challenge the original trained comparison.
    write_scores(tmp_path / "reports", initialized=True, defect=defect if initialized else None)  # Local: independently challenge the post-hoc controls.
    assert headlines.radiative_highlight() is None  # Global: never advertise an unsupported lowest-error result.
