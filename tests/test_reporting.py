"""Global: report regressions guard scientific comparisons and honest missing-data display.

Local: synthetic aggregate fixtures never enter study outputs or model training.
Sources: docs/protocol.md and https://docs.pytest.org/en/stable/how-to/monkeypatch.html.
"""
import pandas as pd  # Local: inspect the same portable tables consumed by the viewers.
from gphyt_attnres_study import analysis, web_export  # Global: exercise production result transformations.
from gphyt_attnres_study.runtime import write_json  # Local: strict JSON mirrors durable evaluation artifacts.


def test_interventions_compare_identical_trajectory_subsets(tmp_path, monkeypatch):
    """Global: unrelated high-error test trajectories must not bias intervention effect sizes."""
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path / "artifacts"))  # Local: keep fixtures away from experimental results.
    monkeypatch.setattr(analysis, "PROJECT", tmp_path)  # Local: generated CSV files stay isolated.
    (tmp_path / "reports").mkdir()  # Local: imitate the public report destination.
    def trajectory(identity, error):
        """Local: construct known per-trajectory endpoints for all three horizons."""
        return dict(dataset="fixture", parameter="p1", source_sha256=identity, nmse={str(h): error for h in (1, 12, 24)})  # Local: one identity is one independent trajectory within a dataset.
    baseline = dict(run="block_seed0", intervention=None, datasets={"fixture": {str(h): 51. for h in (1, 12, 24)}}, trajectories=[trajectory("selected", 2.), trajectory("outside", 100.)])  # Global: the whole-test average is deliberately very different.
    intervention = dict(run="block_seed0", intervention="uniform", datasets={"fixture": {str(h): 4. for h in (1, 12, 24)}}, trajectories=[trajectory("selected", 4.)])  # Local: routing doubles error on the same selected trajectory.
    folder = tmp_path / "artifacts/evaluation"  # Local: production discovery reads this directory.
    write_json(folder / "block_seed0.json", baseline)  # Local: original trained-model reference.
    write_json(folder / "block_seed0_uniform.json", intervention)  # Local: altered routing keeps the original model identity.
    analysis.secondary_tables()  # Global: compare only matched identities through the actual analysis function.
    table = pd.read_csv(tmp_path / "reports/interventions.csv")  # Local: check published rather than transient internal values.
    assert table.ratio.tolist() == [2.] * 3  # Global: using the whole-test mean would produce the incorrect ratio 4/51.
    assert table.trajectories.tolist() == [1] * 3  # Local: no hidden extra trajectories enter the counterfactual.


def test_public_tables_preserve_missing_values(tmp_path, monkeypatch):
    """Global: JSON exports must not turn pending or divergent measurements into zeros."""
    monkeypatch.setattr(web_export, "PROJECT", tmp_path)  # Local: isolate the public table reader.
    (tmp_path / "reports").mkdir()  # Local: create a normal report directory.
    (tmp_path / "reports/metrics.csv").write_text("run,nmse\npending,\nmeasured,0.25\n", encoding="utf-8")  # Local: distinguish a missing result from a measured positive error.
    rows = web_export.table_records("metrics.csv")  # Local: production CSV-to-JSON adaptation.
    assert rows == [dict(run="pending", nmse=None), dict(run="measured", nmse=.25)]  # Global: browsers can show missing states honestly.
