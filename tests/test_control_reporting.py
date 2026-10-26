"""Global: random-control presentation must preserve evaluation scope and causal forecasting.

Local: known fixture scores and fields never become public experiment measurements.
Sources: docs/protocol.md and https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reindex.html.
"""
import numpy as np  # Local: exact fixture arrays expose future-frame leakage.
import pandas as pd  # Local: mirror the already matched intervention CSV schema.
import pytest  # Local: incompatible control comparisons must fail explicitly.
from gphyt_attnres_study.results import intervention_scores  # Global: production reduction must use matched-subset absolute scores.
from gphyt_attnres_study.comparisons import persistence_arrays, matched_fields  # Global: production preview uses only the final observed context frame.


def control_rows():
    """Local: two seeds with exactly known matched original and intervened errors."""
    return pd.DataFrame([dict(run=f"block_seed{seed}", dataset="fixture", horizon=12, intervention=kind, trajectories=4, original_nmse=seed + 1., intervened_nmse=(seed + 1.) * multiplier) for seed in (0, 1) for kind, multiplier in (("uniform", 2), ("shuffle", 3))])  # Local: both interventions share the same subset and original score.


def test_intervention_table_uses_matched_absolute_scores():
    """Global: report absolute diagnostic NMSE, not ratios or unrelated full-test errors."""
    table = intervention_scores(control_rows(), [0, 1]).set_index("mode")  # Local: expected seed IDs are fixed independently of observed rows.
    assert table.loc["block", "mean"] == 1.5  # Global: the reference is original Block on the diagnostic subset.
    assert table.loc["block_uniform", "mean"] == 3.  # Local: fixed equal routing increases the constructed absolute error.
    assert table.loc["block_shuffle", "mean"] == 4.5  # Local: the random control remains a separately labeled intervention.


def test_intervention_table_preserves_missing_seed():
    """Global: a missing random-control seed must not yield a favorable survivor-only mean."""
    rows = control_rows()  # Local: remove one planned intervention, not its baseline.
    rows = rows[~((rows.run == "block_seed1") & (rows.intervention == "shuffle"))]  # Local: one scientific outcome is unavailable.
    table = intervention_scores(rows, [0, 1]).set_index("mode")  # Global: the planned denominator stays two.
    assert pd.isna(table.loc["block_shuffle", "mean"])  # Global: undefined is not zero or an average over one seed.
    assert table.loc["block_shuffle", "n_planned"] == 2  # Local: preserve the missing seed in completion counts.


def test_intervention_table_rejects_different_references():
    """Global: equal-looking labels cannot hide different original subset scores."""
    rows = control_rows()  # Local: alter only the uniform intervention's reference error.
    rows.loc[0, "original_nmse"] = 99.  # Local: this would make one shared-reference row misleading.
    with pytest.raises(ValueError, match="reference"):
        intervention_scores(rows, [0, 1])  # Global: require actual agreement rather than averaging incompatible baselines.


def test_persistence_preview_has_no_future_frame_dependence(tmp_path):
    """Global: persistence must repeat observation four, never a forecast target."""
    trajectory = np.broadcast_to(np.arange(40, dtype=np.float32)[:, None, None, None], (40, 3, 2, 5)).copy()  # Local: future times have distinguishable values.
    path = tmp_path / "trajectory.npy"  # Local: all fixture data stay isolated.
    np.save(path, trajectory)  # Local: exercise the real read-only Sampler rather than a mocked normalizer.
    record = dict(dataset="fixture", split="test", parameter="fixed", length=40, path=str(path), active=[True, False, True, True, True])  # Local: include an absent field to verify the zero mask.
    prepared = dict(records=[record], normalization={"fixture": dict(mean=[1.] * 5, std=[2.] * 5)})  # Global: fixed training statistics do not depend on future observations.
    first = persistence_arrays(prepared, "fixture")  # Local: reconstruct only the already prescribed first case.
    trajectory[4:] = 1e6  # Local: modify every future value while leaving context unchanged.
    other = tmp_path / "changed_future.npy"  # Local: use a different path to avoid Windows memory-map sharing concerns.
    np.save(other, trajectory)  # Local: no original fixture is deleted or mutated on disk.
    record["path"] = str(other)  # Local: preserve the same case structure with only altered future frames.
    second = persistence_arrays(prepared, "fixture")  # Global: changing targets may change truth but cannot change persistence.
    np.testing.assert_array_equal(first["prediction"], second["prediction"])  # Global: the full 24-step forecast is causally invariant to future values.
    assert first["prediction"].shape == (24, 3, 2, 5) and np.all(first["prediction"][..., 0] == 1.)  # Local: observation index three normalizes to (3-1)/2.
    assert np.all(first["prediction"][..., 1] == 0.) and not np.array_equal(first["truth"], second["truth"])  # Local: absent fields remain zero while changed targets are retained.


def test_random_control_uses_same_case_validation(tmp_path):
    """Global: a shuffled-routing forecast must not bypass the common-target guard."""
    values = dict(truth=np.zeros((12, 3, 2, 5)), prediction=np.ones((12, 3, 2, 5)), active=np.ones(5, dtype=bool))  # Local: small complete numeric archives.
    np.savez(tmp_path / "block_seed0_fixture.npz", **values)  # Local: original trained model.
    values["truth"][0, 0, 0, 0] = 1.  # Local: mismatched target need not be on the displayed frame.
    np.savez(tmp_path / "block_seed0_shuffle_fixture.npz", **values)  # Local: deliberately incompatible random-control fixture.
    with pytest.raises(ValueError, match="incompatible"):
        matched_fields(tmp_path, "fixture", 2, runs=("block_seed0", "block_seed0_shuffle"))  # Global: generic control rendering must retain exact case validation.
