"""Global: independent checks of leakage, normalization, sampling, and budget safety."""
import numpy as np  # Local: small arrays represent known physical fields.
import pytest  # Local: exceptions are explicit parts of the contract.
from gphyt_attnres_study.data import canonical_fields, normalization, window_starts, validate_splits  # Global: public data adapter functions.
from gphyt_attnres_study.runtime import Budget, BudgetExceeded, read_json  # Global: use the same ledger implementation as training.


def test_field_mapping_keeps_buoyancy_distinct():
    """Local: synthetic values distinguish field meanings; global: never rename buoyancy as temperature."""
    fields = {"pressure": np.ones((3, 2, 2)), "buoyancy": np.full((3, 2, 2), 7), "velocity": np.zeros((3, 2, 2, 2))}  # Local: a convection-shaped fixture.
    array, active = canonical_fields(fields)  # Local: production canonical ordering.
    assert active == [True, False, True, True, True]  # Global: density is absent, buoyancy is present.
    assert np.all(array[..., 2] == 7) and np.all(array[..., 1] == 0)  # Local: values remain in the correct channels.


def test_normalization_is_fitted_to_training_only(tmp_path):
    """Local: extreme test values must not affect fitted statistics."""
    records = []  # Local: explicit split labels drive the normalization filter.
    for split, value in [("train", 2.), ("valid", 100.), ("test", 1000.)]:
        path = tmp_path / f"{split}.npy"  # Local: each split owns a separate trajectory.
        np.save(path, np.full((5, 2, 2, 5), value, dtype=np.float32))  # Local: known means expose leakage.
        records.append(dict(dataset="fixture", split=split, path=str(path), active=[True] * 5))  # Local: production manifest shape.
    stats = normalization(records)["fixture"]  # Global: fit through the same function used for real data.
    np.testing.assert_allclose(stats["mean"], [2.] * 5)  # Local: validation and test values were excluded.
    assert min(stats["std"]) > 0  # Local: constant fields cannot divide by zero.


def test_duplicate_splits_abort():
    """Global: an identical complete trajectory cannot cross splits."""
    records = [dict(dataset="d", split="train", sha256="same", initial_sha256="i"), dict(dataset="d", split="test", sha256="same", initial_sha256="i")]  # Local: deliberately contaminated fixture.
    with pytest.raises(ValueError, match="duplicate"):
        validate_splits(records)  # Global: fail before training on contaminated data.


def test_windows_never_read_future_targets():
    """Local: fixed starts leave room for 4 context plus 24 forecast frames."""
    assert window_starts(101) == [0, 36, 73]  # Local: floor fractions of the last valid zero-based start.
    assert max(window_starts(101)) + 4 + 24 == 101  # Global: forecast targets remain within the trajectory.


def test_budget_accounts_failure_and_refuses_excess(tmp_path):
    """Local: a deliberately tiny cap checks the failure path without GPU execution."""
    path = tmp_path / "budget.json"  # Local: tests never consume the real study ledger.
    with pytest.raises(BudgetExceeded):
        with Budget("fixture", path, cap_seconds=0):
            pass  # Local: entry must fail before computational work.
    assert read_json(path)["sessions"]  # Global: rejected attempts remain visible.
