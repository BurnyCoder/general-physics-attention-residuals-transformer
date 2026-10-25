"""Global: independent checks of leakage, normalization, sampling, and budget safety."""
import numpy as np  # Local: small arrays represent known physical fields.
import pytest  # Local: exceptions are explicit parts of the contract.
import h5py  # Local: exercise the actual official-format reader, not only detached mapping helpers.
import torch  # Local: compare tensor batches from paired sampling generators.
from gphyt_attnres_study.data import canonical_fields, normalization, window_starts, validate_splits  # Global: public data adapter functions.
from gphyt_attnres_study.runtime import Budget, BudgetExceeded, read_json  # Global: use the same ledger implementation as training.


def test_hdf5_to_normalized_paired_batches(tmp_path, monkeypatch):
    """Global: cover a user's preprocessing-to-sampling path without touching real study artifacts."""
    from gphyt_attnres_study.data import preprocess_file, Sampler  # Local: invoke production modules through their public API.
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path / "artifacts"))  # Local: isolate the complete workflow in the test directory.
    path, rng = tmp_path / "fixture.hdf5", np.random.default_rng(42)  # Local: reproducible finite HDF5 fields.
    with h5py.File(path, "w") as file:
        file["dimensions/x"], file["dimensions/y"] = np.arange(8), np.arange(8)  # Local: known uniform grid in the stored axis order.
        file["t0_fields/pressure"] = rng.normal(size=(1, 40, 8, 8)).astype("float32")  # Local: one scalar in the official field group.
        file["t0_fields/buoyancy"] = rng.normal(size=(1, 40, 8, 8)).astype("float32")  # Local: keep buoyancy physically distinct from missing density.
        file["t1_fields/velocity"] = rng.normal(size=(1, 40, 8, 8, 2)).astype("float32")  # Local: two vector components, explicit sample/time axes.
    records = preprocess_file(dict(path=str(path), filename=path.name, dataset="fixture", split="train", sha256="fixture-bytes", url="https://example.invalid/fixture"), [4, 4])  # Local: actual area downsampling and trajectory hashing.
    prepared = dict(records=records, normalization=normalization(records))  # Local: fit only the fixture's training trajectory.
    a, b = Sampler(prepared, 0), Sampler(prepared, 0)  # Global: same seed reproduces the complete sampling hierarchy.
    for left, right in zip(a.batch(4), b.batch(4)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)  # Local: exact paired samples independent of execution order.
    assert np.load(records[0]["path"]).shape == (40, 4, 4, 5)  # Global: output matches channels-last physics model input.
    assert records[0]["active"] == [True, False, True, True, True]  # Local: absent density remains explicit through the full adapter.


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


def test_segmented_partial_is_never_accepted_by_file_size(tmp_path, monkeypatch):
    """Global: an aria2 sparse file can have final length before its pieces are downloaded."""
    from types import SimpleNamespace  # Local: supply only the HTTP metadata used by the downloader.
    from gphyt_attnres_study import data  # Local: exercise the production completion gate.
    from gphyt_attnres_study.runtime import write_json  # Local: create the normal resumable identity record.
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path))  # Local: no real download or study file is modified.
    monkeypatch.setenv("ARIA2C", str(tmp_path / "missing-aria2.exe"))  # Local: force the HTTP-only fallback.
    folder = tmp_path / "raw/fixture/train"  # Local: mirror the owned raw-file directory.
    folder.mkdir(parents=True)  # Local: create an isolated partial-download fixture.
    (folder / "sample.part").write_bytes(b"holes")  # Local: length alone falsely looks complete.
    (folder / "sample.part.aria2").write_bytes(b"incomplete pieces")  # Global: this control file signals unfinished segmented data.
    write_json(folder / "sample.partial.json", dict(bytes=5, last_modified="fixed"))  # Local: match server identity so the test isolates segment completeness.
    monkeypatch.setattr(data.requests, "head", lambda *args, **kwargs: SimpleNamespace(headers={"Content-Length": "5", "Last-Modified": "fixed"}, raise_for_status=lambda: None))  # Local: deterministic metadata without a network request.
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)  # Local: retry policy need not delay a unit test.
    with pytest.raises(ValueError, match="segmented"):
        data.download_file(dict(dataset="fixture", split="train", filename="sample.hdf5", url="https://example.invalid/sample"))  # Global: incomplete bytes must never be promoted to accepted HDF5.
    assert not (folder / "sample.hdf5").exists()  # Local: the final filename remains reserved for verified completion.


def test_budget_watchdog_stops_a_noncooperating_work_unit(tmp_path):
    """Global: a hung work unit cannot defeat the hard cap by failing to call check()."""
    import os  # Local: pass only the isolated test artifact location to the child process.
    import subprocess  # Local: a watchdog test must not terminate the test runner itself.
    import sys  # Local: use the project's already configured interpreter.
    path = tmp_path / "watchdog.json"  # Local: the real study budget is never touched.
    code = "from gphyt_attnres_study.runtime import Budget; import os,time;\nwith Budget('fixture', os.environ['TEST_LEDGER'], cap_seconds=15.6): time.sleep(30)"  # Local: simulate one noncooperating CPU work unit without executing any GPU code.
    try:
        result = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, TEST_LEDGER=str(path)), timeout=17, capture_output=True, text=True)  # Global: independently bound the test even if the watchdog is absent.
    except subprocess.TimeoutExpired:
        pytest.fail("The budget watchdog did not stop the noncooperating work unit")  # Local: an absent hard stop is a meaningful regression failure.
    assert result.returncode != 0  # Local: process termination must not masquerade as successful work.
    import time  # Local: the independent helper writes its final accounting after process teardown.
    for _ in range(30):
        if read_json(path)["sessions"][-1]["status"] == "hard_budget_stop":
            break  # Local: completion is recognized from durable evidence rather than a fixed sleep.
        time.sleep(.1)  # Local: bound this file-publication race to three seconds without GPU execution.
    assert read_json(path)["sessions"][-1]["status"] == "hard_budget_stop"  # Global: hard termination must retain auditable budget evidence.


def test_normal_budget_exit_cancels_only_its_supervisor(tmp_path):
    """Global: completed work must not leave a stale process capable of terminating a later phase."""
    with Budget("fixture", tmp_path / "normal-budget.json") as budget:
        supervisor = budget.watchdog  # Local: retain the exact child process started by this lease.
        budget.check()  # Local: the normal cooperative path remains usable with independent enforcement.
    assert supervisor.poll() is not None  # Local: the helper has stopped when its owner releases the phase.
    assert read_json(tmp_path / "normal-budget.json")["sessions"][-1]["status"] == "complete"  # Global: normal completion remains distinct from a hard stop.
