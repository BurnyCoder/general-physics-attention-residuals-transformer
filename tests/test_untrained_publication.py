"""Global: initialized releases must not be confused with trained or incomplete checkpoints.

Local: tiny byte fixtures test staging without constructing or running a model.
Source: https://huggingface.co/docs/huggingface_hub/en/guides/upload.
"""
from gphyt_attnres_study.publishing import stage_initializations  # Local: exercise the production allowlisted staging function.
from gphyt_attnres_study.runtime import digest, read_json, write_json  # Local: use exact checkpoint and JSON conventions.


def test_initialization_package_has_zero_updates_and_skips_incomplete_runs(tmp_path, monkeypatch):
    """Global: publish only verified initialized states, with their own unambiguous configuration."""
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path))  # Local: fixtures remain outside scientific outputs.
    run = tmp_path / "untrained/runs/standard_seed0"  # Local: match the real artifact layout.
    run.mkdir(parents=True)  # Local: create only the completed fixture's folder.
    (run / "model.safetensors").write_bytes(b"initialized fixture")  # Local: no tensor deserialization is needed for packaging.
    (run / "credentials.env").write_text("MUST_NOT_UPLOAD", encoding="utf-8")  # Global: unknown files must not enter the release.
    status = dict(status="complete", training_updates=0, initial_state_sha256="unchanged", final_state_sha256="unchanged", checkpoint_sha256=digest(run / "model.safetensors"))  # Local: successful execution metadata must match the copied bytes.
    write_json(run / "status.json", status)  # Local: real staging reads this same record.
    write_json(tmp_path / "untrained/evaluation/standard_seed0.json", status)  # Local: only a measured initialized checkpoint qualifies.
    write_json(tmp_path / "untrained/runs/standard_seed1/status.json", dict(status="failed"))  # Global: failed attempts remain visible but have no published weights.
    target = tmp_path / "release/standard"  # Local: the trained top-level configuration must not be reused for zero-update files.
    statuses = stage_initializations(target, "standard", dict(config={"model": {}}, order=[dict(mode="standard", seed=0),dict(mode="standard", seed=1)]))  # Local: same frozen identities, no invented seeds.
    assert len(statuses) == 2 and statuses[1]["status"] == "failed"  # Global: failures are preserved rather than silently removed.
    assert read_json(target / "untrained/config.json")["updates"] == 0  # Global: consumers cannot mistake this checkpoint for a trained model.
    assert (target / "untrained/seed0/model.safetensors").read_bytes() == b"initialized fixture"  # Local: preserve exact initialized bytes.
    assert not (target / "untrained/seed0/credentials.env").exists() and not (target / "untrained/seed1").exists()  # Global: only allowlisted completed artifacts are uploaded.
