"""Global: initialization-only controls must never load learned weights or update parameters.

Local: tiny CPU fixtures exercise orchestration without executing a scientific model.
Sources: https://docs.pytest.org/en/stable/how-to/monkeypatch.html and docs/untrained-baselines.md.
"""
import pytest  # Local: failures must remain explicit outcomes.
import torch  # Local: tiny tensors expose any accidental weight mutation.
from gphyt_attnres_study import untrained  # Global: test the production initialization-only phase.
from gphyt_attnres_study.runtime import read_json, write_json  # Local: inspect durable records rather than internal bookkeeping.


def fixture_run(tmp_path, monkeypatch, mutate=False):
    """Local: isolate the artifact tree and substitute a tiny constructor/evaluator; global: no fixture enters study evidence."""
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path))  # Local: every generated file is test-owned.
    seen = []  # Local: record constructor arguments to protect the frozen seed/config contract.
    def constructor(cfg, mode, seed):
        """Local: deterministic CPU model with no checkpoint loading or optimizer."""
        seen.append((cfg, mode, seed))  # Global: construction must receive exactly the frozen inputs.
        model = torch.nn.Linear(2, 2)  # Local: fixture size is unrelated to the research architecture.
        def reject(*args, **kwargs):
            """Global: any attempt to restore trained weights invalidates this baseline."""
            raise AssertionError("untrained control loaded a checkpoint")  # Local: fail on the forbidden execution path.
        model.load_state_dict = reject  # Local: test the loaded model object itself, not just imports.
        return model  # Local: production code must switch this initially training-mode model to evaluation mode.
    def evaluate(model, prepared, records, budget, capture, label):
        """Local: inspect inference-only state and optionally simulate an erroneous weight update."""
        assert not model.training and not any(p.requires_grad for p in model.parameters())  # Global: initialize and freeze, never optimize.
        assert capture and label == "untrained_standard_seed2" and records == ["test-record"]  # Global: reuse ordinary evaluation with a separate preview identity.
        if mutate:
            model.weight.add_(1)  # Local: a regression must be detected before publishing metrics.
        return dict(datasets={"fixture": {"12": 1.}}, trajectories=[], routing=[])  # Local: known test values remain inside tmp_path.
    monkeypatch.setattr(untrained, "make_model", constructor)  # Local: do not execute the full scientific architecture in a fixture.
    monkeypatch.setattr(untrained, "evaluate_model", evaluate)  # Local: inspect the real orchestration boundary.
    frozen = dict(config={"fixture": True}, data_sha256="fixture-manifest")  # Local: no production configuration is overwritten.
    return seen, dict(mode="standard", seed=2), frozen  # Local: one planned identity with a distinctive seed.


def test_untrained_run_has_zero_updates_and_is_idempotent(tmp_path, monkeypatch):
    """Global: one initialization-only evaluation is durable and cannot silently rerun."""
    seen, spec, frozen = fixture_run(tmp_path, monkeypatch)  # Local: prepare an isolated inference-only run.
    result = untrained.run_untrained(spec, frozen, {}, ["test-record"], None, device="cpu")  # Local: exercise production save/metadata logic without CUDA.
    assert result["training_updates"] == 0 and result["post_hoc"] and result["status"] == "complete"  # Global: the control cannot masquerade as a trained or preregistered result.
    assert seen == [(frozen["config"], "standard", 2)]  # Global: reuse the exact architecture configuration and seed.
    assert result["initial_state_sha256"] == result["final_state_sha256"]  # Global: actual parameter bytes must remain unchanged through evaluation.
    assert (tmp_path / "untrained/evaluation/standard_seed2.json").exists() and not (tmp_path / "evaluation").exists()  # Global: original trained-model evaluations are never overwritten.
    assert (tmp_path / "untrained/runs/standard_seed2/model.safetensors").exists()  # Local: the initialization can be reproduced independently from saved weights.
    untrained.run_untrained(spec, frozen, {}, ["test-record"], None, device="cpu")  # Local: repeating a completed phase is a no-op for this run.
    assert len(seen) == 1  # Global: completed scientific evaluations are not silently repeated.


def test_untrained_weight_mutation_is_recorded_and_not_retried(tmp_path, monkeypatch):
    """Global: an accidental update invalidates the control and remains visible."""
    seen, spec, frozen = fixture_run(tmp_path, monkeypatch, mutate=True)  # Local: inject one specific implementation failure.
    with pytest.raises(RuntimeError, match="changed"):
        untrained.run_untrained(spec, frozen, {}, ["test-record"], None, device="cpu")  # Global: refuse to publish a purported untrained result after mutation.
    status = read_json(tmp_path / "untrained/runs/standard_seed2/status.json")  # Local: inspect the durable failure record.
    assert status["status"] == "failed" and status["training_updates"] == 0  # Global: zero intended updates does not excuse mutated weights.
    assert not (tmp_path / "untrained/evaluation/standard_seed2.json").exists()  # Local: invalid scores cannot enter later tables.
    untrained.run_untrained(spec, frozen, {}, ["test-record"], None, device="cpu")  # Local: resume preserves the failed attempt.
    assert len(seen) == 1  # Global: failure is not silently replaced by a fresh random model.


def test_state_hash_detects_changes_and_ignores_routing_for_shared_hash():
    """Global: paired backbone checks separate common weights from routing initialization."""
    model = torch.nn.Module()  # Local: a tiny state dictionary exercises the same naming contract as GPhyT.
    model.backbone = torch.nn.Linear(2, 2)  # Local: this parameter belongs in both hashes.
    model.attention_blocks = torch.nn.Module()  # Local: use the real adapter's routing namespace.
    model.attention_blocks.reads = torch.nn.Linear(2, 2)  # Local: only this namespace is excluded from shared-backbone comparisons.
    initial, shared = untrained.state_hash(model), untrained.state_hash(model, shared=True)  # Local: preserve original fingerprints.
    with torch.no_grad():
        model.attention_blocks.reads.weight.add_(1)  # Local: mimic a routing-only difference.
    assert untrained.state_hash(model) != initial and untrained.state_hash(model, shared=True) == shared  # Global: architecture-specific routing cannot create a false backbone mismatch.
    with torch.no_grad():
        model.backbone.weight.add_(1)  # Local: now alter a genuinely shared tensor.
    assert untrained.state_hash(model, shared=True) != shared  # Global: a different initial backbone must be detectable.


def test_recorded_matrix_does_not_acquire_gpu_budget(tmp_path, monkeypatch):
    """Global: resuming completed or failed controls cannot repeat inference or consume another lease."""
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path))  # Local: isolate all recorded attempts.
    order = [dict(mode="standard", seed=0), dict(mode="block", seed=0)]  # Local: two terminal identities exercise both success and failure.
    write_json(tmp_path / "frozen.json", dict(order=order))  # Local: only the orchestration-relevant fixture schema is needed.
    write_json(tmp_path / "prepared.json", {})  # Local: the constructor must never read any arrays in this completed-matrix case.
    monkeypatch.setattr(untrained, "validate_frozen", lambda *args: None)  # Local: scientific provenance validation is covered independently.
    for spec, outcome in zip(order, ["complete", "failed"]):
        write_json(tmp_path / "untrained/runs" / f"{spec['mode']}_seed0" / "status.json", dict(status=outcome))  # Global: failed attempts must remain as visible as successes.
    def forbidden(*args, **kwargs):
        """Local: reject any attempt to open another GPU session."""
        raise AssertionError("completed matrix acquired GPU budget")  # Global: idempotency protects the original hard allowance.
    monkeypatch.setattr(untrained, "Budget", forbidden)  # Local: fail before any real CUDA or watchdog access could occur.
    untrained.untrained()  # Global: a repeat CLI phase is a safe no-op for an already recorded matrix.
