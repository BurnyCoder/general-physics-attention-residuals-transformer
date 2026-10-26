"""Global: evaluate post-hoc initialization-only baselines without changing the confirmatory study.

Local: reuse the frozen constructor, test evaluator, exclusive budget, and watchdog.
Sources: docs/untrained-baselines.md, https://docs.pytorch.org/docs/2.14/notes/randomness.html,
and https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.eval.
"""
import hashlib  # Local: fingerprint actual initialized tensor bytes before and after inference.
import json  # Local: preserve names, dtypes, and shapes in a deterministic hash contract.
import time  # Local: measured elapsed time includes construction, storage, and evaluation.
import torch  # Local: frozen inference uses the same PyTorch runtime as training.
from safetensors.torch import save_file  # Local: public initialized weights contain tensors, never executable pickles.
from .model import make_model  # Global: call exactly the same initialization function used by the original training runner.
from .evaluation import evaluate_model  # Global: identical windows, active fields, precision, metrics, and failure handling.
from .provenance import validate_frozen  # Global: no change to data, normalization, architecture, or original seed schedule.
from .runtime import Budget, BudgetExceeded, LOG, PROJECT, digest, read_json, revision, root, stamp, write_json  # Global: original ledger and complete provenance.


def state_hash(model, shared=False):
    """Local: hash canonical tensor metadata and bytes; global: detect updates and verify shared initialization."""
    checksum = hashlib.sha256()  # Local: standard hash is independent of checkpoint container timestamps.
    for name, tensor in sorted(model.state_dict().items()):
        if shared and name.startswith("attention_blocks.reads."):
            continue  # Local: routing parameters are absent from the original backbone and are compared separately.
        value = tensor.detach().cpu().contiguous()  # Local: explicit CPU bytes avoid device/stride-dependent serialization.
        checksum.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())  # Global: distinct names, shapes, and dtypes cannot share a misleading concatenation.
        checksum.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())  # Local: byte views support every tensor dtype, including BF16.
    return checksum.hexdigest()  # Local: portable fingerprint accompanies saved initialized weights.


def run_untrained(spec, frozen, prepared, records, budget, device="cuda"):
    """Global: one zero-update evaluation with no optimizer or checkpoint load; local: retain completed and failed attempts."""
    name = f"{spec['mode']}_seed{spec['seed']}"  # Local: retain original model/seed identities under a separate namespace.
    folder = root() / "untrained/runs" / name  # Global: initialized controls cannot overwrite trained checkpoints.
    previous = read_json(folder / "status.json", {})  # Local: resumability is at completed-run boundaries.
    if previous:
        LOG.info("Untrained control %s already recorded: %s; not repeated", name, previous["status"])  # Global: failures and interrupted attempts are not silently replaced.
        return previous  # Local: repeated phase execution skips recorded scientific attempts.
    folder.mkdir(parents=True, exist_ok=True)  # Local: make only this explicit project-owned artifact destination.
    started, model = time.monotonic(), None  # Local: include initialization and serialization in conservative phase cost.
    status = dict(spec, run="untrained_" + name, training_updates=0, post_hoc=True, status="running", started=stamp(), commit=revision(), data_sha256=frozen["data_sha256"], model_source_sha256=digest(PROJECT / "src/gphyt_attnres_study/model.py"))  # Global: distinguish fresh reconstruction from a historical saved step-zero checkpoint.
    write_json(folder / "status.json", status)  # Local: a crash or forced budget stop leaves a visible unfinished attempt.
    try:
        model = make_model(frozen["config"], spec["mode"], spec["seed"]).eval().requires_grad_(False)  # Global: same constructor/seed, zero training updates, no learned checkpoint restoration.
        for parameter_name, value in model.state_dict().items():
            if parameter_name.startswith("attention_blocks.reads."):
                expected = torch.zeros_like(value) if parameter_name.endswith(".query") else torch.ones_like(value)  # Local: pseudo-queries start at zero and key RMS scales at one.
                if not torch.equal(value, expected):
                    raise RuntimeError("Routing initialization differs from the specified zero-query/unit-scale baseline")  # Global: never quietly evaluate a learned routing state.
        status.update(initial_state_sha256=state_hash(model), shared_backbone_sha256=state_hash(model, shared=True))  # Local: record actual initial tensor identities before device transfer.
        for path in (root() / "untrained/runs").glob(f"*_seed{spec['seed']}/status.json"):
            other = read_json(path, {})  # Local: previously constructed variants of this seed define the paired backbone.
            if other.get("shared_backbone_sha256") not in (None, status["shared_backbone_sha256"]):
                raise RuntimeError("Shared backbone initialization changed across architecture controls")  # Global: reject a broken paired-initialization comparison.
        save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(folder / "model.safetensors"))  # Local: retain the exact evaluated initialization for independent reproduction.
        status["checkpoint_sha256"] = digest(folder / "model.safetensors")  # Global: tie evaluation to exact public tensor bytes.
        write_json(folder / "status.json", status)  # Local: persist initialized-weight provenance before potentially lengthy forecasts.
        model.to(device)  # Local: production uses CUDA; isolated orchestration tests use tiny CPU fixtures.
        LOG.info("Untrained control start: %s updates=0 backbone=%s", name, status["shared_backbone_sha256"])  # Global: clearly identify the additional scientific work.
        result = evaluate_model(model, prepared, records, budget, capture=True, label="untrained_" + name)  # Global: all official test trajectories use the unchanged causal rollout evaluator.
        if device == "cuda":
            torch.cuda.synchronize()  # Local: queued device work must finish before timing or final hashing.
        status["final_state_sha256"] = state_hash(model)  # Local: include all buffers as well as parameters in the no-update check.
        if status["final_state_sha256"] != status["initial_state_sha256"]:
            raise RuntimeError("Initialized model state changed during evaluation")  # Global: an accidentally updated model cannot be published as untrained.
        status.update(status="complete", ended=stamp(), seconds=time.monotonic() - started)  # Local: completion includes saved finite or undefined scientific endpoints.
        result.update(status, intervention=None, initialization="reconstructed_from_original_constructor_and_seed")  # Global: disclose reconstruction rather than claiming an original archived initialization.
        write_json(root() / "untrained/evaluation" / f"{name}.json", result)  # Local: separate raw results cannot enter the original primary analysis discovery.
        LOG.info("Untrained control complete: %s seconds=%.2f datasets=%s", name, status["seconds"], result["datasets"])  # Global: measured outcomes and failures remain visible.
    except Exception as error:
        status.update(status="budget_stopped" if isinstance(error, BudgetExceeded) else "failed", error=repr(error), ended=stamp(), seconds=time.monotonic() - started)  # Global: implementation failures and allocation stops remain distinct recorded outcomes.
        LOG.exception("Untrained control failed: %s", name)  # Local: preserve the entire traceback for transparent repair.
        raise  # Global: do not continue or retry a failed scientific run silently.
    finally:
        write_json(folder / "status.json", status)  # Local: every normal, divergent, or failed attempt retains its metadata.
        del model  # Local: release the current model before the next architecture.
        if device == "cuda":
            torch.cuda.empty_cache()  # Local: bound device memory without changing any weights or outcomes.
    return status  # Local: tests and phase orchestration inspect the durable outcome.


def untrained():
    """Global: evaluate the original 18 architecture/seed identities at initialization; local: no new confirmatory seeds or training."""
    frozen, prepared = read_json(root() / "frozen.json"), read_json(root() / "prepared.json")  # Local: the existing freeze defines the additional control matrix.
    validate_frozen(frozen, prepared)  # Global: unchanged scientific data and model settings are mandatory.
    pending = [spec for spec in frozen["order"] if not (root() / "untrained/runs" / f"{spec['mode']}_seed{spec['seed']}" / "status.json").exists()]  # Local: each recorded attempt is preserved, including failures.
    if not pending:
        LOG.info("All initialization-only control attempts are already recorded; no GPU execution")  # Global: a repeated command cannot consume more compute or replace outcomes.
        return  # Local: no budget lease or CUDA context is needed for a completed matrix.
    records = [record for record in prepared["records"] if record["split"] == "test"]  # Global: evaluate the same complete official test population as the trained models.
    with Budget("evaluate") as budget:
        from .training import gpu_setup  # Local: reuse the original deterministic CUDA/BF16 setup, without constructing an optimizer.
        gpu_setup()  # Global: all device operations follow exclusive ledger acquisition and watchdog launch.
        for spec in pending:
            budget.check()  # Local: preserve the original two-hour evaluation allocation and global 24-hour cap.
            run_untrained(spec, frozen, prepared, records, budget)  # Global: fixed order and seeds are independent of the newly observed outcomes.
