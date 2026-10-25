"""Global: paired, resumable fixed-update optimization of the pinned physics model.

Local: checkpoint model, optimizer, and independent data RNG together.
Sources: https://docs.pytorch.org/docs/stable/amp.html and
https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html.
"""
import gc  # Local: release completed models before the next budgeted experiment.
import json  # Local: append complete, portable step records.
import math  # Local: the frozen schedule is an explicit cosine formula.
from pathlib import Path  # Local: model artifacts remain under the configured root.
import time  # Local: measure synchronized end-to-end optimizer-update cost.
import numpy as np  # Local: randomization and data sampling use independent generators.
import torch  # Local: official optimizer, autocast, and checkpoint APIs.
from safetensors.torch import save_file  # Local: public final weights contain tensors, not executable pickle objects.
from .data import Sampler  # Global: identical sampling code serves every variant.
from .model import make_model  # Global: model differences are confined to depth routing.
from .runtime import Budget, BudgetExceeded, LOG, PROJECT, config, digest, read_json, revision, root, stamp, write_json  # Global: shared study contracts.


def masked_loss(prediction, target, active):
    """Local: mean squared error per active field, then per example; global: equal example weights."""
    fields = (prediction.float() - target.float()).square().mean(dim=(1, 2, 3))  # Local: reduce time and space without mixing channels.
    return ((fields * active).sum(-1) / active.sum(-1)).mean()  # Local: datasets with more channels do not receive larger weights.


def learning_rate(step, updates, cfg):
    """Local: one-based update schedule; global: every run receives the same learning rates."""
    warmup = max(1, math.ceil(updates * cfg["warmup_fraction"]))  # Local: short profiles still have a nonzero warmup.
    if step <= warmup:
        return cfg["learning_rate"] * step / warmup  # Local: linear warmup reaches the specified peak.
    progress = (step - warmup) / max(1, updates - warmup)  # Local: final update corresponds exactly to cosine phase one.
    return cfg["minimum_lr"] + .5 * (cfg["learning_rate"] - cfg["minimum_lr"]) * (1 + math.cos(math.pi * progress))  # Local: no untracked scheduler state.


def save_checkpoint(path, model, optimizer, rng, step, metadata):
    """Local: atomic trusted-local resume snapshot; global: retain optimizer and sample continuity."""
    path = Path(path)  # Local: normalize the caller's destination.
    path.parent.mkdir(parents=True, exist_ok=True)  # Local: create the run-owned directory.
    temporary = path.with_suffix(".tmp")  # Local: never overwrite the last valid checkpoint midway through serialization.
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), sample_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [], step=step, metadata=metadata), temporary)  # Global: capture every stochastic continuation state used by this runner.
    temporary.replace(path)  # Local: publish a complete checkpoint atomically.


def restore_checkpoint(path, model, optimizer, rng, device):
    """Local: restore only this project's trusted resume files; global: public downloads use safetensors."""
    state = torch.load(path, map_location=device, weights_only=False)  # Local: optimizer/RNG metadata requires trusted pickle loading.
    model.load_state_dict(state["model"])  # Local: restore all backbone and routing parameters.
    optimizer.load_state_dict(state["optimizer"])  # Local: preserve Adam moments and per-parameter step counters.
    rng.bit_generator.state = state["sample_rng"]  # Local: the next minibatch matches uninterrupted execution.
    torch.set_rng_state(state["torch_rng"].cpu())  # Local: restore CPU RNG without changing tensor device assumptions.
    if device != "cpu" and state["cuda_rng"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])  # Local: continue any CUDA random operations consistently.
    return state["step"], state["metadata"]  # Global: the caller validates frozen metadata before continuing.


def gpu_setup():
    """Local: verify the approved device/precision; global: fail instead of silently moving training to CPU."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This study requires a CUDA GPU with BF16 support")  # Global: a different compute regime is not an equivalent experiment.
    torch.backends.cudnn.benchmark = False  # Local: avoid input-dependent autotuning and nondeterministic algorithm selection.
    torch.use_deterministic_algorithms(True)  # Global: unsupported nondeterminism fails visibly instead of silently changing repeatability.
    torch.backends.cuda.matmul.allow_tf32 = False  # Local: FP32 operations remain FP32; autocast controls BF16 explicitly.
    LOG.info("GPU=%s torch=%s CUDA=%s", torch.cuda.get_device_name(), torch.__version__, torch.version.cuda)  # Global: exact runtime enters the timestamped log.


def optimizer_for(model, cfg):
    """Local: official AdamW with fixed defaults; global: routing has the same optimization policy as the backbone."""
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], betas=(.9, .999), eps=1e-8, foreach=False)  # Local: explicit settings avoid implementation-dependent optimizer selection.


def train_step(model, optimizer, sampler, cfg, microbatch, budget):
    """Local: accumulate equal microbatches; global: every update represents sixteen examples."""
    model.train()  # Local: upstream dropout is zero, but mode remains semantically correct.
    optimizer.zero_grad(set_to_none=True)  # Local: clear gradients before accumulating the effective batch.
    accumulation = cfg["batch_size"] // microbatch  # Local: selected candidates divide the fixed effective batch exactly.
    losses = []  # Local: detached microbatch losses support honest training curves.
    for _ in range(accumulation):
        budget.check()  # Global: refuse another GPU work unit when the cap is exhausted.
        x, y, active = (value.to("cuda") for value in sampler.batch(microbatch))  # Local: paired CPU data schedules do not depend on architecture timing.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = masked_loss(model(x), y, active)  # Local: reduction casts to FP32 for stable accumulation.
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")  # Global: retain failure instead of substituting a finite value.
        (loss / accumulation).backward()  # Local: accumulated gradients equal the effective-batch mean.
        losses.append(float(loss.detach()))  # Local: synchronize the scalar so timing includes completed work.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)  # Local: clip once after accumulation.
    optimizer.step()  # Local: Adam counters advance once per effective batch.
    torch.cuda.synchronize()  # Global: measured duration cannot omit queued GPU kernels.
    budget.check()  # Local: persist timing after the bounded update.
    return float(np.mean(losses)), float(norm)  # Local: both fit and stability enter the log.


@torch.no_grad()
def validation_loss(model, prepared, seed, microbatch, budget):
    """Local: a fixed validation sample; global: monitoring never changes training schedules or checkpoint choice."""
    sampler = Sampler(prepared, seed + 100000, split="valid")  # Local: recreate the same validation RNG for each checkpoint.
    model.eval()  # Local: inference semantics for the monitoring pass.
    values = []  # Local: this is secondary monitoring, not the preregistered test endpoint.
    for _ in range(16 // microbatch):
        budget.check()  # Local: validation consumes the same GPU ledger.
        x, y, active = (value.to("cuda") for value in sampler.batch(microbatch))  # Local: validation trajectories remain outside training.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values.append(float(masked_loss(model(x), y, active)))  # Local: no optimizer or gradients enter monitoring.
    return float(np.mean(values))  # Global: never choose the best validation checkpoint post hoc.


def execution_order(cfg):
    """Local: fixed randomization before outcomes; global: primary pairs precede exploratory controls."""
    rng = np.random.default_rng(20261025)  # Local: execution order is reproducible independently of model seeds.
    order = [(mode, seed) for seed in cfg["seeds"] for mode in rng.permutation(["standard", "block"])]  # Local: randomize thermal/time order within each primary pair.
    order += [(mode, seed) for seed in cfg["control_seeds"] for mode in rng.permutation(["full", "uniform_block"])]  # Global: exploratory work cannot displace unfinished primary pairs.
    return [dict(mode=str(mode), seed=seed) for mode, seed in order]  # Local: JSON-compatible frozen run matrix.


def run_one(spec, frozen, prepared, budget):
    """Local: resume or train one fixed-horizon run; global: failures and final checkpoints remain auditable."""
    cfg, mode, seed = frozen["config"], spec["mode"], spec["seed"]  # Local: consume the frozen configuration, not mutable defaults.
    folder = root() / "runs" / f"{mode}_seed{seed}"  # Local: one durable directory per planned experiment.
    folder.mkdir(parents=True, exist_ok=True)  # Local: create the run folder before any recorded work.
    previous = read_json(folder / "status.json", {})  # Local: completed runs are not accidentally retrained.
    if previous.get("status") in {"complete", "failed"}:
        return previous  # Global: a scientific failure is not silently retried with new settings.
    model = make_model(cfg, mode, seed).cuda()  # Local: shared initialization is created before device transfer.
    optimizer, sampler = optimizer_for(model, cfg), Sampler(prepared, seed)  # Local: optimizer and sample RNG are fresh or jointly restored.
    metadata = dict(spec, updates=frozen["updates"], microbatch=frozen["microbatch"], freeze_sha256=digest(root() / "frozen.json"), commit=revision())  # Global: bind the run to its preregistered horizon.
    step = 0  # Local: update indices are one-based after the first completed optimizer step.
    checkpoint = folder / "resume.pt"  # Local: this file is private trusted-local continuation state.
    if checkpoint.exists():
        step, old = restore_checkpoint(checkpoint, model, optimizer, sampler.rng, "cuda")  # Local: resume both optimizer and data sequence.
        if old["freeze_sha256"] != metadata["freeze_sha256"]:
            raise ValueError("Frozen training settings changed across resume")  # Global: no mixed-protocol checkpoint may continue.
    started = time.monotonic()  # Local: record this attempt's end-to-end run cost.
    torch.cuda.reset_peak_memory_stats()  # Local: measure each run's peak independently.
    status = dict(metadata, status="running", started=stamp(), step=step, prior_seconds=previous.get("seconds", 0))  # Local: retain interrupted-run timing in the budget ledger too.
    write_json(folder / "status.json", status)  # Global: dashboard sees the active experiment immediately.
    try:
        with (folder / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) as stream:
            for step in range(step + 1, frozen["updates"] + 1):
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(step, frozen["updates"], cfg)  # Local: reconstruct schedule exactly from the completed update number.
                tick = time.monotonic()  # Local: include sample IO, transfer, backward, and optimizer work.
                loss, gradient = train_step(model, optimizer, sampler, cfg, frozen["microbatch"], budget)  # Global: identical update semantics for all variants.
                row = dict(time=stamp(), step=step, loss=loss, gradient_norm=gradient, lr=optimizer.param_groups[0]["lr"], seconds=time.monotonic() - tick)  # Local: store every update without truncation.
                if step % cfg["validation_every"] == 0:
                    row["validation_loss"] = validation_loss(model, prepared, seed, frozen["microbatch"], budget)  # Local: monitoring does not alter stopping or selection.
                stream.write(json.dumps(row, allow_nan=False) + "\n")  # Global: a crash retains all completed measurements.
                if step % cfg["log_every"] == 0 or step == 1:
                    LOG.info("%s step=%d/%d loss=%.6g grad=%.4g seconds=%.3f", folder.name, step, frozen["updates"], loss, gradient, row["seconds"])  # Local: readable realtime progress.
                if step % cfg["checkpoint_every"] == 0 or step == frozen["updates"]:
                    save_checkpoint(checkpoint, model, optimizer, sampler.rng, step, metadata)  # Local: checkpoint only after complete optimizer updates.
                    status.update(step=step, seconds=status["prior_seconds"] + time.monotonic() - started)  # Local: dashboard cost includes preceding attempts.
                    write_json(folder / "status.json", status)  # Global: provide a durable progress heartbeat.
        save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(folder / "model.safetensors"))  # Local: safely publish final fixed-update weights.
        status["status"] = "complete"  # Global: only a saved final checkpoint makes a run complete.
    except BudgetExceeded:
        status.update(status="budget_stopped", error="GPU allocation exhausted")  # Global: omissions stay visible and resume requires available budget.
        raise  # Local: stop the run matrix rather than spending the evaluation reserve.
    except Exception as error:
        status.update(status="failed", error=repr(error))  # Global: numerical/software failures remain part of the experiment record.
        LOG.exception("Run failed: %s", folder.name)  # Local: retain the full traceback in the timestamped log.
    finally:
        status.update(step=step, seconds=status["prior_seconds"] + time.monotonic() - started, peak_memory_bytes=torch.cuda.max_memory_allocated(), ended=stamp())  # Local: record cost for successes and failures.
        write_json(folder / "status.json", status)  # Global: terminal outcomes are durable.
        del optimizer, model  # Local: release references before the next run.
        gc.collect()  # Local: clear any released Python-owned tensor cycles.
        torch.cuda.empty_cache()  # Local: return unused cached allocations for the next variant.
    return status  # Local: orchestration can proceed without hiding failed runs.
