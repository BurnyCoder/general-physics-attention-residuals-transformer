"""Global: convert measured end-to-end costs into one frozen eighteen-run horizon.

Local: all profiling uses the same optimizer, data adapter, and validation implementation.
Sources: docs/protocol.md and https://docs.pytorch.org/docs/stable/notes/cuda.html.
"""
import gc  # Local: release profiling models between variants.
import time  # Local: synchronized wall time includes CPU data and checkpoint overhead.
import numpy as np  # Local: robust cost summaries from actual optimizer updates.
import torch  # Local: inspect CUDA allocation and handle memory exhaustion explicitly.
from .data import Sampler  # Global: profiling includes real memory-mapped data IO.
from .model import make_model  # Global: profile all architectures at the requested model scale.
from .provenance import data_fingerprint  # Global: scientific identity survives moving the prepared files to another host.
from .training import execution_order, gpu_setup, optimizer_for, save_checkpoint, train_step, validation_loss  # Global: no profiling-only surrogate computation.
from .runtime import Budget, LOG, PROJECT, config, digest, read_json, revision, root, stamp, write_json  # Global: publish frozen settings before confirmatory training.


def profile_variant(mode, microbatch, cfg, prepared, budget):
    """Local: warm then measure actual updates; global: retain each architecture's memory and runtime cost."""
    model = make_model(cfg, mode, 10000).cuda()  # Local: profiling initialization is outside confirmatory seeds 0--5.
    optimizer, sampler = optimizer_for(model, cfg), Sampler(prepared, 10000)  # Local: profile samples do not advance any confirmatory schedule.
    torch.cuda.reset_peak_memory_stats()  # Local: one isolated peak per candidate.
    durations = []  # Local: exclude two warmups, retain sixteen measured updates.
    try:
        for step in range(18):
            start = time.monotonic()  # Local: include CPU sampling and optimizer update.
            loss, norm = train_step(model, optimizer, sampler, cfg, microbatch, budget)  # Local: exactly the training code path.
            if step >= 2:
                durations.append(time.monotonic() - start)  # Local: warm cache and CUDA initialization do not dominate the estimate.
            LOG.info("profile mode=%s microbatch=%d step=%d loss=%.6g gradient=%.4g", mode, microbatch, step + 1, loss, norm)  # Global: every GPU profiling run remains visible.
        start = time.monotonic()  # Local: measure actual fixed validation overhead.
        validation = validation_loss(model, prepared, 10000, microbatch, budget)  # Local: validation uses only the official valid split.
        validation_seconds = time.monotonic() - start  # Local: include data loading and synchronization.
        start = time.monotonic()  # Local: checkpoint serialization also consumes training wall time.
        save_checkpoint(root() / "profile" / f"{mode}.pt", model, optimizer, sampler.rng, 18, {"profile": True})  # Local: exercise the same complete resume artifact as real runs.
        checkpoint_seconds = time.monotonic() - start  # Local: amortize this overhead in the common-update calculation.
        return dict(mode=mode, microbatch=microbatch, durations=durations, median_seconds=float(np.median(durations)), p90_seconds=float(np.quantile(durations, .9)), validation_seconds=validation_seconds, checkpoint_seconds=checkpoint_seconds, peak_memory_bytes=torch.cuda.max_memory_allocated(), parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), validation_loss=validation)  # Global: measured costs are distinguishable from budget projections.
    finally:
        del optimizer, model  # Local: release large CUDA references even when a candidate runs out of memory.
        gc.collect()  # Local: collect any graph cycles before the next candidate.
        torch.cuda.empty_cache()  # Local: return unused cached blocks to the allocator.


def profile():
    """Local: select only microbatch and common updates; global: do not tune scientific settings from test outcomes."""
    if (root() / "frozen.json").exists():
        return read_json(root() / "frozen.json")  # Global: a frozen study is immutable across resumes.
    cfg, prepared = config(), read_json(root() / "prepared.json")  # Local: require completed verified data before cost measurement.
    if prepared is None:
        raise RuntimeError("Run preparation before profiling")  # Local: synthetic IO is not a valid real-data cost estimate.
    measurements, failures = [], []  # Local: OOM candidates are retained as profiling outcomes.
    with Budget("profile") as budget:
        gpu_setup()  # Local: initialize CUDA only inside the GPU accounting context.
        for microbatch in cfg["microbatch_candidates"]:
            measurements = []  # Local: all variants must pass at one common microbatch.
            try:
                for mode in ("standard", "block", "full", "uniform_block"):
                    measurements.append(profile_variant(mode, microbatch, cfg, prepared, budget))  # Local: inspect all four paths before freezing settings.
                break  # Global: use the largest approved microbatch that fits every variant.
            except torch.OutOfMemoryError as error:
                failures.append(dict(microbatch=microbatch, error=str(error), time=stamp()))  # Global: memory failures remain in the profiling record.
                LOG.warning("microbatch=%d exceeded memory; trying the next prespecified candidate", microbatch)  # Local: only the approved fallback changes.
        if len(measurements) != 4:
            raise RuntimeError("No approved common microbatch fits all variants")  # Global: do not quietly omit the costly comparison.
    counts = {"standard": 6, "block": 6, "full": 3, "uniform_block": 3}  # Local: exact preregistered eighteen-run multiplicities.
    per_update = sum(counts[row["mode"]] * (row["p90_seconds"] + row["validation_seconds"] / cfg["validation_every"] + row["checkpoint_seconds"] / cfg["checkpoint_every"]) for row in measurements)  # Local: conservative measured cost plus amortized overhead across the whole matrix.
    fixed = sum(counts[row["mode"]] * (row["checkpoint_seconds"] + 15) for row in measurements)  # Local: reserve final checkpoint and per-run initialization costs.
    updates = int((cfg["gpu_hours"]["train"] * 3600 * cfg["schedule_safety_factor"] - fixed) // per_update)  # Global: twenty-percent safety margin is fixed before outcomes.
    if updates < 20:
        raise RuntimeError("Measured runtime leaves too few updates for the approved study")  # Local: report infeasibility instead of mislabeling a smoke test as training.
    frozen = dict(created=stamp(), commit=revision(), config=cfg, config_sha256=digest(PROJECT / "configs/study.json"), data_sha256=data_fingerprint(prepared), updates=updates, microbatch=microbatch, order=execution_order(cfg), measurements=measurements, failures=failures, predicted_training_seconds=updates * per_update + fixed, device=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda)  # Global: freeze cost-selected settings and portable data identity before any confirmatory training.
    write_json(root() / "frozen.json", frozen)  # Local: all model runs load this immutable artifact.
    write_json(PROJECT / "reports/frozen_protocol.json", frozen)  # Global: commit and push this file before running the training phase.
    LOG.info("FROZEN updates=%d microbatch=%d projected training hours=%.3f; publish reports/frozen_protocol.json before training", updates, microbatch, frozen["predicted_training_seconds"] / 3600)  # Local: make the final preregistration action explicit.
    return frozen  # Local: pipeline callers can inspect the exact run matrix.
