"""Global: budgeted full-scale CUDA smoke checks before real-data profiling.

Local: analytical synthetic inputs test execution, not scientific prediction accuracy.
Source: https://docs.pytorch.org/docs/stable/notes/randomness.html.
"""
import gc  # Local: isolate CUDA memory measurements between variants.
import torch  # Local: exercise actual BF16 forward, backward, and optimizer operations.
from .model import make_model  # Global: smoke checks construct the requested full GPhyT-S dimensions.
from .training import gpu_setup, masked_loss, optimizer_for  # Global: reuse production precision and optimization contracts.
from .runtime import Budget, LOG, PROJECT, config, stamp, write_json  # Global: all GPU validation consumes the profiling allocation.


def verify_gpu():
    """Local: execute one backward update per variant; global: distinguish execution tests from measured study results."""
    cfg, rows = config(), []  # Local: use full approved architecture, not a toy model.
    with Budget("profile") as budget:
        gpu_setup()  # Local: charge CUDA initialization to the same GPU ledger.
        generator = torch.Generator().manual_seed(20261025)  # Local: identical deterministic synthetic inputs for every variant.
        context = torch.randn(1, 4, *cfg["resolution"], 5, generator=generator).cuda()  # Local: approved input tensor shape.
        target, active = context[:, -1:].clone(), torch.ones(1, 5, dtype=torch.bool, device="cuda")  # Local: known finite target, without pretending to simulate physics.
        for mode in ("standard", "block", "full", "uniform_block"):
            budget.check()  # Global: no smoke test bypasses accounting.
            model = make_model(cfg, mode, 10001).cuda()  # Local: initialization is separate from confirmatory seeds.
            optimizer = optimizer_for(model, cfg)  # Local: production AdamW settings.
            torch.cuda.reset_peak_memory_stats()  # Local: record the actual tensor allocation footprint.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(context)  # Local: retain upstream outer next-state update.
                loss = masked_loss(output, target, active)  # Local: test the same supervised objective implementation.
            loss.backward()  # Local: ensure FLA reference routing supports CUDA autograd.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)  # Local: fail immediately on invalid gradients.
            optimizer.step()  # Local: initialize and validate Adam moments on the actual GPU.
            assert output.shape == target.shape and torch.isfinite(loss)  # Global: execution shape and numerical sanity are required before profiling.
            torch.cuda.synchronize()  # Local: include all queued work in the budget measurement.
            rows.append(dict(mode=mode, synthetic_loss=float(loss.detach()), gradient_norm=float(norm), parameters=sum(p.numel() for p in model.parameters()), peak_memory_bytes=torch.cuda.max_memory_allocated()))  # Global: label synthetic quantities so they cannot become paper accuracy results.
            LOG.info("CUDA smoke passed mode=%s parameters=%d memory_MiB=%.1f", mode, rows[-1]["parameters"], rows[-1]["peak_memory_bytes"] / 2**20)  # Local: concise verification evidence.
            del output, loss, optimizer, model  # Local: free the complete autograd graph before the next model.
            gc.collect()  # Local: clear Python-owned cycles.
            torch.cuda.empty_cache()  # Local: release unused GPU cache.
    write_json(PROJECT / "reports/gpu_verification.json", dict(created=stamp(), purpose="Execution checks only; synthetic losses are not scientific outcomes", rows=rows))  # Global: preserve smoke checks separately from experiment metrics.
