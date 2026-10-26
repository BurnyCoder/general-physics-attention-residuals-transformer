"""Global: verify diagnostic arithmetic on CUDA before interpreting interventions.

Local: reuse the published regression assertions under the original profiling allowance.
Sources: https://docs.pytorch.org/docs/2.14/amp.html and
https://docs.pytest.org/en/stable/how-to/monkeypatch.html.
"""
import runpy  # Local: load the repository's existing tests without duplicating assertions.
import pytest  # Local: scope the identity permutation patch to its own test.
import torch  # Local: synchronize actual CUDA work before releasing the budget lease.
from gphyt_attnres_study.runtime import Budget, LOG, PROJECT, cpu_threads, revision, start_logging, stamp, write_json  # Global: retain ordinary provenance and accounting.
from gphyt_attnres_study.training import gpu_setup  # Local: use the production deterministic CUDA policy.


def main():
    """Local: execute sixteen exact-equality checks; global: these are implementation probes, not physics outcomes."""
    start_logging("verify_cuda_interventions")  # Local: retain complete commands, outputs, and failures.
    cpu_threads()  # Local: keep the audit's CPU use bounded.
    tests = runpy.run_path(str(PROJECT / "tests/test_intervention_precision.py"))  # Local: reuse the tests that exposed the original arithmetic confound.
    rows = []  # Local: collect successful assertions without inventing prediction metrics.
    with Budget("profile") as budget:  # Global: serialize GPU access and preserve the original two-hour validation allocation.
        gpu_setup()  # Local: initialize CUDA only after the independent supervisor is active.
        for amp in (False, True):  # Local: cover FP32 and the study's BF16 autocast regime.
            for count in (1, 5, 7):  # Local: include the embedding-only and final Block reads.
                for intervention in ("uniform", "shuffle"):
                    budget.check()  # Global: every probe remains charged to the shared hard cap.
                    tests["test_unchanged_uniform_weights_preserve_output"](amp, count, intervention, device="cuda")  # Local: unchanged weights must produce bitwise-identical outputs.
                    rows.append(dict(test="neutral_routing", amp=amp, sources=count, intervention=intervention, passed=True))  # Local: record the exact tested case.
            with pytest.MonkeyPatch.context() as patch:  # Local: automatically restore torch.randperm before subsequent checks.
                tests["test_identity_shuffle_preserves_learned_routing"](amp, patch, device="cuda")  # Local: nonuniform weights also require unchanged arithmetic.
            rows.append(dict(test="learned_identity_shuffle", amp=amp, passed=True))  # Local: distinguish this stronger counterfactual probe.
            tests["test_ordinary_routing_and_gradients_remain_exactly_fla"](amp, device="cuda")  # Global: ordinary forward and backward paths must still match pinned FLA exactly.
            rows.append(dict(test="ordinary_fla_forward_and_gradients", amp=amp, passed=True))  # Local: record exact reference agreement separately.
            LOG.info("CUDA routing checks passed amp=%s cumulative_checks=%d", amp, len(rows))  # Local: expose progress without synthetic accuracy values.
        torch.cuda.synchronize()  # Global: pending GPU kernels count before the lease ends.
    write_json(PROJECT / "reports/cuda_intervention_verification.json", dict(created=stamp(), commit=revision(), purpose="Implementation equality probes; not physics-accuracy measurements", device=torch.cuda.get_device_name(), torch_version=torch.__version__, rows=rows, budget_seconds=budget.entry["seconds"]))  # Local: publish actual evidence and its charged duration.
    LOG.info("CUDA intervention verification complete: %d checks", len(rows))  # Global: a terminal success message follows all assertions.


if __name__ == "__main__":
    main()  # Local: only this explicit script invocation may use the GPU; ordinary pytest remains CPU-only.
