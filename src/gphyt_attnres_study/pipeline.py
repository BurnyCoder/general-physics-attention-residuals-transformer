"""Global: readable public entry point for preparation, execution, analysis, and release.

Local: phase implementations remain in focused modules and can resume independently.
Source: https://docs.python.org/3/library/argparse.html.
"""
import argparse  # Local: standard-library CLI keeps configuration discoverable.
import subprocess  # Local: verify the public preregistration before confirmatory execution.
from .runtime import Budget, LOG, PROJECT, cpu_threads, read_json, root, run_logged, start_logging  # Global: consistent execution and provenance.
from .provenance import validate_frozen  # Global: training, evaluation, and replication use one scientific identity contract.


def train():
    """Local: execute the frozen matrix; global: all primary pairs precede exploratory controls."""
    from .training import gpu_setup, run_one  # Local: heavy training imports stay inside their phase.
    frozen, prepared = read_json(root() / "frozen.json"), read_json(root() / "prepared.json")  # Local: require explicit completed preparation and profiling.
    validate_frozen(frozen, prepared)  # Global: protect the published schedule, splits, and normalization while allowing portable paths.
    tracked = subprocess.check_output(["git", "show", "HEAD:reports/frozen_protocol.json"], cwd=PROJECT)  # Local: require the frozen protocol in version control.
    if tracked.replace(b"\r\n", b"\n") != (PROJECT / "reports/frozen_protocol.json").read_bytes().replace(b"\r\n", b"\n"):
        raise RuntimeError("Commit the exact frozen protocol before training")  # Global: prevent an unrecorded update-count change.
    subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", "@{upstream}"], cwd=PROJECT, check=True)  # Local: refuse unpushed training implementations.
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "vendor", "configs", "docs/protocol.md", "pyproject.toml", "uv.lock"], cwd=PROJECT, check=True)  # Global: execution code, locked dependencies, and scientific settings must match the published revision.
    if subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "src", "vendor", "configs"], cwd=PROJECT).strip():
        raise RuntimeError("Publish all execution source files before training")  # Global: untracked importable modules cannot evade the clean-source check.
    with Budget("train") as budget:
        gpu_setup()  # Local: device operations begin only after budget acquisition.
        for spec in frozen["order"]:
            budget.check()  # Global: every planned run shares one finite allocation.
            run_one(spec, frozen, prepared, budget)  # Local: resume complete optimizer/sample state when needed.


def main():
    """Local: dispatch one named phase; global: expose a replicable user workflow."""
    parser = argparse.ArgumentParser(description="Preregistered GPhyT attention-residual study")  # Local: --help explains the public interface.
    parser.add_argument("phase", choices=["prepare", "audit", "power", "verify", "profile", "reproduce", "train", "evaluate", "untrained", "analyze", "report", "animate", "dashboard", "publish"])  # Global: each durable phase can resume independently.
    args = parser.parse_args()  # Local: reject misspelled phases rather than guessing.
    start_logging(args.phase)  # Global: capture complete stdout, stderr, configuration identity, and timestamps.
    cpu_threads()  # Local: leave CPU capacity for the operating system and dashboard.
    if args.phase == "prepare":
        from .data import prepare  # Local: official download, preprocessing, split validation, normalization.
        from .data_audit import audit  # Local: quantify initial-condition reuse after preserving the official split checks.
        audit(prepare())  # Global: this phase never trains a model or changes the selected trajectories.
    elif args.phase == "audit":
        from .data_audit import audit  # Local: regenerate overlap diagnostics without reprocessing immutable data.
        audit(read_json(root() / "prepared.json"))  # Global: audit output is distinct from the frozen scientific inputs.
    elif args.phase == "power":
        from .analysis import power  # Local: prospective sensitivity has no dependency on outcomes.
        power()  # Global: publish the power grid before confirmatory training.
    elif args.phase == "verify":
        from .verification import verify_gpu  # Local: validate full-scale CUDA execution before real-data profiling.
        verify_gpu()  # Global: synthetic smoke losses are not scientific study outcomes.
    elif args.phase == "profile":
        from .profiling import profile  # Local: benchmark all variants and freeze one common horizon.
        profile()  # Global: training remains a separate phase so its settings can be published first.
    elif args.phase == "train":
        train()  # Local: fixed matrix with paired initialization and sampling.
    elif args.phase == "reproduce":
        from .provenance import reproduce  # Local: a clone can adopt the released horizon without re-profiling it.
        reproduce()  # Global: verify data and configuration before installing the published freeze.
    elif args.phase == "evaluate":
        from .evaluation import evaluate  # Local: final checkpoints, persistence, and fixed interventions.
        evaluate()  # Global: test outcomes never select training checkpoints.
    elif args.phase == "analyze":
        from .analysis import analyze  # Local: seed-level effects, confidence intervals, and exact tests.
        analyze()  # Global: incomplete results remain explicitly undefined.
    elif args.phase == "untrained":
        from .untrained import untrained  # Local: post-hoc zero-update controls reuse the existing constructor and evaluation protocol.
        untrained()  # Global: additional control artifacts stay separate from the original confirmatory results.
    elif args.phase == "report":
        from .reporting import report  # Local: reproducible figures, experiment reports, and paper tables.
        report()  # Global: measured artifacts are the only source of numerical claims.
    elif args.phase == "animate":
        from .animations import rollout_gifs  # Local: replay saved model, initialized, and intervention fields without inference.
        from .web_export import export_site  # Global: README supplements and the public viewer use identical generated GIFs.
        rollout_gifs()  # Local: CPU-only rendering does not acquire or modify a GPU lease.
        export_site()  # Local: publish the animation inventory through the existing shared viewer exporter.
    elif args.phase == "dashboard":
        import sys  # Local: launch Streamlit with the current locked interpreter.
        run_logged([sys.executable, "-m", "streamlit", "run", str(PROJECT / "src/gphyt_attnres_study/dashboard.py"), "--server.address", "127.0.0.1"])  # Global: the local UI stays on the user's machine and its complete logs enter the audit trail.
    elif args.phase == "publish":
        from .publishing import publish  # Local: public tensor checkpoints and reproducibility metadata.
        publish()  # Global: credentials stay in the existing Hugging Face authentication store.
    LOG.info("Phase completed: %s", args.phase)  # Local: distinguish successful completion from an interrupted log.
