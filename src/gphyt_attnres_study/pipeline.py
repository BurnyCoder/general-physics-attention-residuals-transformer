"""Global: readable public entry point for preparation, execution, analysis, and release.

Local: phase implementations remain in focused modules and can resume independently.
Source: https://docs.python.org/3/library/argparse.html.
"""
import argparse  # Local: standard-library CLI keeps configuration discoverable.
import subprocess  # Local: verify the public preregistration before confirmatory execution.
from .runtime import Budget, LOG, PROJECT, cpu_threads, digest, read_json, root, start_logging  # Global: consistent execution and provenance.


def train():
    """Local: execute the frozen matrix; global: all primary pairs precede exploratory controls."""
    from .training import gpu_setup, run_one  # Local: heavy training imports stay inside their phase.
    frozen, prepared = read_json(root() / "frozen.json"), read_json(root() / "prepared.json")  # Local: require explicit completed preparation and profiling.
    if frozen is None or prepared is None:
        raise RuntimeError("Preparation and profiling must complete first")  # Global: training cannot invent missing settings.
    if frozen["data_sha256"] != digest(root() / "prepared.json"):
        raise RuntimeError("Data manifest changed after freezing")  # Global: protect the frozen split and normalization contract.
    tracked = subprocess.check_output(["git", "show", "HEAD:reports/frozen_protocol.json"], cwd=PROJECT)  # Local: require the frozen protocol in version control.
    if tracked.replace(b"\r\n", b"\n") != (PROJECT / "reports/frozen_protocol.json").read_bytes().replace(b"\r\n", b"\n"):
        raise RuntimeError("Commit the exact frozen protocol before training")  # Global: prevent an unrecorded update-count change.
    subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", "@{upstream}"], cwd=PROJECT, check=True)  # Local: refuse unpushed training implementations.
    with Budget("train") as budget:
        gpu_setup()  # Local: device operations begin only after budget acquisition.
        for spec in frozen["order"]:
            budget.check()  # Global: every planned run shares one finite allocation.
            run_one(spec, frozen, prepared, budget)  # Local: resume complete optimizer/sample state when needed.


def main():
    """Local: dispatch one named phase; global: expose a replicable user workflow."""
    parser = argparse.ArgumentParser(description="Preregistered GPhyT attention-residual study")  # Local: --help explains the public interface.
    parser.add_argument("phase", choices=["prepare", "power", "verify", "profile", "train", "evaluate", "analyze", "report", "dashboard", "publish"])  # Global: each durable phase can resume independently.
    args = parser.parse_args()  # Local: reject misspelled phases rather than guessing.
    start_logging(args.phase)  # Global: capture complete stdout, stderr, configuration identity, and timestamps.
    cpu_threads()  # Local: leave CPU capacity for the operating system and dashboard.
    if args.phase == "prepare":
        from .data import prepare  # Local: official download, preprocessing, split validation, normalization.
        prepare()  # Global: this phase never trains a model.
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
    elif args.phase == "evaluate":
        from .evaluation import evaluate  # Local: final checkpoints, persistence, and fixed interventions.
        evaluate()  # Global: test outcomes never select training checkpoints.
    elif args.phase == "analyze":
        from .analysis import analyze  # Local: seed-level effects, confidence intervals, and exact tests.
        analyze()  # Global: incomplete results remain explicitly undefined.
    elif args.phase == "report":
        from .reporting import report  # Local: reproducible figures, experiment reports, and paper tables.
        report()  # Global: measured artifacts are the only source of numerical claims.
    elif args.phase == "dashboard":
        import sys  # Local: launch Streamlit with the current locked interpreter.
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(PROJECT / "src/gphyt_attnres_study/dashboard.py"), "--server.address", "127.0.0.1"], check=True)  # Global: the local UI stays on the user's machine.
    elif args.phase == "publish":
        from .publishing import publish  # Local: public tensor checkpoints and reproducibility metadata.
        publish()  # Global: credentials stay in the existing Hugging Face authentication store.
    LOG.info("Phase completed: %s", args.phase)  # Local: distinguish successful completion from an interrupted log.
