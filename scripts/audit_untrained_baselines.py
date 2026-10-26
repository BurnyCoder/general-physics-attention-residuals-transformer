"""Global: verify the added zero-update controls against durable evidence without model execution.

Local: run after `untrained` and `report`; hashes, full populations, means, and images are checked.
Sources: docs/untrained-baselines.md; https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html.
"""
import json  # Local: compare the original published primary analysis exactly apart from its timestamp.
from pathlib import Path  # Local: saved protected paths are within the designated project directories.
import subprocess  # Local: Git provides the pre-extension primary result without another model run.
import numpy as np  # Local: check exact matched fields and independently recomputed arithmetic means.
import pandas as pd  # Local: inspect the published machine-readable comparison table.
from gphyt_attnres_study.comparisons import MODES, matched_fields  # Global: validate complete targets, active masks, and forecast dimensions before showing examples.
from gphyt_attnres_study.runtime import LOG, PROJECT, digest, read_json, root, stamp, start_logging, write_json  # Local: one timestamped audit trail and explicit artifact roots.


def audit():
    """Global: fail on changed original evidence or mislabeled controls; local: write only completed checks."""
    start_logging("untrained_evidence_audit")  # Local: preserve all assertions/failures in the normal log directory.
    before = read_json(root() / "untrained_before.json")  # Local: snapshot recorded before the additional GPU phase.
    for path, expected in before["protected_inputs"].items():
        assert digest(Path(path)) == expected, path  # Global: original evaluations and frozen/prepared inputs must remain byte-identical.
    original = json.loads(subprocess.check_output(["git", "show", "eb17eabb51b4e294de8d55486eff9dfae605b78a:reports/primary_analysis.json"], cwd=PROJECT))  # Local: source/protocol publication commit fixes the original primary result.
    current = read_json(PROJECT / "reports/primary_analysis.json")  # Local: reporting may update only the generation timestamp.
    original.pop("created"); current.pop("created")  # Global: compare all scientific fields, excluding one metadata timestamp.
    assert original == current  # Global: the later controls cannot change preregistered inference.
    frozen, records, backbones = read_json(root() / "frozen.json"), [], {}  # Local: original ordered identities determine the complete added matrix.
    identity = lambda row: tuple(row[key] for key in ("dataset", "parameter", "trajectory", "source_sha256"))  # Global: compare actual trajectory identity, never source-file hash alone.
    for spec in frozen["order"]:
        name = f"{spec['mode']}_seed{spec['seed']}"  # Local: initialized and trained records share only this identity.
        folder = root() / "untrained/runs" / name  # Local: initialized files are isolated from trained artifacts.
        status = read_json(folder / "status.json")  # Local: every planned attempt must be present for this completed-study audit.
        assert status["status"] == "complete" and status["training_updates"] == 0 and status["post_hoc"] is True, name  # Global: incomplete/mislabeled controls cannot pass as a complete release.
        assert status["initial_state_sha256"] == status["final_state_sha256"], name  # Global: no parameter or buffer changed during inference.
        assert status["commit"] == "eb17eabb51b4e294de8d55486eff9dfae605b78a", name  # Local: all scientific execution used the published preexecution source revision.
        assert status["checkpoint_sha256"] == digest(folder / "model.safetensors"), name  # Global: exact measured initialized weights accompany every result.
        backbones.setdefault(spec["seed"], set()).add(status["shared_backbone_sha256"])  # Local: all variants within a seed must share the same backbone.
        initialized = read_json(root() / "untrained/evaluation" / f"{name}.json")  # Local: additional measurements are separate from primary discovery.
        trained = read_json(root() / "evaluation" / f"{name}.json")  # Local: original matched trained counterpart.
        assert [identity(row) for row in initialized["trajectories"]] == [identity(row) for row in trained["trajectories"]], name  # Global: identical test population and ordering in each comparison.
        assert len(initialized["trajectories"]) == 24 and all(len(row["windows"]) == 3 for row in initialized["trajectories"]), name  # Local: each of the 24 official test trajectories has three fixed forecast origins.
        assert digest(root() / "untrained/evaluation" / f"{name}.json") == digest(PROJECT / "reports/untrained/evaluation" / f"{name}.json"), name  # Global: public raw measurements are exact copies.
        records.append(dict(run=name, seed=spec["seed"], mode=spec["mode"], checkpoint_sha256=status["checkpoint_sha256"], initial_state_sha256=status["initial_state_sha256"], seconds=status["seconds"], trajectories=24, windows=72, finite_24_windows=sum(window["finite_forecast_steps"] == 24 for row in initialized["trajectories"] for window in row["windows"])))  # Local: retain divergence rather than assuming every completed inference is finite.
    assert len(records) == 18 and all(len(values) == 1 for values in backbones.values())  # Global: original seeds and paired shared initialization remain invariant.
    table = pd.read_csv(PROJECT / "reports/untrained/comparison.csv")  # Local: verify the scores actually displayed in README, paper, and site.
    for row in table.itertuples(index=False):
        seeds = frozen["config"]["seeds"] if row.mode in ("standard", "block") else frozen["config"]["control_seeds"]  # Local: planned seed counts are fixed independently of measured outcomes.
        for namespace, reported in (("untrained/evaluation", row.mean_initial), ("evaluation", row.mean_trained)):
            values = [read_json(root() / namespace / f"{row.mode}_seed{seed}.json")["datasets"][row.dataset][str(row.horizon)] for seed in seeds]  # Local: independently read every original endpoint.
            expected = float(np.mean(np.asarray(values, dtype=float)))  # Global: null entries propagate as NaN, never a survivor-only mean.
            np.testing.assert_allclose(reported, expected, rtol=1e-12, atol=1e-12, equal_nan=True)  # Local: table values must match the raw measured endpoints.
    examples = []  # Local: audit all eight architecture/dataset rows, not a selected attractive example.
    for mode in MODES:
        for dataset, channel in (("rayleigh_benard_uniform", 2), ("turbulent_radiative_layer_2D", 1)):
            runs = ("persistence", f"untrained_{mode}_seed0", f"{mode}_seed0")  # Local: the figure prepends ground truth to these three predictions.
            fields = matched_fields(root() / "visuals", dataset, channel, runs=runs)  # Global: complete arrays and masks, not just the plotted frame, must agree.
            assert fields.shape == (4, 128, 64)  # Local: original resolution and four plotted panels.
            examples.append(dict(mode=mode, dataset=dataset, channel=channel, step=12, aligned=True, finite_cells=int(np.isfinite(fields).sum()), total_cells=int(fields.size)))  # Global: any divergent pixels remain visible in the evidence receipt.
    for seed in (0, 1, 2):
        block = read_json(root() / "untrained/evaluation" / f"block_seed{seed}.json")  # Local: zero-query learned routing should equal fixed uniform routing before training.
        uniform = read_json(root() / "untrained/evaluation" / f"uniform_block_seed{seed}.json")  # Local: same paired seed and backbone.
        assert block["initial_state_sha256"] == uniform["initial_state_sha256"] and block["datasets"] == uniform["datasets"]  # Global: equal initial models are not independent evidence for architecture quality.
        for dataset in frozen["config"]["data"]:
            with np.load(root() / "visuals" / f"untrained_block_seed{seed}_{dataset}.npz") as left, np.load(root() / "visuals" / f"untrained_uniform_block_seed{seed}_{dataset}.npz") as right:
                assert np.array_equal(left["prediction"], right["prediction"], equal_nan=True)  # Local: verify full 24-frame equality, not just aggregate coincidence.
    ledger = read_json(root() / "gpu_budget.json")  # Local: retain every original and additional charged session.
    sessions = [row for row in ledger["sessions"] if row["pid"] == 58828]  # Local: the recorded initialized-control worker identifies this extension's actual lease.
    assert len(sessions) == 1 and sessions[0]["status"] == "complete"  # Global: never claim the GPU phase ended while it is active.
    usage = {phase: sum(row["seconds"] for row in ledger["sessions"] if row["phase"] == phase) for phase in ("profile", "train", "evaluate", "contingency")}  # Local: original phase allocations remain distinct.
    assert usage["evaluate"] <= 7200 and sum(usage.values()) <= 86400  # Global: no budget extension or reset is permitted.
    receipt = dict(created=stamp(), protected_input_files_unchanged=len(before["protected_inputs"]), primary_analysis_unchanged_except_timestamp=True, source_commit="eb17eabb51b4e294de8d55486eff9dfae605b78a", additional_gpu_seconds=sessions[0]["seconds"], phase_seconds=usage, total_gpu_hours=sum(usage.values()) / 3600, initialized_models=records, examples=examples, shared_backbone_hashes={str(seed): next(iter(values)) for seed, values in backbones.items()}, block_uniform_initialization_and_forecasts_equal_for_seeds=[0,1,2], all_24_comparison_rows_recomputed=True)  # Global: record only completed numerical/provenance checks; visual and remote checks are added after they occur.
    write_json(PROJECT / "reports/untrained_baseline_verification.json", receipt)  # Local: portable public audit record excludes secret or raw data content.
    LOG.info("Verified %d initialized models, %d example rows, unchanged primary inference; additional GPU %.3f s", len(records), len(examples), sessions[0]["seconds"])  # Local: concise terminal outcome with full evidence saved.


if __name__ == "__main__":
    audit()  # Local: explicit CPU-only entry point; imports never rerun an experiment.
