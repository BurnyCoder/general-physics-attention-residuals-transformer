"""Global: publish tensor checkpoints and verify a downloaded model by evaluation.

Local: authenticated official Hugging Face CLI; no credential is read or serialized here.
Source: https://huggingface.co/docs/huggingface_hub/en/guides/cli.
"""
import json  # Local: parse CLI metadata and write reproducibility configurations.
import os  # Local: only the public account namespace comes from environment configuration.
from pathlib import Path  # Local: locate the virtual-environment executable directly.
import shutil  # Local: stage only explicitly approved, nonsecret release files.
import sys  # Local: locate the CLI from this project's locked virtual environment.
import torch  # Local: verify the downloaded tensor checkpoint with real inference.
from safetensors.torch import load_file  # Local: deserialize public weights without executable pickle content.
from .evaluation import evaluate_model  # Global: release verification uses production metrics and causal forecasting.
from .model import make_model  # Global: reconstruct the published model from frozen configuration.
from .runtime import Budget, LOG, PROJECT, digest, read_json, root, run_logged, stamp, write_json  # Global: publication and verification remain auditable.


def hf(*arguments):
    """Local: capture complete CLI output; global: publishing uses the existing authenticated account."""
    executable = str(Path(sys.executable).parent / ("hf.exe" if os.name == "nt" else "hf"))  # Local: use the project's pinned Hub CLI.
    return run_logged([executable, *arguments])  # Local: shared streaming preserves full CLI output without duplicate process wrappers.


def stage_release(mode, frozen, prepared):
    """Local: create an allowlisted model package; global: raw data, environments, and credentials never enter upload folders."""
    folder = root() / "release" / mode  # Local: one model repository groups this variant's completed seeds.
    folder.mkdir(parents=True, exist_ok=True)  # Local: stage under the project-owned artifact root.
    runs = []  # Local: missing/failed runs are described but never replaced by unfinished weights.
    for spec in frozen["order"]:
        if spec["mode"] != mode:
            continue  # Local: each repository has one unambiguous architecture.
        run = root() / "runs" / f"{mode}_seed{spec['seed']}"  # Local: locate the final model from the frozen run matrix.
        status = read_json(run / "status.json", {"status": "not_started", **spec})  # Global: retain planned omissions in model metadata.
        runs.append(status)  # Local: card and metadata report the complete variant matrix.
        if status["status"] == "complete":
            destination = folder / f"seed{spec['seed']}"  # Local: stable seed directories support exact checkpoint selection.
            destination.mkdir(exist_ok=True)  # Local: create only explicit release paths.
            shutil.copy2(run / "model.safetensors", destination / "model.safetensors")  # Local: publish tensor weights, never trusted-local optimizer pickles.
            write_json(destination / "training_status.json", status)  # Global: attach exact update count, runtime, seed, and revision.
            evaluation = root() / "evaluation" / f"{run.name}.json"  # Local: measured metrics accompany the same model.
            if evaluation.exists():
                shutil.copy2(evaluation, destination / "evaluation.json")  # Global: model users can inspect per-trajectory evidence.
            shutil.copy2(run / "metrics.jsonl", destination / "metrics.jsonl")  # Global: preserve complete update-level training measurements with the checkpoint.
    if not any(run["status"] == "complete" for run in runs):
        return None  # Global: do not publish an empty repository pretending to contain trained models.
    write_json(folder / "config.json", dict(frozen["config"], residual_mode=mode, updates=frozen["updates"], microbatch=frozen["microbatch"]))  # Local: architecture and training horizon travel with the weights.
    write_json(folder / "normalization.json", prepared["normalization"])  # Global: inference must use training-only statistics, not new evaluation fits.
    write_json(folder / "runs.json", runs)  # Local: failures remain part of the public variant record.
    shutil.copy2(PROJECT / "reports/frozen_protocol.json", folder / "frozen_protocol.json")  # Local: published settings match preregistration.
    shutil.copy2(PROJECT / "reports/data_manifest.json", folder / "data_manifest.json")  # Local: selected data hashes are public, raw data are not uploaded.
    shutil.copy2(PROJECT / "LICENSE", folder / "LICENSE")  # Local: retain the project license with released artifacts.
    card = f"""---
license: mit
library_name: pytorch
tags:
- physics
- simulation
- attention-residuals
- gphyt
---
# GPhyT-S: {mode}

Final fixed-update checkpoints from the paired GPhyT Attention Residuals study by Libor Burian.
The architecture uses twelve layers, width 192, MLP width 768, three heads, four context frames,
128x64 fields, and five canonical channels. Mode: `{mode}`. Training updates: {frozen['updates']}.

## Use and evaluation

Clone the [study repository](https://github.com/BurnyCoder/general-physics-attention-residuals-transformer),
install with `uv sync --locked`, and construct `make_model(config, residual_mode, seed)` from
`gphyt_attnres_study.model`. Load `seedN/model.safetensors` with `safetensors.torch.load_file`.
Normalize each active channel using this release's per-dataset `normalization.json`; absent
channels remain zero and must be masked out of metrics and autoregressive feedback.
The CLI evaluation phase and tests specify the exact inference and scoring procedure.

Each `seedN/evaluation.json`, when available, reports measured one-step, 12-step, and 24-step
errors. `runs.json` retains failed or omitted seeds. No unmeasured improvement is asserted here.
These models were trained from scratch on the preregistered subset of
[The Well](https://arxiv.org/abs/2412.00568), with official trajectory splits and train-only normalization.
They are research surrogates, not numerical solvers or validated predictors for unseen physics.
Coarse spatial resolution, a bounded local compute budget, and six primary seed pairs limit inference.
Buoyancy is a separate channel and is not temperature.

## Sources and attribution

- [GPhyT paper](https://arxiv.org/abs/2509.13805) and [official MIT code](https://github.com/FloWsnr/General-Physics-Transformer).
- [Attention Residuals](https://arxiv.org/abs/2603.15031) and [FLA MIT operator](https://github.com/fla-org/flash-linear-attention).
- Exact pinned revisions, licenses, reports, and the paper are in the study repository.
"""  # Global: model-card claims are bounded by the released artifacts and exact configuration.
    (folder / "README.md").write_text(card, encoding="utf-8")  # Local: each variant has a human-readable public model card.
    return folder  # Local: the uploader sees only this allowlisted staging directory.


def publish():
    """Local: release completed models and round-trip one checkpoint; global: verify public availability empirically."""
    frozen, prepared = read_json(root() / "frozen.json"), read_json(root() / "prepared.json")  # Local: release only a real prepared/frozen study.
    if frozen is None or prepared is None:
        raise RuntimeError("Prepare, train, and evaluate before model publication")  # Global: no placeholder model release.
    namespace = os.environ.get("HF_NAMESPACE", "BurnyCoder")  # Local: public namespace is not a secret credential.
    state_path = root() / "publication.json"  # Local: durable state makes upload retries idempotent.
    state = read_json(state_path, {"models": {}, "created": stamp()})  # Local: preserve already published repository identifiers.
    if "collection" not in state:
        response = json.loads(hf("collections", "create", "GPhyT Attention Residuals", "--namespace", namespace, "--description", "Paired physics forecasting study: fixed-update checkpoints, configurations and measured evaluation.", "--exists-ok", "--json"))  # Local: official CLI creates a public collection with existing account authorization.
        state["collection"] = response["slug"]  # Local: retain the exact server-assigned collection identity.
        write_json(state_path, state)  # Global: a retry reuses the same collection.
    for mode in ("standard", "block", "full", "uniform_block"):
        folder = stage_release(mode, frozen, prepared)  # Local: completed seeds only, with explicit omissions.
        if folder is None:
            continue  # Local: skip empty model packages without inventing checkpoints.
        repo = f"{namespace}/gphyt-attnres-{mode.replace('_', '-')}"  # Local: stable public repository identity per architecture.
        hf("upload", repo, folder, ".", "--repo-type", "model", "--commit-message", "Publish fixed-update physics study checkpoints and measured metadata")  # Global: only staged allowlisted artifacts leave this project.
        hf("collections", "add-item", state["collection"], repo, "model", "--exists-ok")  # Local: organize every published variant into the requested collection.
        state["models"][mode] = repo  # Local: record confirmed upload identity.
        write_json(state_path, state)  # Global: partial publication is resumable.
    if "verified" not in state and "standard" in state["models"]:
        repository = state["models"]["standard"]  # Local: verify the prespecified standard seed zero, not the best-performing model.
        downloaded = root() / "release_verification"  # Local: a separate directory proves an actual Hub round trip.
        hf("download", repository, "seed0/model.safetensors", "--local-dir", downloaded)  # Local: retrieve the published checkpoint through the official client.
        path = downloaded / "seed0/model.safetensors"  # Local: use the actual downloaded artifact for inference.
        if digest(path) != digest(root() / "runs/standard_seed0/model.safetensors"):
            raise ValueError("Published checkpoint hash differs from the measured local model")  # Global: do not claim verification of mismatched bytes.
        records = [row for dataset in sorted(frozen["config"]["data"]) for row in [r for r in prepared["records"] if r["split"] == "test" and r["dataset"] == dataset][:1]]  # Local: verify fixed first trajectories from both datasets.
        with Budget("evaluate") as budget:
            model = make_model(frozen["config"], "standard", 0).cuda().eval()  # Local: charge download-verification inference to the shared budget.
            model.load_state_dict(load_file(str(path), device="cuda"))  # Local: evaluate the retrieved weights, not a cached model object.
            result = evaluate_model(model, prepared, records, budget)  # Global: use the same causal metrics and fixed windows as the main study.
            del model  # Local: release GPU memory after verification.
            torch.cuda.empty_cache()  # Local: leave the device ready for the user's next task.
        state["verified"] = dict(time=stamp(), sha256=digest(path), datasets=result["datasets"])  # Global: concrete measured evidence supports the public round-trip claim.
        write_json(root() / "release_verification/evaluation.json", result)  # Local: retain per-trajectory verification details.
        write_json(state_path, state)  # Local: durable success record prevents unnecessary repeated GPU work.
    write_json(PROJECT / "reports/publication.json", state)  # Global: final links and verification evidence become public reproducibility artifacts.
    return state  # Local: CLI reports completion only after requested actions succeed.
