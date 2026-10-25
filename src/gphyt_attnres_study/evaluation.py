"""Global: fixed-origin autoregressive evaluation with trajectory-level aggregation.

Local: forecasting accepts past context only; future truth enters metrics afterward.
Sources: docs/protocol.md and https://polymathic-ai.org/the_well/.
"""
import contextlib  # Local: CPU tests and CUDA evaluation share the causal forecast function.
import time  # Local: synchronized inference timing is a secondary outcome.
import numpy as np  # Local: portable field metrics and physical-coordinate derivatives.
import torch  # Local: no-gradient inference through the same trained model.
from safetensors.torch import load_file  # Local: public model weights never execute pickle content.
from .data import Sampler, window_starts  # Global: use frozen normalization and deterministic origins.
from .model import make_model, ResidualStack  # Global: interventions operate on the trained routing modules.
from .runtime import Budget, LOG, PROJECT, config, read_json, root, stamp, write_json  # Global: share provenance and hard budget.


@torch.no_grad()
def forecast(model, context, active, horizon, budget=None):
    """Local: append only predictions; global: future frames cannot enter this API."""
    history, predictions = context.clone(), []  # Local: leave caller context unchanged for paired comparisons.
    mask = active[:, None, None, None, :]  # Local: structural field validity broadcasts across time and space.
    for step in range(horizon):
        if budget is not None:
            budget.check()  # Local: evaluation and interventions consume the shared GPU allocation.
        manager = torch.autocast("cuda", dtype=torch.bfloat16) if context.is_cuda else contextlib.nullcontext()  # Local: BF16 is used only on the approved CUDA path.
        with manager:
            prediction = torch.where(mask, model(history).float(), 0.)  # Local: absent channels remain zero even if an ignored output is nonfinite.
        if not torch.isfinite(prediction).all():
            predictions.extend([torch.full_like(prediction, float("nan")) for _ in range(horizon - step)])  # Global: divergent remaining steps stay undefined, never clipped.
            break  # Local: avoid feeding NaNs into the upstream assertion.
        predictions.append(prediction)  # Local: preserve every completed predicted frame.
        history = torch.cat((history[:, 1:], prediction), dim=1)  # Global: shift the context using this model's output only.
    return torch.cat(predictions, dim=1)  # Local: shape is batch, horizon, height, width, fields.


def nmse_frames(prediction, target, active, epsilon=1e-6):
    """Local: spatial NMSE per frame and active field; global: match the prespecified standardized-space denominator."""
    error = (prediction.double() - target.double()).square().mean(dim=(2, 3))  # Local: accumulate metrics in FP64 independently of inference precision.
    energy = target.double().square().mean(dim=(2, 3)) + epsilon  # Local: stabilized target second moment, not centered test variance.
    values = ((error / energy) * active[:, None, :]).sum(-1) / active.sum(-1)[:, None]  # Local: fields are equally weighted within each example.
    return values.detach().cpu().numpy()  # Global: never carry metric tensors into model feedback.


def finite_mean(values):
    """Local: strict mean; global: do not improve results by discarding divergent trajectories."""
    array = np.asarray(values, dtype=float)  # Local: convert missing values to NaN through the explicit array representation.
    return float(array.mean()) if array.size and np.isfinite(array).all() else None  # Global: any undefined constituent makes the aggregate undefined.


def physical_diagnostics(prediction, target, record, stats):
    """Local: coarsened-grid diagnostics; global: avoid claiming conservation for radiative/open systems."""
    mean, scale = np.asarray(stats["mean"]), np.asarray(stats["std"])  # Local: recover physical field units with training-only statistics.
    pred, truth = prediction * scale + mean, target * scale + mean  # Local: predictions and targets use identical inverse transforms.
    if not np.isfinite(pred).all():
        return {"status": "nonfinite"}  # Global: do not silently drop unstable diagnostic cases.
    if record["dataset"] == "turbulent_radiative_layer_2D":
        return dict(negative_pressure_fraction=float((pred[..., 0] < 0).mean()), negative_density_fraction=float((pred[..., 1] < 0).mean()))  # Local: positivity violations are interpretable without a false closed-system law.
    dx = np.mean(np.diff(record["coordinates"]["x"])) * len(record["coordinates"]["x"]) / pred.shape[1]  # Local: area coarsening preserves the physical extent of cell-centered grids.
    dy = np.mean(np.diff(record["coordinates"]["y"])) * len(record["coordinates"]["y"]) / pred.shape[2]  # Local: use stored axes rather than assuming unit square coordinates.
    pred_div = np.gradient(pred[..., 3], dx, axis=1) + np.gradient(pred[..., 4], dy, axis=2)  # Local: first component differentiates along stored x, second along y.
    true_div = np.gradient(truth[..., 3], dx, axis=1) + np.gradient(truth[..., 4], dy, axis=2)  # Local: coarsened truth has its own discretization residual.
    return dict(divergence_rms=float(np.sqrt(np.mean(pred_div ** 2))), target_divergence_rms=float(np.sqrt(np.mean(true_div ** 2))), divergence_error_rms=float(np.sqrt(np.mean((pred_div - true_div) ** 2))))  # Global: report numerical residuals relative to the same-grid target.


def evaluate_model(model, prepared, records, budget=None, capture=False, label="diagnostic"):
    """Local: evaluate fixed trajectory windows; global: hierarchy prevents frame-level pseudoreplication."""
    sampler = Sampler(prepared, 0, split="test")  # Local: reuse normalization only; evaluation never calls the random batch method.
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")  # Local: persistence has no trainable parameters.
    rows, routes = [], []  # Local: raw trajectory outcomes and compact routing diagnostics remain separate.
    for index, record in enumerate(records):
        dataset_rank = sum(previous["dataset"] == record["dataset"] for previous in records[:index])  # Local: diagnostic subsets include four trajectories from each dataset.
        trajectory = sampler.array(record)  # Local: memory-map the complete held-out trajectory.
        active = torch.tensor([record["active"]], device=device)  # Local: fixed field semantics accompany every window.
        windows, diagnostics = [], []  # Local: average windows before trajectories, as preregistered.
        for start in window_starts(record["length"]):
            past = sampler.normalize(trajectory[start:start + 4], record)  # Global: normalization is fitted exclusively to training data.
            truth = sampler.normalize(trajectory[start + 4:start + 28], record)  # Local: future truth is held separately from forecast inputs.
            context = torch.from_numpy(past[None]).to(device)  # Local: only these four frames enter model execution.
            stack = getattr(model, "attention_blocks", None)  # Local: persistence and standard models need no routing machinery.
            if isinstance(stack, ResidualStack):
                stack.capture = capture and dataset_rank < 4  # Local: bounded diagnostics represent both datasets equally.
            prediction = forecast(model, context, active, 24, budget)  # Global: forecast has no access to truth or trajectory references.
            values = nmse_frames(prediction, torch.from_numpy(truth[None]).to(device), active)[0]  # Local: score only after autoregressive prediction.
            windows.append({str(h): finite_mean(values[:h]) for h in (1, 12, 24)})  # Local: each horizon averages its own first h forecast steps.
            diagnostics.append(physical_diagnostics(prediction[0].cpu().numpy(), truth, record, prepared["normalization"][record["dataset"]]))  # Local: physical units are secondary diagnostics only.
            if isinstance(stack, ResidualStack) and stack.capture:
                routes.append(dict(dataset=record["dataset"], trajectory=index, start=start, final_forecast_step=24, trace=stack.trace))  # Local: record which autoregressive step the heatmap describes.
            if dataset_rank == 0 and start == window_starts(record["length"])[0] and capture:
                root().joinpath("visuals").mkdir(exist_ok=True)  # Local: representative fields are saved for review and videos.
                np.savez_compressed(root() / "visuals" / f"{label}_{record['dataset']}.npz", prediction=prediction[0].cpu().numpy(), truth=truth, active=np.asarray(record["active"]))  # Local: reporting can compare fixed cases across models without additional GPU work.
        row = dict(dataset=record["dataset"], parameter=record["parameter"], trajectory=record["trajectory"], source_sha256=record["sha256"], windows=windows, diagnostics=diagnostics)  # Global: every estimate traces to one official trajectory.
        row["nmse"] = {str(h): finite_mean([window[str(h)] for window in windows]) for h in (1, 12, 24)}  # Local: strict aggregation preserves divergent cases.
        rows.append(row)  # Local: retain all selected trajectories in original manifest order.
        LOG.info("evaluation trajectory=%d/%d dataset=%s NMSE12=%s", index + 1, len(records), record["dataset"], row["nmse"]["12"])  # Local: explicit progress and instability evidence.
    datasets = {name: {str(h): finite_mean([row["nmse"][str(h)] for row in rows if row["dataset"] == name]) for h in (1, 12, 24)} for name in sorted({row["dataset"] for row in rows})}  # Global: trajectories, not windows, determine dataset weights.
    return dict(created=stamp(), trajectories=rows, datasets=datasets, routing=routes)  # Local: downstream analysis consumes measured numbers only.


class Persistence(torch.nn.Module):
    """Global: a zero-training baseline that repeats the final context frame."""
    def forward(self, context):
        """Local: the next prediction equals the most recently available field."""
        return context[:, -1:]  # Local: persistence has no access to the forecast target.


def evaluate():
    """Local: evaluate final checkpoints and fixed interventions; global: never select checkpoints by test performance."""
    frozen, prepared = read_json(root() / "frozen.json"), read_json(root() / "prepared.json")  # Local: require completed preparation and frozen training.
    records = [record for record in prepared["records"] if record["split"] == "test"]  # Global: preserve official test membership.
    destination = root() / "evaluation"  # Local: resumable per-run output files avoid repeated evaluation.
    destination.mkdir(exist_ok=True)  # Local: result storage is separate from checkpoints.
    persistence = destination / "persistence.json"  # Local: CPU baseline uses no project GPU time.
    if not persistence.exists():
        write_json(persistence, evaluate_model(Persistence(), prepared, records))  # Local: same windows and metrics as learned models.
    with Budget("evaluate") as budget:
        for spec in frozen["order"]:
            name = f"{spec['mode']}_seed{spec['seed']}"  # Local: stable run identifier aligns all result artifacts.
            folder = root() / "runs" / name  # Local: load only a final successfully completed checkpoint.
            if read_json(folder / "status.json", {}).get("status") != "complete":
                continue  # Global: failed and omitted runs remain missing rather than receiving substitute estimates.
            model = make_model(frozen["config"], spec["mode"], spec["seed"]).cuda().eval()  # Local: reconstruct exact architecture from frozen configuration.
            model.load_state_dict(load_file(str(folder / "model.safetensors"), device="cuda"))  # Local: safe tensor-only public checkpoint format.
            for intervention in ([None, "uniform", "shuffle"] if spec["mode"] == "block" else [None]):
                output = destination / f"{name}{'_' + intervention if intervention else ''}.json"  # Local: within-model interventions cannot overwrite ordinary metrics.
                if output.exists():
                    continue  # Local: completed evaluation is resumable.
                selected = records  # Local: primary/secondary ordinary evaluations use every held-out trajectory.
                if intervention:
                    selected = [row for dataset in sorted(frozen["config"]["data"]) for row in [r for r in records if r["dataset"] == dataset][:4]]  # Global: fixed diagnostic subset is selected without prediction outcomes.
                if isinstance(model.attention_blocks, ResidualStack):
                    model.attention_blocks.intervention = intervention  # Local: weights change, source values and trained parameters remain fixed.
                    model.attention_blocks.generator.manual_seed(20261025)  # Local: depth permutations are identical across repeat evaluations.
                tick = time.monotonic()  # Local: capture complete evaluation cost.
                result = evaluate_model(model, prepared, selected, budget, capture=True, label=output.stem)  # Global: all interventions use the same causal rollout implementation.
                torch.cuda.synchronize()  # Local: include queued GPU work in timing.
                result.update(run=name, intervention=intervention, seconds=time.monotonic() - tick)  # Local: separate inference cost from training cost.
                write_json(output, result)  # Global: preserve raw per-trajectory results before analysis.
            del model  # Local: do not keep previous networks resident.
            torch.cuda.empty_cache()  # Local: bound GPU memory across the full matrix.
            budget.check()  # Global: evaluation cannot bypass the cumulative cap.
