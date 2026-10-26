"""Global: compare saved physics forecasts without new inference or outcome-based case selection.

Local: identical targets, frames, fields, and color limits make trained variants and interventions comparable.
Sources: https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html and
https://matplotlib.org/stable/gallery/images_contours_and_fields/multi_image.html.
"""
import matplotlib.pyplot as plt  # Local: render measured arrays with ordinary publication plotting tools.
from matplotlib.colors import Normalize  # Local: one normalization per field gives every model the same color mapping.
import numpy as np  # Local: preserve full saved precision until raster rendering.
from .runtime import LOG, root, read_json  # Global: use only the project's existing, measured evaluation artifacts.
from .data import Sampler, window_starts  # Global: persistence reuses the original data ordering and training-only transform.

MODES = ("standard", "block", "full", "uniform_block")  # Local: baseline first; controls retain their preregistered identities.


def matched_fields(folder, dataset, channel, step=12, runs=None):
    """Local: load a common frame, or all times with step=None; global: reject mismatched cases."""
    paths = [folder / f"{run}_{dataset}.npz" for run in (runs or tuple(f"{mode}_seed0" for mode in MODES))]  # Local: callers specify fixed models or interventions, never favorable cases.
    if not all(path.exists() for path in paths):
        LOG.info("All-model field comparison unavailable: dataset=%s", dataset)  # Global: missing evidence is explicit in the report log.
        return None  # Local: no partial panel is presented as a complete comparison.
    cases = []  # Local: materialize arrays before closing each compressed file.
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            cases.append({key: archive[key] for key in ("truth", "prediction", "active")})  # Local: only numeric arrays enter plotting.
    reference = cases[0]  # Local: the no-Attention-Residuals baseline identifies the shared target.
    for case in cases:
        compatible = np.array_equal(case["truth"], reference["truth"]) and np.array_equal(case["active"], reference["active"])  # Global: compare complete saved targets, not just the displayed slice.
        compatible = compatible and case["prediction"].shape == reference["truth"].shape and case["truth"].ndim == 4  # Local: forecast dimensions must align exactly.
        if not compatible or not case["active"][channel] or (step is not None and not 1 <= step <= len(case["truth"])):
            raise ValueError(f"incompatible saved field comparison: {dataset}")  # Global: never silently crop or compare an absent physical field.
    time = slice(None) if step is None else step - 1  # Local: full sequences preserve exact time ordering for animations; still figures retain their original frame.
    return np.stack([reference["truth"][time, ..., channel], *[case["prediction"][time, ..., channel] for case in cases]])  # Local: retain raw values, including nonfinite forecasts, in truth/baseline/treatment/control order.


def persistence_arrays(prepared, dataset):
    """Global: repeat only the final context frame; local: np.repeat preserves forecast shape (https://numpy.org/doc/stable/reference/generated/numpy.repeat.html)."""
    record = next(row for row in prepared["records"] if row["dataset"] == dataset and row["split"] == "test")  # Local: the same first test case used by saved model previews.
    sampler = Sampler(prepared, 0, split="test")  # Global: apply the unchanged training-only normalization and absent-field mask.
    start, trajectory = window_starts(record["length"])[0], sampler.array(record)  # Local: select the first prespecified window without inspecting outcomes.
    past = sampler.normalize(trajectory[start:start + 4], record)  # Global: prediction depends only on the four observed frames.
    return dict(prediction=np.repeat(past[-1:], 24, axis=0), truth=sampler.normalize(trajectory[start + 4:start + 28], record), active=np.asarray(record["active"]))  # Local: targets are retained only for comparison; they never enter persistence.


def comparison_figure(save_figure, runs, titles, name):
    """Local: one renderer serves trained models and controls; global: identical cases and unclipped scales."""
    specifications = (("rayleigh_benard_uniform", 2, "Convection\nbuoyancy"), ("turbulent_radiative_layer_2D", 1, "Radiative layer\ndensity"))  # Local: buoyancy is distinct from temperature; these fields were already fixed for previews.
    panels = [matched_fields(root() / "visuals", dataset, channel, runs=runs) for dataset, channel, _ in specifications]  # Local: reuse seed zero, first test trajectory/window, step twelve.
    if any(values is None for values in panels):
        return False  # Global: an unavailable comparison remains omitted rather than synthesized.
    figure, axes = plt.subplots(2, len(titles) + 1, figsize=(2 * len(titles) + 3, 5), gridspec_kw={"width_ratios": [1] * len(titles) + [.06]})  # Local: dedicated colorbar axes avoid unequal panel sizes.
    color = plt.get_cmap("viridis").with_extremes(bad="magenta")  # Global: nonfinite predictions are visible, never interpolated into plausible fields.
    for row, (values, (_, _, label)) in enumerate(zip(panels, specifications)):
        finite = values[np.isfinite(values)]  # Local: finite targets anchor the range even if a model diverges.
        norm = Normalize(vmin=float(finite.min()), vmax=float(finite.max()))  # Global: a shared range covers every selected panel, with no clipped extrema.
        for column, (field, title) in enumerate(zip(values, titles)):
            axis = axes[row, column]  # Local: each model occupies the same column across both datasets.
            image = axis.imshow(np.ma.masked_invalid(field.T), origin="lower", interpolation="nearest", norm=norm, cmap=color)  # Local: saved [x,y] arrays are transposed so x runs horizontally, without spatial smoothing.
            axis.set(title=title if row == 0 else "", xlabel="x grid index", xticks=[0, field.shape[0] - 1], yticks=[0, field.shape[1] - 1])  # Local: coordinates are explicitly grid indices, not invented physical units.
            axis.title.set_fontsize(16)  # Local: titles remain eight-point text when scaled to a 6.5-inch paper column.
            axis.xaxis.label.set_fontsize(14)  # Local: keep coordinate labels readable after the same paper scaling.
            axis.tick_params(labelsize=14)  # Local: shared grid coordinates remain legible without adding extra ticks.
            if column == 0:
                axis.set_ylabel(label + "\ny grid index", fontsize=14)  # Local: field identity appears once per common color scale.
        bar = figure.colorbar(image, cax=axes[row, -1])  # Local: one colorbar describes every panel in the row.
        bar.set_label("Standardized value", fontsize=14)  # Global: standardized signs are not interpreted as physical positivity.
        bar.ax.tick_params(labelsize=14)  # Local: color limits remain readable at publication size.
    figure.suptitle("Matched forecasts: step 12, seed 0, first test trajectory/window", fontsize=16)  # Global: display selection is explicit and independent of performance.
    save_figure(figure, name)  # Local: shared PNG/PDF exporter feeds README, website, and paper.
    return True  # Local: callers include the figure only after actual saved examples were rendered.


def field_comparison_figure(save_figure):
    """Global: preserve the four trained architecture comparison; local: reuse shared plotting and case validation."""
    return comparison_figure(save_figure, tuple(f"{mode}_seed0" for mode in MODES), ("Ground truth", "GPhyT without\nAttention Residuals", "Block AttnRes", "Full AttnRes", "Uniform Block"), "field_comparison")  # Local: stable original order and output filename.


def control_comparison_figure(save_figure):
    """Global: contrast ordinary forecasts with fixed and random interventions; local: no GPU or new neural inference."""
    prepared = read_json(root() / "prepared.json", {})  # Local: use actual selected trajectories, never synthetic substitutes.
    if not prepared:
        return False  # Global: pending preparation cannot produce an observed persistence example.
    for dataset in prepared["normalization"]:
        np.savez_compressed(root() / "visuals" / f"persistence_{dataset}.npz", **persistence_arrays(prepared, dataset))  # Local: reconstruct the deterministic baseline solely from the final observed frame.
    runs = ("persistence", "standard_seed0", "block_seed0", "block_seed0_uniform", "block_seed0_shuffle")  # Global: the random control is a within-model intervention, not a retrained random architecture.
    titles = ("Ground truth", "Persistence", "GPhyT without\nAttention Residuals", "Block AttnRes", "Block: uniform\nrouting intervention", "Block: shuffled\n(random) routing")  # Local: visibly distinguish trained Uniform Block from intervened Block.
    return comparison_figure(save_figure, runs, titles, "control_comparison")  # Global: one scale includes baseline, method, and control extrema.
