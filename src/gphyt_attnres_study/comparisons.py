"""Global: compare saved physics forecasts without new inference or outcome-based case selection.

Local: identical targets, frames, fields, and color limits make all four variants comparable.
Sources: https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html and
https://matplotlib.org/stable/gallery/images_contours_and_fields/multi_image.html.
"""
import matplotlib.pyplot as plt  # Local: render measured arrays with ordinary publication plotting tools.
from matplotlib.colors import Normalize  # Local: one normalization per field gives every model the same color mapping.
import numpy as np  # Local: preserve full saved precision until raster rendering.
from .runtime import LOG, root  # Global: use only the project's existing, measured evaluation artifacts.

MODES = ("standard", "block", "full", "uniform_block")  # Local: baseline first; controls retain their preregistered identities.


def matched_fields(folder, dataset, channel, step=12):
    """Local: load one common seed-zero forecast frame; global: fail if saved targets or masks disagree."""
    paths = [folder / f"{mode}_seed0_{dataset}.npz" for mode in MODES]  # Local: no search for visually favorable seeds or trajectories.
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
        if not compatible or not case["active"][channel] or not 1 <= step <= len(case["truth"]):
            raise ValueError(f"incompatible saved field comparison: {dataset}")  # Global: never silently crop or compare an absent physical field.
    return np.stack([reference["truth"][step - 1, ..., channel], *[case["prediction"][step - 1, ..., channel] for case in cases]])  # Local: retain raw values, including nonfinite forecasts, in truth/baseline/treatment/control order.


def field_comparison_figure(save_figure):
    """Local: draw the same representative fields as the saved rollout previews; global: no new scientific endpoint."""
    specifications = (("rayleigh_benard_uniform", 2, "Convection\nbuoyancy"), ("turbulent_radiative_layer_2D", 1, "Radiative layer\ndensity"))  # Local: buoyancy is distinct from temperature; these fields were already fixed for previews.
    panels = [matched_fields(root() / "visuals", dataset, channel) for dataset, channel, _ in specifications]  # Local: reuse seed zero, first test trajectory/window, step twelve.
    if any(values is None for values in panels):
        return False  # Global: an unavailable comparison remains omitted rather than synthesized.
    figure, axes = plt.subplots(2, 6, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, .06]})  # Local: dedicated colorbar axes avoid unequal panel sizes.
    titles = ("Ground truth", "GPhyT without\nAttention Residuals", "Block AttnRes", "Full AttnRes", "Uniform Block")  # Global: the unmodified baseline is unmistakable beside every trained alternative.
    color = plt.get_cmap("viridis").with_extremes(bad="magenta")  # Global: nonfinite predictions are visible, never interpolated into plausible fields.
    for row, (values, (_, _, label)) in enumerate(zip(panels, specifications)):
        finite = values[np.isfinite(values)]  # Local: finite targets anchor the range even if a model diverges.
        norm = Normalize(vmin=float(finite.min()), vmax=float(finite.max()))  # Global: a shared range covers all five panels, with no clipped extrema.
        for column, (field, title) in enumerate(zip(values, titles)):
            axis = axes[row, column]  # Local: each model occupies the same column across both datasets.
            image = axis.imshow(np.ma.masked_invalid(field.T), origin="lower", interpolation="nearest", norm=norm, cmap=color)  # Local: saved [x,y] arrays are transposed so x runs horizontally, without spatial smoothing.
            axis.set(title=title if row == 0 else "", xlabel="x grid index", xticks=[0, field.shape[0] - 1], yticks=[0, field.shape[1] - 1])  # Local: coordinates are explicitly grid indices, not invented physical units.
            axis.title.set_fontsize(16)  # Local: titles remain eight-point text when scaled to a 6.5-inch paper column.
            axis.xaxis.label.set_fontsize(14)  # Local: keep coordinate labels readable after the same paper scaling.
            axis.tick_params(labelsize=14)  # Local: shared grid coordinates remain legible without adding extra ticks.
            if column == 0:
                axis.set_ylabel(label + "\ny grid index", fontsize=14)  # Local: field identity appears once per common color scale.
        bar = figure.colorbar(image, cax=axes[row, -1])  # Local: one colorbar describes all five panels in the row.
        bar.set_label("Standardized value", fontsize=14)  # Global: standardized signs are not interpreted as physical positivity.
        bar.ax.tick_params(labelsize=14)  # Local: color limits remain readable at publication size.
    figure.suptitle("Matched forecasts: step 12, seed 0, first test trajectory/window", fontsize=16)  # Global: display selection is explicit and independent of performance.
    save_figure(figure, "field_comparison")  # Local: shared PNG/PDF exporter feeds README, website, and paper.
    return True  # Local: callers include the figure only after actual saved examples were rendered.
