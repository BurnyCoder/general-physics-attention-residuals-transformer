"""Global: turn complete measured outcomes into reproducible paper tables and figures.

Local: missing/divergent seeds remain undefined; no frame is an inferential replicate.
Sources: docs/protocol.md, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html,
and https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html.
"""
import json  # Local: read complete update records for measured throughput.
import numpy as np  # Local: seed statistics and routing matrices use explicit arrays.
import pandas as pd  # Local: portable CSV and compact LaTeX tables share the same values.
from scipy.stats import sem, t  # Local: standard seed-level descriptive uncertainty.
from .evaluation import finite_mean  # Global: reuse strict propagation of missing or divergent constituents.
from .runtime import PROJECT, config, read_json, root  # Global: reporting reads saved evidence only.

LABELS = {"standard": "Standard", "block": "Block", "full": "Full", "uniform_block": "Uniform", "persistence": "Persistence"}  # Local: stable short labels make tables readable.
DATASETS = {"rayleigh_benard_uniform": "Convection", "turbulent_radiative_layer_2D": "Radiative"}  # Local: full dataset identities remain in all CSV files.


def aggregate_estimates(values, expected):
    """Local: summarize seeds only when the planned set is finite; global: no survivor-only architecture means."""
    array = np.asarray(values, dtype=float)  # Local: None becomes NaN and remains a missing constituent.
    row = dict(n_finite=int(np.isfinite(array).sum()), n_planned=expected, mean=None, sd=None, ci_low=None, ci_high=None)  # Global: expose availability even when inference is undefined.
    if len(array) != expected or not np.isfinite(array).all():
        return row  # Global: a failed or omitted seed cannot silently improve the reported mean.
    row["mean"] = float(array.mean())  # Local: arithmetic dataset error across the complete planned seed set.
    if expected > 1:
        margin = float(t.ppf(.975, expected - 1) * sem(array))  # Local: descriptive t interval assumes independent approximately normal seed means.
        row.update(sd=float(array.std(ddof=1)), ci_low=row["mean"] - margin, ci_high=row["mean"] + margin)  # Global: do not clip a negative lower confidence bound to disguise approximation limits.
    return row  # Local: persistence has one deterministic measurement and no invented seed interval.


def measured_tables():
    """Local: create accuracy, cost, and physical-diagnostic tables; global: preserve all variants and planned seeds."""
    cfg, accuracy, costs, physical, stability = config(), [], [], [], []  # Local: one shared collection of reporting evidence.
    for dataset in cfg["data"]:
        for mode in LABELS:
            seeds = [None] if mode == "persistence" else cfg["seeds"] if mode in {"standard", "block"} else cfg["control_seeds"]  # Global: inferential counts come from the plan, not successful files.
            for horizon in (1, 12, 24):
                values = [read_json(root() / "evaluation" / ("persistence.json" if seed is None else f"{mode}_seed{seed}.json"), {}).get("datasets", {}).get(dataset, {}).get(str(horizon)) for seed in seeds]  # Local: missing evaluations remain explicit.
                accuracy.append(dict(dataset=dataset, mode=mode, horizon=horizon, **aggregate_estimates(values, len(seeds))))  # Global: complete-matrix descriptive seed statistics.
    for folder in sorted((root() / "runs").glob("*")):
        status = read_json(folder / "status.json", {})  # Local: preserve partial and failed runs alongside completions.
        if not status:
            continue  # Local: an empty directory is not an execution measurement.
        path = folder / "metrics.jsonl"  # Local: per-update times include sample IO and synchronized GPU work.
        updates = pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]).drop_duplicates("step", keep="last") if path.exists() and path.stat().st_size else pd.DataFrame()  # Local: repeated checkpoint intervals count once in descriptive throughput.
        median = float(updates.seconds.median()) if len(updates) else None  # Local: empty attempts have no observed update speed.
        costs.append(dict(run=folder.name, status=status["status"], updates=status["step"], training_seconds=status.get("seconds"), median_update_seconds=median, examples_per_second=cfg["batch_size"] / median if median else None, peak_memory_mib=status.get("peak_memory_bytes", 0) / 2 ** 20))  # Global: throughput and allocated peak memory are separate outcomes.
    for path in sorted((root() / "evaluation").glob("*.json")):
        result = read_json(path)  # Local: physical diagnostics use original units saved during evaluation.
        for dataset in result["datasets"]:
            trajectories = [row for row in result["trajectories"] if row["dataset"] == dataset]  # Local: aggregate windows before trajectories.
            windows = [window for row in trajectories for window in row["windows"]]  # Local: descriptive failure counts retain their actual correlated-window unit.
            stability.append(dict(run=result.get("run", "persistence"), intervention=result.get("intervention"), dataset=dataset, windows=len(windows), undefined_12=sum(window["12"] is None for window in windows), undefined_24=sum(window["24"] is None for window in windows)))  # Global: completed training does not conceal unstable forecasts.
            keys = ["negative_pressure_fraction", "negative_density_fraction"] if dataset == "turbulent_radiative_layer_2D" else ["divergence_rms", "target_divergence_rms", "divergence_error_rms"]  # Global: use physics-appropriate diagnostics without assuming closed-system conservation.
            for key in keys:
                values = [finite_mean([window.get(key) for window in row["diagnostics"]]) for row in trajectories]  # Local: divergent or missing windows remain undefined.
                physical.append(dict(run=result.get("run", "persistence"), intervention=result.get("intervention"), dataset=dataset, diagnostic=key, trajectories=len(trajectories), value=finite_mean(values)))  # Global: never improve stability by dropping unstable trajectories.
    frames = dict(accuracy_summary=pd.DataFrame(accuracy), costs=pd.DataFrame(costs, columns=["run", "status", "updates", "training_seconds", "median_update_seconds", "examples_per_second", "peak_memory_mib"]), physical_diagnostics=pd.DataFrame(physical, columns=["run", "intervention", "dataset", "diagnostic", "trajectories", "value"]), rollout_stability=pd.DataFrame(stability, columns=["run", "intervention", "dataset", "windows", "undefined_12", "undefined_24"]))  # Local: even pending tables have a valid schema.
    for name, frame in frames.items():
        frame.to_csv(PROJECT / "reports" / f"{name}.csv", index=False)  # Global: every displayed summary has a reusable machine-readable source.
    return frames  # Local: figures and LaTeX use these exact reductions.


def empirical_figures(frames, primary, save_figure):
    """Local: create evidence plots only when measurements exist; global: retain all seed effects and missing-result annotations."""
    import matplotlib.pyplot as plt  # Local: the reporting entry point already selected a noninteractive backend.
    accuracy = frames["accuracy_summary"]  # Local: architecture averages never substitute for the paired primary test.
    if accuracy["mean"].notna().any():
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))  # Local: separate physical systems rather than averaging incomparable errors.
        for axis, (dataset, title) in zip(axes, DATASETS.items()):
            for mode, label in LABELS.items():
                group = accuracy[(accuracy.dataset == dataset) & (accuracy["mode"] == mode)]  # Local: one curve per prespecified architecture.
                axis.plot(group.horizon, group["mean"], "o-", label=label)  # Local: linearly display the three fixed horizons; missing estimates break the line.
            axis.set(title=title, xlabel="Forecast horizon (steps)", ylabel="Mean seed NMSE", xticks=[1, 12, 24], ylim=(0, None))  # Global: start at zero and disclose that each horizon averages its full forecast prefix.
        axes[-1].legend(frameon=False, fontsize=8)  # Local: label all tested architectures and persistence.
        save_figure(figure, "forecast_errors")  # Local: numerical summary and plot share one table.
    if primary["status"] == "complete":
        figure, axis = plt.subplots(figsize=(7, 3.8))  # Local: the pooled effect and all six seed pairs share one axis.
        axis.scatter(range(6), 100 * (1 - np.exp(primary["effects"])), label="Paired seed effects")  # Local: show every independent pair.
        mean, (low, high) = primary["improvement"], primary["improvement_ci95"]  # Local: transformed paired log-ratio interval from the primary analysis.
        axis.errorbar([7], [100 * mean], yerr=[[100 * (mean - low)], [100 * (high - mean)]], fmt="D", color="black", capsize=5, label="Primary effect, 95% CI")  # Global: uncertainty is across seeds, not pixels or correlated frames.
        axis.axhline(0, color="black", linewidth=.8)  # Local: negative effects explicitly indicate worse forecasts.
        axis.axhline(5, color="tab:orange", linestyle="--", label="Prespecified 5% threshold")  # Global: distinguish practical importance from a test of any difference.
        axis.set(xticks=[0, 1, 2, 3, 4, 5, 7], xticklabels=["0", "1", "2", "3", "4", "5", "Mean"], xlabel="Paired seed / pooled estimate", ylabel="Block error reduction (%)")  # Local: use the mathematically correct direction of improvement.
        axis.legend(frameon=False, fontsize=8)  # Local: explain point and interval semantics.
        save_figure(figure, "paired_effects")  # Global: this is the only confirmatory comparison figure.
    for name in ("block_seed0", "full_seed0"):
        result = read_json(root() / "evaluation" / f"{name}.json", {})  # Local: predetermined representative seeds, never the most attractive pattern.
        if not result.get("routing"):
            continue  # Local: absent measured routing never becomes a synthetic heatmap.
        route = result["routing"][0]  # Local: first diagnostic trajectory and first fixed window in manifest order.
        trace = route["trace"]  # Local: each row corresponds to one attention/MLP/final read.
        matrix = np.full((len(trace), max(row["sources"] for row in trace)), np.nan)  # Local: unavailable future sources remain blank.
        for index, row in enumerate(trace):
            matrix[index, :row["sources"]] = np.asarray(row["weights"], dtype=float)  # Local: undefined measured weights remain NaN, not zero.
        figure, axis = plt.subplots(figsize=(6.4, 5))  # Local: allow the 25 read sites to remain distinguishable.
        color = axis.imshow(matrix, vmin=0, vmax=1, aspect="auto", interpolation="nearest", cmap="viridis")  # Global: fixed probability limits support comparison without exaggerated color scales.
        figure.colorbar(color, ax=axis, label="Token-mean routing weight")  # Local: the color represents actual normalized depth weight.
        axis.set(xlabel="Available source index (0 = embedding)", ylabel="Read site", title=f"{name}, first fixed window, step {route['final_forecast_step']}")  # Global: the figure discloses its exact selection and actual attempted step.
        save_figure(figure, f"routing_{name}")  # Local: deterministic names link manuscript and standalone output.


def measured_latex(frames):
    """Local: render concise empirical tables; global: use explicit undefined labels rather than numeric placeholders."""
    accuracy, rows = frames["accuracy_summary"], []  # Local: retain one row per dataset and architecture.
    for dataset, title in DATASETS.items():
        for mode, label in LABELS.items():
            row = {"Dataset": title, "Model": label}  # Local: compact labels are defined in the methods and CSV metadata.
            for horizon in (1, 12, 24):
                value = accuracy[(accuracy.dataset == dataset) & (accuracy["mode"] == mode) & (accuracy.horizon == horizon)].iloc[0]  # Local: exactly one planned summary per endpoint.
                row[f"{horizon} step"] = f"{value['mean']:.4g}" if pd.notna(value["mean"]) else "undefined"  # Global: incomplete or divergent architectures stay unranked.
                if horizon == 12:
                    row["Finite/planned"] = f"{int(value.n_finite)}/{int(value.n_planned)}"  # Local: disclose the primary-horizon availability without inflating the sample size.
            rows.append(row)  # Local: all five architectures, including persistence, enter each dataset table.
    table = pd.DataFrame(rows)[["Dataset", "Model", "Finite/planned", "1 step", "12 step", "24 step"]]  # Local: put completion before accuracy so missingness is visible.
    text = "\\begin{table}[ht]\\centering\\small\n" + table.to_latex(index=False, escape=True) + "\\caption{Arithmetic mean seed NMSE for each fixed horizon. A horizon averages all forecast steps up to that horizon. Architecture means require every planned seed to have a finite endpoint. Finite/planned counts refer to twelve steps; persistence is a deterministic baseline, not a one-seed uncertainty estimate. Complete descriptive seed SDs and t intervals are in accuracy\\_summary.csv.}\\end{table}\n"  # Global: descriptive errors are distinct from the paired log-ratio primary estimand.
    if accuracy["mean"].notna().any():
        text += "\\begin{figure}[ht]\\centering\\includegraphics[width=\\linewidth]{../reports/figures/forecast_errors.pdf}\\caption{Fixed-horizon trajectory-averaged errors for both datasets. Missing values remain missing.}\\end{figure}\n"  # Local: include only a figure backed by measurements.
    if len(frames["costs"]):
        cost = frames["costs"][["run", "updates", "training_seconds", "examples_per_second", "peak_memory_mib"]].copy()  # Local: show every run rather than selecting the fastest.
        cost.columns = ["Run", "Updates", "Wall seconds", "Examples/s", "Peak MiB"]  # Local: distinguish allocated memory and wall-time throughput.
        text += "\\begin{table}[ht]\\centering\\scriptsize\n" + cost.to_latex(index=False, escape=True, float_format="%.2f", na_rep="undefined") + "\\caption{Per-run cost, including failed or incomplete runs. Examples/s uses the median synchronized update time; wall seconds also include validation and checkpointing. Peak memory is PyTorch allocated device memory.}\\end{table}\n"  # Global: equal updates do not imply equal compute cost.
    interventions = pd.read_csv(PROJECT / "reports/interventions.csv")  # Local: these ratios already match ordinary and intervened trajectory subsets.
    if len(interventions):
        rows = []  # Local: summarize six within-model effects per dataset and intervention, without primary-test reuse.
        for (dataset, kind), group in interventions[interventions.horizon == 12].groupby(["dataset", "intervention"]):
            estimate = aggregate_estimates(group.ratio.tolist(), len(config()["seeds"]))  # Global: a missing intervention run cannot create a successful-subset mean.
            rows.append({"Dataset": DATASETS[dataset], "Routing": kind, "Finite/planned": f"{estimate['n_finite']}/{estimate['n_planned']}", "Mean error ratio": estimate["mean"], "Seed SD": estimate["sd"]})  # Local: ratios above one indicate harm from intervening on routing.
        text += "\\begin{table}[ht]\\centering\\small\n" + pd.DataFrame(rows).to_latex(index=False, escape=True, float_format="%.4g", na_rep="undefined") + "\\caption{Within-model twelve-step routing interventions on the same first four test trajectories per dataset. Values are intervened/original NMSE ratios, averaged over trained Block seeds. These are exploratory diagnostics, not independently trained controls or evidence of a unique mechanism.}\\end{table}\n"  # Global: disclose the causal intervention's exact scope.
    physical = frames["physical_diagnostics"]  # Local: diagnostics in physical units were saved independently of NMSE aggregation.
    if len(physical):
        names = {"divergence_rms": "Predicted divergence RMS", "target_divergence_rms": "Target divergence RMS", "divergence_error_rms": "Divergence error RMS", "negative_pressure_fraction": "Negative pressure fraction", "negative_density_fraction": "Negative density fraction"}  # Local: readable labels retain the original quantitative meanings.
        rows = []  # Local: focus the paper's physical table on primary architectures and persistence; full controls remain in CSV.
        for (dataset, diagnostic), group in physical[physical.intervention.isna()].groupby(["dataset", "diagnostic"]):
            row = {"Dataset": DATASETS[dataset], "Diagnostic": names[diagnostic]}  # Local: positivity and divergence are not interchangeable physical constraints.
            for mode in ("standard", "block", "persistence"):
                selected = group[group.run.eq("persistence") if mode == "persistence" else group.run.str.startswith(mode + "_seed")]  # Local: exclude intervention runs and other architectures.
                row[LABELS[mode]] = aggregate_estimates(selected.value.tolist(), 1 if mode == "persistence" else len(config()["seeds"]))["mean"]  # Global: average only a complete planned seed set.
            rows.append(row)  # Local: one physical quantity receives one explicit row.
        text += "\\begin{table}[ht]\\centering\\scriptsize\n" + pd.DataFrame(rows).to_latex(index=False, escape=True, float_format="%.3g", na_rep="undefined") + "\\caption{Secondary physical diagnostics averaged over all 24 forecast steps, then windows, trajectories, and complete planned seed sets. Divergence uses the stored physical coordinates and coarsened target; positivity rates are fractions. These diagnostics do not establish exact conservation. All architectures and interventions are retained in physical\\_diagnostics.csv.}\\end{table}\n"  # Global: physical claims remain limited to the measured proxy.
    stability = frames["rollout_stability"]  # Local: raw failure counts complement undefined aggregate NMSE.
    if len(stability):
        ordinary = stability[stability.intervention.isna()]  # Local: distinguish ordinary forecasts from deliberate interventions.
        text += f"Across the available ordinary evaluations, {int(ordinary.undefined_12.sum())}/{int(ordinary.windows.sum())} forecast windows have undefined twelve-step error, and {int(ordinary.undefined_24.sum())}/{int(ordinary.windows.sum())} have undefined twenty-four-step error. These descriptive counts include persistence and repeated model evaluations; they are not independent replicates or an inferential sample size. Per-run counts are preserved in rollout\\_stability.csv.\n"  # Global: unstable forecasts remain visible even when all training runs completed.
    return text  # Local: the reporting phase inserts this artifact into the measured-results section.
