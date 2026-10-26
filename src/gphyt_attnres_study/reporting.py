"""Global: regenerate reports and publication figures from measured artifacts.

Local: missing results remain missing; synthetic verification never enters accuracy tables.
Sources: https://matplotlib.org/stable/users/index.html and https://imageio.readthedocs.io/.
"""
import json  # Local: read complete per-update logs without replacing raw evidence.
import os  # Local: choose an explicitly configured compiler without hardcoding this machine's application path.
import shutil  # Local: copy measured result files into the public report tree.
import imageio.v2 as imageio  # Local: encode fixed-case autoregressive videos from saved arrays.
import matplotlib  # Local: use a headless plotting backend for reproducible batch reports.
matplotlib.use("Agg")  # Local: plotting never opens interactive native windows.
import matplotlib.pyplot as plt  # Local: standard publication graphics with explicit axes.
import numpy as np  # Local: heatmaps, binning, and numerical video color mapping.
import pandas as pd  # Local: tabular output and exact conversion to LaTeX.
from .analysis import analyze, power  # Global: one inferential implementation feeds all reporting surfaces.
from .comparisons import field_comparison_figure, control_comparison_figure  # Global: identical held-out cases show trained architectures and routing controls beside baselines.
from .untrained_reporting import untrained_report  # Global: initialization-only controls remain a separate, explicitly post-hoc result.
from .headlines import radiative_highlight  # Global: radiative headline is descriptive and cannot redefine primary inference.
from .animations import rollout_gifs  # Global: synchronized GIFs replay every saved architecture/control alongside truth and persistence.
from .training import execution_order  # Global: report planned failed and omitted runs as well as successes.
from .results import measured_tables, empirical_figures, measured_latex  # Global: reusable reductions keep paper tables and figures consistent.
from .runtime import LOG, PROJECT, config, read_json, root, run_logged, stamp, write_json  # Global: shared durable artifacts and timestamps.


def save_figure(figure, name):
    """Local: save raster preview and vector paper figure; global: avoid screenshot-based scientific charts."""
    folder = PROJECT / "reports/figures"  # Local: all generated figures have one public home.
    folder.mkdir(parents=True, exist_ok=True)  # Local: create only project-owned output directories.
    figure.tight_layout()  # Local: fit axis labels and legends inside the canvas.
    for extension in ("png", "pdf"):
        temporary = folder / f"{name}.tmp.{extension}"  # Local: readers and the live LaTeX preview never see a partially written final figure.
        figure.savefig(temporary, dpi=180)  # Local: sharp raster previews and vector PDF use standard Matplotlib exporters.
        temporary.replace(folder / f"{name}.{extension}")  # Global: publish a complete figure atomically after export succeeds.
    plt.close(figure)  # Local: release memory across many experiment plots.


def compile_paper():
    """Local: export a real PDF using an installed compiler; global: the static viewer never silently ships stale paper text."""
    executable = os.environ.get("LATEX_EXECUTABLE") or shutil.which("tectonic") or shutil.which("pdflatex")  # Local: support portable installations and an explicit local runtime path.
    if not executable:
        raise RuntimeError("Install Tectonic or pdfLaTeX, or set LATEX_EXECUTABLE in .env, to complete paper reporting")  # Global: missing compilation is explicit rather than a claimed PDF deliverable.
    folder = PROJECT / "paper"  # Local: compiler reads the dedicated multifile project.
    if "tectonic" in executable.lower():
        run_logged([executable, "--keep-logs", "main.tex"], cwd=folder)  # Local: Tectonic automatically reruns for references and writes the PDF beside the source.
    else:
        for _ in range(2):
            run_logged([executable, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=folder)  # Local: two pdfLaTeX passes resolve cross-references without requiring Perl/latexmk.


def power_figure():
    """Local: visualize prospective assumptions; global: never label simulated power as observed accuracy."""
    values = pd.DataFrame(power()["rows"])  # Local: reuse the preregistered simulation, not a new outcome-informed calculation.
    figure, axis = plt.subplots(figsize=(6.4, 3.6))  # Local: readable single-column-like chart dimensions.
    for effect, group in values.groupby("reduction"):
        axis.plot(group.seed_log_sd, group.sign_flip_power, "o-", label=f"{effect:.0%} improvement")  # Local: each curve is one explicitly assumed effect.
    axis.set(xlabel="Assumed SD of paired seed log-error ratio", ylabel="Exact-test rejection probability", ylim=(0, 1.02), title="Prospective power with six independent pairs")  # Global: bounded probability scale makes limited power visible.
    axis.legend(frameon=False)  # Local: labels identify hypothetical effect sizes.
    save_figure(figure, "power_sensitivity")  # Local: same figure appears in the paper and reports.


def learning_figures():
    """Local: bin all training updates uniformly; global: avoid selecting favorable loss segments."""
    figure, axis = plt.subplots(figsize=(8, 4.5))  # Local: room for all eighteen runs without overlapping the legend.
    measured = False  # Local: do not emit an empty chart as if training occurred.
    for path in sorted((root() / "runs").glob("*/metrics.jsonl")):
        rows = []  # Local: tolerate only an incomplete final append, not invalid scientific values.
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))  # Local: preserve each complete logged update.
            except json.JSONDecodeError:
                LOG.warning("Incomplete training log line: %s", path)  # Global: a concurrent/failed write remains visible.
        if not rows:
            continue  # Local: empty logs supply no evidence.
        frame = pd.DataFrame(rows).drop_duplicates("step", keep="last")  # Local: interrupted unsaved intervals may be replayed; retain their latest complete measurement.
        groups = frame.groupby((frame.step - 1) // 50).agg(step=("step", "max"), loss=("loss", "mean"))  # Local: disjoint fixed fifty-update bins include all observed steps.
        axis.plot(groups.step, groups.loss, linewidth=1, label=path.parent.name)  # Local: no best-seed selection.
        measured = True  # Local: a real update log exists.
    if measured:
        axis.set(xlabel="Optimizer updates", ylabel="Training MSE (50-update means)", yscale="log", title="All recorded training runs")  # Global: logarithmic axis and averaging are explicit.
        axis.legend(fontsize=6, ncol=3, frameon=False)  # Local: compact legend keeps every run identifiable.
        save_figure(figure, "learning_curves")  # Local: publish the same inclusive learning view.
    else:
        plt.close(figure)  # Local: missing training is represented in text, not fabricated curves.


def rollout_videos():
    """Local: render predetermined saved cases; global: common color limits preserve amplitude comparison."""
    folder = PROJECT / "reports/videos"  # Local: short videos are public paper supplements.
    folder.mkdir(parents=True, exist_ok=True)  # Local: create only the report-owned destination.
    for path in sorted((root() / "visuals").glob("*.npz")):
        if not path.name.startswith(("standard_seed0_", "block_seed0_")) or "_uniform_" in path.name or "_shuffle_" in path.name:
            continue  # Global: prespecified seed zero examples, not visually selected best cases.
        arrays = np.load(path)  # Local: saved evaluation arrays avoid extra GPU executions.
        channel = 2 if arrays["active"][2] else 1  # Local: buoyancy for convection, density for the radiative layer.
        truth, prediction = arrays["truth"][..., channel], arrays["prediction"][..., channel]  # Local: identical physical field on both sides.
        low, high = float(np.min(truth)), float(np.max(truth))  # Local: one target-derived scale across the complete clip.
        frames = []  # Local: twenty-four small frames fit comfortably in memory.
        for time_index in range(len(truth)):
            fields = np.concatenate([truth[time_index], prediction[time_index]], axis=1)  # Local: truth left, model right, same time index.
            image = (plt.get_cmap("viridis")(np.clip((fields - low) / max(high - low, 1e-12), 0, 1))[..., :3] * 255).astype(np.uint8)  # Local: fixed color mapping with a safe constant-field denominator.
            image[~np.isfinite(fields)] = [255, 0, 255]  # Global: divergent pixels are visibly magenta, never clipped into plausible fields.
            frames.append(np.repeat(np.repeat(image, 3, axis=0), 3, axis=1))  # Local: nearest-neighbor enlargement preserves cell values.
        imageio.mimsave(folder / f"{path.stem}.mp4", frames, fps=6, macro_block_size=1)  # Local: a short four-second clip shows all 24 forecast steps.
    (folder / "README.md").write_text("# Predetermined rollout videos\n\nTruth is on the left and autoregressive prediction on the right. Each clip uses the first test trajectory and first fixed window for its dataset, seed 0, all 24 forecast steps, and one target-derived color scale across time. Convection displays buoyancy; the radiative layer displays density. Values are in training-standardized coordinates; magenta marks nonfinite predictions. See [the protocol](../../docs/protocol.md).\n", encoding="utf-8")  # Global: video selection and encoding are fully disclosed.


def experiment_reports():
    """Local: one report per planned run; global: failures and omissions cannot disappear from the consolidated record."""
    frozen, cfg = read_json(root() / "frozen.json"), config()  # Local: permit a transparent pre-results report before profiling.
    order = frozen["order"] if frozen else execution_order(cfg)  # Local: the approved matrix exists even before execution.
    ledger = read_json(root() / "gpu_budget.json", {"sessions": []})  # Local: identify budget-driven omissions from recorded execution, not guessed intent.
    exhausted = any(row["phase"] == "train" and (row["status"] == "hard_budget_stop" or row["status"] == "failed" and ("allocation" in row.get("error", "") or "cap" in row.get("error", ""))) for row in ledger["sessions"])  # Local: only an explicit cooperative or independently enforced training budget failure explains an omitted run.
    folder = PROJECT / "reports/experiments"  # Local: individual reports do not duplicate the full protocol.
    folder.mkdir(parents=True, exist_ok=True)  # Local: keep generated per-run documents organized.
    rows = []  # Local: build a consolidated outcome table from all planned runs.
    for spec in order:
        name = f"{spec['mode']}_seed{spec['seed']}"  # Local: align checkpoint, log, and report identities.
        status = read_json(root() / "runs" / name / "status.json", {"status": "budget_omitted" if exhausted else "not_started", "step": 0})  # Global: preserve budget-driven exclusions without inventing executed updates.
        if exhausted and status["status"] == "running":
            status = dict(status, status="budget_stopped", error="Independent hard stop; progress reflects the last saved run status")  # Global: a terminated worker must not remain displayed as currently training.
        evaluation = read_json(root() / "evaluation" / f"{name}.json", {})  # Local: primary errors come only from final-checkpoint evaluation.
        values = {dataset: evaluation.get("datasets", {}).get(dataset, {}).get("12") for dataset in cfg["data"]}  # Local: missing estimates stay None.
        rows.append(dict(run=name, status=status["status"], updates=status["step"], gpu_phase_seconds=status.get("seconds"), **values))  # Global: include all planned seeds in the consolidated table.
        body = f"# {name}\n\nStatus: **{status['status']}**. Completed updates: {status['step']}.\n\n"  # Local: lead with observed completion rather than an implied success.
        body += "Method, data, optimization rationale, fixed endpoints, and analysis assumptions are specified once in [the protocol](../../docs/protocol.md). This run shares its backbone initialization and data schedule with the other variants at the same seed.\n\n"  # Global: avoid duplicated, drifting descriptions across eighteen reports.
        body += "## Measured twelve-step errors\n\n" + "\n".join(f"- {dataset}: {value if value is not None else 'not available / undefined'}" for dataset, value in values.items()) + "\n\n"  # Local: no fabricated numerical placeholders.
        body += "## Interpretation and learnings\n\n"  # Local: distinguish observations from causal explanations.
        body += ("This single run cannot establish a seed-robust architecture effect. Interpret it with the preregistered paired analysis and all failures. No explanation of its error is inferred from a single curve.\n" if status["status"] == "complete" else "No complete fixed-checkpoint evaluation is established for this run. It contributes no invented accuracy estimate.\n")  # Global: match claim strength to the available evidence.
        body += "\nThe [completed-study interpretation](../interpretation.md) discusses absolute persistence comparisons, physical failures, controls, and the post-hoc metric diagnosis.\n"  # Local: share measured learnings without duplicating or drifting across eighteen reports.
        if "error" in status:
            body += f"\nRecorded failure: `{status['error']}`. The attempt remains visible; no new seed is substituted.\n"  # Local: preserve failure information without claiming an unverified diagnosis.
        body += "\nExact run metadata:\n\n```json\n" + json.dumps(status, indent=2) + "\n```\n"  # Local: revision, cost, and frozen-config identity accompany the report.
        (folder / f"{name}.md").write_text(body, encoding="utf-8")  # Global: regenerate from durable evidence, not conversational memory.
    frame = pd.DataFrame(rows)  # Local: all eighteen planned rows enter one table.
    frame.to_csv(PROJECT / "reports/experiments.csv", index=False)  # Local: reusable machine-readable run index.
    return frame  # Local: consolidated Markdown and paper tables share the same evidence.


def report():
    """Local: rebuild tables, figures, reports, and numerical paper sections; global: factual claims follow saved evidence."""
    summary = analyze()  # Global: a single primary analysis definition serves all outputs.
    runs = experiment_reports()  # Local: failed, incomplete, and successful runs are all included.
    power_figure()  # Local: prospective sensitivity exists even while training is pending.
    learning_figures()  # Local: generate a learning chart only when real update logs exist.
    measured = measured_tables()  # Global: descriptive accuracy, cost, and physical outcomes use the full planned matrix.
    empirical_figures(measured, summary, save_figure)  # Local: include measured error curves, primary confidence intervals, and fixed-case routing.
    field_comparison_available = field_comparison_figure(save_figure)  # Local: saved arrays provide fair examples without additional model execution.
    control_comparison_available = control_comparison_figure(save_figure)  # Local: existing intervention forecasts and deterministic persistence use the same held-out cases.
    initialized_results = untrained_report(measured["accuracy_summary"], save_figure)  # Local: the existing persistence previews enable matched zero-update comparisons without new inference.
    highlight = radiative_highlight()  # Local: require all nine actual full-test comparator means before highlighting a winner.
    write_json(PROJECT / "reports/radiative_highlight.json", highlight)  # Global: every public surface shares the same guarded claim.
    rollout_videos()  # Local: generate clips only from saved evaluation predictions.
    rollout_gifs()  # Local: include the requested animated supplement in ordinary report reproductions.
    public_evaluation = PROJECT / "reports/evaluation"  # Local: raw trajectory measurements accompany aggregate claims.
    public_evaluation.mkdir(exist_ok=True)  # Local: public output directory is explicit.
    for path in (root() / "evaluation").glob("*.json"):
        shutil.copy2(path, public_evaluation / path.name)  # Global: independent readers can recompute every table.
    ledger = read_json(root() / "gpu_budget.json", {"sessions": []})  # Local: include all recorded failures and GPU phases.
    write_json(PROJECT / "reports/gpu_budget.json", ledger)  # Global: publish actual cost separately from requested allocation.
    complete = int((runs.status == "complete").sum())  # Local: explicit completion count prevents mislabeling a launched job as finished.
    text = f"# Consolidated experiment record\n\nGenerated {stamp()}. {complete}/18 planned runs completed; primary inference status: **{summary['status']}**.\n\n"  # Local: publication clearly states current evidence availability.
    if highlight:
        text += "## " + highlight["title"] + "\n\n" + highlight["summary"] + "\n\n" + highlight["scope"] + " See [trained scores](accuracy_summary.csv) and [initialized scores](untrained/accuracy_summary.csv).\n\n"  # Local: lead with the requested descriptive result while retaining both evidence sources.
    text += "This is a from-scratch, supervised next-frame regression study. A **seed pair** is two models with matching initial backbone weights and training examples. **NMSE** divides spatial squared error by target energy; lower is better. A **rollout** repeatedly feeds predictions back as inputs. A **confidence interval** quantifies uncertainty under the stated seed-level assumptions, not a guarantee for unseen physics.\n\n"  # Global: explain key terms without diluting technical definitions in the protocol.
    text += "| Network | Depth composition | Algorithm/objective | Data/curriculum | Completion/accuracy |\n|---|---|---|---|---|\n"  # Local: a compact architecture comparison addresses all tested networks.
    for mode, composition in [("standard", "Upstream additive residuals"), ("block", "Learned attention over four-sublayer sums"), ("full", "Learned attention over every raw output"), ("uniform_block", "Fixed averaging over the Block source bank")]:
        subset = runs[runs.run.str.startswith(mode + "_seed")]  # Local: match exact variant prefix.
        text += f"| {mode} | {composition} | AdamW / masked MSE | Two fixed physics datasets; balanced sampling; strides 1-8 | {int((subset.status == 'complete').sum())}/{len(subset)} complete; see measured CSV |\n"  # Global: no rank or accuracy claim precedes results.
    text += "\nAll models share twelve layers, width 192, MLP width 768, three heads, and 1x16x16 patches. Fill/completion benchmarks and reinforcement-learning reward are not applicable. Detailed raw outcomes are in [experiments.csv](experiments.csv), [metrics.csv](metrics.csv), and the [individual reports](experiments).\n\n"  # Local: distinguish requested generic metrics from this regression domain.
    text += "## Primary analysis\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n\n"  # Global: confidence limits and undefined status remain exact.
    text += "## Interpretation and learnings\n\nSee the [completed-study interpretation](interpretation.md) for the relative gain, severe convection failure against persistence, physical diagnostics, exploratory controls, and explicitly post-hoc metric decomposition. The numerical record here and that source-linked interpretation form the consolidated report.\n\n"  # Local: keep detailed empirical interpretation in one maintained location.
    if initialized_results:
        text += "The later user-requested [initialization-only baseline report](untrained/summary.md) compares all four architectures before and after training on matching seeds and test trajectories. These additional measurements are post-hoc; they do not change the original paired endpoint.\n\n"  # Global: the consolidated record includes the extension without merging experimental populations.
    text += "## Sources and audit\n\nSee [method and prespecified decisions](../docs/protocol.md), [literature/code review](../docs/sources.md), [data preparation and initial-state reuse](data-preparation.md), [development learnings](development.md), and the [manuscript](../paper/main.tex). Raw data and checkpoints are excluded from Git; public checkpoint links appear in publication.json after a successful upload.\n"  # Local: centralize explanations instead of copying the protocol into every report.
    (PROJECT / "reports/summary.md").write_text(text, encoding="utf-8")  # Global: a single cumulative summary stays synchronized with actual artifacts.
    result_text = f"At this report revision, {complete} of 18 planned training runs are complete. "  # Local: numerical paper section is regenerated from status files.
    if highlight:
        result_text = "\\subsection{Radiative-flow highlight: lowest mean error among tested baselines}\n" + highlight["summary"].replace("%", r"\%") + "\n" + highlight["scope"] + "\n\\subsection{Unchanged preregistered two-dataset comparison}\n" + result_text  # Global: results lead with the observed ranking without assigning it the primary p-value.
    if summary["status"] == "complete":
        low, high = summary["improvement_ci95"]  # Local: transformed interval already has correctly ordered limits.
        result_text += f"Block Attention Residuals reduce the primary error endpoint by {100 * summary['improvement']:.2f}\\% relative to GPhyT without Attention Residuals, with 95\\% confidence interval [{100 * low:.2f}, {100 * high:.2f}]\\% and exact paired p-value {summary['permutation_p']:.5f}. "  # Global: identify the matched baseline with the effect size and uncertainty.
        result_text += "These estimates condition on the selected datasets and fixed evaluation windows.\n"  # Local: restrict the interpretation to the independent units actually varied.
        threshold = "The interval supports the prespecified minimum 5\\% improvement." if low > .05 else "The interval lies below the prespecified 5\\% practical-improvement threshold." if high < .05 else "The interval overlaps the prespecified 5\\% threshold; practical importance remains inconclusive."  # Global: decide the practical claim using the criterion published before outcomes.
        result_text += threshold + "\n\\begin{figure}[ht]\\centering\\includegraphics[width=.95\\linewidth]{../reports/figures/paired_effects.pdf}\\caption{All six paired seed effects and the primary transformed t confidence interval. The dashed line is the prespecified practical threshold.}\\end{figure}\n"  # Local: show uncertainty rather than merely describing significance.
    else:
        result_text += "Primary effect and uncertainty estimates are undefined because complete finite endpoints for all six preregistered pairs are not yet established. No accuracy improvement is claimed.\n"  # Global: never turn an incomplete study into a null finding.
    result_text += measured_latex(measured)  # Global: accuracy and cost tables are rendered directly from saved measurements.
    if control_comparison_available:
        result_text += "\\begin{figure}[ht]\\centering\\includegraphics[width=\\linewidth]{../reports/figures/control_comparison.pdf}\\caption{Baselines, new method, and controls on the same first test trajectory/window at step twelve: persistence, GPhyT without Attention Residuals, ordinary Block, and Block with uniform or randomly shuffled depth weights. Interventions use the same trained seed-zero Block checkpoint without retraining; shuffling preserves learned weight magnitudes but permutes source assignments with fixed RNG seed 20261025. Persistence repeats the last observed context frame. Rows show convection buoyancy and radiative density with one shared, unclipped color scale per row in training-standardized units. This post-result presentation reuses fixed cases and saved intervention forecasts; it introduces no new endpoint or GPU inference.}\\end{figure}\n"  # Global: distinguish random routing intervention from a separately trained random control.
    if field_comparison_available:
        result_text += "\\begin{figure}[ht]\\centering\\includegraphics[width=\\linewidth]{../reports/figures/field_comparison.pdf}\\caption{Ground truth and all four trained architectures on identical held-out cases: seed zero, first test trajectory and first fixed window, forecast step twelve. Each row uses one common color range spanning truth and all predictions, without clipping, in training-standardized units; magenta would mark nonfinite values. Convection shows buoyancy (not temperature); radiative flow shows density. These illustrative fields do not replace the full paired evaluation. The all-model layout was added after results at the user's request, reusing the already fixed preview cases.}\\end{figure}\n"  # Global: disclose post-result presentation while retaining fixed, matched examples.
    for name in ("block_seed0", "full_seed0"):
        if read_json(root() / "evaluation" / f"{name}.json", {}).get("routing"):
            result_text += "\\begin{figure}[ht]\\centering\\includegraphics[width=.72\\linewidth]{../reports/figures/routing_" + name + ".pdf}\\caption{Routing in the prespecified seed-zero, first-trajectory, first-window case. White cells represent unavailable or undefined weights; a heatmap alone does not establish causal importance.}\\end{figure}\n"  # Global: selection is fixed independently of routing appearance.
    if (root() / "runs").exists() and any((root() / "runs").glob("*/metrics.jsonl")):
        result_text += "\\begin{figure}[ht]\\centering\\includegraphics[width=\\linewidth]{../reports/figures/learning_curves.pdf}\\caption{All recorded runs with fixed fifty-update bins. Full update records accompany the checkpoints. Supervised MSE is a training objective, not a reinforcement-learning reward.}\\end{figure}\n"  # Local: loss curves cannot substitute for held-out forecast evaluation.
    result_text += "\\begin{figure}[ht]\n\\centering\\includegraphics[width=.8\\linewidth]{../reports/figures/power_sensitivity.pdf}\n\\caption{Prospective power under explicitly assumed independent Gaussian seed effects. This is a sensitivity calculation, not observed prediction accuracy.}\\end{figure}\n"  # Local: label the simulated figure unambiguously.
    result_text += "For a 5\\% improvement, exact-test simulated power is 0.37124 at assumed log-ratio standard deviation 0.05 and 0.11563 at 0.10 (100,000 draws per condition). All sensitivity assumptions and Monte Carlo errors are published.\n"  # Global: fixed prospective results are backed by power_sensitivity.json.
    result_text += "\\clearpage\n" + initialized_results if initialized_results else ""  # Local: separate the added controls from original figures and inference.
    (PROJECT / "paper/results.tex").write_text(result_text, encoding="utf-8")  # Local: update only the machine-generated numerical section.
    abstract = "We compare GPhyT with Block Attention Residuals against GPhyT without Attention Residuals, reusing the released GPhyT backbone and Flash Linear Attention reference operator. Both models train from scratch with six paired seeds, identical shared-parameter initialization and sampled training examples, the same optimizer schedule, effective batch size 16, and 3,175 updates. Standard GPhyT retains its original additive residuals and transformer self-attention. Three-seed Full and uniform-routing variants provide exploratory controls. The selected convection and radiative-flow datasets use official trajectory splits, training-only normalization, and $128\\times64$ resolution. The primary endpoint compares equally weighted dataset log ratios of twelve-step autoregressive NMSE. The total project limit is 24 GPU-hours. "  # Global: make the matched no-Attention-Residuals baseline and tested scope explicit.
    if highlight:
        abstract = highlight["summary"].replace("%", r"\%") + " This descriptive ranking, including post-hoc initialized controls, is not a significance claim against every comparator. " + abstract  # Global: the abstract leads with the requested result and its actual inferential status.
    abstract += f"At this revision, {complete}/18 planned training runs are complete. "  # Local: do not equate completed training with completed test inference.
    if summary["status"] == "complete":
        abstract += f"Relative to GPhyT without Attention Residuals, Block reduces the primary error endpoint by {100 * summary['improvement']:.2f}\\% (95\\% CI [{100 * low:.2f}, {100 * high:.2f}]\\%; exact paired $p={summary['permutation_p']:.5f}$). " + threshold + " Claims are restricted to this model scale, selected data, resolution, and training budget.\n"  # Global: empirical claims identify the baseline, uncertainty, and explicit scope.
    else:
        abstract += "Complete finite endpoints for all six primary pairs are not established, so the primary effect is undefined. Implementation checks and prospective power do not establish an accuracy or stability improvement.\n"  # Global: partial evidence never becomes a null or positive empirical conclusion.
    (PROJECT / "paper/abstract-results.tex").write_text(abstract, encoding="utf-8")  # Local: abstract numbers are regenerated from the same verified analysis as the results.
    compile_paper()  # Global: the exported manuscript corresponds to the just-generated measured results.
    from .web_export import export_site  # Local: static Pages output shares all measured report artifacts.
    export_site()  # Global: public and local viewers cannot silently drift onto separate result calculations.
    LOG.info("Reports generated: completed=%d/18 primary=%s", complete, summary["status"])  # Global: completion and scientific availability remain distinct.
