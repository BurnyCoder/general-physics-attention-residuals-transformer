"""Global: report measured initialization-only controls separately from confirmatory inference.

Local: reuse complete-seed statistics and matched-field plotting for all four architectures.
Sources: docs/untrained-baselines.md and https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html.
"""
import shutil  # Local: publish only explicit measured JSON and initialized checkpoint metadata.
import pandas as pd  # Local: one reduction feeds machine-readable tables, README, and paper.
from .comparisons import MODES, comparison_figure  # Global: reuse the identical-case guard and shared unclipped color scales.
from .results import aggregate_estimates, DATASETS  # Global: missing seeds and divergence follow the existing reporting contract.
from .runtime import PROJECT, config, read_json, root, stamp  # Local: separate post-hoc paths preserve the original study records.

NAMES = {"standard": "GPhyT without AttnRes", "block": "Block AttnRes", "full": "Full AttnRes", "uniform_block": "Uniform Block"}  # Local: short labels retain each baseline/control identity.


def markdown_table(frame):
    """Local: format small tables with standard strings; global: preserve the frozen environment and undefined outcomes."""
    rows = [frame.columns.tolist(), ["---"] * len(frame.columns), *frame.values.tolist()]  # Local: CommonMark header and separator precede measured rows.
    lines = []  # Local: one shared renderer serves compact and per-run reports without optional dependencies.
    for row in rows:
        cells = ["undefined" if pd.isna(value) else f"{value:.8g}" if isinstance(value, float) else str(value) for value in row]  # Global: absent measurements cannot become finite zeros.
        lines.append("| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in cells) + " |")  # Local: literal cell separators cannot corrupt the table's columns.
    return "\n".join(lines)  # Local: the caller controls document-level spacing.


def untrained_tables():
    """Global: retain every planned zero-update control; local: reduce only saved measurements."""
    cfg, metrics, runs = config(), [], []  # Local: planned denominators come from the unchanged configuration.
    destination = PROJECT / "reports/untrained"  # Global: new endpoints cannot overwrite the original primary CSV files.
    destination.mkdir(parents=True, exist_ok=True)  # Local: public reports remain in the repository.
    for mode in MODES:
        seeds = cfg["seeds"] if mode in ("standard", "block") else cfg["control_seeds"]  # Global: match each architecture's original trained seed set.
        for seed in seeds:
            name = f"{mode}_seed{seed}"  # Local: common identity aligns initialized and trained model metadata.
            result = read_json(root() / "untrained/evaluation" / f"{name}.json", {})  # Local: never substitute the trained result when an initialization is unavailable.
            if result and (result.get("training_updates") != 0 or result.get("post_hoc") is not True):
                raise ValueError(f"Expected a post-hoc zero-update evaluation: {name}")  # Global: prevent accidental relabeling of trained checkpoint errors.
            status = read_json(root() / "untrained/runs" / name / "status.json", {})  # Local: retain failures and incomplete attempts alongside metrics.
            runs.append(dict(run=name, mode=mode, seed=seed, training_updates=0, status=status.get("status", result.get("status", "not_started")), seconds=status.get("seconds"), initial_state_sha256=status.get("initial_state_sha256"), final_state_sha256=status.get("final_state_sha256"), shared_backbone_sha256=status.get("shared_backbone_sha256"), checkpoint_sha256=status.get("checkpoint_sha256"), error=status.get("error")))  # Global: no absence is silently interpreted as success.
            for dataset in cfg["data"]:
                for horizon in (1, 12, 24):
                    metrics.append(dict(run=name, mode=mode, seed=seed, dataset=dataset, horizon=horizon, nmse=result.get("datasets", {}).get(dataset, {}).get(str(horizon))))  # Local: missing/divergent endpoints remain missing.
    metrics = pd.DataFrame(metrics)  # Local: one row per planned seed/dataset/horizon preserves the experimental unit.
    accuracy = [dict(mode=mode, dataset=dataset, horizon=horizon, **aggregate_estimates(group.nmse.tolist(), len(cfg["seeds"] if mode in ("standard", "block") else cfg["control_seeds"]))) for (mode, dataset, horizon), group in metrics.groupby(["mode", "dataset", "horizon"])]  # Global: descriptive intervals use independent seeds, never frames or pixels.
    frames = dict(accuracy_summary=pd.DataFrame(accuracy), experiments=pd.DataFrame(runs), metrics=metrics)  # Local: status, raw seed metrics, and reductions remain separate tables.
    for name, frame in frames.items():
        frame.to_csv(destination / f"{name}.csv", index=False)  # Global: every displayed score can be reproduced from public numerical records.
    return frames  # Local: the same frames feed subsequent comparisons and publication.


def initialization_comparison(initial, trained):
    """Global: compare identical planned seed populations; local: merge one-to-one without dropping unavailable baselines."""
    joined = initial.merge(trained, on=["mode", "dataset", "horizon"], how="left", suffixes=("_initial", "_trained"), validate="one_to_one")  # Local: retain initial rows even if a trained summary is malformed or missing.
    if not joined.n_planned_initial.eq(joined.n_planned_trained).all():
        raise ValueError("Initialized and trained comparisons require matching planned seed sets")  # Global: six-seed and three-seed means cannot masquerade as matched learning effects.
    return joined  # Local: keep both absolute scores and their descriptive uncertainty separately labeled.


def display_table(comparison):
    """Local: format the primary-horizon absolute scores; global: no added inferential test or changed estimand."""
    rows = []  # Local: use the same display ordering in Markdown and LaTeX.
    for dataset, title in DATASETS.items():
        for mode in MODES:
            value = comparison[(comparison.dataset == dataset) & (comparison["mode"] == mode) & (comparison.horizon == 12)].iloc[0]  # Local: twelve-step scores use all their forecast prefixes, not just the twelfth frame.
            rows.append({"Dataset": title, "Model": NAMES[mode], "Finite/planned": f"{int(value.n_finite_initial)}/{int(value.n_planned_initial)}", "Initialized (0 updates)": f"{value.mean_initial:.6g}" if pd.notna(value.mean_initial) else "undefined", "Trained (3,175 updates)": f"{value.mean_trained:.6g}" if pd.notna(value.mean_trained) else "undefined"})  # Global: incomplete initialization means remain undefined beside the unchanged trained counterpart.
    return pd.DataFrame(rows)  # Local: table serialization cannot diverge across paper and README.


def untrained_report(trained, save_figure):
    """Global: publish actual initialized baselines and fixed examples; local: return an optional paper section."""
    if not (root() / "untrained").exists():
        return ""  # Local: an ordinary pre-extension reproduction cannot fabricate extra scientific results.
    frames = untrained_tables()  # Local: generated tables retain missing and failed attempts.
    compared = initialization_comparison(frames["accuracy_summary"], trained)  # Global: initialized/trained comparisons use the same architecture and planned seeds.
    compared.to_csv(PROJECT / "reports/untrained/comparison.csv", index=False)  # Local: all horizons and descriptive uncertainties accompany the compact table.
    table, figures = display_table(compared), []  # Local: no result-dependent example selection.
    for mode in MODES:
        name = f"initialization_{mode}"  # Local: stable figure names preserve reproducible references.
        label = "Without AttnRes" if mode == "standard" else NAMES[mode]  # Local: short three-line labels fit each panel; captions retain the complete baseline identity.
        if comparison_figure(save_figure, ("persistence", f"untrained_{mode}_seed0", f"{mode}_seed0"), ("Ground truth", "Persistence", label + "\ninitialized\n0 updates", label + "\ntrained\n3,175 updates"), name):
            figures.append(name)  # Global: include only figures backed by complete matching saved targets.
    destination = PROJECT / "reports/untrained"  # Local: raw additional measurements are explicitly separated from original evaluation files.
    for path in (root() / "untrained/evaluation").glob("*.json"):
        (destination / "evaluation").mkdir(exist_ok=True)  # Local: preserve actual trajectory-level measurements for external checking.
        shutil.copy2(path, destination / "evaluation" / path.name)  # Global: copy measured records verbatim, including undefined values and provenance.
    individual = destination / "runs"  # Local: one small report per planned initialized experiment satisfies the existing audit convention.
    individual.mkdir(exist_ok=True)  # Local: keep these reports distinct from the trained-run reports.
    for row in frames["experiments"].itertuples(index=False):
        scores = frames["metrics"][frames["metrics"].run == row.run][["dataset", "horizon", "nmse"]]  # Local: retain every planned horizon, including missing values.
        detail = f"# Initialized {row.run}\n\nStatus: **{row.status}**. Training updates: **0**. This reconstructed initialization is a post-hoc baseline, specified before its execution but after the trained results were known.\n\n"  # Global: distinguish the added descriptive experiment from preregistration.
        detail += markdown_table(scores) + f"\n\nThe [fixed additional method](../../../docs/untrained-baselines.md) describes the constructor, seeds, full test population, and budget. The [paired trained experiment](../../experiments/{row.run}.md) uses the same architecture and seed. [Raw initialized evaluation](../evaluation/{row.run}.json).\n\nInitial state hash: `{row.initial_state_sha256}`. Final state hash: `{row.final_state_sha256}`. Checkpoint SHA-256: `{row.checkpoint_sha256}`. Recorded error: `{row.error}`.\n"  # Local: measured hashes establish zero parameter/buffer updates and identify exact weight bytes.
        (individual / f"{row.run}.md").write_text(detail, encoding="utf-8")  # Global: failed or absent controls retain reports rather than disappearing.
    complete = int(frames["experiments"].status.eq("complete").sum())  # Local: evaluation completion is distinct from finite scientific endpoints.
    introduction = f"**Post-hoc initialization-only controls: {complete}/{len(frames['experiments'])} evaluations complete.** These models have zero training updates and were reconstructed with the original constructor, frozen configuration, and seeds. They were requested after the trained results were known. Both columns use the same full test population and the same seeds for each architecture. Lower 12-step NMSE is better; finite/planned counts describe initialized seed endpoints.\n\n"  # Global: never present the added control as preregistered or as a replacement primary endpoint.
    markdown = introduction + markdown_table(table) + "\n\nAll 1/12/24-step scores, seed SDs, and descriptive intervals are in [the comparison CSV](reports/untrained/comparison.csv). Missing or divergent seeds remain undefined. The original six-pair primary comparison is unchanged. Block and Uniform have identical initial routing on their shared seeds 0–2; their overall means differ because Block also includes seeds 3–5. [Additional protocol](docs/untrained-baselines.md).\n\n"  # Local: each displayed number links to its machine-readable source and fixed post-hoc protocol.
    for name in figures:
        markdown += f"![Ground truth, persistence, initialized and trained {NAMES[name.removeprefix('initialization_')]} on matching convection and radiative cases](reports/figures/{name}.png)\n\n"  # Global: display both datasets for every initialized architecture, without selecting attractive cases.
    markdown += "Examples use seed 0, first test trajectory/window, forecast step 12: convection buoyancy above, radiative density below. Each row has one common unclipped color range; magenta marks nonfinite predictions.\n"  # Local: state both the matching rule and color convention.
    (destination / "summary.md").write_text("# Initialization-only baseline results\n\nGenerated " + stamp() + ".\n\n" + markdown.replace("](reports/", "](../").replace("](docs/", "](../../docs/"), encoding="utf-8")  # Local: one generated summary reuses the same numerical table with repository-relative links.
    readme = PROJECT / "README.md"  # Local: replace only the explicitly managed new-results block.
    start, end = "<!-- initialization-results:start -->", "<!-- initialization-results:end -->"  # Global: preserve all human prose outside this bounded generated block.
    if readme.exists() and start in readme.read_text(encoding="utf-8"):
        content = readme.read_text(encoding="utf-8")  # Local: inspect the current document, preserving unrelated edits.
        before, remainder = content.split(start, 1)  # Local: retain the user's introduction and primary results verbatim.
        _, after = remainder.split(end, 1)  # Local: require the matching closing marker rather than truncating an incomplete document.
        readme.write_text(before + start + "\n\n" + markdown + "\n" + end + after, encoding="utf-8")  # Global: regenerate measured values without duplicating hand-maintained tables.
    latex_table = table.rename(columns={"Initialized (0 updates)": "Initialized", "Trained (3,175 updates)": "Trained"})  # Local: compact paper headers retain full update counts in the caption.
    latex = "\\subsection{Post-hoc initialization-only baselines}\n" + f"The additional user-requested zero-update evaluation completed {complete}/{len(frames['experiments'])} planned controls. These controls were specified after observing the trained-model results and do not alter the original confirmatory comparison. The models were reconstructed with the original constructor, configuration, and seeds, without loading trained weights or applying optimizer updates. Parameter and buffer hashes were checked before and after inference.\n"  # Global: explicitly disclose timing and reconstruction limits in the paper.
    latex += "\\begin{table}[ht]\\centering\\scriptsize\n" + latex_table.to_latex(index=False, escape=True) + "\\caption{Twelve-step NMSE for initialized models (zero updates) and their trained counterparts (3,175 updates), on the same complete test population. Standard/Block use seeds 0--5; Full/Uniform use seeds 0--2. Finite/planned counts refer to initialized endpoints; incomplete or divergent seed sets remain undefined. These are descriptive post-hoc controls, not additional confirmatory tests. All horizons and seed uncertainty are in reports/untrained/comparison.csv.}\\end{table}\n"  # Global: training effects cannot borrow significance from the original architecture comparison.
    for name in figures:
        latex += "\\begin{figure}[ht]\\centering\\includegraphics[width=\\linewidth]{../reports/figures/" + name + ".pdf}\\caption{Initialized versus trained " + NAMES[name.removeprefix("initialization_")] + ": ground truth, persistence, zero-update model, and final trained checkpoint. Both rows use the same seed-zero first test trajectory/window at forecast step twelve, with one common unclipped color range per row. Top: convection buoyancy; bottom: radiative density. Magenta denotes nonfinite values. These fixed illustrative cases do not replace evaluation over all trajectories and seeds.}\\end{figure}\n"  # Local: retain explicit stage labels and matching rules beside every figure.
    return latex  # Local: the ordinary report phase inserts this separate post-hoc section after its original results.
