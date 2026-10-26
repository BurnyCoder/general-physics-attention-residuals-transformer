"""Global: shared descriptive headline from complete saved scores, separate from primary inference.

Local: no new evaluation or significance test; https://neurips.cc/public/guides/PaperChecklist.
"""
import csv  # Local: consume the existing report schema; https://docs.python.org/3/library/csv.html.
import math  # Local: nonfinite outcomes cannot support a favorable ranking.
from .runtime import PROJECT  # Global: every publication reads the same project evidence.


def radiative_highlight():
    """Local: validate all nine full-test means; global: suppress unsupported winner claims."""
    scores = {}  # Local: initialized and trained versions remain distinct comparators.
    for stage, relative in (("trained", "accuracy_summary.csv"), ("initialized", "untrained/accuracy_summary.csv")):
        path = PROJECT / "reports" / relative  # Local: interventions use a different population and are deliberately excluded.
        if not path.exists():
            return None  # Global: pending controls cannot be described as beaten.
        with path.open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["dataset"] == "turbulent_radiative_layer_2D" and row["horizon"] == "12"]  # Local: one fixed dataset and horizon define this descriptive highlight.
        expected = dict(standard=6, block=6, full=3, uniform_block=3)  # Local: original fixed seed matrix, not result-selected counts.
        if stage == "trained":
            expected["persistence"] = 1  # Local: a deterministic comparator has one endpoint, not one random seed.
        if len(rows) != len(expected) or {row["mode"] for row in rows} != set(expected):
            return None  # Global: every planned comparator must be represented exactly once.
        for row in rows:
            mean = float(row["mean"]) if row["mean"] else math.nan  # Local: empty CSV cells stay undefined.
            if int(row["n_finite"]) != expected[row["mode"]] or int(row["n_planned"]) != expected[row["mode"]] or not math.isfinite(mean) or mean < 0:
                return None  # Global: missing, divergent, or invalid endpoints cannot improve a ranking by omission.
            scores[stage, row["mode"]] = mean  # Local: preserve all nine actual means for comparison.
    block, standard, persistence = (scores["trained", mode] for mode in ("block", "standard", "persistence"))  # Local: only final trained Block is the highlighted method.
    if any(block >= value for key, value in scores.items() if key != ("trained", "block")):
        return None  # Global: ties or better competitors invalidate the stated strict win.
    standard_reduction, persistence_reduction = (100 * (1 - block / value) for value in (standard, persistence))  # Local: descriptive reductions of arithmetic means, never the primary paired log estimand.
    title = "GPhyT + Block Attention Residuals beats GPhyT without Attention Residuals and all other tested baselines on radiative flow"  # Global: tested scope accompanies the positive headline.
    summary = f"On this study's radiative-flow full test set, GPhyT + Block Attention Residuals has the lowest mean 12-step NMSE among all nine tested trained, initialized, and persistence models: {block:.5f}. That is {standard_reduction:.2f}% lower than GPhyT without Attention Residuals ({standard:.5f}) and {persistence_reduction:.2f}% lower than persistence ({persistence:.5f})."  # Local: derive every displayed number from measured tables.
    scope = "This is a descriptive mean ranking, emphasized after results, not a win in every seed. Standard/Block use six seeds and Full/Uniform three; initialized controls are post-hoc. The preregistered two-dataset test remains separate and does not establish significance against every comparator."  # Global: changing presentation does not redefine the scientific endpoint.
    return dict(title=title, summary=summary, scope=scope, block_nmse=block, standard_reduction_percent=standard_reduction, persistence_reduction_percent=persistence_reduction, comparators=len(scores))  # Local: paper, cards, and viewers share a single supported result.
