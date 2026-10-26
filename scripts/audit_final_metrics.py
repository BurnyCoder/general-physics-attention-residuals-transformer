"""Global: independently verify the released primary inference and explain a metric edge case.

Local: use saved results and fixed example arrays only; never run or select a model.
The field decomposition is explicitly post hoc and does not replace the frozen endpoint.
Sources: docs/protocol.md and https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html.
"""
from itertools import product  # Local: independently enumerate every one of the 64 paired sign assignments.
import numpy as np  # Local: reconstruct ratios and per-field contributions with explicit FP64 arrays.
from scipy.stats import t  # Local: compute the prespecified Student-t interval with five degrees of freedom.
from gphyt_attnres_study.runtime import PROJECT, read_json, root, start_logging, stamp, write_json  # Global: all observations retain source and timestamp provenance.


def main():
    """Local: assert reported inference and fixed-case decompositions; global: no test-driven tuning follows these observations."""
    start_logging("audit_final_metrics")  # Local: preserve complete audit output.
    datasets = ("rayleigh_benard_uniform", "turbulent_radiative_layer_2D")  # Local: retain the two equally weighted protocol datasets.
    effects = []  # Local: each independent observation is one paired seed.
    for seed in range(6):
        values = [read_json(root() / "evaluation" / f"{mode}_seed{seed}.json")["datasets"] for mode in ("standard", "block")]  # Local: read raw aggregates independently of analysis tables.
        effects.append(np.mean([np.log(values[1][dataset]["12"] / values[0][dataset]["12"]) for dataset in datasets]))  # Global: dataset averaging precedes seed inference.
    effects = np.asarray(effects)  # Local: make six independent log ratios explicit.
    probability = np.mean([abs(np.mean(effects * signs)) >= abs(effects.mean()) - 1e-12 for signs in product((-1, 1), repeat=6)])  # Local: calculate the exact two-sided probability without calling the study analysis function.
    radius = t.ppf(.975, 5) * effects.std(ddof=1) / np.sqrt(6)  # Local: match the preregistered normal-effect confidence interval.
    interval = 1 - np.exp([effects.mean() + radius, effects.mean() - radius])  # Local: reverse endpoints under the decreasing transformation.
    summary = read_json(PROJECT / "reports/primary_analysis.json")  # Local: compare independently reconstructed numbers with the published inference.
    np.testing.assert_allclose(effects, summary["effects"], rtol=1e-13, atol=1e-13)  # Global: no seed or dataset can silently disappear.
    np.testing.assert_allclose(interval, summary["improvement_ci95"], rtol=1e-13, atol=1e-13)  # Local: protect interval order and scale.
    assert probability == summary["permutation_p"]  # Global: exact enumeration must reproduce the reported p-value.
    fields, cases = ["pressure", "density", "buoyancy", "velocity_x", "velocity_y"], []  # Local: use canonical semantic labels.
    for mode in ("standard", "block"):
        for dataset in datasets:
            arrays = np.load(root() / "visuals" / f"{mode}_seed0_{dataset}.npz")  # Local: these examples were fixed before outcomes: seed zero, first test trajectory and window.
            truth, prediction = arrays["truth"][:12].astype(float), arrays["prediction"][:12].astype(float)  # Local: reuse saved first-twelve-step standardized fields without inference.
            mse, energy = ((prediction - truth) ** 2).mean((1, 2)), (truth ** 2).mean((1, 2))  # Local: spatial averages retain separate frames and fields.
            terms = (mse / (energy + 1e-6)).mean(0)  # Global: identical epsilon and averaging order to the frozen metric.
            original = next(row for row in read_json(root() / "evaluation" / f"{mode}_seed0.json")["trajectories"] if row["dataset"] == dataset)  # Local: identify the first trajectory in original evaluation order.
            np.testing.assert_allclose(terms[arrays["active"]].mean(), original["windows"][0]["12"], rtol=1e-12)  # Local: confirm the saved-case decomposition reproduces the reported window exactly.
            cases.append(dict(run=f"{mode}_seed0", dataset=dataset, source_sha256=original["source_sha256"], window="first fixed window", horizon=12, fields=[dict(field=name, mse=float(mse[:, i].mean()), minimum_target_energy=float(energy[:, i].min()), maximum_target_energy=float(energy[:, i].max()), nmse=float(terms[i])) for i, name in enumerate(fields) if arrays["active"][i]]))  # Global: preserve every active field, including unattractive errors.
    controls = []  # Local: exploratory controls use only their three shared seeds, never all six Block runs.
    for mode in ("full", "uniform_block"):
        paired = []  # Local: retain each control-versus-Block paired dataset log ratio.
        for seed in range(3):
            control = read_json(root() / "evaluation" / f"{mode}_seed{seed}.json")["datasets"]  # Local: the control's final checkpoint endpoint.
            block = read_json(root() / "evaluation" / f"block_seed{seed}.json")["datasets"]  # Local: same seed and data schedule for the comparator.
            paired.append(float(np.mean([np.log(control[dataset]["12"] / block[dataset]["12"]) for dataset in datasets])))  # Local: use the primary weighting convention descriptively.
        controls.append(dict(mode=mode, comparator="block", seeds=[0, 1, 2], log_ratios=paired, geometric_error_ratio=float(np.exp(np.mean(paired)))))  # Global: no confirmatory p-value is assigned to three-seed controls.
    write_json(PROJECT / "reports/final_metric_audit.json", dict(created=stamp(), primary_verified=True, effects=effects.tolist(), exact_p=float(probability), improvement_ci95=interval.tolist(), exploratory_controls=controls, diagnostic_status="Post hoc fixed-case field decomposition; not a new primary endpoint or a population-wide explanation", cases=cases))  # Local: expose numerical evidence and its inferential limits.
    print("Independent six-pair inference matches; all four fixed-case decompositions reproduce their saved window errors.")  # Global: report the checked scope precisely.


if __name__ == "__main__":
    main()  # Local: explicit CPU-only audit command, independent of GPU accounting.
