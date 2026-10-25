"""Global: prespecified seed-level inference and prospective power sensitivity.

Local: no frame-level p-values, outcome-dependent seed additions, or post-hoc power.
Sources: https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.html
and https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html.
"""
import itertools  # Local: enumerate every paired sign assignment exactly.
import math  # Local: translate between log-error effects and relative improvement.
import numpy as np  # Local: transparent numerical simulation and aggregation.
import pandas as pd  # Local: publish machine-readable tables from raw measurements.
from scipy.stats import t  # Local: Student-t interval for the paired seed-level mean.
from statsmodels.stats.power import TTestPower  # Local: reuse a validated noncentral-t power implementation.
from .runtime import LOG, PROJECT, config, read_json, root, stamp, write_json  # Global: results are timestamped and reproducible.


def sign_matrix(n):
    """Local: all 2**n sign assignments; global: no Monte Carlo approximation in the primary test."""
    return np.asarray(list(itertools.product([-1., 1.], repeat=n)))  # Local: paired exchangeability swaps treatment and baseline within a seed.


def paired_summary(effects):
    """Local: inference over independent paired seeds; global: uncertainty does not treat trajectories as replicates."""
    values = np.asarray(effects, dtype=float)  # Local: each input is the equally weighted two-dataset log ratio for one seed.
    if values.size != 6 or not np.isfinite(values).all():
        return dict(status="undefined", n_pairs=int(values.size), reason="Primary inference requires all six finite preregistered pairs")  # Global: incomplete or divergent pairs cannot be silently excluded.
    mean, sd = float(values.mean()), float(values.std(ddof=1))  # Local: estimate paired variation across initialization/sampling seeds.
    margin = float(t.ppf(.975, 5) * sd / math.sqrt(6))  # Local: t interval assumes approximately normal independent seed effects.
    null = np.abs(sign_matrix(6) @ values / 6)  # Local: exact randomization distribution for the absolute mean log ratio.
    probability = float(np.mean(null >= abs(mean) - 1e-12))  # Local: include the observed assignment and ties.
    return dict(status="complete", n_pairs=6, effects=values.tolist(), mean_log_ratio=mean, sd_log_ratio=sd, improvement=1 - math.exp(mean), improvement_ci95=[1 - math.exp(mean + margin), 1 - math.exp(mean - margin)], permutation_p=probability, interval_assumption="Independent approximately normal paired seed log ratios; fixed datasets and windows")  # Global: monotone transformation reverses confidence-limit order.


def power():
    """Local: prospective assumption grid; global: quantify uncertainty before training outcomes exist."""
    path = PROJECT / "reports/power_sensitivity.json"  # Local: the preregistration includes reproducible numerical sensitivity.
    if path.exists():
        return read_json(path)  # Global: do not silently replace the prospective simulation after outcomes.
    rng, rows = np.random.default_rng(20261025), []  # Local: Monte Carlo seed is independent of training seeds.
    signs, simulations = sign_matrix(6), 100000  # Local: exact per-draw permutation tests over a fixed simulation budget.
    for reduction in (.05, .10, .20):
        for sd in (.02, .05, .10, .20):
            mean = math.log1p(-reduction)  # Local: improvements map to negative log-error ratios.
            draws = rng.normal(mean, sd, (simulations, 6))  # Local: explicitly assume independent normal seed-pair effects.
            probabilities = (np.abs(draws @ signs.T / 6) >= np.abs(draws.mean(1))[:, None] - 1e-12).mean(1)  # Local: every draw uses the same exact two-sided test as the experiment.
            exact_power = float(np.mean(probabilities <= .05))  # Local: quantify rejection frequency under the stated alternative.
            rows.append(dict(reduction=reduction, seed_log_sd=sd, standardized_effect=abs(mean) / sd, t_power=float(TTestPower().power(abs(mean) / sd, nobs=6, alpha=.05)), sign_flip_power=exact_power, simulation_se=math.sqrt(exact_power * (1 - exact_power) / simulations)))  # Global: report simulation uncertainty instead of false precision.
    result = dict(created=stamp(), seeds=6, simulations=simulations, alpha=.05, minimum_two_sided_p=2 / 64, assumptions="Independent Gaussian paired log-error ratios; variance assumptions are hypothetical, not fitted to test outcomes", rows=rows)  # Global: six pairs impose a coarse exact-test resolution.
    write_json(path, result)  # Local: publish before confirmatory training.
    pd.DataFrame(rows).to_csv(PROJECT / "reports/power_sensitivity.csv", index=False)  # Local: tables are reusable without Python-specific formats.
    LOG.info("Prospective power grid written: %s", path)  # Local: timestamp and provenance appear in the process log.
    return result  # Local: reporting and dashboard consume the same numbers.


def analyze():
    """Local: aggregate existing results only; global: missing experiments remain explicitly missing."""
    cfg, records, effects = config(), [], []  # Local: the planned seeds are fixed, not inferred from successful result files.
    for path in sorted((root() / "evaluation").glob("*.json")):
        result = read_json(path)  # Local: each result retains raw trajectory evidence.
        for dataset, metrics in result["datasets"].items():
            for horizon, value in metrics.items():
                records.append(dict(run=result.get("run", "persistence"), intervention=result.get("intervention"), dataset=dataset, horizon=int(horizon), nmse=value))  # Local: one row per aggregate makes filtering transparent.
    for seed in cfg["seeds"]:
        ratios = []  # Local: each seed requires both datasets and both primary variants.
        for dataset in cfg["data"]:
            baseline = read_json(root() / "evaluation" / f"standard_seed{seed}.json", {}).get("datasets", {}).get(dataset, {}).get("12")  # Local: missing baseline is not imputed.
            treatment = read_json(root() / "evaluation" / f"block_seed{seed}.json", {}).get("datasets", {}).get(dataset, {}).get("12")  # Local: missing or divergent treatment is not dropped.
            ratios.append(math.log(treatment / baseline) if baseline and treatment and baseline > 0 and treatment > 0 else float("nan"))  # Local: undefined/zero ratios remain explicitly undefined.
        effects.append(float(np.mean(ratios)))  # Global: equal dataset weights precede seed-level inference.
    summary = paired_summary(effects)  # Local: six-pair test or a clearly undefined result.
    summary.update(created=stamp(), minimum_meaningful_improvement=.05, scope="Two selected datasets at 128x64, fixed training budget and GPhyT-S scale")  # Global: prohibit extrapolation beyond this experiment.
    write_json(PROJECT / "reports/primary_analysis.json", summary)  # Local: strict JSON prevents unnoticed NaN serialization.
    pd.DataFrame(records, columns=["run", "intervention", "dataset", "horizon", "nmse"]).to_csv(PROJECT / "reports/metrics.csv", index=False)  # Global: missing values stay blank, not zero.
    LOG.info("Primary analysis status=%s", summary["status"])  # Local: reporting cannot confuse an incomplete study with evidence of no effect.
    return summary  # Local: the paper renderer uses measured analysis output.
