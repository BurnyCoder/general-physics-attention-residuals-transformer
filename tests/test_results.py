"""Global: paper tables must retain planned sample sizes and undefined outcomes.

Local: fixed synthetic values test reporting arithmetic without generating study evidence.
Source: docs/protocol.md and https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html.
"""
from gphyt_attnres_study.results import aggregate_estimates  # Global: figures and manuscript tables share one reporting reduction.


def test_running_run_memory_is_missing_until_measured(tmp_path, monkeypatch):
    """Global: an unfinished run must not acquire a fabricated zero-memory measurement."""
    import pandas as pd  # Local: use the same missing-value semantics as CSV and LaTeX export.
    from gphyt_attnres_study import results  # Local: exercise the complete production cost-table reduction.
    from gphyt_attnres_study.runtime import write_json  # Local: fixture status files use the real artifact schema.
    monkeypatch.setattr(results, "PROJECT", tmp_path)  # Local: generated fixture tables cannot enter public reports.
    monkeypatch.setattr(results, "root", lambda: tmp_path)  # Local: no real training artifacts are modified.
    (tmp_path / "reports").mkdir()  # Local: provide only the report destination required by the reducer.
    write_json(tmp_path / "runs/running/status.json", dict(status="running", step=250, seconds=10.))  # Local: active status omits peak memory until the training attempt ends.
    write_json(tmp_path / "runs/complete/status.json", dict(status="complete", step=500, seconds=20., peak_memory_bytes=2 ** 20))  # Local: a known completed measurement must remain numeric.
    costs = results.measured_tables()["costs"].set_index("run")  # Global: test the actual table that feeds the paper and viewer.
    assert pd.isna(costs.loc["running", "peak_memory_mib"])  # Global: absence is not evidence of zero allocation.
    assert costs.loc["complete", "peak_memory_mib"] == 1.  # Local: retain the established byte-to-MiB conversion.
    exported = pd.read_csv(tmp_path / "reports/costs.csv").set_index("run")  # Local: missingness must survive the public interchange format.
    assert pd.isna(exported.loc["running", "peak_memory_mib"])  # Global: downstream LaTeX and JSON can display undefined instead of zero.


def test_missing_seed_is_not_silently_dropped_from_variant_mean():
    """Global: a missing or divergent seed prevents a complete-matrix architecture mean."""
    complete = aggregate_estimates([1., 2., 3.], 3)  # Local: known sample mean and unbiased sample standard deviation.
    assert complete["mean"] == 2. and complete["sd"] == 1.  # Local: sample statistics use seeds, not correlated frames.
    assert complete["ci_low"] < 2. < complete["ci_high"]  # Local: a nonzero variance has a nondegenerate interval.
    missing = aggregate_estimates([1., 2.], 3)  # Local: one planned run has no result.
    divergent = aggregate_estimates([1., 2., None], 3)  # Local: one evaluated run has an undefined error.
    assert missing["mean"] is None and divergent["mean"] is None  # Global: neither omission can improve the architecture's reported average.
    assert missing["n_finite"] == divergent["n_finite"] == 2  # Local: readers can see the exact available sample count.


def test_measured_analysis_to_paper_artifacts_with_isolated_fixtures(tmp_path, monkeypatch):
    """Global: exercise populated analysis, physical tables, figures, and LaTeX without fabricating public study results."""
    import numpy as np  # Local: independent expected values check generated aggregates.
    import matplotlib  # Local: fixture figures never open native application windows.
    matplotlib.use("Agg")  # Local: use the same noninteractive rendering policy as production reporting.
    import matplotlib.pyplot as plt  # Local: inspect actual figure generation, not a mocked plotting API.
    from gphyt_attnres_study import analysis, results  # Local: compose the user's actual analysis and report phases.
    from gphyt_attnres_study.runtime import config, write_json  # Local: fixture files use real schemas and the frozen seed matrix.
    monkeypatch.setenv("STUDY_ROOT", str(tmp_path / "artifacts"))  # Global: synthetic fixtures cannot enter real experimental outputs.
    monkeypatch.setattr(analysis, "PROJECT", tmp_path)  # Local: analysis outputs stay in the test-owned tree.
    monkeypatch.setattr(results, "PROJECT", tmp_path)  # Local: paper tables read only those test outputs.
    (tmp_path / "reports").mkdir()  # Local: create the normal public-artifact layout in isolation.
    cfg = config()  # Local: test the complete prescribed seed and dataset matrix.
    for mode in results.LABELS:
        seeds = [None] if mode == "persistence" else cfg["seeds"] if mode in {"standard", "block"} else cfg["control_seeds"]  # Local: exact planned multiplicities.
        for seed in seeds:
            name = "persistence" if seed is None else f"{mode}_seed{seed}"  # Local: production discovery uses these stable identifiers.
            value = (1 + .01 * (seed or 0)) * (.9 if mode == "block" else 1.)  # Local: deliberately known ten-percent paired improvement, test fixture only.
            endpoints = {str(h): value for h in (1, 12, 24)}  # Local: isolate reporting from horizon-dependent model behavior.
            diagnostic = dict(divergence_rms=.2, target_divergence_rms=.1, divergence_error_rms=.1, negative_density_fraction=0., negative_pressure_fraction=0.)  # Local: analytically specified secondary fixture values.
            trajectories = [dict(dataset=dataset, parameter="fixed", source_sha256=dataset, nmse=endpoints, windows=[dict(endpoints, finite_forecast_steps=24)] * 3, diagnostics=[diagnostic] * 3) for dataset in cfg["data"]]  # Local: explicit windows are correlated within the same synthetic trajectory.
            write_json(tmp_path / "artifacts/evaluation" / f"{name}.json", dict(run=name, intervention=None, trajectories=trajectories, datasets={dataset: endpoints for dataset in cfg["data"]}, routing=[]))  # Local: no fixture leaves the temporary test directory.
    primary = analysis.analyze()  # Global: exercise the same exact paired inference used by the public paper.
    frames = results.measured_tables()  # Local: aggregate all planned seeds and physical diagnostics.
    selected = frames["accuracy_summary"].query("mode == 'block' and horizon == 12")  # Local: one complete architecture mean per dataset.
    np.testing.assert_allclose(selected["mean"], .9225)  # Global: independent arithmetic catches seed/group aggregation errors.
    np.testing.assert_allclose(primary["improvement"], .1)  # Local: paired ratio inference matches the constructed ten-percent effect.
    def save_fixture(figure, name):
        """Local: render actual graphics under isolated fixture names; global: no fabricated plot becomes a study deliverable."""
        figure.savefig(tmp_path / f"fixture_{name}.png")  # Local: plotting errors surface during the end-to-end test.
        plt.close(figure)  # Local: release fixture-only graphics memory.
    results.empirical_figures(frames, primary, save_fixture)  # Local: execute populated error and confidence-interval plotting branches.
    latex = results.measured_latex(frames)  # Local: physical and accuracy tables must render with real measured-table schemas.
    assert "0.9225" in latex and "Negative density fraction" in latex  # Global: numerical and physical summaries both reach the paper output.
    assert (tmp_path / "fixture_paired_effects.png").is_file() and (tmp_path / "fixture_forecast_errors.png").is_file()  # Local: actual rendered evidence checks the plotting workflow.
