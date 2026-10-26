"""Global: local interactive inspection of measured outcomes, failures, and costs.

Local: read durable artifacts without changing training or evaluation state.
Sources: https://docs.streamlit.io/develop/api-reference and https://plotly.com/python/.
"""
import json  # Local: consume append-only training records and public result files.
import numpy as np  # Local: fixed-array field visualization and heatmap padding.
import pandas as pd  # Local: tabular filtering keeps raw outcomes inspectable.
import plotly.express as px  # Local: interactive axes, tooltips, and color scales.
import streamlit as st  # Local: the dashboard is a local Python application.
from gphyt_attnres_study.runtime import PROJECT, read_json, root  # Global: use the same .env paths as the CLI.

st.set_page_config(page_title="GPhyT Attention Residuals", layout="wide")  # Local: make room for paired charts and field comparisons.
st.title("Attention Residuals for GPhyT")  # Global: identify the bounded architecture study.
st.caption("Six paired seeds · two physics datasets · fixed checkpoints · 24 GPU-hour cap")  # Local: keep the study scope visible.
st.sidebar.button("Refresh measurements")  # Local: Streamlit reruns on interaction and reads current artifacts.
ledger = read_json(root() / "gpu_budget.json", {"sessions": []})  # Local: empty ledger is an unstarted study, not zero-cost training.
hours = sum(row["seconds"] for row in ledger["sessions"]) / 3600  # Local: all recorded GPU phases contribute to the same total.
st.sidebar.metric("Recorded GPU hours", f"{hours:.3f} / 24")  # Global: display the hard compute budget alongside results.
st.sidebar.caption("Active sessions update at bounded work-unit checkpoints.")  # Local: explain small live-reporting delays.
statuses = [dict(read_json(path), run=path.parent.name) for path in sorted((root() / "runs").glob("*/status.json"))]  # Local: failed experiments remain in the same table as successes.
if statuses:
    st.subheader("Experiment status")  # Local: progress and failures precede numerical claims.
    st.dataframe(pd.DataFrame(statuses)[[column for column in ("run", "status", "step", "updates", "seconds", "peak_memory_bytes", "error") if any(column in row for row in statuses)]], hide_index=True, width="stretch")  # Local: only available fields are displayed; missing values remain missing.
else:
    st.info("No training runs have started. Preparation, tests, and prospective power results are shown below.")  # Global: never substitute synthetic scores for absent experiments.
tabs = st.tabs(["Errors and effects", "Learning and cost", "Fields and routing", "Protocol and power"])  # Local: separate independent inspection tasks without duplicating evidence.

with tabs[0]:
    summary = read_json(PROJECT / "reports/primary_analysis.json", {})  # Local: analysis owns the inferential calculation.
    if summary.get("status") == "complete":
        st.metric("Block relative improvement", f"{100 * summary['improvement']:.2f}%")  # Local: sign and scale match the prespecified log-ratio transformation.
        st.write({"95% CI (%)": [100 * value for value in summary["improvement_ci95"]], "Exact paired p": summary["permutation_p"], "Independent pairs": summary["n_pairs"]})  # Global: uncertainty accompanies the effect.
        st.caption("Completed-study interpretation: all trained variants lose to persistence on convection under the fixed metric, and radiative forecasts frequently have negative density. The relative gain does not establish a reliable physics surrogate; see reports/interpretation.md.")  # Local: show the observed absolute failures alongside the relative architecture effect.
    else:
        st.warning("Primary inference is unavailable until all six paired endpoints are finite and evaluated.")  # Global: undefined estimates are explicit.
    path = PROJECT / "reports/metrics.csv"  # Local: one normalized table is shared with the paper.
    if path.exists():
        metrics = pd.read_csv(path)  # Local: blank values remain NaN, not successful zero-error results.
        if not metrics.empty:
            datasets = st.multiselect("Datasets", sorted(metrics.dataset.unique()), default=sorted(metrics.dataset.unique()))  # Local: interactive dataset comparisons.
            horizon = st.selectbox("Forecast horizon", [1, 12, 24], index=1)  # Local: default to the primary twelve-step endpoint.
            filtered = metrics[metrics.dataset.isin(datasets) & (metrics.horizon == horizon) & metrics.intervention.isna()]  # Local: main errors always use all test trajectories, excluding differently scoped diagnostic subsets.
            st.plotly_chart(px.scatter(filtered, x="run", y="nmse", color="dataset", title="Full-test trajectory-averaged NMSE; each dot is a run/dataset"), width="stretch")  # Global: do not plot frame-level pseudo-replicate error bars.
            st.dataframe(filtered, hide_index=True, width="stretch")  # Local: expose exact values and missing estimates.
    initial_path = PROJECT / "reports/untrained/comparison.csv"  # Global: additional zero-update controls remain separate from the primary analysis.
    if initial_path.exists():
        st.subheader("Initialized versus trained models (post-hoc)")  # Local: make the added control's timing and stage explicit.
        st.caption("Zero versus 3,175 optimizer updates; matching architecture, original seeds, and complete test population. The Fields tab includes the saved untrained forecasts. Missing estimates remain undefined.")  # Global: avoid confusing reconstruction with historical step-zero archives or new confirmatory evidence.
        st.dataframe(pd.read_csv(initial_path), hide_index=True, width="stretch")  # Local: the README, paper, Pages, and dashboard share one measured comparison table.
    intervention_path = PROJECT / "reports/interventions.csv"  # Local: analysis constructs correctly matched within-model subsets.
    if intervention_path.exists():
        interventions = pd.read_csv(intervention_path)  # Local: the baseline column uses the same four trajectories as the altered model.
        if not interventions.empty:
            st.subheader("Within-model interventions on matched diagnostic trajectories")  # Global: do not imply these are separately trained comparisons.
            st.dataframe(interventions, hide_index=True, width="stretch")  # Local: exact original/altered errors and ratios remain inspectable.

with tabs[1]:
    available = sorted(path.parent.name for path in (root() / "runs").glob("*/metrics.jsonl"))  # Local: only actual logs become selectable.
    selected = st.multiselect("Runs", available, default=available[:2])  # Local: compare paired trajectories without hiding other runs.
    rows = []  # Local: combine full logs; line-level parse failures remain visible.
    for name in selected:
        for line in (root() / "runs" / name / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                rows.append(dict(json.loads(line), run=name))  # Local: preserve original update indices.
            except json.JSONDecodeError:
                st.warning(f"Incomplete log line in {name}; refresh after the active write.")  # Local: do not crash on a concurrent append.
    if rows:
        frame = pd.DataFrame(rows).drop_duplicates(["run", "step"], keep="last")  # Local: resumed replay of unsaved updates is shown once, while raw logs remain intact.
        st.plotly_chart(px.line(frame, x="step", y="loss", color="run", log_y=True, title="Training MSE (log scale; every recorded update)"), width="stretch")  # Local: logarithmic scale is labeled explicitly.
        st.plotly_chart(px.line(frame, x="step", y="gradient_norm", color="run", log_y=True, title="Gradient norm before clipping"), width="stretch")  # Global: stability evidence accompanies fit.
        st.plotly_chart(px.box(frame, x="run", y="seconds", title="End-to-end seconds per optimizer update"), width="stretch")  # Local: runtime variation is visible instead of a single favorable sample.
    st.caption("No reward curve: this study uses supervised next-frame regression, without reinforcement learning.")  # Global: do not invent an inapplicable metric.

with tabs[2]:
    files = sorted((root() / "visuals").glob("*.npz"))  # Local: visualization data originate from predetermined evaluation cases.
    if files:
        chosen = st.selectbox("Saved rollout", files, format_func=lambda path: path.stem)  # Local: compare available measured predictions.
        arrays = np.load(chosen)  # Local: compressed numerical arrays contain no executable pickle objects.
        channel = st.selectbox("Field", np.flatnonzero(arrays["active"]).tolist(), format_func=lambda value: ["Pressure", "Density", "Buoyancy", "Velocity x", "Velocity y"][value])  # Global: missing fields are excluded from the selector.
        step = st.slider("Predicted frame", 1, len(arrays["truth"]), 12) - 1  # Local: display one-based horizon labels.
        truth, prediction = arrays["truth"][step, ..., channel], arrays["prediction"][step, ..., channel]  # Local: same field and time for both views.
        minimum, maximum = float(np.nanmin(truth)), float(np.nanmax(truth))  # Local: common target-derived color limits enable honest visual comparison.
        left, right = st.columns(2)  # Local: aligned truth/prediction panels.
        left.plotly_chart(px.imshow(truth, zmin=minimum, zmax=maximum, title="Ground truth (training-standardized units)"), width="stretch")  # Local: explicitly label the units.
        right.plotly_chart(px.imshow(prediction, zmin=minimum, zmax=maximum, title="Autoregressive prediction"), width="stretch")  # Global: do not autoscale each prediction to conceal amplitude errors.
        st.plotly_chart(px.imshow(prediction - truth, color_continuous_scale="RdBu", color_continuous_midpoint=0, title="Signed prediction error"), width="stretch")  # Local: error remains separately inspectable.
    routing_files = [path for path in (root() / "evaluation").glob("*.json") if read_json(path).get("routing")]  # Local: learned and intervention routing retain distinct files.
    if routing_files:
        chosen = st.selectbox("Routing experiment", sorted(routing_files), format_func=lambda path: path.stem)  # Local: select a real measured routing trace.
        route = read_json(chosen)["routing"][0]  # Local: predetermined first diagnostic window, not strongest-looking routing.
        trace = route["trace"]  # Local: each sublayer read reports a different causal source set.
        heatmap = np.full((len(trace), max(row["sources"] for row in trace)), np.nan)  # Local: unavailable future sources remain blank, not zero-weight observations.
        for row in trace:
            heatmap[row["site"], :row["sources"]] = row["weights"]  # Local: align completed block/source indices across depth.
        st.plotly_chart(px.imshow(heatmap, zmin=0, zmax=1, labels={"x": "Available source index (0 = embedding)", "y": "Read site", "color": "Mean routing weight"}, title=f"Routing at forecast step {route['final_forecast_step']}"), width="stretch")  # Global: distinguish token-averaged diagnostic weights from a universal routing rule.

with tabs[3]:
    st.image(str(PROJECT / "assets/architecture.svg"), caption="Only depth composition changes; upstream GPhyT transforms are reused.")  # Local: architecture is shared with the repository overview.
    sensitivity = read_json(PROJECT / "reports/power_sensitivity.json")  # Local: prospective simulation is independent of observed training results.
    if sensitivity:
        frame = pd.DataFrame(sensitivity["rows"])  # Local: one row per explicit effect/variance assumption.
        st.plotly_chart(px.line(frame, x="seed_log_sd", y="sign_flip_power", color="reduction", markers=True, range_y=[0, 1], title="Prospective exact-test power under Gaussian paired effects"), width="stretch")  # Global: a bounded probability axis avoids exaggerating sensitivity.
        st.dataframe(frame, hide_index=True, width="stretch")  # Local: show numerical assumptions alongside the plot.
    st.markdown("[Preregistered protocol](https://github.com/BurnyCoder/general-physics-attention-residuals-transformer/blob/main/docs/protocol.md) · [Code and source attribution](https://github.com/BurnyCoder/general-physics-attention-residuals-transformer)")  # Global: direct links make claims auditable.
