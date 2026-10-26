# Interpretation of the completed study

All 18 preregistered runs completed 3,175 updates, and all 31 evaluation files are available: 18 ordinary models, 12 within-model routing interventions, and persistence. No seed was added, no final checkpoint was selected by accuracy, and no evaluated forecast became nonfinite through 24 steps. These statements describe completion and arithmetic, not physical reliability. Sources are the [training receipt](training_completion.json), [raw evaluations](evaluation), and [rollout counts](rollout_stability.csv).

## Primary result

Block AttnRes reduced the equally weighted dataset geometric error ratio against Standard GPhyT by **24.52%**, with a **95% confidence interval of 11.56–35.58%**, and exact two-sided paired **p = 0.03125**. All six seed effects favor Block. The interval lies above the preregistered 5% practical threshold under the stated approximately normal independent-seed assumptions, conditional on these fixed test trajectories. Six pairs permit only 64 sign assignments: 0.03125 is the minimum attainable two-sided p-value. This is neither a 24.52% improvement in every field nor evidence about arbitrary physics systems. See [primary analysis](primary_analysis.json), [the frozen endpoint](../docs/protocol.md), and the [independent numerical reconstruction](final_metric_audit.json).

## Absolute failures matter

| Dataset, 12-step NMSE | Standard, 6 seeds | Block, 6 seeds | Persistence |
|---|---:|---:|---:|
| Uniform convection | 39,284.16 | 30,005.97 | 0.21738 |
| Radiative layer | 0.39277 | 0.28917 | 0.41901 |

These are arithmetic mean seed errors, distinct from the primary geometric paired estimand. Every trained architecture performs much worse than persistence on convection. Radiative Block forecasts improve on both Standard and persistence at 12 and 24 steps. The mean 24-step radiative errors are 0.4270, 0.7028, and 0.5403 for Block, Standard, and persistence respectively. Full tables, including all exploratory seeds and uncertainty, are in [accuracy_summary.csv](accuracy_summary.csv).

The large convection numbers deserve an explicit metric diagnosis. A **post-hoc** decomposition uses only the already saved first test trajectory/window at seed zero; selection of this visual example preceded outcomes. For Standard's first twelve steps, pressure and buoyancy NMSE are 0.502 and 0.0394, while the two velocity terms are about 15,707 and 34,062. Minimum standardized target energies are 1.19e-7 and 2.67e-8 for those velocities, below the fixed 1e-6 denominator stabilizer. Small target energy amplifies velocity errors in the specified metric. The reconstruction reproduces the saved aggregate exactly within floating-point tolerance. It explains one example's contributions, not every large error. No epsilon, window, normalization, or field weight was changed. The [audit script](../scripts/audit_final_metrics.py) and [all active-field values for four fixed cases](final_metric_audit.json) make this check reproducible without another GPU run.

The physical diagnostics also preclude a reliability claim. Mean convection divergence error RMS is 4.134 for Standard and 2.823 for Block, versus 0.228 for persistence. Over 24-step radiative forecasts, mean negative-density fractions are **26.04% for Standard and 27.30% for Block**, versus zero for persistence. There were no negative-pressure cells in ordinary evaluations. Thus improved NMSE does not improve every physical diagnostic. Predictions were neither clipped nor repaired. See [physical_diagnostics.csv](physical_diagnostics.csv); these proxies do not establish conservation or numerical-solver accuracy.

## Controls and interventions answer different questions

Across matched seeds 0–2, Full/Block has geometric error ratio 0.8349 and Uniform/Block has ratio 1.0238, using equal dataset log weights. The three Uniform/Block seed effects include both signs. These small exploratory comparisons leave the incremental value of learned selection over fixed averaging unresolved. They cannot borrow the six-pair primary p-value. Their individual paired values are retained in [final_metric_audit.json](final_metric_audit.json).

On the prescribed first four test trajectories per dataset, changing trained Block routing to uniform increases 12-step error by mean factors 2.565 (convection) and 7.490 (radiative); shuffled depth weights give 2.630 and 9.524. Every measured seed/dataset ratio exceeds one. These interventions demonstrate dependence on routing in those already trained networks and that subset. They do not establish that learned routing uniquely mediates the Standard–Block training effect, and they do not contradict a uniformly routed model learning a different backbone from scratch. See [matched intervention ratios](interventions.csv) and the [precision audit](architecture-audit.md).

## Cost, scope, and learnings

Training used 11.068 of its allocated 18 GPU-hours. Mean per-run wall times were 1,570/2,460/2,756/2,455 seconds for Standard/Block/Full/Uniform, with mean run-level throughputs of 34.42/21.19/19.09/21.41 examples/s. Peak PyTorch allocated device memory was 278.46/365.88/546.00/345.86 MiB. This is one laptop under varying host conditions; these measurements are not universal kernel benchmarks. Equal updates match data exposure, not compute cost. See [costs.csv](costs.csv) and the [complete GPU ledger](gpu_budget.json).

The experiment supports a bounded residual-composition improvement, alongside severe absolute failures. It does not establish unseen-initial-condition generalization: all held-out trajectories reuse a first frame from training at a different physical parameter setting. Coarse resolution, short training, six primary seeds, a mixed stride objective without a stride token, and the standardized low-energy metric limit interpretation. The fixed plan was not extended after outcomes. A future study could preregister low-energy-aware metrics, explicit temporal conditioning, and physical admissibility constraints, but none of those alternatives was tested here. See [data overlap evidence](data-preparation.md), [methods](../docs/protocol.md), and [the paper](../paper/main.pdf).
