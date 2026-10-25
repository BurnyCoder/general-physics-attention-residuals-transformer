# Implementation and verification learnings

## 2026-10-25: residual accumulation precision

A final source-level precision trace identified a material potential confound before profiling or confirmatory training. The upstream positional addition produces an FP32 residual stream, so Standard adds AMP sublayer outputs into FP32. A direct Block sum of BF16 writer outputs would instead accumulate in BF16. A failing analytical fixture demonstrated lost increments: four writers emitting `1, 1/256, 1/256, 1/256` gave a final two-source mean of `0.5` instead of `(1+3/256)/2`. Promoting raw outputs to the embedding stream's dtype before summing fixes this without altering their values or the source schedule. Full and uniform modes use the same promotion; Standard remains untouched. This follows [PyTorch's mixed-precision promotion rules](https://docs.pytorch.org/docs/stable/accelerator/amp.html) and the inspected pinned upstream positional-addition code. The clarification is recorded before the training freeze; no test-accuracy result motivated it.

## 2026-10-25: portable replication, populated reporting, and a hard deadline

The portable fingerprint retains scientific inputs while excluding host paths and preparation timestamps. A regression verifies that moving files preserves identity but changing normalization or split membership does not. Clones reuse the published source registry; `reproduce` adopts the original frozen horizon instead of recomputing it from a different GPU's speed. Private and published frozen schedules must match.

The reporting workflow now generates complete-matrix seed summaries, descriptive confidence intervals, cost tables, physical diagnostics, rollout-failure counts, primary confidence-interval plots, and fixed-case routing heatmaps. An isolated synthetic fixture exercises populated analysis, tables, plotting, and LaTeX rendering. Fixture values never enter the public experiment record. A Windows figure-open error occurred during simultaneous PDF inspection; complete temporary figure files are now atomically promoted, and the full report command succeeds.

A cooperative budget check alone cannot stop a hung work unit. A failing subprocess regression demonstrated this limitation. An independent, hidden process supervisor now waits on the exact worker PID and creation time, terminates it before the phase/total cap if necessary, and preserves the hard stop in the ledger. Normal context exit cancels its own supervisor. The ordinary loop retains an earlier cooperative margin for checkpoint/cleanup. Sources: [psutil process supervision](https://psutil.readthedocs.io/en/latest/), [Python subprocess](https://docs.python.org/3/library/subprocess.html).

The [focused review](../docs/review.md) records scientific assumptions, resolved findings, and remaining empirical verification. The public Pages deployment succeeded in [Actions run 40206948848](https://github.com/BurnyCoder/general-physics-attention-residuals-transformer/actions/runs/40206948848), and its hosted dataset/horizon controls were exercised.

## 2026-10-25: public viewer and segmented-download regression

The GitHub Pages viewer reuses generated analysis tables and saved forecast arrays. Its local dataset, horizon, run, and curve selectors were exercised in the browser; pending measurements remain explicitly missing. GitHub Pages is configured to deploy only the allowlisted `site/` directory through official Actions.

A new failing regression fixture reproduced a sparse-download hazard: an aria2 partial file can already have the server's final length while its `.aria2` control file still records missing pieces. The downloader now refuses an HTTP-only resume in this state and requires the control file to disappear after successful aria2 completion. This follows [aria2's control-file contract](https://aria2.github.io/manual/en/html/aria2c.html#control-file). Active downloads already use aria2; none has been accepted through this faulty fallback path.

A matched-subset analysis fixture also verifies that routing interventions compare the exact same trajectories with ordinary routing: changing an unrelated test trajectory's error cannot change the within-model ratio. A separate export fixture verifies that missing CSV metrics remain JSON null rather than becoming zero.

The focused correctness review found a second numerical edge case: multiplying an absent field's NaN error by a zero mask still yields NaN. A failing CPU regression now checks both loss gradients and reported errors with an absent NaN channel. Masking before squaring fixes the fixture, following [PyTorch's autograd guidance](https://docs.pytorch.org/docs/stable/notes/autograd.html). Nonfinite active forecasts remain divergent; diagnostic traces retain nulls and the actual last attempted step instead of falsely labeling every trace as step 24. The resulting suite passes **19 CPU tests**.

This log describes implementation evidence, not physics accuracy. Scientific settings remain in the [preregistered protocol](../docs/protocol.md); measurements and source links are centralized in the [experiment summary](summary.md) and [source review](../docs/sources.md).

## 2026-10-25: source reuse and causal composition

We retained the pinned GPhyT modules and wrapped only depth composition. Independent equation/gradient tests cover FLA's operator. Analytical constant-output writers check raw-source retention, one embedding, a partial final block, final aggregation, and embedding gradients. An exact standard-path check compares the original forward with the identity wrapper. These tests are preferable to a shape-only test because an incorrect cumulative-state implementation can still return the expected shape. The FLA operator's contiguous-view requirement is satisfied at the wrapper boundary. Initial pseudo-queries yield averaging, not the standard additive skip path; the uniform control therefore tests a distinct normalization change.

## 2026-10-25: data transfer

The official server supports byte ranges but long file transfers were slow. We added bounded range resumption, persisted server identity, and independently hashed complete files. Old partial files lacking identity metadata were preserved separately and never used as accepted data. An optional official aria2 1.37.0 binary then provided four segmented connections per file, with three files in flight. This transport change does not alter dataset selection, splits, byte-level acceptance, normalization, or outcomes. Sources: [The Well distribution discussion](https://github.com/PolymathicAI/the_well/discussions/8), [reported download limitations](https://github.com/PolymathicAI/the_well/issues/94), and [aria2's documented segmentation/resume](https://aria2.github.io/manual/en/html/aria2c.html). The Hugging Face search returned no matching uniform-grid mirror; the nonuniform dataset was not substituted.

## 2026-10-25: execution and checkpointing

The initial TDD collection failed before the data module existed, then passed once implemented. Subsequent contracts cover actual HDF5 preprocessing, paired sampling, train-only normalization, split duplication, masked metrics, prediction-only feedback, and optimizer/RNG restoration. CPU checkpoint testing compares the next optimizer update against uninterrupted continuation. Full-scale BF16 checks ran all four variants on the RTX 5070 inside the GPU ledger. Their synthetic losses are execution checks only. Measured parameter counts are 7,919,045 for the five-channel standard backbone and 7,928,645 total for the routing variants; uniform-routing parameters are frozen.

## 2026-10-25: UI, reporting, and power

Streamlit's first-run email prompt prevented unattended startup; a project-local headless configuration fixed it and disabled usage telemetry. The app browser showed the dashboard and its protocol/power tab; a screenshot is retained in `assets/dashboard-power.png`. Empty-result states explicitly avoid fabricated scores. A fixed 100,000-draw prospective simulation quantifies six-pair power under the declared Gaussian effect/variance assumptions. All 18 planned runs receive reports even before execution, labeled `not_started`. The manuscript is provisionally labeled until confirmatory outcomes exist. The app's LaTeX compiler successfully compiled the current source after a stale-project retry.

## Pending scientific evidence

The final pre-profile CPU suite passes **25 tests**. The precision regression was observed failing before promotion of raw sublayer outputs to the embedding residual dtype, then passing after that change. GPU phases now hold a process-scoped Windows automatic-sleep request and restore the preceding state on exit; this does not alter persistent power settings. Source: [SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate). The independent budget supervisor and its termination/cancellation tests remain active.

Training, final-checkpoint evaluation, interventions, empirical effect interpretation, public checkpoint round-trip verification, and final artifact review remain pending until their corresponding artifacts establish completion. No positive or negative physics prediction result is inferred from execution tests or power sensitivity.
