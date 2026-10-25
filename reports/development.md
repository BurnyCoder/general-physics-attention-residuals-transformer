# Implementation and verification learnings

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

Training, final-checkpoint evaluation, interventions, empirical effect interpretation, public checkpoint round-trip verification, and final artifact review remain pending until their corresponding artifacts establish completion. No positive or negative physics prediction result is inferred from execution tests or power sensitivity.
