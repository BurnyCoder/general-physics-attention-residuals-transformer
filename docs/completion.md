# Completion checklist

- [x] Pinned upstream code and licenses; environment locked
- [x] Public protocol and initial implementation pushed before model execution
- [x] Equation, gradient, integration, data-adapter, and resume tests pass (44 CPU tests including the final CLI logging regression; four pretraining budgeted full-scale CUDA checks)
- [x] Requested independent architecture audit completed; intervention arithmetic corrected and mixed-precision discrepancy disclosed without changing training (see [audit evidence](../reports/architecture-audit.md))
- [x] Corrected neutral routing interventions verified on CUDA within the existing budget after training (16 exact-equality checks; [evidence](../reports/cuda_intervention_verification.json))
- [x] Official data downloaded, hashed, grid/split checked, and normalized from training only; cross-parameter initial-state reuse is disclosed in the data audit
- [x] Prospective power sensitivity published
- [x] Profile measured; common schedule frozen and published (3,175 updates, microbatch 4, effective batch 16)
- [x] Six baseline/Block seed pairs completed at the frozen update count
- [x] Three Full/Uniform seed pairs completed at the frozen update count; [all 18 checkpoints and logs hashed](../reports/training_completion.json)
- [x] Rollout, persistence, physical, efficiency, and intervention analyses complete (all 31 evaluations; independent reconstruction of the six paired effects)
- [x] Interactive dashboard and figures/videos verified (four tabs, dataset/horizon/field/routing controls; four 24-frame videos)
- [x] Populated interactive public viewer deployed to GitHub Pages and verified (live dataset/horizon, cost/routing, and forecast-frame controls; remote paper and all preview data match committed bytes)
- [x] Individual and consolidated reports generated (18 experiment reports, shared interpretation, and reproducible raw tables)
- [x] Paper compiled, visually checked, and claims audited (all 20 pages in the initialization-baseline revision, including negative results and narrow scope)
- [x] Checkpoints and collection published; downloaded checkpoint re-evaluated (all 18; Standard seed zero round trip)
- [x] Requested baseline emphasis added: explicit “without Attention Residuals” headline/abstract, matched baseline tables, all-four-model field examples, and the six-seed public baseline model card ([focused verification](../reports/baseline_presentation_verification.json))
- [x] Requested control coverage added: all-model/persistence score table, separate matched-subset table for original/uniform/random-shuffled Block, and matched field examples ([focused verification](../reports/control_presentation_verification.json))
- [x] Control-presentation revision deployed and checked on the public site ([public byte checks and browser evidence](../reports/public_deployment.json))
- [ ] Additional user-requested initialization-only baselines evaluated and included in README/paper tables and matched field examples ([post-hoc protocol](untrained-baselines.md))
- [x] Final review, functional PRs merged, and all claims limited to measured scope ([release PR #9](https://github.com/BurnyCoder/general-physics-attention-residuals-transformer/pull/9) and [verification PR #10](https://github.com/BurnyCoder/general-physics-attention-residuals-transformer/pull/10), both merged)

The [focused implementation review](review.md), [execution record](../reports/execution.md), and [final verification](../reports/final-verification.md) link each completed item to evidence. The original planned deliverables are complete; their charged GPU time was 11.745949 hours. The later initialization-only extension adds 0.462928 hours, bringing total usage to **12.208877 hours** under the unchanged phase allocations. Its separate checklist item records release completion. No synthetic fixtures enter public results. The findings remain conditional on the frozen scale, data, resolution, and training budget, and include major negative results.

A requested completion recheck on 2026-10-26 verified the actual files and public releases again: all 18 local and public checkpoint hashes, 57,150 update records, 31 evaluation files, unchanged scientific inputs, collection membership, and current public paper/data. The [machine-readable receipt](../reports/plan_completion_recheck.json) records the scope and outcomes. No additional model execution was needed.
