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
- [ ] Populated interactive public viewer deployed to GitHub Pages and verified (final local snapshot verified; public deployment follows the release merge)
- [x] Individual and consolidated reports generated (18 experiment reports, shared interpretation, and reproducible raw tables)
- [x] Paper compiled, visually checked, and claims audited (all 15 pages, including negative results and narrow scope)
- [x] Checkpoints and collection published; downloaded checkpoint re-evaluated (all 18; Standard seed zero round trip)
- [ ] Final review, functional PRs merged, and all claims limited to measured scope

The [focused implementation review](review.md), [execution record](../reports/execution.md), and [final verification](../reports/final-verification.md) link each completed item to evidence. The remaining release merge and deployed-view checks are explicitly open until they occur. No synthetic fixtures enter public results.
