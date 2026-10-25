# Completion checklist

- [x] Pinned upstream code and licenses; environment locked
- [x] Public protocol and initial implementation pushed before model execution
- [x] Equation, gradient, integration, data-adapter, and resume tests pass (43 CPU tests; four pretraining budgeted full-scale CUDA checks)
- [x] Requested independent architecture audit completed; intervention arithmetic corrected and mixed-precision discrepancy disclosed without changing training (see [audit evidence](../reports/architecture-audit.md))
- [ ] Corrected neutral routing interventions verified on CUDA within the existing budget after training, before reporting mechanism effects
- [x] Official data downloaded, hashed, grid/split checked, and normalized from training only; cross-parameter initial-state reuse is disclosed in the data audit
- [x] Prospective power sensitivity published
- [x] Profile measured; common schedule frozen and published (3,175 updates, microbatch 4, effective batch 16)
- [ ] Six baseline/Block seed pairs completed or failures explicitly documented
- [ ] Three Full/Uniform seed pairs completed or exclusions explicitly documented
- [ ] Rollout, persistence, physical, efficiency, and intervention analyses complete
- [ ] Interactive dashboard and figures/videos verified
- [x] Interactive public viewer deployed to GitHub Pages and verified (pending-results snapshot; measured-result views still require final verification)
- [ ] Individual and consolidated reports generated
- [ ] Paper compiled, visually checked, and claims audited
- [ ] Checkpoints and collection published; downloaded checkpoint re-evaluated
- [ ] Final review, functional PRs merged, and all claims limited to measured scope

Partial artifacts: local dashboard tabs and power visualization were visually checked; an eight-page provisional PDF was compiled and all pages inspected. The hosted pending-results explorer and its controls were verified after PR #2 deployed successfully. Populated report branches were exercised using isolated fixtures, which never enter real results. Empirical figures, final paper claims, and model releases await completed experiments. The [focused implementation review](review.md) is complete; final empirical verification remains open.
