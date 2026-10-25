# Completion checklist

- [x] Pinned upstream code and licenses; environment locked
- [x] Public protocol and initial implementation pushed before model execution
- [x] Equation, gradient, integration, data-adapter, and resume tests pass (25 CPU tests; four budgeted full-scale CUDA checks)
- [ ] Official data downloaded, hashed, checked, and normalized without leakage
- [x] Prospective power sensitivity published
- [ ] Profile measured; common schedule frozen and published
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
