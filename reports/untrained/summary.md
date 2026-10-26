# Initialization-only baseline results

Generated 2026-10-26T15:34:18+00:00.

**Post-hoc initialization-only controls: 18/18 evaluations complete.** These models have zero training updates and were reconstructed with the original constructor, frozen configuration, and seeds. They were requested after the trained results were known. Both columns use the same full test population and the same seeds for each architecture. Lower 12-step NMSE is better; finite/planned counts describe initialized seed endpoints.

| Dataset | Model | Finite/planned | Initialized (0 updates) | Trained (3,175 updates) |
| --- | --- | --- | --- | --- |
| Convection | GPhyT without AttnRes | 6/6 | 350742 | 39284.2 |
| Convection | Block AttnRes | 6/6 | 8173.5 | 30006 |
| Convection | Full AttnRes | 3/3 | 1332 | 18397.8 |
| Convection | Uniform Block | 3/3 | 7780.76 | 24987 |
| Radiative | GPhyT without AttnRes | 6/6 | 7.02088 | 0.392769 |
| Radiative | Block AttnRes | 6/6 | 0.572704 | 0.289174 |
| Radiative | Full AttnRes | 3/3 | 0.413095 | 0.317353 |
| Radiative | Uniform Block | 3/3 | 0.524838 | 0.347345 |

All 1/12/24-step scores, seed SDs, and descriptive intervals are in [the comparison CSV](../untrained/comparison.csv). Missing or divergent seeds remain undefined. The original six-pair primary comparison is unchanged. Block and Uniform have identical initial routing on their shared seeds 0–2; their overall means differ because Block also includes seeds 3–5. [Additional protocol](../../docs/untrained-baselines.md).

![Ground truth, persistence, initialized and trained GPhyT without AttnRes on matching convection and radiative cases](../figures/initialization_standard.png)

![Ground truth, persistence, initialized and trained Block AttnRes on matching convection and radiative cases](../figures/initialization_block.png)

![Ground truth, persistence, initialized and trained Full AttnRes on matching convection and radiative cases](../figures/initialization_full.png)

![Ground truth, persistence, initialized and trained Uniform Block on matching convection and radiative cases](../figures/initialization_uniform_block.png)

Examples use seed 0, first test trajectory/window, forecast step 12: convection buoyancy above, radiative density below. Each row has one common unclipped color range; magenta marks nonfinite predictions.
