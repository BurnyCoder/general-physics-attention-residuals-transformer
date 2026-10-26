# Consolidated experiment record

Generated 2026-10-26T11:27:10+00:00. 18/18 planned runs completed; primary inference status: **complete**.

This is a from-scratch, supervised next-frame regression study. A **seed pair** is two models with matching initial backbone weights and training examples. **NMSE** divides spatial squared error by target energy; lower is better. A **rollout** repeatedly feeds predictions back as inputs. A **confidence interval** quantifies uncertainty under the stated seed-level assumptions, not a guarantee for unseen physics.

| Network | Depth composition | Algorithm/objective | Data/curriculum | Completion/accuracy |
|---|---|---|---|---|
| standard | Upstream additive residuals | AdamW / masked MSE | Two fixed physics datasets; balanced sampling; strides 1-8 | 6/6 complete; see measured CSV |
| block | Learned attention over four-sublayer sums | AdamW / masked MSE | Two fixed physics datasets; balanced sampling; strides 1-8 | 6/6 complete; see measured CSV |
| full | Learned attention over every raw output | AdamW / masked MSE | Two fixed physics datasets; balanced sampling; strides 1-8 | 3/3 complete; see measured CSV |
| uniform_block | Fixed averaging over the Block source bank | AdamW / masked MSE | Two fixed physics datasets; balanced sampling; strides 1-8 | 3/3 complete; see measured CSV |

All models share twelve layers, width 192, MLP width 768, three heads, and 1x16x16 patches. Fill/completion benchmarks and reinforcement-learning reward are not applicable. Detailed raw outcomes are in [experiments.csv](experiments.csv), [metrics.csv](metrics.csv), and the [individual reports](experiments).

## Primary analysis

```json
{
  "status": "complete",
  "n_pairs": 6,
  "effects": [
    -0.14668375154119728,
    -0.2792059750393223,
    -0.2882002923990926,
    -0.44303401893880767,
    -0.07947003323909095,
    -0.4512263735091465
  ],
  "mean_log_ratio": -0.28130340744444293,
  "sd_log_ratio": 0.15095809619491443,
  "improvement": 0.2452007109885831,
  "improvement_ci95": [
    0.11563251982290934,
    0.35578593801520586
  ],
  "permutation_p": 0.03125,
  "interval_assumption": "Independent approximately normal paired seed log ratios; fixed datasets and windows",
  "created": "2026-10-26T11:26:57+00:00",
  "minimum_meaningful_improvement": 0.05,
  "scope": "Two selected datasets at 128x64, fixed training budget and GPhyT-S scale"
}
```

## Interpretation and learnings

See the [completed-study interpretation](interpretation.md) for the relative gain, severe convection failure against persistence, physical diagnostics, exploratory controls, and explicitly post-hoc metric decomposition. The numerical record here and that source-linked interpretation form the consolidated report.

## Sources and audit

See [method and prespecified decisions](../docs/protocol.md), [literature/code review](../docs/sources.md), [data preparation and initial-state reuse](data-preparation.md), [development learnings](development.md), and the [manuscript](../paper/main.tex). Raw data and checkpoints are excluded from Git; public checkpoint links appear in publication.json after a successful upload.
