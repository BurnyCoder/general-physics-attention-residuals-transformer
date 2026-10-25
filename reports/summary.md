# Consolidated experiment record

Generated 2026-10-25T14:53:10+00:00. 0/18 planned runs completed; primary inference status: **undefined**.

This is a from-scratch, supervised next-frame regression study. A **seed pair** is two models with matching initial backbone weights and training examples. **NMSE** divides spatial squared error by target energy; lower is better. A **rollout** repeatedly feeds predictions back as inputs. A **confidence interval** quantifies uncertainty under the stated seed-level assumptions, not a guarantee for unseen physics.

| Network | Depth composition | Algorithm/objective | Data/curriculum | Completion/accuracy |
|---|---|---|---|---|
| standard | Upstream additive residuals | AdamW / masked MSE | Two fixed physics datasets; balanced sampling; strides 1-8 | 0/6 complete; see measured CSV |
| block | Learned attention over four-sublayer sums | AdamW / masked MSE | Two fixed physics datasets; balanced sampling; strides 1-8 | 0/6 complete; see measured CSV |
| full | Learned attention over every raw output | AdamW / masked MSE | Two fixed physics datasets; balanced sampling; strides 1-8 | 0/3 complete; see measured CSV |
| uniform_block | Fixed averaging over the Block source bank | AdamW / masked MSE | Two fixed physics datasets; balanced sampling; strides 1-8 | 0/3 complete; see measured CSV |

All models share twelve layers, width 192, MLP width 768, three heads, and 1x16x16 patches. Fill/completion benchmarks and reinforcement-learning reward are not applicable. Detailed raw outcomes are in [experiments.csv](experiments.csv), [metrics.csv](metrics.csv), and the [individual reports](experiments).

## Primary analysis

```json
{
  "status": "undefined",
  "n_pairs": 6,
  "reason": "Primary inference requires all six finite preregistered pairs",
  "created": "2026-10-25T14:53:08+00:00",
  "minimum_meaningful_improvement": 0.05,
  "scope": "Two selected datasets at 128x64, fixed training budget and GPhyT-S scale"
}
```

## Sources and audit

See [method and prespecified decisions](../docs/protocol.md), [literature/code review](../docs/sources.md), [development learnings](development.md), and the [manuscript](../paper/main.tex). Raw data and checkpoints are excluded from Git; public checkpoint links appear in publication.json after a successful upload.
