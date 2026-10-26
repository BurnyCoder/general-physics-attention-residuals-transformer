# standard_seed4

Status: **complete**. Completed updates: 3175.

Method, data, optimization rationale, fixed endpoints, and analysis assumptions are specified once in [the protocol](../../docs/protocol.md). This run shares its backbone initialization and data schedule with the other variants at the same seed.

## Measured twelve-step errors

- rayleigh_benard_uniform: 34419.98364379827
- turbulent_radiative_layer_2D: 0.3957106210162784

## Interpretation and learnings

This single run cannot establish a seed-robust architecture effect. Interpret it with the preregistered paired analysis and all failures. No explanation of its error is inferred from a single curve.

The [completed-study interpretation](../interpretation.md) discusses absolute persistence comparisons, physical failures, controls, and the post-hoc metric diagnosis.

Exact run metadata:

```json
{
  "mode": "standard",
  "seed": 4,
  "updates": 3175,
  "microbatch": 4,
  "freeze_sha256": "fcdfc583a123a39706e60fc87aa83bce07be52baa2e13154bc5f02e2c9d2eda4",
  "commit": "4583931575ec13683da4beb18a66ccb4fd4c2d3e",
  "status": "complete",
  "started": "2026-10-25T21:49:45+00:00",
  "step": 3175,
  "prior_seconds": 0,
  "seconds": 1488.4742067000007,
  "peak_memory_bytes": 291984384,
  "ended": "2026-10-25T22:14:33+00:00"
}
```
