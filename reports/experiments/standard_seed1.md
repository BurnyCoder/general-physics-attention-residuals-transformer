# standard_seed1

Status: **complete**. Completed updates: 3175.

Method, data, optimization rationale, fixed endpoints, and analysis assumptions are specified once in [the protocol](../../docs/protocol.md). This run shares its backbone initialization and data schedule with the other variants at the same seed.

## Measured twelve-step errors

- rayleigh_benard_uniform: not available / undefined
- turbulent_radiative_layer_2D: not available / undefined

## Interpretation and learnings

This single run cannot establish a seed-robust architecture effect. Interpret it with the preregistered paired analysis and all failures. No explanation of its error is inferred from a single curve.

Exact run metadata:

```json
{
  "mode": "standard",
  "seed": 1,
  "updates": 3175,
  "microbatch": 4,
  "freeze_sha256": "fcdfc583a123a39706e60fc87aa83bce07be52baa2e13154bc5f02e2c9d2eda4",
  "commit": "d43e6479e8a9b9174eeeb536a22ef863b7862df0",
  "status": "complete",
  "started": "2026-10-25T18:38:46+00:00",
  "step": 3175,
  "prior_seconds": 0,
  "seconds": 1282.0112898000007,
  "peak_memory_bytes": 291984384,
  "ended": "2026-10-25T19:00:08+00:00"
}
```
