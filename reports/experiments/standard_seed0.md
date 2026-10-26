# standard_seed0

Status: **complete**. Completed updates: 3175.

Method, data, optimization rationale, fixed endpoints, and analysis assumptions are specified once in [the protocol](../../docs/protocol.md). This run shares its backbone initialization and data schedule with the other variants at the same seed.

## Measured twelve-step errors

- rayleigh_benard_uniform: 31067.703249100996
- turbulent_radiative_layer_2D: 0.3832423027578889

## Interpretation and learnings

This single run cannot establish a seed-robust architecture effect. Interpret it with the preregistered paired analysis and all failures. No explanation of its error is inferred from a single curve.

The [completed-study interpretation](../interpretation.md) discusses absolute persistence comparisons, physical failures, controls, and the post-hoc metric diagnosis.

Exact run metadata:

```json
{
  "mode": "standard",
  "seed": 0,
  "updates": 3175,
  "microbatch": 4,
  "freeze_sha256": "fcdfc583a123a39706e60fc87aa83bce07be52baa2e13154bc5f02e2c9d2eda4",
  "commit": "5ceae76f399c6504bf61e3cb1649d201e4c99056",
  "status": "complete",
  "started": "2026-10-25T17:20:58+00:00",
  "step": 3175,
  "prior_seconds": 0,
  "seconds": 2191.0929419999993,
  "peak_memory_bytes": 291984384,
  "ended": "2026-10-25T17:57:29+00:00"
}
```
