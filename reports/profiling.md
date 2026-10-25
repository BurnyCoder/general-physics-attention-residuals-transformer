# Measured training schedule

Frozen on **2026-10-25 at 17:17:23 UTC**, before any confirmatory run. The immutable machine-readable record is [frozen_protocol.json](frozen_protocol.json), tied to source commit `f129be8afb7202c6806ff5e6e61a00e4880bf8b4` and the portable data fingerprint.

All four variants passed at the first approved candidate, **microbatch 4**. Effective batch size remains **16**, with four accumulation steps per optimizer update. The cost rule selects **3,175 updates for every run**, or 50,800 sampled training examples per run. No accuracy-based hyperparameter selection was performed.

| Variant | Median update seconds | 90th-percentile update seconds | Validation seconds | Checkpoint seconds | Peak allocated bytes |
|---|---:|---:|---:|---:|---:|
| Standard | 0.7760 | 0.8767 | 0.6762 | 1.4033 | 291,984,384 |
| Block | 0.7913 | 0.8592 | 0.3537 | 1.5554 | 383,652,864 |
| Full | 0.8914 | 0.9673 | 0.3881 | 1.5016 | 572,519,424 |
| Uniform Block | 0.8471 | 0.9304 | 0.4174 | 1.6293 | 362,658,816 |

The device was an NVIDIA GeForce RTX 5070 Laptop GPU, using PyTorch `2.14.1+cu130` and CUDA 13.0. Each variant used profiling seed 10000, two warm-up updates, and sixteen timed real-data updates. The profiling path reuses the actual optimizer, effective batch, validation sampler, and complete resume-checkpoint writer. Peak allocated bytes are PyTorch tensor allocations, not total system/GPU residency. Sixteen timings are a short scheduling measurement; their differences do not establish stable throughput rankings.

For variant counts 6/6/3/3, the fixed rule sums each variant's 90th-percentile update cost, amortized validation every 500 updates, and checkpointing every 250 updates. It reserves a final checkpoint plus 15 seconds of initialization per run, then applies the preregistered 20% safety margin to the 18-hour training allocation. The resulting projection is **51,839.981 seconds (14.400 hours)**. Timing may change during sustained operation; the hard phase supervisor still enforces the 18-hour allocation and cumulative 24-hour cap. Missing runs remain visible rather than receiving extra time or replacement seeds.

Real-data profiling consumed **71.144 budgeted seconds**. Including the earlier four-path synthetic execution verification, the ledger contained **74.064 seconds** before confirmatory training. Profiling losses are execution diagnostics, not final physics prediction results. Their values and every timed update are retained in the frozen artifact and timestamped log.

Sources: [prespecified budget rule](../docs/protocol.md), [profiling implementation](../src/gphyt_attnres_study/profiling.py), and [PyTorch CUDA memory/timing documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html).
