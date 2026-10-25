# Data preparation and split audit

Preparation completed on 2026-10-25 at 16:46:49 UTC. All 36 selected official HDF5 files were transferred, size-checked, SHA-256 hashed, and processed into 240 trajectories. Total raw size is approximately 38.607 GB (decimal). Exact source identities, file hashes, original coordinates, trajectory hashes, active channels, and training-only normalization statistics are in [data_manifest.json](data_manifest.json); selection provenance is in [source_manifest.json](source_manifest.json).

| Dataset | Train | Validation | Test | Original x/y grid | Frames per trajectory |
|---|---:|---:|---:|---|---:|
| Uniform Rayleigh–Bénard convection | 120 | 15 | 15 | 512 × 128 | 200 |
| Turbulent radiative layer, 2D | 72 | 9 | 9 | 128 × 384 | 101 |

All inspected spatial coordinates pass the prescribed uniform-spacing check. The common prepared grid is 128 × 64. Active fields and fitting counts agree with the original dataset schemas and selected training records. Normalization values are finite. The source documentation is [uniform convection](https://github.com/PolymathicAI/the_well/tree/master/datasets/rayleigh_benard_uniform) and [radiative layer](https://github.com/PolymathicAI/the_well/tree/master/datasets/turbulent_radiative_layer_2D).

## Initial-condition reuse and interpretation

There are **no exact complete-trajectory hashes shared across official splits**. However, the raw first-frame hashes form **33 distinct cross-split groups**: 25 convection groups and eight radiative groups. Every validation and test trajectory shares its initial field state with at least one training trajectory at another parameter setting. No cross-split initial-state group repeats within the same parameter setting. Exact member identities and denominators are in [data_audit.json](data_audit.json).

The original preparation log reports 39 overlap witnesses because its sequential check records split transitions for repeated hashes. That count is not the number of distinct initial-state groups. The additional audit groups hashes explicitly and distinguishes these units. A regression verifies this distinction and also detects actual four-frame duplicates when inserted into an isolated fixture.

No identical **first four consecutive prepared frames** occur across splits. This is a narrowly scoped context check, not an exhaustive comparison of every strided training window against every evaluation window or a claim that trajectories are statistically unrelated. The shared initial states restrict interpretation: these official splits do not test generalization to entirely unseen initial conditions. The paired architecture comparison remains conditional on the fixed official splits and parameter mixture. We preserve the approved selection, endpoints, and normalization policy; this audit precedes profiling and confirmatory training and does not use model prediction errors.

## Reproduction and learning

Run the documented `prepare` phase, or rerun `audit` after preparation; see [operational instructions](../docs/data-preparation.md). The audit reuses hashes from preparation and memory-maps the first four resized frames. It writes a separate artifact tied to the portable data fingerprint without changing the scientific inputs. Grouping is implemented with [Python collections](https://docs.python.org/3/library/collections.html#collections.defaultdict), exact byte identities with [hashlib](https://docs.python.org/3/library/hashlib.html), and bounded reads with [NumPy memory maps](https://numpy.org/doc/stable/reference/generated/numpy.load.html).

The important learning is that an official trajectory split and absence of duplicate complete trajectories do not imply disjoint initial conditions across a multi-parameter simulation collection. This limitation must accompany the final measured claims.
