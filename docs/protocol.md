# Preregistered protocol, version 1.0

Status: specified before the first model execution. Git history records amendments; test outcomes cannot justify changing endpoints, seeds, or datasets. The project implements the human-approved plan from 2026-10-25.

## Question and scope

Does Block Attention Residuals improve the small General Physics Transformer's 12-step autoregressive prediction accuracy under a fixed local training budget? This is an architecture-transfer experiment, not a reproduction of full GPhyT pretraining or a test of universal physics generalization. Sources and code provenance are in [sources.md](sources.md).

## Data and model

Use the official The Well train/valid/test trajectories for `rayleigh_benard_uniform` at Pr=1 and Ra={1e6,1e8,1e10}, and `turbulent_radiative_layer_2D` at every cooling setting. Do not mix original and uniformly resampled versions of the same simulations. Store file SHA-256 hashes, per-trajectory hashes, source coordinates, parameter settings, field ordering, and trajectory counts. Abort on an exact full-trajectory duplicate crossing splits. Check initial-state duplication separately and describe any overlap; it does not establish identity of governing dynamics.

Map original fields to [pressure, density, buoyancy, velocity_x, velocity_y]. Buoyancy is not temperature; radiative data has no buoyancy and convection data has no density. Missing channels are zero, excluded from the loss and errors, and reset to zero on autoregressive feedback. Normalize each present field with the selected training trajectories' mean and population standard deviation, floored at 1e-6. No validation or test statistics enter normalization.

Area-average uniform-grid data to 128x64 in its stored axis order, retaining coordinate metadata and explicitly recording orientation. These are coarse-grid errors, not numerical-solver accuracy at the original resolution. Four context frames predict one future frame. Draw dataset, parameter setting, trajectory, stride (1 through 8), then valid start index uniformly at their respective levels. No flips are used.

GPhyT-S: 12 transformer layers, embedding 192, MLP 768, 3 heads, patches [1,16,16], absolute position embeddings, upstream finite-difference features, and no dropout. Preserve the upstream outer `differentiate(x)+x` update and last-frame output. The stencil features are not calibrated physical derivatives; the upstream forward ignores the integrator object and step_size. Do not claim RK4 integration.

Reuse FLA's reference aggregation, RMS epsilon 1e-6, learned key-normalization scales initialized to 1, and zero pseudo-queries. Full routing keeps raw sublayer outputs; Block routing sums each group of four sublayers. Every read includes the embedding once, excludes empty partial blocks, and ends with a learned final read before detokenization. Uniform Block uses the identical source schedule with fixed zero queries. Mathematically it produces a mean, not the standard additive residual sum.

BF16 autocast applies to the upstream matrix operations. Positional addition creates the original FP32 residual stream; raw sublayer outputs are promoted to that stream's dtype before block summation, matching the standard path's accumulation precision. This promotion preserves each raw BF16 value and avoids adding an unintended low-precision accumulation difference to the architecture comparison.

**Post-freeze implementation disclosure, 2026-10-25:** the original text incorrectly described FLA routing as entirely FP32. The frozen implementation calls FLA inside BF16 autocast; its `.float()` conversions do not prevent eligible contractions from being autocast. This is a deviation from the stated precision description, not a change made to ongoing training. The FP32 raw-block sums above remain unchanged. Before test evaluation, an independent architecture audit also corrected intervention recombination to use FLA's same flattened contraction, removing an unintended arithmetic difference when routing weights were unchanged. The [audit and regression evidence](../reports/architecture-audit.md) distinguish CPU measurements from pending CUDA verification. Hypotheses, checkpoints, data, seeds, update counts, and ordinary model execution are unchanged.

## Runs, optimization, and budget

Train standard and block modes at seeds 0--5; train full and uniform_block at seeds 0--2. Copy identical shared backbone initialization for paired seeds and use a separate sample RNG. Use AdamW (betas .9/.999), lr=1e-4, weight decay=1e-4, effective batch 16, BF16, gradient clipping 1, warmup 5% of updates, then cosine decay to 1e-6. Use the final fixed-update checkpoint, never the best test checkpoint.

Reserve 2 GPU-hours for profile/validation, 18 for training, 2 for evaluation, and 2 contingency; cap all GPU task wall time at 24 hours. CPU preparation and downloads do not consume that cap. Benchmark the four modes with microbatch 4, falling back jointly to 2 or 1. Set a common update count from their measured costs and the full run matrix, with a 20% timing margin and validation/checkpoint overhead. Save the frozen schedule before confirmatory training; do not recalculate it based on accuracy. Randomize standard/block order within each seed using a fixed scheduling RNG. Finish all primary pairs before controls. Keep interrupted/failing attempts and budget exclusions in the ledger.

## Primary endpoint and inference

Evaluate 12-step free-running rollouts at stride 1 from three fixed starts: floor(f*(T-4-24)) for f={0,.5,1}. The same starts support all reported horizons. NMSE for each frame/active field is spatial MSE divided by spatial mean squared standardized target plus 1e-6. Average fields, frames, and windows within each trajectory, then trajectories within each dataset. No clipping is applied.

For seed s, define D_s = (log(E_block,RBC/E_standard,RBC) + log(E_block,TRL/E_standard,TRL))/2. The primary effect is 1-exp(mean(D_s)); positive means improvement. Compute a two-sided exact paired sign-permutation test over all 2^6 assignments using SciPy and a Student-t 95% confidence interval for the mean seed log-ratio, transformed to improvement. The interval assumes approximately normal independent seed effects conditional on these fixed test datasets. Publish every seed effect. Frames and pixels are not independent replicates.

Five percent is the prespecified practically meaningful improvement. A confidence interval entirely above 5% supports that threshold; entirely below 5% rules it out at this interval's level; overlap is inconclusive. Statistical evidence of any improvement and practical importance are separate statements. For nonfinite rollouts report failures and leave affected arithmetic inference undefined; never silently discard or reseed them.

Before training, report power sensitivity at improvements {5%,10%,20%} and assumed seed-log-ratio SDs {.02,.05,.1,.2}; these are assumptions, not variance estimates. Use paired-t power and an independently seeded simulation of the exact sign-permutation test. Do not report post-hoc observed power.

## Secondary and mechanism analyses

Report one-step and 24-step errors, per-dataset and per-parameter errors, last-frame persistence, loss curves, gradient/activation magnitudes, routing entropy, speed, and peak memory. Compare radiative negative density/pressure rates and convection divergence error against coarsened ground truth with its physical coordinates; do not assume radiative energy or open-boundary mass is conserved.

For the first four test trajectories per dataset in manifest order, evaluate trained Block models with uniform routing and depth-shuffled learned weights (fixed intervention RNG). These interventions reuse each recipient's own sources and change only its routing. They establish intervention effects within those trained models, not a unique explanation for all gains. Full and Uniform trained comparisons use only their paired seeds and are exploratory; do not reuse the primary p-value for them.

## Artifacts and integrity

Persist raw trajectory metrics, full configurations, commit/hash provenance, checkpoints, budget ledger, and timestamped logs. Report positive, negative, failed, and incomplete experiments. Publish an empirical paper only after results exist; provisional text must be visibly labeled. Every numeric claim must be reproducible from saved measurements. Public releases contain no credentials or private chat transcript.

Operational details fixed before confirmatory training: an independent process supervisor enforces each GPU phase's deadline even if a work unit stops calling the cooperative budget check; hard stops remain in the ledger. The checkpoint round-trip verification uses the first completed and evaluated run in frozen execution order (normally Standard seed 0), never a model selected by accuracy. Portable data fingerprints permit exact-schedule replication on another filesystem without changing scientific inputs. These implementation clarifications do not change the hypotheses, endpoints, run matrix, or statistical analysis.
