# Independent architecture audit and intervention correction

Date: 2026-10-25. Audited source: `d43e6479e8a9b9174eeeb536a22ef863b7862df0`. This audit was requested during confirmatory training and did not use earlier audit conclusions or held-out prediction errors. Its numerical probes are implementation checks, not physics-accuracy results.

## Architecture findings

The comparison used the [GPhyT paper, version 4](https://arxiv.org/abs/2509.13805v4), its [pinned implementation](https://github.com/FloWsnr/General-Physics-Transformer/tree/df1527c4773ababc6f8af0d117fb5803fa3bf74b), [Attention Residuals, equations 2--6](https://arxiv.org/abs/2603.15031v1), the [official AttnRes pseudocode](https://github.com/MoonshotAI/Attention-Residuals), and the [pinned FLA operator](https://github.com/fla-org/flash-linear-attention/blob/a400aff6ec0eb81cc88721aedba782cd7d535357/fla/ops/attnres/naive.py). Freshly fetched pinned vendor sources matched the local bytes. No architecture-breaking mismatch was found.

| Component | Independently checked behavior |
|---|---|
| GPhyT backbone | Original patch projections, absolute positional embeddings, difference features, pre-LayerNorm attention/MLP, and final `differentiate(x) + x` last-frame update are retained. |
| Standard residual path | Initial parameters and CPU forward output matched a directly constructed pinned upstream model exactly. |
| Depth attention | Scores use learned pseudo-queries against RMS-normalized keys, softmax over sources, and unnormalized source values. Queries start at zero and key scales at one. |
| Full / Block source schedule | Full retains individual raw outputs. Block sums groups of four sublayers, includes the embedding once, clears completed partial sums, and includes a separate final read. Twenty-four writers form six blocks; the final Block read has seven sources. |
| State and pairing | Source history resets between calls; Standard, Block, Full, and Uniform shared backbone initialization matched for the checked seed. |
| Actual saved models | Completed Standard and Block seed-0 safetensors loaded into the intended architectures. All parameters were finite. Block had nonzero queries at its 24 nontrivial read sites; the first read has only one source. |

A separate CPU reference reconstructed banks from the complete list of raw outputs and grouped slices, rather than copying the incremental production loop. It used explicit RMS normalization, dot products, softmax, and weighted sums. Cases covered Full, four-sublayer Block, a partial final block, one-sublayer blocks, Uniform Block, and twelve-layer depth. Across these probes, maximum output and parameter-gradient differences were respectively `3.58e-7` and `4.53e-6`; forward-call reset was exact. A separate reconstruction of patch projections, difference features, depth composition, detokenization, and the outer update on the completed checkpoints differed by at most `2.27e-6` on the tested CPU input. These are finite numerical checks, not a proof over all inputs.

The upstream model's `step_size` and stored integrator object do not enter its forward update. Its difference stencil is a feature, not a calibrated derivative. These upstream behaviors are preserved and already bounded by the protocol; the study does not claim RK4 integration or reproduction of the original large-scale pretraining.

## Precision discrepancy and correction

The original protocol and paper incorrectly described FLA routing as entirely FP32. The pinned operator casts values and keys to FP32, but its contractions remain inside the caller's BF16 autocast context. Explicit tensor casts do not disable autocast for eligible matrix operations; see [PyTorch 2.14's operation rules](https://docs.pytorch.org/docs/2.14/amp.html#autocast-op-reference). A CPU dispatch trace observed BF16 contractions. CPU and CUDA autocast rules differ for some other operations, so CPU discrepancies are not reported as measured CUDA errors.

The preexisting intervention branch used a pointwise FP32 weighted sum instead of FLA's flattened contraction. Consequently, even when an intervention left all weights unchanged, it could change outputs through arithmetic alone. A zero-query uniform intervention and an identity shuffle of nonuniform weights reproduced this problem.

The correction uses the same flattened `einsum` contraction, source layout, autocast context, output shape, and final cast as ordinary FLA routing. It changes only diagnostic recombination. Vendored code, the normal training/read branch, block accumulation, initialization, sample schedules, optimizer, frozen update count, and checkpoints are unchanged. The active training worker was neither stopped nor restarted. No test evaluation artifacts existed when this correction was made.

This is a disclosed post-freeze implementation correction and a correction to the precision description. Training continues with the original mixed-precision implementation; switching it to FP32 routing midway would create a different experiment. No favorable result or significance claim follows from this audit.

## Regression evidence

The permanent regressions are in [test_intervention_precision.py](../tests/test_intervention_precision.py). They cover embedding-only and multi-source reads, uniform and shuffled routing, a nonuniform identity permutation, and exact ordinary-path output/gradient agreement with the unchanged FLA operator. Both CPU FP32 and CPU BF16 autocast are exercised.

| UTC start | Check | Observed result |
|---|---|---|
| 2026-10-25 21:11:56 | New tests before the correction | 5 failed, 11 passed; all failures were BF16 identity-intervention cases. |
| 2026-10-25 21:12:39 | Same tests after the correction | 16 passed in 3.19 seconds. |
| 2026-10-25 21:13:55 | Complete CPU suite | 42 passed in 40.51 seconds. |
| 2026-10-25 21:22:42 | Complete CPU suite after the reporting correction below | 43 passed in 19.09 seconds. |

The largest failing regression discrepancy was `0.005163431` on a synthetic nonuniform identity-shuffle probe. The corrected tests require bitwise equality rather than relaxing tolerances. Full-suite download warnings and its analysis-completion message come from isolated fixtures; they do not report a real download failure or completed scientific analysis.

Reproduce the targeted and complete checks with:

```powershell
uv run --no-sync pytest -q tests/test_intervention_precision.py
uv run --no-sync pytest -q
```

Complete timestamped local output is retained under the configured artifact root's `logs/` directory as `20261025T211156Z_audit_intervention_red_40620.log`, `20261025T211239Z_audit_intervention_green_13136.log`, and `20261025T211355Z_audit_intervention_suite_1288.log`. The logged invocations set one CPU thread and call `pytest.main` with `-q --tb=short`; no GPU kernels were launched for these checks.

## Report refresh

Rendering the updated manuscript exposed a stale pretraining run count, so the existing `report` phase regenerated the provisional paper, individual reports, learning curves, and public viewer from saved run artifacts. The populated cost table then exposed a missing-value bug: an active run without recorded peak memory displayed `0.00 MiB`. A new failing fixture reproduced this; missing memory now remains undefined in the table and CSV. The final 43-test suite includes that regression, with complete output in `20261025T212242Z_audit_final_suite_47576.log`. Existing LaTeX `na_rep` handling and JSON null conversion retain this missingness, following the [pandas table-export contract](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_latex.html). These are reporting corrections; no checkpoint or measured training record was rewritten.

The report phase completed at 2026-10-25 21:23:45 UTC with six completed runs in its saved snapshot and undefined primary inference. The resulting ten-page provisional PDF was rendered and visually inspected; its missing peak memory is labeled undefined. The local Pages preview displayed actual learning curves, and switching to `block_seed3` and `Seconds per update` changed both the chart and run status as expected. Held-out error and intervention panels remained visibly pending. The publication is a timestamped snapshot, not a live progress claim.

## CUDA verification after training

On 2026-10-26 at 04:35 UTC, after all 18 training runs finished and released the GPU, the same 16 regression cases passed on the RTX 5070 Laptop GPU with PyTorch 2.14.1+cu130. The checks required bitwise equality for unchanged uniform weights, identity shuffles of nonuniform learned weights, and ordinary FLA outputs and gradients, under both FP32 and CUDA BF16 autocast. The original profiling/validation allocation was charged 1.2083 seconds through the existing Budget ledger and independent supervisor; the allowance was not reset. These finite probes verify the tested arithmetic identities, not equality over every possible activation.

The [machine-readable evidence](cuda_intervention_verification.json) records each case and the published source revision. Reproduce with `uv run --no-sync python scripts/verify_cuda_interventions.py` when no other GPU phase is active. Ordinary pytest remains CPU-only. Complete output is in `20261026T043523Z_verify_cuda_interventions_2040.log` under the artifact root's logs directory. Held-out evaluation started afterward. The final paper and model cards retain the mixed-precision disclosure; no seeds, training settings, or primary endpoints changed.
