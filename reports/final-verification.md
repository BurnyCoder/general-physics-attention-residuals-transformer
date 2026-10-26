# Final verification

Checked on 2026-10-26 against saved measurements. This is a focused release check, not external peer review or a new experiment. The original [implementation review](../docs/review.md) and requested [architecture audit](architecture-audit.md) remain separate historical records.

| Deliverable | Observed evidence |
|---|---|
| Fixed training matrix | All 18 runs complete at 3,175 updates; checkpoint and complete per-update log hashes in [training_completion.json](training_completion.json). No failures, added seeds, or budget omissions. |
| Evaluation coverage | 18 ordinary models, 12 routing interventions, and persistence: 31 raw JSON files in [evaluation](evaluation), with finite forecasts through 24 steps. Physical and baseline failures remain visible. |
| Numerical inference | [Independent CPU reconstruction](final_metric_audit.json) reads the raw endpoints, reproduces all six effects and the transformed confidence interval, and explicitly enumerates all 64 sign assignments. Four fixed-case field decompositions reproduce their saved window errors. Those decompositions are labeled post hoc. |
| Tests | 44 CPU tests passed in 21.77 seconds after a failing CLI stream regression; [development record](development.md). All 16 relevant CUDA parity/intervention checks passed before test evaluation under the existing allowance; [CUDA evidence](cuda_intervention_verification.json). |
| Paper | The final 15-page PDF was compiled at 05:44 UTC. Every page was rendered with Poppler and visually inspected: equations, tables, figures, references, appendices, and the requested author statement are readable without clipping. The final compiler log contains no overfull/underfull boxes or undefined references. |
| Figures and videos | Seven paper figures were inspected in context. All four fixed Standard/Block seed-zero videos decode to 24 frames at 6 fps; first and last frames were visually compared. Left is truth, right is prediction, on the same target-derived color scale. |
| Local dashboard | All four tabs were exercised. Radiative dataset filtering at horizon 24 shows Block seed-zero NMSE 0.4907376865 and Standard seed-zero 0.6872162680; these match saved metrics. Density selection, final-frame slider, Full routing selection, and power view work. No browser console errors were recorded. |
| Public-view local preview | Measured 18/18 status, primary effect/CI/p, and 11.746/24 recorded hours are populated. Dataset/horizon, run/cost curve, Full routing, radiative rollout, and frame 1/24 controls update their charts. No browser console errors were recorded. Deployment verification is recorded below after the release merge. |
| Model publication | All four public variant repositories belong to the [Hugging Face collection](https://huggingface.co/collections/BurnyCoder/gphyt-attention-residuals-6adee5fe6ceb5f9ae1ccb873). The [publication receipt](publication.json) records the actual downloaded checkpoint hash and reproduced three-horizon errors on fixed trajectories from both datasets. |
| Budget and provenance | [Ledger](gpu_budget.json): 42,285.415381 seconds, or 11.745949 hours, with original phase allocations intact. A recovered publication-only parsing failure remains logged; it did not restart training/evaluation or duplicate the collection. |

The paper, README, reports, dashboard, viewer, and model cards explicitly distinguish the relative paired improvement from poor absolute convection forecasts, negative radiative densities, three-seed exploratory controls, and reused initial conditions. They do not claim physical reliability, a general advantage over fixed averaging, or unseen-initial-condition generalization. Numerical claims trace to the [interpretation and source tables](interpretation.md).

The complete user workflow has now run on real data: prepare, power, verify, profile/freeze, train, evaluate, analyze, report, dashboard, and publish. No external LLM is called by that pipeline. Full process logs remain in the designated artifact directory; unavailable chat transcripts are not fabricated. Credentials, raw datasets, environments, optimizer pickle checkpoints, and machine-local paths are excluded from the public viewer and model staging directories.

## Public deployment

Final release merge and remote artifact/UI checks are pending at this revision. They must pass before the completion checklist is closed.
