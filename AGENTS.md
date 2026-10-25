# Working rules

Implement the approved protocol in `docs/protocol.md`. The human specification is the study's authority. Keep implementation small, modular, reusable, and documented; use upstream code rather than reinventing established components. Research changes against primary documentation and cite them.

Use `uv`, the repository-local `.venv`, and `.env`; work only in this repository and its designated storage directory under `D:\AI`. Never commit credentials. Preserve vendored sources and licenses verbatim; explanatory local/global comments belong in our adapter and orchestration code.

Write tests before new research behavior. Push source, tests, methodology, and README before model execution. Keep paired initialization and sample schedules invariant. Never tune on test data, silently omit failures, manufacture significance, or extend the GPU budget. All findings must trace to measured artifacts or cited sources.

Record commands, configurations, results, failures, timestamps, and complete external LLM prompts/responses when used. Do not store secrets or fabricate inaccessible chat transcripts. No external LLM is needed for model training.

Use functional commits and feature PRs; merge completed PRs. Keep README concise and link detailed methods, sources, reports, and the paper. Review correctness, reliability, tests, and scientific validity once before release. Update the completion checklist honestly.

The replacement human instructions from 2026-10-25 additionally require deploying an interactive results viewer to GitHub Pages. Keep its numerical content generated from the same measured artifacts as the local dashboard. Empty or incomplete results must remain visibly labeled. The approved fixed-endpoint protocol and scientific integrity govern significance claims; never force a favorable result.
