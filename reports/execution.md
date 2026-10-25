# Confirmatory execution record

Training launched on **2026-10-25 at 17:20:58 UTC** from public commit `5ceae76f399c6504bf61e3cb1649d201e4c99056`, after [PR #5](https://github.com/BurnyCoder/general-physics-attention-residuals-transformer/pull/5) published and merged the verified data manifest and fixed schedule. The first run is Standard seed 0; the next is Block seed 0. Every planned run receives 3,175 updates with effective batch size 16, following [the frozen execution order](frozen_protocol.json).

The initial training updates have finite losses and gradient norms. They do not establish final accuracy or a comparison between architectures. Per-update measurements, optimizer checkpoints, run statuses, and the shared GPU ledger are written continuously under the configured artifact root. Complete numerical reports will be regenerated after evaluation; earlier generated reports remain timestamped snapshots rather than live progress feeds. The local dashboard reads the live artifacts.

The 18-hour training deadline and cumulative 24-hour project cap remain enforced independently of model progress. Actual sustained runtime can differ from the short profiling measurement; the 3,175-update horizon is fixed and is not adjusted after observing run behavior. Any budget-driven omissions will be retained explicitly, with primary pairs attempted before exploratory controls.
