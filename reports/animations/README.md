# Animated forecast supplement

The [README gallery](../../README.md#animated-physics-fields) embeds all six GIFs; the [public viewer](https://burnycoder.github.io/general-physics-attention-residuals-transformer/#animations) offers a selector and explicit playback/first-frame controls. These are post-result presentation additions using the previously fixed seed-zero cases, not new experiments or replacement endpoints.

| Group | Radiative density | Convection buoyancy | Included predictions |
|---|---|---|---|
| Trained | [GIF](trained_radiative_flow.gif) | [GIF](trained_convection.gif) | GPhyT without Attention Residuals, Block, Full, Uniform Block, persistence |
| Initialized | [GIF](initialized_radiative_flow.gif) | [GIF](initialized_convection.gif) | The same four architectures at zero updates, persistence |
| Interventions | [GIF](interventions_radiative_flow.gif) | [GIF](interventions_convection.gif) | Trained GPhyT baseline, original Block, uniform-routed Block, randomly shuffled Block, persistence |

Every panel follows the same first test trajectory and first predetermined window through forecast steps 1–24, with synchronized ground truth. Persistence repeats the final observed context frame. Interventions reuse the same trained Block checkpoint; shuffled routing preserves learned weight magnitudes with the existing fixed RNG seed 20261025. Initialized controls remain explicitly post-hoc. See the [evaluation protocol](../../docs/protocol.md), [initialization extension](../../docs/untrained-baselines.md), and [quantitative interpretation](../interpretation.md).

Values are in training-standardized units, so a negative displayed density does not itself imply negative physical density. Buoyancy is not temperature. Images transpose the saved `[x,y]` field into horizontal x/vertical y with nearest-neighbor display, without spatial smoothing. The shared linear color range includes all finite extrema of every panel and all 24 frames within that GIF; it never rescales between frames or models. Different GIFs have different labeled ranges. Large forecast errors can consequently make accurate panels appear almost uniform; no outliers are clipped to improve appearance. Nonfinite predictions are magenta.

The clips loop at five forecast frames per second (200 ms per frame, 4.8 seconds per cycle). This playback rate is illustrative and does not represent seconds of physical simulation. GIF palette quantization is only a display encoding; metric calculations use the original arrays. PNG companions show the actual first frame for motion-free viewing. Fixed single-trajectory examples cannot establish seed-level or population-level superiority; the full evaluation tables remain authoritative.

Rebuild from the saved `visuals/*.npz` arrays after evaluation and reporting:

```powershell
uv run --no-sync python -m gphyt_attnres_study animate
```

The normal `report` phase also generates the GIFs. Both paths use [the same animation module](../../src/gphyt_attnres_study/animations.py), shared full-target/mask/dimension validation, [Matplotlib's common normalization](https://matplotlib.org/stable/gallery/images_contours_and_fields/multi_image.html), and its [PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html). [Pillow's GIF documentation](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif) defines frame duration, looping, and palette encoding. The [manifest](manifest.json) records input/output SHA-256 hashes, frame counts, color limits, group membership, and unavailable groups. No model is rerun, no test score is recomputed, and no GPU lease is acquired.

Verification: two new tests first failed for the absent animation implementation, then passed with the existing comparison/control tests (**13 passed in 12.17 seconds** in the final run). The CLI completed on real saved arrays. All six exported GIFs decode to 24 distinct 1200×650 frames, with 200 ms duration per frame and infinite looping; their source hashes agree. A visual pass checked labels, representative first/final fields, fixed scales, and negative examples. It found slight panel movement from repeated layout calculation, corrected by fixing layout after the first draw. All 152 protected artifacts—including saved forecasts, all 49 raw evaluations, the original 21-page PDF, and the GPU ledger—remain byte-identical. The local viewer's six selections and play/first-frame controls work without console errors, and the dashboard serves the actual GIF in its Fields tab. [Machine-readable verification](../animation_verification.json).

[Release PR #23](https://github.com/BurnyCoder/general-physics-attention-residuals-transformer/pull/23) merged as `2ce9e473ac6bb0d54dd9950e04e91782eb64c453`; its [Pages workflow](https://github.com/BurnyCoder/general-physics-attention-residuals-transformer/actions/runs/40382406525) succeeded. All 16 public interface/media files match the merged Git bytes, including all six GIFs and six still posters. The live browser plays the radiative comparison, switches to the convection routing controls, and returns to a still first frame without console errors. [Deployment receipt](../public_deployment.json) · [Playback screenshot](../../assets/public-animated-physics.png).
