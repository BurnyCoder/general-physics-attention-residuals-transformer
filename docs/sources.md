# Sources and reuse review

Reviewed 2026-10-25. The two main papers are [GPhyT](https://arxiv.org/abs/2509.13805v4) and [Attention Residuals](https://arxiv.org/abs/2603.15031v1). This study transfers an existing mechanism; no novelty priority claim is made.

| Source | Reused contribution | Assessment |
|---|---|---|
| [Official GPhyT](https://github.com/FloWsnr/General-Physics-Transformer/tree/df1527c4773ababc6f8af0d117fb5803fa3bf74b) | MIT model modules, tokenizer, differences, position encoding | Preserve backbone and outer update; published pretraining scale exceeds this study |
| [FLA reference](https://github.com/fla-org/flash-linear-attention/blob/a400aff6ec0eb81cc88721aedba782cd7d535357/fla/ops/attnres/naive.py) | MIT standalone `naive_attnres` | Selected: pure PyTorch/einops, arbitrary leading dimensions, routing weights; explicit FP32 casts do not exclude contractions from outer autocast (see the [precision audit](../reports/architecture-audit.md)) |
| [FLA tests](https://github.com/fla-org/flash-linear-attention/blob/a400aff6ec0eb81cc88721aedba782cd7d535357/tests/ops/test_attnres.py) | Forward/backward comparison methodology | Supplement with scheduling and integration tests |
| [Official AttnRes repo](https://github.com/MoonshotAI/Attention-Residuals) | Reference pseudocode and paper | No complete training package in inspected repository; equations specify sublayer routing |
| [Fast-AttnRes](https://github.com/jon123boss/fast-attnres) | Alternative optimized implementation | Current inspected API requires CUDA BF16 and Triton; avoid an extra backend in this Windows study |
| [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/attn_res.py) | Alternative optimized operator | Potential later performance comparison; not part of the experiment |
| [nktkt](https://github.com/nktkt/attention-residuals/blob/1bc66fc5e6606c7bc827384c7e7df0a378320477/attention_residuals/model.py) | Community implementation reviewed | Its model mixes residual-containing states at block boundaries; do not copy that loop as the paper's schedule |
| [The Well paper](https://arxiv.org/abs/2412.00568) and [code](https://github.com/PolymathicAI/the_well) | Official simulations, metadata, splits | Record exact registry revision and downloaded hashes |
| [Uniform convection](https://github.com/PolymathicAI/the_well/tree/master/datasets/rayleigh_benard_uniform) | Convection fields and geometry | Use uniform resampling to make area resizing and coordinate-aware diagnostics valid |
| [Radiative layer](https://github.com/PolymathicAI/the_well/tree/master/datasets/turbulent_radiative_layer_2D) | Density, pressure, velocities | Cooling and open boundaries preclude generic energy-conservation claims |

The [Well API](https://polymathic-ai.org/the_well/api/) warns of training/test duplication in shear_flow v1; that dataset is excluded. The [release notes](https://github.com/PolymathicAI/the_well/releases) explain corrected convection coordinates and the uniform-grid release. General web, GitHub, Hugging Face, discussion, and Kaggle searches did not establish an existing identical GPhyT+AttnRes study; absence in these searches is not proof of novelty.

Implementation documentation: [uv/PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/), [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [HDF5 datasets](https://docs.h5py.org/en/stable/high/dataset.html), [SciPy paired permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), [statsmodels paired-t power](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.html), [Streamlit charts](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart), [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

Reporting guidance: [Lones, ML pitfalls](https://arxiv.org/abs/2108.02497), [Lipton and Steinhardt](https://arxiv.org/abs/1807.03341), [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist). Cite original dataset simulators in the paper: [Dedalus](https://doi.org/10.1103/PhysRevResearch.2.023068) and [Fielding et al.](https://doi.org/10.3847/2041-8213/ab8d2c).
