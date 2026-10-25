"""Global: add depth routing while preserving the pinned GPhyT computation.

Local: FLA implements the weighted read; this module owns only the source schedule.
Sources: docs/sources.md and https://arxiv.org/abs/2603.15031, equations 2--6.
"""
import math  # Local: serialize undefined diagnostic scalars explicitly without changing model computation.
import torch  # Local: use native modules and automatic differentiation.
from torch import nn  # Local: routing parameters participate in ordinary state_dict checkpoints.
from . import runtime  # Local: register the project-owned vendor directory before upstream imports.
from fla_attnres import naive_attnres  # Global: reuse the pinned MIT reference operator verbatim.
from gphyt.models.transformer.model import PhysicsTransformer  # Global: preserve upstream forward and tokenization.


class DepthRead(nn.Module):
    """Global: one content-dependent depth read; local: no sequence-attention replacement."""
    def __init__(self, width, uniform=False):
        """Local: zero queries and unit key scales implement paper initialization."""
        super().__init__()  # Local: register parameters with PyTorch.
        self.query = nn.Parameter(torch.zeros(width), requires_grad=not uniform)  # Global: zero means a mean, not an additive sum.
        self.rms_weight = nn.Parameter(torch.ones(width), requires_grad=not uniform)  # Local: learned normalization only for learned routing.
        self.weights = None  # Local: diagnostic weights are retained only when explicitly requested.

    def forward(self, sources, intervention=None, capture=False, generator=None):
        """Local: apply FLA or a routing intervention; global: never change source values."""
        sources = [x.contiguous() for x in sources]  # Local: satisfy FLA's view contract for rearranged GPhyT tensors.
        result, weights = naive_attnres(self.query, sources, self.rms_weight, rms_eps=1e-6, return_weights=True)  # Global: verified upstream equation and gradients.
        if intervention == "uniform":  # Local: remove content-dependent selection without changing the source bank.
            weights = torch.full_like(weights, 1 / len(sources))  # Local: preserve the normalized aggregation scale.
        elif intervention == "shuffle":  # Local: preserve weight magnitudes but disrupt source-depth assignments.
            order = torch.randperm(len(sources), generator=generator, device="cpu").to(weights.device)  # Local: fixed CPU RNG gives reproducible permutations.
            weights = weights.index_select(0, order)  # Local: permute source labels, not examples or values.
        if intervention is not None:  # Local: only diagnostic forwards replace the reference result.
            result = (weights[..., None] * torch.stack(sources).float()).sum(0).to(sources[0].dtype)  # Global: intervention changes only routing.
        self.weights = weights.detach() if capture else None  # Local: avoid keeping autograd histories in diagnostics.
        return result  # Global: pass the mixture to the unchanged sublayer normalization.


class ResidualStack(nn.Module):
    """Global: causal-in-depth Full/Block schedules over the original sublayer modules."""
    def __init__(self, blocks, width, mode, block_size=4):
        """Local: reuse each upstream block; global: shared backbone names remain stable."""
        super().__init__()  # Local: register children normally.
        if mode not in {"standard", "block", "full", "uniform_block"} or block_size < 1:
            raise ValueError("Invalid residual mode or block size")  # Local: fail on ambiguous architectures.
        for index, block in enumerate(blocks):  # Local: numeric child names preserve upstream state_dict keys.
            self.add_module(str(index), block)  # Global: copying initialization never depends on routing RNG.
        self.count, self.mode, self.block_size = len(blocks), mode, block_size  # Local: distinguish transformer layers and sublayers.
        self.reads = nn.ModuleList([DepthRead(width, mode == "uniform_block") for _ in range(2 * self.count + 1)]) if mode != "standard" else nn.ModuleList()  # Local: include the final output read.
        self.capture, self.intervention, self.trace = False, None, []  # Local: diagnostics are opt-in and reset for every call.
        self.generator = torch.Generator().manual_seed(20261025)  # Local: a separate RNG prevents intervention effects on training samples.

    def forward(self, x):
        """Local: raw writes feed depth reads; global: history never crosses forecast calls."""
        if self.mode == "standard":  # Local: preserve the upstream two-addition implementation exactly.
            for index in range(self.count):
                x = self._modules[str(index)](x)  # Local: call the original AttentionBlock.forward unchanged.
            return x  # Global: parity tests protect the baseline path.
        completed, partial, self.trace = [x], None, []  # Local: include the embedding once and no fictitious zero source.
        for site in range(2 * self.count):  # Global: attention and MLP are separate depth-writing sublayers.
            sources = completed + ([partial] if partial is not None else [])  # Local: read completed blocks and the current partial sum.
            h = self.reads[site](sources, self.intervention, self.capture, self.generator)  # Local: aggregate before executing the sublayer.
            self._record(site, sources, h)  # Local: no-op unless diagnostics were explicitly requested.
            block = self._modules[str(site // 2)]  # Local: retrieve the original transformer layer.
            output = block.sd(block.attention(block.norm1(h))) if site % 2 == 0 else block.sd(block.mlp(block.norm2(h)))  # Global: retain upstream transformations without their old skip additions.
            if self.mode == "full":  # Local: Full retains every raw writer output independently.
                completed.append(output)  # Global: never store an already-aggregated residual state.
            else:
                partial = output if partial is None else partial + output  # Local: ordinary addition accumulates within a block.
                if (site + 1) % self.block_size == 0:  # Local: boundary accounting is explicitly in sublayers.
                    completed.append(partial)  # Local: completed representation is a sum, not an average.
                    partial = None  # Global: do not count the same block twice at the next read.
        sources = completed + ([partial] if partial is not None else [])  # Local: include an incomplete final group if necessary.
        result = self.reads[-1](sources, self.intervention, self.capture, self.generator)  # Global: the last writer must influence the output.
        self._record(2 * self.count, sources, result)  # Local: output routing appears in diagnostic heatmaps.
        return result  # Local: upstream detokenization and outer next-state update remain unchanged.

    def _record(self, site, sources, output):
        """Local: summarize weights; global: avoid retaining large activation graphs."""
        if self.capture:  # Local: synchronizing diagnostics never slow normal training.
            weights = self.reads[site].weights  # Local: detached weights were retained by the read.
            means = weights.float().reshape(len(sources), -1).mean(1).cpu().tolist()  # Local: retain token-mean depth weights without autograd history.
            entropy = float(-(weights * weights.clamp_min(1e-30).log()).sum(0).mean())  # Local: entropy measures actual per-token routing before averaging tokens.
            rms = float(output.detach().float().square().mean().sqrt())  # Local: activation magnitude is a descriptive stability diagnostic.
            self.trace.append(dict(site=site, sources=len(sources), weights=[value if math.isfinite(value) else None for value in means], entropy=entropy if math.isfinite(entropy) else None, rms=rms if math.isfinite(rms) else None))  # Global: divergent diagnostics remain JSON null instead of breaking failure reporting or inventing finite values.


def make_model(cfg, mode, seed):
    """Local: construct the same backbone for a paired seed; global: isolate routing changes."""
    with torch.random.fork_rng(devices=[]):  # Local: building a model cannot change the sample RNG.
        torch.manual_seed(seed)  # Local: every mode receives identical original parameters.
        model = PhysicsTransformer(num_fields=5, hidden_dim=cfg["hidden_dim"], mlp_dim=cfg["mlp_dim"], num_heads=cfg["heads"], num_layers=cfg["layers"], patch_size=tuple(cfg["patch_size"]), img_size=(cfg["input_steps"], *cfg["resolution"]), use_derivatives=True, pos_enc_mode="absolute", tokenizer_mode="linear", detokenizer_mode="linear")  # Global: use the pinned GPhyT model rather than a look-alike.
        if mode != "standard":  # Local: baseline stays completely untouched for reference parity.
            model.attention_blocks = ResidualStack(model.attention_blocks, cfg["hidden_dim"], mode, cfg["block_size_sublayers"])  # Local: replace only the depth-composition layer.
    return model  # Global: model.forward keeps its upstream public signature.
