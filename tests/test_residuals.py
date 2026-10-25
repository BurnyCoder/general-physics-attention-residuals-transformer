"""Global: specification-first tests for the new residual schedule; CPU only.

Local: hand-computable sources catch semantic errors that shape checks miss.
Source: https://arxiv.org/abs/2603.15031, equations 2--6.
"""
import torch  # Local: test tensors and autograd independently of GPU profiling.
from gphyt_attnres_study.model import DepthRead, ResidualStack, make_model  # Global: the public adapter surface.
from gphyt_attnres_study.runtime import config  # Local: use the published model specification.


def test_zero_query_mean_and_gradients():
    """Local: zero queries average values; global: initialization must not claim a sum."""
    read = DepthRead(3)  # Local: a tiny independently testable routing operator.
    a = torch.randn(2, 2, 3, requires_grad=True)  # Local: distinct sources make their derivatives observable.
    b = torch.randn(2, 2, 3, requires_grad=True)  # Local: require gradients through every source.
    actual = read([a, b])  # Local: invoke the production FLA-backed path.
    torch.testing.assert_close(actual, (a + b) / 2)  # Global: exact mathematical initialization contract.
    actual.sum().backward()  # Local: test source and query autograd.
    torch.testing.assert_close(a.grad, torch.full_like(a, .5))  # Local: neither source may be detached.
    assert read.query.grad is not None and torch.isfinite(read.query.grad).all()  # Global: routing is trainable.


def test_block_accumulation_matches_the_upstream_residual_dtype():
    """Global: BF16 sublayer outputs must not lower the FP32 upstream residual-stream accumulation precision."""
    class BF16Writer(torch.nn.Module):
        """Local: expose the quantization-sensitive addition independently of attention kernels."""
        def __init__(self, value):
            """Local: each writer emits an exactly representable BF16 scalar."""
            super().__init__()  # Local: register a standard callable module.
            self.value = value  # Local: raw outputs are fixed analytical values.
        def forward(self, x):
            """Local: imitate AMP's BF16 matrix output with an FP32 residual input."""
            return torch.full_like(x, self.value, dtype=torch.bfloat16)  # Local: no hidden residual addition occurs inside this writer.
    blocks = []  # Local: four sublayers make one complete Block sum.
    for attention, mlp in [(1., 1 / 256), (1 / 256, 1 / 256)]:
        block = torch.nn.Module()  # Local: match the original block's public sublayer interface.
        block.norm1 = block.norm2 = block.sd = torch.nn.Identity()  # Local: isolate accumulation precision.
        block.attention, block.mlp = BF16Writer(attention), BF16Writer(mlp)  # Local: low-precision addition would round each small increment away.
        blocks.append(block)  # Local: retain both transformer layers in their original order.
    result = ResidualStack(blocks, 2, "block", 4)(torch.zeros(1, 1, 2))  # Local: FP32 embedding plus raw BF16 writers, as in upstream AMP positional addition.
    torch.testing.assert_close(result, torch.full_like(result, (1 + 3 / 256) / 2), rtol=0, atol=0)  # Global: the final read averages one embedding and a sum accumulated at the original residual dtype.


def test_routing_matches_independent_equation():
    """Local: compare outputs and gradients with explicit tensor operations."""
    read = DepthRead(4)  # Local: production normalization has a learned channel scale.
    with torch.no_grad():  # Local: deterministic nonzero routing probes the nonlinear case.
        read.query.copy_(torch.tensor([.2, -.4, .1, .3]))  # Local: avoid a test restricted to uniform routing.
    sources = [torch.randn(2, 3, 4, requires_grad=True) for _ in range(3)]  # Local: three independent depth sources.
    values = torch.stack(sources)  # Local: reference equation operates on an explicit depth axis.
    keys = values / (values.square().mean(-1, keepdim=True) + 1e-6).sqrt() * read.rms_weight  # Local: RMS-normalized keys, original values.
    weights = (keys * read.query).sum(-1).softmax(0)  # Local: softmax is over depth, never tokens.
    expected = (weights[..., None] * values).sum(0)  # Local: independent aggregation expression.
    actual = read(sources)  # Global: validate reused code through our adapter.
    torch.testing.assert_close(actual, expected)  # Local: outputs must agree within FP32 tolerance.
    variables = sources + [read.query, read.rms_weight]  # Local: validate parameter gradients as well as input gradients.
    for got, want in zip(torch.autograd.grad(actual.sum(), variables, retain_graph=True), torch.autograd.grad(expected.sum(), variables)):
        torch.testing.assert_close(got, want)  # Global: matching outputs alone cannot validate training.


def test_raw_writer_values_embedding_once_and_incomplete_final_block():
    """Local: hand-computable constant writers expose duplicate embeddings or cumulative-state writes."""
    class Writer(torch.nn.Module):
        """Local: return a known raw sublayer output independent of its read."""
        def __init__(self, value):
            """Local: retain one scalar for an analytical source bank."""
            super().__init__()  # Local: use the normal PyTorch module protocol.
            self.value = value  # Local: each sublayer has a distinct known output.
        def forward(self, x):
            """Local: constant raw output preserves only shape and dtype."""
            return torch.full_like(x, self.value)  # Local: no hidden residual state enters the writer.
    blocks = []  # Local: three layers yield a complete four-writer block and a partial two-writer block.
    for layer in range(3):
        block = torch.nn.Module()  # Local: expose the same named sublayer interface as upstream GPhyT.
        block.norm1 = block.norm2 = block.sd = torch.nn.Identity()  # Local: isolate depth composition from normalization.
        block.attention, block.mlp = Writer(2 * layer + 1), Writer(2 * layer + 2)  # Local: raw outputs are exactly 1 through 6.
        blocks.append(block)  # Local: preserve the stated writer order.
    for mode, divisor in [("block", 3), ("uniform_block", 3), ("full", 7)]:
        embedding = torch.full((1, 2, 3), 10., requires_grad=True)  # Local: a distinguishable nonzero embedding catches duplicate inclusion.
        result = ResidualStack(blocks, 3, mode, 4)(embedding)  # Local: final read must include every writer exactly once at block granularity.
        torch.testing.assert_close(result, torch.full_like(result, 31 / divisor))  # Global: Block has [10,10,11]; Full has [10,1,2,3,4,5,6].
        result.sum().backward()  # Local: ensure the embedding contributes through the final aggregation.
        torch.testing.assert_close(embedding.grad, torch.full_like(embedding, 1 / divisor))  # Global: source inclusion has the expected analytical derivative.


def test_standard_parity_and_shared_initialization():
    """Global: intervention changes routing only; local: compare the reference backbone."""
    cfg = dict(config(), layers=2, hidden_dim=12, mlp_dim=24, heads=3, resolution=[16, 16])  # Local: cheap CPU fixture.
    standard = make_model(cfg, "standard", 4)  # Local: fixed seed defines the shared backbone.
    block = make_model(cfg, "block", 4)  # Local: routing creation must not perturb backbone initialization.
    for name, tensor in standard.state_dict().items():  # Global: every original tensor must be shared exactly.
        torch.testing.assert_close(tensor, block.state_dict()[name], rtol=0, atol=0)  # Local: detect RNG ordering mistakes.
    x = torch.randn(1, 4, 16, 16, 5)  # Local: a valid upstream five-channel input.
    expected = standard(x)  # Local: upstream path is the standard reference.
    stack = standard.attention_blocks  # Local: save original modules before wrapping them.
    standard.attention_blocks = ResidualStack(stack, 12, "standard", 4)  # Local: exercise the identity adapter path.
    torch.testing.assert_close(standard(x), expected, rtol=0, atol=0)  # Global: no baseline drift.
    a, b = block(x), block(x)  # Local: history must be per-forward rather than stored across calls.
    torch.testing.assert_close(a, b, rtol=0, atol=0)  # Global: independent forecasts cannot contaminate one another.


def test_block_schedule_and_final_read():
    """Local: trace source counts; global: catch embedding duplication and missing final reads."""
    cfg = dict(config(), layers=4, hidden_dim=12, mlp_dim=24, heads=3, resolution=[16, 16])  # Local: two complete groups.
    model = make_model(cfg, "block", 0)  # Local: four sublayers per routing block.
    stack = model.attention_blocks  # Local: inspect the public diagnostic trace only.
    stack.capture = True  # Local: opt into trace collection outside training.
    model(torch.randn(1, 4, 16, 16, 5))  # Local: one valid input triggers every routing site.
    assert [entry["sources"] for entry in stack.trace] == [1, 2, 2, 2, 2, 3, 3, 3, 3]  # Global: eight sublayers plus final read.
    full = make_model(cfg, "full", 0)  # Local: each completed sublayer becomes its own source.
    full.attention_blocks.capture = True  # Local: retain only small diagnostics.
    full(torch.randn(1, 4, 16, 16, 5))  # Local: run a separate history.
    assert [entry["sources"] for entry in full.attention_blocks.trace] == list(range(1, 10))  # Global: final read includes the last output.
