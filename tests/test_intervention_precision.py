"""Global: routing interventions must not introduce an arithmetic-precision intervention.

Local: CPU autocast exposes the same contraction eligibility without spending GPU time.
Sources: https://docs.pytorch.org/docs/2.14/amp.html#autocast-op-reference and vendor/fla_attnres.py.
"""
import pytest  # Local: cover normal FP32 execution and BF16 autocast with the same assertions.
import torch  # Local: use actual autocast kernels and gradients, not simulated output dtypes.
from gphyt_attnres_study.model import DepthRead  # Global: exercise the production intervention entry point.
from fla_attnres import naive_attnres  # Local: the unchanged vendored operator defines ordinary execution.


@pytest.mark.parametrize("amp", [False, True])  # Local: a matching formula must also preserve autocast behavior.
@pytest.mark.parametrize("count", [1, 5, 7])  # Local: include the embedding-only read and the final seven-source Block read.
@pytest.mark.parametrize("intervention", ["uniform", "shuffle"])  # Global: both interventions must be identities when weights are unchanged.
def test_unchanged_uniform_weights_preserve_output(amp, count, intervention):
    """Local: zero queries make both interventions weight-preserving; global: isolate numerical confounding."""
    generator = torch.Generator().manual_seed(9137)  # Local: fixed probes make failures reproducible without touching sample RNGs.
    reader = DepthRead(12)  # Local: initialization supplies exact zero pseudo-queries and unit key scales.
    sources = [torch.randn(1, 4, 2, 2, 12, generator=generator) for _ in range(count)]  # Local: retain GPhyT's five-dimensional activation shape.
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        ordinary = reader(sources, capture=True)  # Local: capture the unmodified operator's output and source weights.
        weights = reader.weights.clone()  # Local: a later diagnostic call replaces the stored weights.
        changed = reader(sources, intervention=intervention, capture=True, generator=generator)  # Local: intervene through the actual production branch.
    torch.testing.assert_close(reader.weights, weights, rtol=0, atol=0)  # Local: establish that routing itself did not change.
    torch.testing.assert_close(changed, ordinary, rtol=0, atol=0)  # Global: any remaining difference would be an arithmetic confound.


@pytest.mark.parametrize("amp", [False, True])  # Local: nonuniform learned weights also need precision-matched recombination.
def test_identity_shuffle_preserves_learned_routing(amp, monkeypatch):
    """Local: force an identity depth permutation; global: test a nonuniform routing counterfactual."""
    generator = torch.Generator().manual_seed(107)  # Local: make source values and learned routing probes repeatable.
    reader = DepthRead(12)  # Local: use the same module as trained checkpoints.
    with torch.no_grad():
        reader.query.copy_(torch.randn(12, generator=generator) * .4)  # Local: leave the uniform initialization regime.
    sources = [torch.randn(1, 4, 2, 2, 12, generator=generator) for _ in range(7)]  # Local: exercise the complete Block source bank.
    monkeypatch.setattr(torch, "randperm", lambda count, **kwargs: torch.arange(count, device=kwargs.get("device")))  # Local: preserve source labels while executing the shuffle branch.
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        ordinary = reader(sources)  # Local: reference output uses learned content-dependent weights.
        changed = reader(sources, intervention="shuffle", generator=generator)  # Local: recombination must leave this identity permutation unchanged.
    torch.testing.assert_close(changed, ordinary, rtol=0, atol=0)  # Global: precision cannot masquerade as a routing effect.


@pytest.mark.parametrize("amp", [False, True])  # Local: protect both tested arithmetic regimes from training-path drift.
def test_ordinary_routing_and_gradients_remain_exactly_fla(amp):
    """Local: compare the adapter directly with pinned FLA; global: intervention repairs cannot alter training."""
    generator = torch.Generator().manual_seed(25)  # Local: deterministic nonzero-query gradient probes.
    reader = DepthRead(12)  # Local: use production parameters and normalization defaults.
    with torch.no_grad():
        reader.query.copy_(torch.randn(12, generator=generator) * .4)  # Local: gradients must cover content-dependent routing.
    sources = [torch.randn(1, 4, 2, 2, 12, generator=generator, requires_grad=True) for _ in range(5)]  # Local: test gradients into every source as well as routing parameters.
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        actual = reader(sources)  # Global: ordinary forwards use no intervention.
        expected = naive_attnres(reader.query, sources, reader.rms_weight, rms_eps=1e-6)  # Local: call the unchanged upstream implementation independently.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)  # Global: the frozen forward computation remains bitwise identical.
    variables = sources + [reader.query, reader.rms_weight]  # Local: cover data gradients and both learned routing vectors.
    for got, want in zip(torch.autograd.grad(actual.sum(), variables), torch.autograd.grad(expected.sum(), variables)):
        torch.testing.assert_close(got, want, rtol=0, atol=0)  # Global: the frozen backward computation also remains bitwise identical.
