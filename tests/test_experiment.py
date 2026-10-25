"""Global: pre-implementation contracts for training, forecasting, and inference.

Local: analytical fixtures expose leakage and state-resumption mistakes.
Sources: docs/protocol.md and https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html.
"""
import numpy as np  # Local: numerical assertions use independent expected values.
import torch  # Local: CPU fixtures need no project GPU budget.
from gphyt_attnres_study.training import masked_loss, learning_rate, save_checkpoint, restore_checkpoint  # Global: production training contracts.
from gphyt_attnres_study.evaluation import forecast, nmse_frames  # Global: forecasting accepts context only.
from gphyt_attnres_study.analysis import paired_summary  # Global: one independent unit is one seed pair.


def test_mask_and_metric():
    """Local: missing channels cannot contribute to optimization or errors."""
    target = torch.ones(2, 1, 2, 2, 5)  # Local: unit targets make the NMSE denominator explicit.
    prediction = target.clone()  # Local: initially perfect prediction.
    prediction[..., 1] = 1000  # Local: an absent density prediction is deliberately wrong.
    active = torch.tensor([[1, 0, 1, 1, 1]] * 2, dtype=torch.bool)  # Local: convection-style channel mask.
    assert masked_loss(prediction, target, active).item() == 0  # Global: absent fields are excluded.
    np.testing.assert_array_equal(nmse_frames(prediction, target, active), np.zeros((2, 1)))  # Local: reporting uses the same mask.


def test_rollout_uses_predictions_and_zeros_absent_channels():
    """Local: the function has no future-data argument; global: teacher forcing is impossible here."""
    class Increment(torch.nn.Module):
        """Local: deterministic next-frame fixture for a multi-step causal test."""
        def forward(self, x):
            """Local: add one to the last observed frame."""
            return x[:, -1:] + 1  # Local: future observations are never available.
    context = torch.zeros(1, 4, 2, 2, 5)  # Local: the only input is four past frames.
    active = torch.tensor([[1, 0, 1, 1, 1]], dtype=torch.bool)  # Local: density remains structurally absent.
    result = forecast(Increment(), context, active, 3)  # Local: three repeated prediction-feedback steps.
    np.testing.assert_array_equal(result[0, :, 0, 0, 0], [1, 2, 3])  # Global: model output, not future truth, enters the next context.
    assert result[..., 1].count_nonzero() == 0  # Local: absent outputs do not create fictitious future inputs.


def test_checkpoint_restores_optimizer_and_sample_rng(tmp_path):
    """Global: interruption cannot silently restart optimization or change paired sampling."""
    model = torch.nn.Linear(2, 1)  # Local: a small trainable CPU model exercises real Adam state.
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)  # Local: same optimizer family as the experiment.
    model(torch.ones(1, 2)).sum().backward()  # Local: materialize gradients and moments.
    optimizer.step()  # Local: checkpoint must contain nonempty optimizer state.
    rng = np.random.default_rng(4)  # Local: independent data-sampling generator.
    path = tmp_path / "resume.pt"  # Local: trusted test checkpoint stays in the fixture directory.
    save_checkpoint(path, model, optimizer, rng, 7, {"updates": 10})  # Global: save the complete continuation state.
    expected = rng.integers(100, size=20)  # Local: expected next samples from the saved state.
    with torch.no_grad():
        model.weight.add_(20)  # Local: make a restoration failure observable.
    step, metadata = restore_checkpoint(path, model, optimizer, rng, "cpu")  # Local: production resume implementation.
    assert step == 7 and metadata["updates"] == 10  # Global: preserve the fixed training horizon.
    np.testing.assert_array_equal(rng.integers(100, size=20), expected)  # Local: sample schedule resumes exactly.
    assert float(model.weight.abs().max()) < 20  # Local: model parameters were actually restored.


def test_learning_schedule_and_exact_sign_test():
    """Global: frozen optimization and seed-level inference have independently known endpoints."""
    cfg = {"warmup_fraction": .05, "learning_rate": 1e-4, "minimum_lr": 1e-6}  # Local: approved settings.
    assert learning_rate(100, 100, cfg) == 1e-6  # Local: the final update reaches the stated floor.
    assert learning_rate(5, 100, cfg) == 1e-4  # Local: five-percent warmup reaches the peak.
    report = paired_summary([-.1] * 6)  # Local: six concordant equal effects have exactly two extreme sign allocations.
    assert report["permutation_p"] == 2 / 64  # Global: repeated frames never inflate the test sample size.
    assert report["n_pairs"] == 6  # Local: the independent unit remains the initialization seed.
