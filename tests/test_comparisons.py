"""Global: field comparisons must show the identical forecast case across architectures.

Local: small synthetic arrays stay in pytest's temporary directory, never public reports.
Sources: https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html.
"""
import numpy as np  # Local: construct exact known targets and distinguishable predictions.
import pytest  # Local: require explicit rejection of incompatible comparison cases.
from gphyt_attnres_study.comparisons import matched_fields  # Global: exercise the loader used by real publication figures.


def cases(folder, changed=None):
    """Local: save four tiny case files; global: all modes begin with identical truth and active fields."""
    truth = np.arange(12 * 3 * 2 * 5, dtype=float).reshape(12, 3, 2, 5)  # Local: distinct time/space/channel values expose wrong slicing.
    for index, mode in enumerate(("standard", "block", "full", "uniform_block")):
        values = dict(truth=truth.copy(), prediction=truth + index + 1, active=np.ones(5, dtype=bool))  # Local: preserve exact model order without identical predictions.
        if mode == "full" and changed is not None:
            changed(values)  # Local: one deliberate mismatch must prevent a misleading comparison.
        np.savez(folder / f"{mode}_seed0_fixture.npz", **values)  # Local: production filenames, isolated fixture contents.
    return truth  # Local: independently specify the expected frame and channel.


def test_matched_fields_preserve_common_frame_and_raw_values(tmp_path):
    """Global: no rescaling, averaging, or alternative forecast frame may enter an example."""
    truth = cases(tmp_path)  # Local: all four models forecast one fixed case.
    panels = matched_fields(tmp_path, "fixture", 2)  # Local: default frame is the primary twelve-step horizon.
    np.testing.assert_array_equal(panels, np.stack([truth[11, ..., 2] + offset for offset in range(5)]))  # Global: truth followed by four raw predictions, in defined order.


@pytest.mark.parametrize("mismatch", ("truth", "active", "prediction"))
def test_matched_fields_reject_incomparable_cases(tmp_path, mismatch):
    """Global: different targets, masks, or prediction dimensions invalidate visual comparisons."""
    def change(values):
        """Local: alter exactly one part of one model's case."""
        if mismatch == "prediction":
            values[mismatch] = values[mismatch][:-1]  # Local: a shorter forecast cannot be aligned by silent cropping.
        else:
            values[mismatch].flat[0] = 0 if mismatch == "active" else -1  # Local: change even an unshown target pixel or field mask.
    cases(tmp_path, change)  # Local: save the invalid case before invoking production code.
    with pytest.raises(ValueError, match="incompatible"):
        matched_fields(tmp_path, "fixture", 2)  # Global: reject instead of presenting different cases side by side.


def test_missing_model_prevents_partial_all_model_figure(tmp_path):
    """Global: partial results must not be presented as a complete four-model comparison."""
    cases(tmp_path)  # Local: start with an otherwise valid comparison.
    (tmp_path / "full_seed0_fixture.npz").unlink()  # Local: remove one fixture file, never a study artifact.
    assert matched_fields(tmp_path, "fixture", 2) is None  # Global: reporting can explicitly skip unavailable evidence.


def test_nonfinite_predictions_are_retained_for_visible_masking(tmp_path):
    """Global: divergence must stay visible rather than being replaced by ground truth or zero."""
    def change(values):
        """Local: insert one invalid forecast pixel at the displayed step."""
        values["prediction"][11, 0, 0, 2] = np.nan  # Local: truth stays finite and identical across models.
    cases(tmp_path, change)  # Local: finite targets with a nonfinite prediction remain comparable.
    assert np.isnan(matched_fields(tmp_path, "fixture", 2)[3, 0, 0])  # Global: the renderer must receive the original nonfinite value.
