"""Global: distinguish reused initial conditions from repeated complete prediction contexts."""
import numpy as np  # Local: exact small arrays make overlap expectations analytical.
from gphyt_attnres_study.data_audit import audit_records  # Global: exercise the public audit without real-study artifacts.


def test_initial_groups_and_context_overlap_are_distinct(tmp_path):
    """Local: three split witnesses share one initial state; global: report groups rather than transition counts."""
    records = []  # Local: all records have the same original first-frame digest.
    for index, split in enumerate(("train", "valid", "test")):
        array = np.zeros((5, 2, 2, 5), dtype=np.float32)  # Local: a canonical toy trajectory with a shared initial frame.
        array[1:] = index  # Local: later context frames differ across all three splits.
        path = tmp_path / f"{split}.npy"  # Local: isolated files prevent any contact with study data.
        np.save(path, array)  # Local: exercise the production memory-mapped context hashing.
        records.append(dict(dataset="fixture", split=split, parameter=str(index), trajectory=0, initial_sha256="same-initial", path=str(path)))  # Local: parameter settings differ across splits.
    result = audit_records(records)  # Global: group identity does not depend on traversal transitions.
    assert len(result["initial_groups"]) == 1 and result["same_parameter_initial_groups"] == 0  # Local: one shared initial state, no within-parameter overlap.
    assert result["first_context_groups"] == []  # Global: shared initial states need not imply identical four-frame contexts.
    np.save(records[-1]["path"], np.load(records[0]["path"]))  # Local: create a true complete-context duplicate for the second assertion.
    assert len(audit_records(records)["first_context_groups"]) == 1  # Global: the audit detects identical prepared inputs when they exist.
