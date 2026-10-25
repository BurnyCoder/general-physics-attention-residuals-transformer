"""Global: exact study replication must survive path changes without accepting changed science.

Local: content fingerprints ignore only machine locations and preparation timestamps.
Source: https://docs.python.org/3/library/hashlib.html and docs/protocol.md.
"""
import copy  # Local: independent fixtures expose accidental mutation of original metadata.
from gphyt_attnres_study.provenance import data_fingerprint  # Global: one identity contract protects freezing, training, and evaluation.


def test_data_identity_is_portable_but_sensitive_to_split_and_statistics():
    """Global: moving the project is harmless; changing normalization or split membership is not."""
    prepared = dict(created="first-time", resolution=[128, 64], files=[dict(path="D:/first/raw.hdf5", sha256="raw", bytes=12)], records=[dict(path="D:/first/array.npy", split="train", sha256="trajectory")], normalization={"field": {"mean": [1.]}}, initial_state_overlaps=[])  # Local: minimal representative provenance, independent of any test outcome.
    moved = copy.deepcopy(prepared)  # Local: simulate an otherwise identical clone on another filesystem.
    moved["created"], moved["files"][0]["path"], moved["records"][0]["path"] = "later-time", "/clone/raw.hdf5", "/clone/array.npy"  # Local: only incidental host metadata changes.
    assert data_fingerprint(prepared) == data_fingerprint(moved)  # Global: exact data identity is portable across hosts.
    moved["normalization"]["field"]["mean"][0] = 2.  # Local: changing train-only statistics changes the model's input contract.
    assert data_fingerprint(prepared) != data_fingerprint(moved)  # Global: a changed normalization cannot silently continue the frozen study.
    moved = copy.deepcopy(prepared)  # Local: isolate the split-membership mutation.
    moved["records"][0]["split"] = "test"  # Local: identical bytes assigned to another split are scientifically different.
    assert data_fingerprint(prepared) != data_fingerprint(moved)  # Global: data leakage cannot hide behind an unchanged file checksum.
