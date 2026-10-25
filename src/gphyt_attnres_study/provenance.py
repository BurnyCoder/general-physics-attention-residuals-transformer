"""Global: bind a frozen study to portable scientific inputs, not one machine's paths.

Local: hashes retain splits, field semantics, coordinates, raw bytes, and fitted statistics.
Sources: https://docs.python.org/3/library/hashlib.html and docs/protocol.md.
"""
import hashlib  # Local: the standard library implements the content digest.
import json  # Local: canonical key order gives one deterministic byte representation.
from .runtime import LOG, PROJECT, config, read_json, root, write_json  # Global: use the same source and artifact destinations as every phase.


def data_fingerprint(prepared):
    """Local: omit only timestamp and filesystem locations; global: preserve the complete scientific data contract."""
    portable = {key: value for key, value in prepared.items() if key not in {"created", "files", "records"}}  # Local: retain resolution, normalization, and overlap diagnostics.
    for kind in ("files", "records"):
        portable[kind] = [{key: value for key, value in row.items() if key != "path"} for row in prepared[kind]]  # Global: ordered source identities and split assignments remain part of the hash.
    encoded = json.dumps(portable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")  # Local: deterministic JSON independent of pretty-printing or newline conventions.
    return hashlib.sha256(encoded).hexdigest()  # Global: changes to the data or preprocessing contract invalidate the freeze.


def validate_frozen(frozen, prepared):
    """Local: check data and settings before expensive work; global: no silent protocol changes across phases."""
    if frozen is None or prepared is None:
        raise RuntimeError("Preparation and a published frozen schedule are required")  # Local: no default model execution when evidence is absent.
    if frozen["data_sha256"] != data_fingerprint(prepared):
        raise RuntimeError("Scientific data manifest changed after freezing")  # Global: path moves are allowed; changed scientific inputs are not.
    if frozen["config"] != config():
        raise RuntimeError("Configuration differs from the frozen protocol")  # Global: endpoints, model size, seeds, and optimizer settings remain fixed.
    if frozen != read_json(PROJECT / "reports/frozen_protocol.json"):
        raise RuntimeError("Local frozen schedule differs from the published protocol")  # Global: changing only the private update count cannot bypass preregistration.


def reproduce():
    """Local: adopt the released update count on a clone; global: replication does not refit timing or choose new endpoints."""
    frozen, prepared = read_json(PROJECT / "reports/frozen_protocol.json"), read_json(root() / "prepared.json")  # Local: the released freeze is the source of truth.
    validate_frozen(frozen, prepared)  # Global: require identical selected data, statistics, and scientific configuration.
    existing = read_json(root() / "frozen.json")  # Local: never overwrite a different active study implicitly.
    if existing is not None and existing != frozen:
        raise RuntimeError("A different local study is already frozen; select a fresh STUDY_ROOT")  # Local: preserve existing experimental artifacts.
    write_json(root() / "frozen.json", frozen)  # Local: copy only after all replication checks succeed.
    LOG.info("Adopted published schedule: updates=%d microbatch=%d", frozen["updates"], frozen["microbatch"])  # Global: replication records the exact original horizon.
