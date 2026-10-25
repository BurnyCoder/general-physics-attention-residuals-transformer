"""Global: quantify official-split initial-condition reuse without changing the study population.

Local: exact hashes distinguish initial frames from complete first prediction contexts.
Sources: https://docs.python.org/3/library/collections.html#collections.defaultdict,
https://docs.python.org/3/library/hashlib.html, and https://numpy.org/doc/stable/reference/generated/numpy.load.html.
"""
from collections import Counter, defaultdict  # Local: group existing identities without introducing a new data dependency.
import hashlib  # Local: exact prepared-context bytes have a reproducible identity.
import numpy as np  # Local: inspect only four memory-mapped frames per trajectory.
from .provenance import data_fingerprint  # Global: bind this diagnostic to the exact prepared study.
from .runtime import LOG, PROJECT, stamp, write_json  # Global: share timestamped logging and atomic public artifacts.


def cross_split_groups(groups):
    """Local: retain distinct hashes spanning splits; global: do not count traversal transitions as independent groups."""
    result = []  # Local: one row represents one exact state/context identity.
    for (dataset, identity), members in sorted(groups.items()):
        if len({row["split"] for row in members}) > 1:
            result.append(dict(dataset=dataset, sha256=identity, members=[{key: row[key] for key in ("split", "parameter", "trajectory")} for row in members]))  # Local: public identifiers exclude machine-specific paths.
    return result  # Global: every reported overlap can be resolved against the public manifest.


def audit_records(records):
    """Local: inspect raw first-frame hashes and prepared first-four-frame hashes; global: disclose the scope of held-out generalization."""
    initials, contexts = defaultdict(list), defaultdict(list)  # Local: separate original-frame and resized-context identities.
    for record in records:
        initials[(record["dataset"], record["initial_sha256"])].append(record)  # Local: reuse the original-field digest computed during preparation.
        context = np.ascontiguousarray(np.load(record["path"], mmap_mode="r", allow_pickle=False)[:4])  # Local: this diagnostic does not inspect prediction errors or fit any statistic.
        identity = hashlib.sha256(context.tobytes()).hexdigest()  # Local: exact canonical FP32 context bytes determine equality.
        contexts[(record["dataset"], identity)].append(record)  # Local: schemas are compared only within one dataset.
    initial_groups = cross_split_groups(initials)  # Local: remove within-split repetitions from the cross-split diagnostic.
    same_parameter = sum(any(len({r["split"] for r in group["members"] if r["parameter"] == parameter}) > 1 for parameter in {r["parameter"] for r in group["members"]}) for group in initial_groups)  # Global: distinguish different dynamics from split reuse at an identical parameter.
    shared = Counter((group["dataset"], row["split"]) for group in initial_groups if any(r["split"] == "train" for r in group["members"]) for row in group["members"] if row["split"] != "train")  # Local: count held-out trajectories, not correlated frames.
    counts = Counter((row["dataset"], row["split"]) for row in records)  # Local: denominator counts come from the same inspected manifest.
    return dict(initial_groups=initial_groups, same_parameter_initial_groups=same_parameter, first_context_groups=cross_split_groups(contexts), split_counts=[dict(dataset=dataset, split=split, trajectories=count, shares_training_initial=shared[(dataset, split)] if split != "train" else None) for (dataset, split), count in sorted(counts.items())], context_scope="First four consecutive prepared frames only; this is not an exhaustive comparison of all possible training and evaluation windows.")  # Global: narrow diagnostic scope prevents overclaiming absence of all possible overlap.


def audit(prepared):
    """Local: save the reproducible audit; global: no data, endpoints, or split membership are changed."""
    result = dict(created=stamp(), data_sha256=data_fingerprint(prepared), **audit_records(prepared["records"]))  # Local: preserve the original prepared artifact unchanged.
    write_json(PROJECT / "reports/data_audit.json", result)  # Global: make overlap evidence public before confirmatory training.
    LOG.info("Data audit: distinct initial groups=%d same-parameter groups=%d first-context groups=%d", len(result["initial_groups"]), result["same_parameter_initial_groups"], len(result["first_context_groups"]))  # Local: communicate quantities with their actual counting units.
    return result  # Local: reporting and tests can consume the same structured artifact.
