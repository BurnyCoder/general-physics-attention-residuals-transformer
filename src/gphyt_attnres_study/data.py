"""Global: official simulation data become audited, normalized training examples.

Local: use h5py slicing, CPU area resizing, and memory-mapped NumPy trajectories.
Sources: https://docs.h5py.org/en/stable/high/dataset.html and docs/sources.md.
"""
from concurrent.futures import ThreadPoolExecutor  # Local: bounded parallel public-file downloads.
import hashlib  # Local: detect identical complete trajectories independently of filenames.
import json  # Local: manifest metadata is inspectable without Python-specific objects.
from pathlib import Path  # Local: data paths stay inside the configured storage root.
import re  # Local: select prespecified parameter files without outcome-dependent filtering.
import time  # Local: back off failed network transfers.
import h5py  # Local: slice official HDF5 without loading entire multi-GB files.
import numpy as np  # Local: arrays, statistics, and independent sample RNG.
import requests  # Local: HTTP range resumption and timeout handling.
import torch  # Local: use the documented interpolation implementation on CPU.
import torch.nn.functional as F  # Local: area averaging downsamples uniform grids.
import yaml  # Local: read the official The Well URL registry.
from .runtime import LOG, PROJECT, config, digest, read_json, root, stamp, write_json  # Global: shared paths and provenance.

SCALARS = ("pressure", "density", "buoyancy")  # Global: buoyancy remains a distinct physical quantity.


def selected_sources():
    """Local: pin The Well registry; global: select parameters before observing results."""
    target = root() / "source_manifest.json"  # Local: reuse one immutable registry snapshot across resumes.
    if target.exists():
        return read_json(target)  # Global: branch changes cannot alter an ongoing study's data selection.
    response = requests.get("https://api.github.com/repos/PolymathicAI/the_well/commits/master", timeout=60)  # Local: resolve the public registry revision.
    response.raise_for_status()  # Local: do not silently use an error page as metadata.
    commit = response.json()["sha"]  # Local: a full commit identifier pins the source registry.
    url = f"https://raw.githubusercontent.com/PolymathicAI/the_well/{commit}/the_well/utils/registry.yaml"  # Local: download at the resolved revision.
    response = requests.get(url, timeout=60)  # Local: bounded public network read.
    response.raise_for_status()  # Global: an unavailable official source is a visible preparation failure.
    registry = yaml.safe_load(response.text)  # Local: YAML data never executes arbitrary constructors.
    sources = []  # Local: retain URL, dataset, split, and source parameter for every file.
    for dataset in config()["data"]:  # Global: only the two preregistered datasets are included.
        for split in ("train", "valid", "test"):  # Local: preserve official split membership.
            for address in registry[dataset][split]:  # Local: choose files from the official registry.
                if dataset == "rayleigh_benard_uniform" and not re.search(r"Rayleigh_1e(6|8|10)_Prandtl_1\.hdf5$", address):
                    continue  # Global: select fixed log-spaced Rayleigh values, not easiest outcomes.
                filename = address.rsplit("/", 1)[-1]  # Local: use only the URL basename as a filename.
                sources.append(dict(dataset=dataset, split=split, url=address, filename=filename))  # Global: maintain a reproducible selection ledger.
    if len(sources) != 36:  # Local: nine convection plus twenty-seven radiative files are expected.
        raise ValueError(f"Unexpected official file selection: {len(sources)}")  # Global: source changes need investigation, not a silent fallback.
    manifest = dict(created=stamp(), registry_commit=commit, registry_url=url, sources=sources)  # Local: store provenance with selection.
    write_json(target, manifest)  # Global: freeze data choices before downloading arrays.
    write_json(PROJECT / "reports/source_manifest.json", manifest)  # Local: publish the small reproducibility manifest.
    return manifest  # Local: downloading and preprocessing share this exact file list.


def download_file(source):
    """Local: resume a public download; global: never accept a truncated HDF5 file."""
    path = root() / "raw" / source["dataset"] / source["split"] / source["filename"]  # Local: different splits have separate directories.
    path.parent.mkdir(parents=True, exist_ok=True)  # Local: create only the selected dataset folder.
    metadata_path = path.with_suffix(".download.json")  # Local: retain checksum and server identity outside Git.
    metadata = read_json(metadata_path)  # Local: completed downloads need no repeated network transfer.
    if metadata and path.exists() and path.stat().st_size == metadata["bytes"]:
        return dict(source, path=str(path), **metadata)  # Local: preprocessing can use the existing verified download.
    partial = path.with_suffix(".part")  # Local: incomplete data never has the final HDF5 filename.
    identity_path = path.with_suffix(".partial.json")  # Local: bind resumed bytes to the original server version.
    for attempt in range(5):  # Local: bound retries instead of hanging indefinitely on a source.
        try:
            head = requests.head(source["url"], timeout=60, allow_redirects=True)  # Local: obtain the expected byte count.
            head.raise_for_status()  # Local: surface inaccessible files.
            expected = int(head.headers["Content-Length"])  # Global: completion requires the exact server-reported length.
            identity = dict(bytes=expected, last_modified=head.headers.get("Last-Modified"))  # Local: server identity guards resumed segments.
            previous_identity = read_json(identity_path)  # Local: compare against the identity that produced existing partial bytes.
            if partial.exists() and partial.stat().st_size and previous_identity != identity:
                raise ValueError("Partial download identity is missing or changed; inspect it before resuming")  # Global: never combine unidentified source versions.
            write_json(identity_path, identity)  # Local: persist identity before writing the first byte.
            offset = partial.stat().st_size if partial.exists() else 0  # Local: resume an earlier verified sequence of byte ranges.
            LOG.info("download %s/%s/%s resume=%d", source["dataset"], source["split"], source["filename"], offset)  # Local: clear live provenance.
            with requests.Session() as session, partial.open("ab") as stream:
                while offset < expected:  # Local: bounded requests avoid stalled unbounded multi-GB transfers.
                    end = min(expected, offset + 8 * 1024 * 1024) - 1  # Local: at most eight MiB reside in memory per downloader.
                    headers = {"Range": f"bytes={offset}-{end}", "If-Range": identity["last_modified"] or ""}  # Local: HTTP ranges are conditional on the original source version.
                    response = session.get(source["url"], headers=headers, timeout=(30, 60))  # Local: each chunk has a bounded network timeout.
                    response.raise_for_status()  # Local: reject HTTP errors before appending bytes.
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{expected}" or len(response.content) != end - offset + 1:
                        raise ValueError("Server returned an incorrect bounded byte range")  # Global: protect exact file concatenation.
                    stream.write(response.content)  # Local: append only a verified complete range.
                    stream.flush()  # Local: make each completed chunk durable and visible to monitoring.
                    offset = end + 1  # Local: the next range starts exactly after the last completed byte.
                    if offset % (256 * 1024 * 1024) == 0:
                        LOG.info("download progress %s %.1f%%", source["filename"], 100 * offset / expected)  # Local: bounded periodic progress without flooding logs.
            if partial.stat().st_size != expected:
                raise ValueError("Downloaded file size differs from the official server")  # Global: incomplete data cannot reach preprocessing.
            checksum = digest(partial)  # Local: identify exact bytes before accepting the file.
            partial.replace(path)  # Local: atomically promote a complete download.
            metadata = dict(bytes=expected, sha256=checksum, last_modified=head.headers.get("Last-Modified"))  # Global: retain integrity evidence.
            write_json(metadata_path, metadata)  # Local: checkpoints make download resumption cheap.
            LOG.info("download complete %s sha256=%s", source["filename"], checksum)  # Local: completion is distinguishable from progress.
            return dict(source, path=str(path), **metadata)  # Local: forward metadata to the prepared manifest.
        except (requests.RequestException, ValueError, OSError) as error:
            LOG.warning("download attempt %d: %s", attempt + 1, error)  # Global: failures remain in the process log.
            if attempt == 4:
                raise  # Local: persistent failures stop preparation with an actionable error.
            time.sleep(min(2 ** attempt, 8))  # Local: small bounded backoff between failed transfers.


def canonical_fields(fields):
    """Local: map original arrays; global: identical channels across the two physics systems."""
    velocity = np.asarray(fields["velocity"], dtype=np.float32)  # Local: velocity defines the common time/grid shape.
    output = np.zeros((*velocity.shape[:-1], 5), dtype=np.float32)  # Local: absent fields have an explicit zero representation.
    active = [name in fields for name in SCALARS] + [True, True]  # Global: metrics and feedback know which channels are meaningful.
    for channel, name in enumerate(SCALARS):
        if name in fields:
            output[..., channel] = fields[name]  # Local: buoyancy occupies its own slot, never temperature.
    output[..., 3:5] = velocity  # Local: preserve the stored vector-component ordering.
    if not np.isfinite(output).all():
        raise ValueError("Official trajectory contains nonfinite values")  # Global: do not silently replace corrupted data with zeros.
    return output, active  # Local: both arrays and channel validity enter the manifest.


def window_starts(length, context=4, horizon=24):
    """Local: predetermined forecast origins; global: identical windows for every variant."""
    if length < context + horizon:
        raise ValueError("Trajectory too short for the preregistered rollout")  # Local: avoid padding with future or repeated targets.
    return sorted(set(int(f * (length - context - horizon)) for f in (0, .5, 1)))  # Local: fixed early/middle/late starts.


def validate_splits(records):
    """Local: inspect trajectory hashes; global: prevent exact cross-split contamination."""
    seen, initials, overlaps = {}, {}, []  # Local: full-trajectory and initial-state overlap have different meanings.
    for record in records:
        key = (record["dataset"], record["sha256"])  # Local: compare the same physical field schema.
        if key in seen and seen[key] != record["split"]:
            raise ValueError(f"Cross-split duplicate trajectory in {record['dataset']}")  # Global: refuse contaminated evaluation.
        seen[key] = record["split"]  # Local: repeated windows from one trajectory are not separate records.
        initial = (record["dataset"], record["initial_sha256"])  # Local: track shared starting conditions separately.
        if initial in initials and initials[initial] != record["split"]:
            overlaps.append(dict(dataset=record["dataset"], splits=[initials[initial], record["split"]]))  # Global: disclose initial-condition overlap.
        initials[initial] = record["split"]  # Local: retain a split witness for each initial-state hash.
    return overlaps  # Global: this diagnostic does not relabel official splits.


def normalization(records):
    """Local: fit stable FP64 statistics; global: validation/test arrays never enter the fit."""
    accumulators = {}  # Local: normalization is per dataset, preserving heterogeneous field scales.
    for record in records:
        if record["split"] != "train":
            continue  # Global: the leakage boundary is explicit and testable.
        total, square, count = accumulators.setdefault(record["dataset"], [np.zeros(5), np.zeros(5), np.zeros(5)])  # Local: count active field samples only.
        array = np.load(record["path"], mmap_mode="r")  # Local: read one trajectory without resident copies.
        active = np.array(record["active"])  # Local: missing fields retain mean 0 and standard deviation 1.
        for start in range(0, len(array), 16):
            chunk = np.asarray(array[start:start + 16], dtype=np.float64).reshape(-1, 5)  # Local: reduce roundoff without loading the whole dataset.
            total[active] += chunk.sum(0)[active]  # Local: accumulate first moments.
            square[active] += np.square(chunk).sum(0)[active]  # Local: accumulate second moments.
            count[active] += len(chunk)  # Local: weight every training grid cell equally within its dataset.
    result = {}  # Local: serialize plain numbers for portable inference.
    for dataset, (total, square, count) in accumulators.items():
        mean = total / np.maximum(count, 1)  # Local: absent channels have zero mean.
        std = np.sqrt(np.maximum(square / np.maximum(count, 1) - mean ** 2, 0))  # Local: clamp only numerical negative variance.
        std = np.where(count > 0, np.maximum(std, 1e-6), 1.)  # Local: avoid zero divisors while preserving absent fields.
        result[dataset] = dict(mean=mean.tolist(), std=std.tolist(), count=count.astype(np.int64).tolist())  # Global: publish the exact training-only transform.
    return result  # Local: training and evaluation share these statistics.


def preprocess_file(source, resolution):
    """Local: slice and resize one file; global: preserve trajectory identity and geometry."""
    folder = root() / "processed" / source["dataset"] / source["split"] / Path(source["filename"]).stem  # Local: one owned output group per official file.
    folder.mkdir(parents=True, exist_ok=True)  # Local: processed artifacts never overwrite raw data.
    record_file = folder / "records.json"  # Local: resume after completing an HDF5 file.
    if record_file.exists():
        cached = read_json(record_file)  # Local: provenance prevents reusing a cache for different data or resolution.
        if cached["source_sha256"] != source["sha256"] or cached["resolution"] != resolution:
            raise ValueError("Processed cache provenance mismatch")  # Global: never silently mix preprocessing configurations.
        return cached["records"]  # Local: completed preprocessing is deterministic and reusable.
    records = []  # Local: collect every official trajectory, including difficult cases.
    with h5py.File(source["path"], "r") as file:
        velocity = file["t1_fields/velocity"]  # Local: official vector field defines sample/time/grid dimensions.
        coordinates = {}  # Local: retain actual coordinates for physical diagnostics.
        for axis in ("x", "y"):
            coordinate = np.asarray(file[f"dimensions/{axis}"]).reshape(-1)  # Local: these selected datasets use one shared static grid.
            spacing = np.diff(coordinate)  # Local: uniformity is checked from data, not the dataset name.
            if not np.allclose(spacing, spacing.mean(), rtol=1e-3, atol=1e-7):
                raise ValueError(f"Nonuniform {axis} coordinate in {source['filename']}")  # Global: area resizing requires a uniform original grid.
            coordinates[axis] = coordinate.tolist()  # Local: original endpoints and lengths remain auditable.
        present = [name for name in SCALARS if name in file["t0_fields"]]  # Local: read only physically present scalars.
        shape = velocity.shape  # Local: expected layout is trajectory,time,x,y,component.
        if len(shape) != 5 or shape[-1] != 2:
            raise ValueError(f"Unexpected velocity layout {shape}")  # Global: fail rather than invent a tensor orientation.
        for trajectory in range(shape[0]):
            output_path = folder / f"trajectory_{trajectory:03}.npy"  # Local: unique path retains original trajectory index.
            output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(shape[1], *resolution, 5))  # Local: incrementally write the smaller trajectory.
            full_hash, initial_hash = hashlib.sha256(), hashlib.sha256()  # Local: original data, not normalized approximations, define identity.
            for start in range(0, shape[1], 8):
                fields = {name: file[f"t0_fields/{name}"][trajectory, start:start + 8] for name in present}  # Local: bounded scalar slices.
                fields["velocity"] = velocity[trajectory, start:start + 8]  # Local: include both components in identity checks.
                for name in (*present, "velocity"):
                    full_hash.update(np.ascontiguousarray(fields[name]).tobytes())  # Local: hash every original frame and field.
                    if start == 0:
                        initial_hash.update(np.ascontiguousarray(fields[name][0]).tobytes())  # Local: separately identify identical initial states.
                canonical, active = canonical_fields(fields)  # Local: one tested mapping handles both systems.
                resized = F.interpolate(torch.from_numpy(canonical).permute(0, 3, 1, 2), size=resolution, mode="area")  # Local: CPU area averaging preserves coarse cell means.
                output[start:start + len(canonical)] = resized.permute(0, 2, 3, 1).numpy()  # Local: keep upstream channels-last convention.
            output.flush()  # Local: finish the memory-mapped array before publishing its record.
            del output  # Local: release the Windows file mapping before later readers open it.
            record = dict(dataset=source["dataset"], split=source["split"], parameter=Path(source["filename"]).stem.replace("rayleigh_benard_uniform_", "").replace("rayleigh_benard_", ""), trajectory=trajectory, path=str(output_path), length=shape[1], active=active, sha256=full_hash.hexdigest(), initial_sha256=initial_hash.hexdigest(), coordinates=coordinates, source_url=source["url"])  # Global: retain identity, geometry, and source semantics.
            records.append(record)  # Local: one statistical trajectory has one manifest record.
            LOG.info("prepared %s %s trajectory=%d/%d", source["dataset"], source["split"], trajectory + 1, shape[0])  # Local: preprocessing stays observable.
    write_json(record_file, dict(source_sha256=source["sha256"], resolution=resolution, records=records))  # Local: resumable file-level checkpoint.
    return records  # Global: normalization and split checks use the same complete records.


def prepare():
    """Global: public data -> checked arrays -> training-only statistics, without GPU execution."""
    if (root() / "prepared.json").exists():
        LOG.info("Preparation already complete; using its frozen manifest")  # Local: do not refit a completed study on changed data.
        return read_json(root() / "prepared.json")  # Global: preparation is idempotent.
    manifest = selected_sources()  # Local: freeze the official source selection.
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(download_file, manifest["sources"]))  # Local: finish required downloads before fitting statistics.
    records = []  # Local: hold only small metadata, not full trajectories.
    for source in files:
        records.extend(preprocess_file(source, config()["resolution"]))  # Local: preprocessing stays serial to bound RAM.
    overlaps = validate_splits(records)  # Global: refuse exact cross-split duplicate trajectories.
    stats = normalization(records)  # Global: fit only selected training trajectories.
    result = dict(created=stamp(), resolution=config()["resolution"], records=records, files=files, normalization=stats, initial_state_overlaps=overlaps)  # Local: one input contract for all later phases.
    write_json(root() / "prepared.json", result)  # Local: publish only after every data check succeeds.
    portable = dict(result, records=[dict(r, path=str(Path(r["path"]).relative_to(root()))) for r in records], files=[{k: v for k, v in f.items() if k != "path"} for f in files])  # Local: public manifests use portable paths.
    write_json(PROJECT / "reports/data_manifest.json", portable)  # Global: anyone cloning can reproduce the selection and normalization.
    LOG.info("Prepared %d trajectories; initial-state overlaps=%d", len(records), len(overlaps))  # Local: summary reports actual inspected counts.
    return result  # Local: optional pipeline chaining reuses the prepared data.


class Sampler:
    """Global: paired runs see identical balanced windows independent of model execution."""
    def __init__(self, prepared, seed, split="train"):
        """Local: group records by physics and parameter, then seed a dedicated RNG."""
        self.prepared, self.split = prepared, split  # Local: retain shared statistics and split identity.
        self.rng = np.random.default_rng(seed + 1717)  # Global: data randomness is separate from Torch initialization.
        self.groups, self.cache = {}, {}  # Local: small dictionaries index memory-mapped trajectories.
        for record in prepared["records"]:
            if record["split"] == split:
                self.groups.setdefault(record["dataset"], {}).setdefault(record["parameter"], []).append(record)  # Local: equalize sampling at every requested level.
        self.datasets = sorted(self.groups)  # Local: canonical ordering makes RNG replay portable.

    def array(self, record):
        """Local: cache read-only memory maps; global: training does not mutate prepared data."""
        if record["path"] not in self.cache:
            self.cache[record["path"]] = np.load(record["path"], mmap_mode="r")  # Local: avoid loading the dataset into RAM.
        return self.cache[record["path"]]  # Local: the OS manages page caching.

    def normalize(self, array, record):
        """Local: apply saved training-only statistics; global: absent channels remain zero."""
        stats = self.prepared["normalization"][record["dataset"]]  # Local: heterogeneous systems keep their own scales.
        result = (np.asarray(array, dtype=np.float32) - np.array(stats["mean"], dtype=np.float32)) / np.array(stats["std"], dtype=np.float32)  # Local: no fitting occurs during sampling.
        result[..., ~np.array(record["active"])] = 0  # Local: enforce zero representation for absent physical fields.
        return result  # Global: model input and target share the same transform.

    def batch(self, size):
        """Local: sample independent windows; global: paired seeds replay exactly the same draws."""
        inputs, targets, masks = [], [], []  # Local: assemble only the requested minibatch.
        for _ in range(size):
            dataset = self.datasets[self.rng.integers(len(self.datasets))]  # Local: equal probability for the two physics systems.
            parameters = sorted(self.groups[dataset])  # Local: avoid dictionary-order dependence.
            group = self.groups[dataset][parameters[self.rng.integers(len(parameters))]]  # Local: uniform parameter setting.
            record = group[self.rng.integers(len(group))]  # Local: uniform trajectory within that setting.
            stride = int(self.rng.integers(1, 9)) if self.split == "train" else 1  # Local: validation uses the reported stride.
            start = int(self.rng.integers(record["length"] - 4 * stride))  # Local: leave four strides between first context and target.
            frames = self.normalize(self.array(record)[start + np.arange(5) * stride], record)  # Local: gather only allowed context/target frames.
            inputs.append(frames[:4])  # Global: no future target enters the context.
            targets.append(frames[4:5])  # Local: retain the upstream singleton output-time axis.
            masks.append(record["active"])  # Local: per-example field validity enters the loss.
        return tuple(torch.from_numpy(np.stack(items)) for items in (inputs, targets, masks))  # Local: conversion shares the compact batch storage.
