"""Global: shared configuration, provenance, logging, and the GPU budget contract.

Local: atomic JSON writes and a file lock keep interrupted runs reviewable.
Sources: https://docs.python.org/3/library/logging.html and https://py-filelock.readthedocs.io/.
"""
import hashlib  # Local: link inputs and outputs to exact bytes.
import json  # Local: use portable human-readable metadata.
import logging  # Local: timestamps appear in both file and terminal messages.
import os  # Local: read configuration without exposing environment secrets.
from pathlib import Path  # Local: construct paths without shell interpolation.
import subprocess  # Local: read the repository revision for provenance.
import sys  # Local: mirror terminal streams and locate vendored imports.
import time  # Local: monotonic duration is independent of wall-clock corrections.
from datetime import datetime, timezone  # Local: persist unambiguous UTC timestamps.
from dotenv import load_dotenv  # Local: honor the requested .env workflow.
from filelock import FileLock  # Local: prevent simultaneous GPU budget writers.
import psutil  # Local: bind the hard-cap supervisor to the exact budget-owning process.

PROJECT = Path(__file__).resolve().parents[2]  # Global: all source paths remain in this project.
load_dotenv(PROJECT / ".env")  # Local: credentials are neither printed nor serialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Local: deterministic CUDA BLAS workspace; see PyTorch reproducibility documentation.
sys.path.insert(0, str(PROJECT / "vendor"))  # Local: import the pinned upstream package without its unrelated dependencies.
LOG = logging.getLogger("study")  # Global: every phase uses one consistent logger.


def stamp():
    """Local: timezone-qualified timestamp; global: provenance is cross-machine readable."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # Local: UTC avoids daylight-saving ambiguity.


def root():
    """Local: resolve configured artifact root; global: large files stay on the allowed data drive."""
    path = Path(os.environ.get("STUDY_ROOT", str(PROJECT / "artifacts"))).resolve()  # Local: clones work without a D: drive.
    path.mkdir(parents=True, exist_ok=True)  # Local: create only the explicitly selected artifact directory.
    return path  # Global: every data/checkpoint phase uses the same root.


def config():
    """Local: load the immutable study inputs; global: defaults live in one versioned file."""
    return json.loads((PROJECT / os.environ.get("STUDY_CONFIG", "configs/study.json")).read_text(encoding="utf-8"))  # Local: no hidden training defaults.


def write_json(path, value):
    """Local: atomic replacement; global: interrupted writes cannot produce a valid-looking partial result."""
    path = Path(path)  # Local: accept both strings and Path instances.
    path.parent.mkdir(parents=True, exist_ok=True)  # Local: make the destination's owned directory.
    temporary = path.with_suffix(path.suffix + ".tmp")  # Local: stage alongside the destination for atomic rename.
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # Local: disallow nonstandard JSON NaN/Infinity.
    temporary.replace(path)  # Global: readers observe either the previous or complete new metadata.


def read_json(path, default=None):
    """Local: read metadata when present; global: missing results stay explicitly missing."""
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default  # Local: do not fabricate an empty measurement.


def digest(path):
    """Local: stream SHA-256; global: large HDF5 files never need to fit in RAM."""
    with Path(path).open("rb") as stream:  # Local: hash exact bytes without normalization.
        return hashlib.file_digest(stream, "sha256").hexdigest()  # Local: the standard library supplies chunked hashing.


def revision():
    """Local: read Git HEAD; global: each run identifies its implementation."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()  # Local: argument arrays avoid shell expansion.


class Tee:
    """Global: preserve complete stdout/stderr; local: forward writes without truncation."""
    def __init__(self, terminal, stream):
        """Local: retain both destinations; global: one append-only log per invocation."""
        self.terminal, self.stream = terminal, stream  # Local: share a single timestamped file for both streams.

    def write(self, text):
        """Local: mirror every byte-like text chunk; global: keep tracebacks visible."""
        self.terminal.write(text)  # Local: preserve realtime terminal output.
        self.stream.write(text)  # Local: preserve the complete output on disk.
        self.flush()  # Local: abrupt interruptions lose at most an in-flight write.
        return len(text)  # Local: satisfy the TextIO write protocol.

    def flush(self):
        """Local: flush both outputs; global: logging is observable during long runs."""
        self.terminal.flush()  # Local: display pending terminal text.
        self.stream.flush()  # Local: make logs readable by the dashboard and operator.

    def isatty(self):
        """Local: use plain logs; global: avoid terminal control codes in files."""
        return False  # Local: progress libraries choose noninteractive output.


def start_logging(phase):
    """Local: attach full stream capture; global: timestamp every pipeline invocation."""
    folder = root() / "logs"  # Local: logs live beside results rather than credentials.
    folder.mkdir(exist_ok=True)  # Local: one directory groups process logs.
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{phase}_{os.getpid()}.log"  # Local: include PID to avoid collisions.
    stream = (folder / name).open("a", encoding="utf-8", buffering=1)  # Local: line buffering supports live inspection.
    sys.stdout, sys.stderr = Tee(sys.stdout, stream), Tee(sys.stderr, stream)  # Local: capture external-library prints too.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout, force=True)  # Local: shared readable format.
    LOG.info("phase=%s utc=%s commit=%s", phase, stamp(), revision())  # Global: identify the executable state before work.
    LOG.info("command=%s", subprocess.list2cmdline(sys.argv))  # Local: record the invoked CLI without printing environment secrets.


class BudgetExceeded(RuntimeError):
    """Global: a budget stop is a recorded outcome, never authorization to spend more."""


class Budget:
    """Global: serialize GPU phases and enforce the cumulative 24-hour contract."""
    def __init__(self, phase, location=None, cap_seconds=86400):
        """Local: allow isolated test ledgers; global: real runs share one persistent ledger."""
        self.path = Path(location or root() / "gpu_budget.json")  # Local: all phases charge the same default ledger.
        self.phase, self.cap = phase, cap_seconds  # Local: preserve phase attribution and an immutable cap.
        self.lock = FileLock(str(self.path) + ".lock", timeout=0)  # Local: only one GPU task may own the budget.
        self.last_write = 0.0  # Local: rate-limit metadata writes without weakening in-memory checks.
        self.watchdog = None  # Local: a rejected lease never starts a supervisor process.

    def __enter__(self):
        """Local: acquire the ledger and start a lease; global: interruptions leave evidence."""
        self.path.parent.mkdir(parents=True, exist_ok=True)  # Local: test and real ledger directories are explicit.
        self.lock.acquire()  # Local: fail instead of racing another GPU process.
        self.ledger = read_json(self.path, {"cap_seconds": self.cap, "sessions": []})  # Local: resume previous accounting.
        for entry in self.ledger["sessions"]:  # Local: settle any lease left by an interrupted process.
            if entry["status"] == "running":  # Local: the file lock proves no cooperating process still owns it.
                entry.update(status="interrupted", seconds=entry["seconds"] + 15, end=stamp())  # Global: conservatively charge the heartbeat interval.
        self.prior = sum(e["seconds"] for e in self.ledger["sessions"])  # Local: include failures and previous tests.
        self.phase_prior = sum(e["seconds"] for e in self.ledger["sessions"] if e["phase"] == self.phase)  # Local: include all attempts in this allocation.
        self.phase_cap = config()["gpu_hours"].get(self.phase, 24) * 3600  # Global: preserve the reserved evaluation and contingency allocations.
        self.start = time.monotonic()  # Local: wall-clock jumps cannot reduce elapsed usage.
        self.entry = dict(phase=self.phase, start=stamp(), pid=os.getpid(), seconds=0, status="running")  # Global: make the active attempt visible.
        self.ledger["sessions"].append(self.entry)  # Local: preserve all attempts instead of overwriting a total.
        try:  # Local: release the lock even if the cap is already exhausted.
            self.check(force=True)  # Local: refuse work before the first GPU operation when out of budget.
        except BaseException as error:
            self.entry.update(status="failed", error=str(error), end=stamp())  # Global: rejected leases are terminal outcomes.
            write_json(self.path, self.ledger)  # Local: persist rejection before relinquishing ownership.
            self.lock.release()  # Local: do not leave a failed context holding ownership.
            raise  # Global: propagate the explicit budget failure.
        deadline = self.start + min(self.cap - self.prior, self.phase_cap - self.phase_prior) - 5  # Global: reserve five seconds for forced GPU-worker teardown before the hard limit.
        command = [sys.executable, "-m", "gphyt_attnres_study.budget_watchdog", str(os.getpid()), str(psutil.Process().create_time()), str(self.start), str(deadline), str(self.path)]  # Local: pass exact PID identity and only nonsecret execution metadata.
        LOG.info("budget supervisor command=%s", subprocess.list2cmdline(command))  # Global: independent enforcement is part of the execution audit.
        self.watchdog = subprocess.Popen(command, cwd=PROJECT, env=dict(os.environ, STUDY_ROOT=str(self.path.parent)), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))  # Local: the hidden helper inherits the terminal stream and writes its own complete timestamped file, without opening a UI window.
        return self  # Local: ordinary loops still stop cooperatively before the independent hard deadline.

    def check(self, force=False):
        """Local: heartbeat and cap check; global: account training, profiling, and evaluation alike."""
        now = time.monotonic()  # Local: use a monotonic clock for duration.
        self.entry["seconds"] = now - self.start  # Local: IO inside a GPU phase counts conservatively as GPU wall time.
        if force or now - self.last_write >= 10:  # Local: persist frequent resumable heartbeats.
            write_json(self.path, self.ledger)  # Global: the ledger survives process termination.
            self.last_write = now  # Local: schedule the next heartbeat.
        if self.prior + self.entry["seconds"] >= self.cap - 15:  # Local: stop cooperatively ten seconds before the hard supervisor, preserving a teardown margin.
            raise BudgetExceeded("The cumulative GPU wall-time cap has been reached")  # Global: never silently exceed authorized compute.
        if self.phase_prior + self.entry["seconds"] >= self.phase_cap - 15:  # Local: leave a small guard before another bounded work unit.
            raise BudgetExceeded(f"The {self.phase} GPU allocation has been reached")  # Global: do not consume another phase's reserved allocation silently.

    def remaining(self):
        """Local: expose a conservative remainder for scheduling; global: budget allocation is explicit."""
        return max(0, self.cap - self.prior - (time.monotonic() - self.start))  # Local: include the active lease.

    def __exit__(self, kind, error, traceback):
        """Local: finalize timing on success or failure; global: neither outcome disappears."""
        if self.watchdog is not None:
            self.watchdog.terminate()  # Local: normal or cooperative completion cancels only the helper started for this lease.
            self.watchdog.wait(timeout=5)  # Local: do not leave a stale supervisor after the GPU phase finishes.
        self.entry.update(seconds=time.monotonic() - self.start, status="failed" if error else "complete", end=stamp())  # Local: preserve actual consumed time.
        if error:
            self.entry["error"] = str(error)  # Global: failed work remains interpretable.
        write_json(self.path, self.ledger)  # Local: commit the final duration before releasing ownership.
        self.lock.release()  # Local: permit the next planned GPU phase.


def cpu_threads():
    """Local: bound CPU threads; global: avoid oversubscribing the laptop during GPU work."""
    import torch  # Local: defer heavy imports until a computational phase requires them.
    torch.set_num_threads(4)  # Local: leave cores for data loading, the UI, and the operating system.


def run_logged(command, cwd=PROJECT):
    """Local: stream complete subprocess output; global: external CLI actions share the timestamped audit log."""
    command = list(map(str, command))  # Local: callers supply an argument array, never a shell-built command.
    LOG.info("command=%s", subprocess.list2cmdline(command))  # Local: caller commands must contain no credential values.
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")  # Local: merge child output into the parent Tee.
    output = []  # Local: preserve complete metadata for commands returning structured output.
    for line in process.stdout:
        print(line, end="", flush=True)  # Global: file and terminal receive complete output in real time.
        output.append(line)  # Local: retain full responses without truncation.
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command, output="".join(output))  # Global: failed commands cannot appear successful.
    return "".join(output)  # Local: structured CLI metadata can be parsed by the caller.
