"""Global: enforce a GPU-phase deadline even if its worker stops checking the budget.

Local: supervise only the explicitly identified parent PID and creation time.
Sources: https://psutil.readthedocs.io/en/latest/#psutil.Process.wait and
https://docs.python.org/3/library/subprocess.html#subprocess.CREATE_NO_WINDOW.
"""
import sys  # Local: the parent supplies exact process identity, deadline, and owned ledger path.
import time  # Local: monotonic clocks share one origin across processes on supported Python 3.13 platforms.
from pathlib import Path  # Local: the ledger remains within the parent's configured artifact root.
import psutil  # Local: documented process identity, waiting, and termination APIs.
from filelock import FileLock  # Local: update the ledger only after the terminated owner releases it.
from .runtime import LOG, read_json, stamp, start_logging, write_json  # Global: hard stops retain the same timestamped audit trail.


def supervise(pid, created, started, deadline, path):
    """Local: wait for this worker or its deadline; global: never identify processes by a broad executable-name match."""
    try:
        process = psutil.Process(pid)  # Local: bind the exact worker that acquired the GPU ledger.
        if process.create_time() != created:
            return  # Global: PID reuse must never terminate another process.
        try:
            process.wait(timeout=max(0., deadline - time.monotonic()))  # Local: the OS wait is independent of a stuck worker's Python interpreter.
            return  # Local: normal worker exit needs no hard-stop intervention.
        except psutil.TimeoutExpired:
            LOG.error("Hard budget deadline reached for study worker pid=%d", pid)  # Global: record why this exact project process is stopped.
        process.terminate()  # Local: stop only the budget-owning worker identified above.
        try:
            process.wait(timeout=2)  # Local: allow bounded process teardown within the five-second reserve.
        except psutil.TimeoutExpired:
            process.kill()  # Global: a noncooperating worker cannot continue consuming the approved allocation.
            process.wait(timeout=2)  # Local: ensure process termination before publishing its final ledger state.
        with FileLock(str(path) + ".lock", timeout=5):
            ledger = read_json(path)  # Local: preserve all prior sessions, including other phases.
            for entry in ledger["sessions"]:
                if entry["pid"] == pid and entry["status"] == "running":
                    entry.update(status="hard_budget_stop", seconds=time.monotonic() - started, end=stamp(), error="Independent watchdog stopped a noncooperating work unit before the budget deadline")  # Global: hard termination is a visible failed attempt, never a successful run.
            write_json(path, ledger)  # Local: atomic accounting update after exclusive ownership ends.
    except psutil.NoSuchProcess:
        return  # Local: an already exited worker cannot consume additional GPU time.


if __name__ == "__main__":
    start_logging("budget-watchdog")  # Global: helper commands and outputs receive their own complete timestamped log.
    supervise(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), Path(sys.argv[5]))  # Local: explicit arguments avoid searching unrelated machine processes.
