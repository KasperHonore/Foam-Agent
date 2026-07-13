"""Unit tests for the background run loop: start_case + case_status (issue #74).

start_case performs exactly the blocking run's preparation and ledger side
effects, launches Allrun detached, writes a pidfile and returns the ledger
run id immediately; case_status polls the run — typed parser progress while
the process lives, then the identical completion path a blocking run takes
(error gate -> done with the parser-backed Result, or debugging). The
in-memory registry is a cache; the pidfile plus the ledger are the truth.

These tests drive the mechanics seam with a TEST-CONTROLLED child process
standing in for Allrun: a tiny Python script that writes solver-log lines
and exits on cue (an 'exit.cue' file appearing in the case directory) —
prior art: the stubbed subprocess boundary in test_ledger_lifecycle.py.
Key-free, no docker, no OpenFOAM, no real solver; runs on Windows and Linux.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import mechanics  # noqa: E402

COLUMNS = ["id", "case", "created", "solver", "mesh", "status", "result", "key_result", "notes"]

# Fake case contents: enough for the deterministic Solver/Mesh inspection
# (and for parse_solver_log to pick log.icoFoam as the solver log).
CONTROL_DICT = """FoamFile
{
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     icoFoam;
endTime         0.5;
"""

ALLRUN = """#!/bin/sh
runApplication blockMesh
runApplication icoFoam
"""

# Every real OpenFOAM log opens with this banner — its "floating point
# exception"/"sigFpe" words must never read as a blow-up (permanent policy
# since the v1 shakedown; see test_ledger_lifecycle.py).
SIGFPE_BANNER = "sigFpe : Enabling floating point exception trapping (FOAM_SIGFPE).\n"

GOOD_LOG = SIGFPE_BANNER + """\
Time = 0.5s

Courant Number mean: 0.222158 max: 0.852134
smoothSolver:  Solving for Ux, Initial residual = 2.3091e-07, Final residual = 2.3091e-07, No Iterations 0
DICPCG:  Solving for p, Initial residual = 8.63844e-07, Final residual = 8.63844e-07, No Iterations 0
End
"""

INCOMPLETE_LOG = SIGFPE_BANNER + """\
Time = 0.1s

Courant Number mean: 0.22117 max: 0.8517
smoothSolver:  Solving for Ux, Initial residual = 0.000274837, Final residual = 6.41057e-06, No Iterations 7
"""  # no End marker — mid-run for a live process, a failure for a dead one


def _rows(runs_root: Path) -> list:
    """Parse the data rows out of a ledger file, as a reader would."""
    text = (runs_root / "ledger.md").read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if line.startswith("| ID") or set(line) <= {"|", "-", " "}:
            continue  # header / separator
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(dict(zip(COLUMNS, cells)))
    return rows


def _row(runs_root: Path, case: str) -> dict:
    return next(r for r in _rows(runs_root) if r["case"] == case)


def _make_case(runs_root, name="cavity") -> str:
    """Resolve a case (ledgers it as planned) and write fake run scripts."""
    case_dir = mechanics.resolve_case_dir(name, run_directory=str(runs_root))
    (Path(case_dir) / "system").mkdir(parents=True)
    (Path(case_dir) / "system" / "controlDict").write_text(CONTROL_DICT)
    (Path(case_dir) / "Allrun").write_text(ALLRUN)
    return case_dir


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    """Fresh in-memory registry per test; kill any child left running."""
    registry = {}
    monkeypatch.setattr(mechanics, "_BACKGROUND_RUNS", registry, raising=False)
    yield registry
    for run in list(registry.values()):
        if run.process.poll() is None:
            run.process.kill()
            run.process.wait(timeout=10)


@pytest.fixture
def fake_allrun(monkeypatch, tmp_path):
    """Substitute the Allrun launch seam with a test-controlled child process.

    The installer takes the solver-log content the 'solver' should write on
    startup; the child then idles until an 'exit.cue' file appears in the
    case directory (or a 30 s self-destruct, so no test can orphan it).
    """
    def _install(log_content, log_name="log.icoFoam"):
        script = tmp_path / "fake_allrun.py"
        script.write_text(
            "import os, time\n"
            f"with open({log_name!r}, 'w') as fh:\n"
            f"    fh.write({log_content!r})\n"
            "deadline = time.time() + 30\n"
            "while time.time() < deadline and not os.path.exists('exit.cue'):\n"
            "    time.sleep(0.05)\n"
        )
        monkeypatch.setattr(mechanics, "_allrun_argv",
                            lambda case_dir: [sys.executable, str(script)])
        return script

    return _install


def _cue_exit(case_dir) -> None:
    (Path(case_dir) / "exit.cue").write_text("")


def _poll_until(run_id, runs_root, predicate, timeout=20.0):
    """Poll case_status until the predicate holds (the external seam)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = mechanics.case_status(run_id, run_directory=str(runs_root))
        if predicate(status):
            return status
        time.sleep(0.05)
    raise AssertionError(f"case_status never reached the expected state within {timeout}s")


def _wait_dead(pid, timeout=20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not mechanics._pid_alive(pid):
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} still alive after {timeout}s")


# ---------------------------------------------------------------------------
# start_case: the blocking run's prep + stamp, then a detached launch
# ---------------------------------------------------------------------------

def test_start_case_returns_run_id_and_flips_row_to_running(tmp_path, fake_allrun):
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)

    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))

    assert start.run_id == "0001"          # the LEDGER id is the run id
    assert start.case == "cavity"
    assert start.status == "running"
    assert start.solver == "icoFoam"       # same inspection as the blocking run
    assert start.mesh == "blockMesh"
    row = _row(tmp_path, "cavity")
    assert row["status"] == "running"
    assert row["result"] == "pending"
    _cue_exit(case_dir)


def test_start_case_writes_a_pidfile_with_a_live_pid(tmp_path, fake_allrun):
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)

    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))

    pidfile = Path(case_dir) / "Allrun.pid"
    assert pidfile.is_file()
    info = json.loads(pidfile.read_text(encoding="utf-8"))
    assert info["run_id"] == start.run_id
    assert info["pid"] == start.pid
    assert "pgid" in info                  # process group (None on Windows)
    assert isinstance(info["started"], float)
    assert mechanics._pid_alive(start.pid)
    _cue_exit(case_dir)


def test_start_case_performs_the_clean_rerun_sweep(tmp_path, fake_allrun):
    case_dir = _make_case(tmp_path)
    (Path(case_dir) / "log.stale").write_text("old log")
    (Path(case_dir) / "Allrun.out").write_text("old out")
    (Path(case_dir) / "Allrun.err").write_text("old err")
    (Path(case_dir) / "0.3").mkdir()       # old time-step folder
    (Path(case_dir) / "0").mkdir()         # initial conditions stay
    fake_allrun(INCOMPLETE_LOG)

    mechanics.start_case(case_dir, run_directory=str(tmp_path))

    assert not (Path(case_dir) / "log.stale").exists()
    assert not (Path(case_dir) / "0.3").exists()
    assert (Path(case_dir) / "0").is_dir()
    # Allrun.out/err are recreated as the detached process's output capture.
    assert (Path(case_dir) / "Allrun.out").read_text() == ""
    assert (Path(case_dir) / "Allrun.err").read_text() == ""
    _cue_exit(case_dir)


def test_start_case_without_allrun_is_a_typed_error(tmp_path):
    case_dir = mechanics.resolve_case_dir("bare", run_directory=str(tmp_path))
    Path(case_dir).mkdir(parents=True)

    with pytest.raises(mechanics.BackgroundRunError, match="Allrun"):
        mechanics.start_case(case_dir, run_directory=str(tmp_path))
    assert _row(tmp_path, "bare")["status"] == "planned"  # row untouched


def test_start_case_on_a_case_with_a_live_run_is_a_typed_error(tmp_path, fake_allrun):
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))

    with pytest.raises(mechanics.BackgroundRunError, match="already"):
        mechanics.start_case(case_dir, run_directory=str(tmp_path))

    assert start.run_id  # the original run is unaffected
    _cue_exit(case_dir)


def test_start_case_out_of_tree_case_dir_is_a_typed_error(tmp_path, fake_allrun):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    case_dir = tmp_path / "elsewhere" / "case"
    case_dir.mkdir(parents=True)
    (case_dir / "Allrun").write_text(ALLRUN)
    fake_allrun(INCOMPLETE_LOG)

    # No ledger row means no run id to hand back — a typed refusal, unlike
    # the blocking run which executes out-of-tree cases untracked.
    with pytest.raises(mechanics.BackgroundRunError, match="runs root"):
        mechanics.start_case(str(case_dir), run_directory=str(runs_root))
    assert not (runs_root / "ledger.md").exists()


# ---------------------------------------------------------------------------
# case_status: typed progress while alive, the identical completion path after
# ---------------------------------------------------------------------------

def test_case_status_mid_run_reports_running_with_typed_progress(tmp_path, fake_allrun):
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))

    # The child writes its log on startup; poll until the parser sees it.
    status = _poll_until(start.run_id, tmp_path,
                         lambda s: s.progress is not None)

    assert status.status == "running"
    assert status.result == "pending"
    assert status.pid == start.pid
    assert status.elapsed_seconds >= 0
    assert status.progress.verdict == "incomplete"   # correct mid-run
    assert status.progress.time.latest_time == pytest.approx(0.1)
    assert [r.field for r in status.progress.residuals] == ["Ux"]
    assert status.errors == []
    _cue_exit(case_dir)


def test_poll_after_clean_exit_stamps_done_with_parsed_result(tmp_path, fake_allrun):
    case_dir = _make_case(tmp_path)
    fake_allrun(GOOD_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    _cue_exit(case_dir)

    status = _poll_until(start.run_id, tmp_path,
                         lambda s: s.status != "running")

    assert status.status == "done"
    assert status.result == "converged"     # the parser-backed Result
    assert status.errors == []
    assert status.pid is None
    row = _row(tmp_path, "cavity")
    assert (row["status"], row["result"]) == ("done", "converged")
    assert not (Path(case_dir) / "Allrun.pid").exists()  # stale truth dropped


def test_poll_after_failed_exit_stamps_debugging_with_extracted_errors(tmp_path, fake_allrun):
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)              # never reaches End -> error gate
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    _cue_exit(case_dir)

    status = _poll_until(start.run_id, tmp_path,
                         lambda s: s.status != "running")

    assert status.status == "debugging"
    assert status.result == "pending"
    assert status.errors                     # same extraction as run_case
    assert status.errors[0]["file"] == "log.icoFoam"
    row = _row(tmp_path, "cavity")
    assert (row["status"], row["result"]) == ("debugging", "pending")


def test_poll_after_completion_is_idempotent_and_deregisters(tmp_path, fake_allrun,
                                                             isolated_registry):
    case_dir = _make_case(tmp_path)
    fake_allrun(GOOD_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    _cue_exit(case_dir)
    _poll_until(start.run_id, tmp_path, lambda s: s.status != "running")

    assert start.run_id not in isolated_registry   # deregistered on the stamp
    again = mechanics.case_status(start.run_id, run_directory=str(tmp_path))

    assert again.status == "done"
    assert again.result == "converged"
    assert again.progress is None
    assert again.elapsed_seconds is None


def test_case_status_accepts_unpadded_run_ids(tmp_path, fake_allrun):
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)
    mechanics.start_case(case_dir, run_directory=str(tmp_path))

    status = mechanics.case_status("1", run_directory=str(tmp_path))

    assert status.run_id == "0001"
    assert status.status == "running"
    _cue_exit(case_dir)


def test_case_status_unknown_run_id_is_a_typed_error(tmp_path):
    _make_case(tmp_path)  # ledger exists, but not this id

    with pytest.raises(mechanics.BackgroundRunError, match="0042"):
        mechanics.case_status("0042", run_directory=str(tmp_path))


# ---------------------------------------------------------------------------
# Registry-miss recovery: the registry is a cache, pidfile + ledger the truth
# ---------------------------------------------------------------------------

def test_registry_miss_with_live_pidfile_reports_running(tmp_path, fake_allrun,
                                                         isolated_registry):
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    process = isolated_registry[start.run_id].process
    isolated_registry.clear()               # the server-restart stand-in

    status = mechanics.case_status(start.run_id, run_directory=str(tmp_path))

    assert status.status == "running"
    assert status.pid == start.pid
    assert status.elapsed_seconds is not None
    _cue_exit(case_dir)
    process.wait(timeout=20)                 # reap, then the next poll heals

    healed = _poll_until(start.run_id, tmp_path, lambda s: s.status != "running")
    assert healed.status == "debugging"      # INCOMPLETE_LOG never reached End
    assert not (Path(case_dir) / "Allrun.pid").exists()


def test_stuck_running_row_with_dead_pid_heals_on_the_next_poll(tmp_path):
    case_dir = _make_case(tmp_path)
    # The run died with the server: row stuck running, pidfile pid dead,
    # clean logs on disk (the solver had finished before the crash).
    mechanics._stamp_running(str(tmp_path), case_dir)
    (Path(case_dir) / "log.icoFoam").write_text(GOOD_LOG)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=20)
    (Path(case_dir) / "Allrun.pid").write_text(json.dumps(
        {"run_id": "0001", "pid": dead.pid, "pgid": None, "started": time.time() - 5}))

    status = mechanics.case_status("0001", run_directory=str(tmp_path))

    assert status.status == "done"
    assert status.result == "converged"
    row = _row(tmp_path, "cavity")
    assert (row["status"], row["result"]) == ("done", "converged")
    assert not (Path(case_dir) / "Allrun.pid").exists()


def test_case_status_on_a_row_that_never_ran_reports_ledger_state(tmp_path):
    _make_case(tmp_path)  # planned row, no run, no pidfile

    status = mechanics.case_status("0001", run_directory=str(tmp_path))

    assert status.status == "planned"
    assert status.result == "pending"
    assert status.pid is None
    assert status.progress is None
    assert status.errors == []


# ---------------------------------------------------------------------------
# run_case guard: one live solver per case directory
# ---------------------------------------------------------------------------

def test_run_case_refuses_a_case_with_a_live_background_run(tmp_path, fake_allrun):
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)
    mechanics.start_case(case_dir, run_directory=str(tmp_path))

    with pytest.raises(mechanics.BackgroundRunError, match="live background run"):
        mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(tmp_path))

    assert _row(tmp_path, "cavity")["status"] == "running"  # untouched
    _cue_exit(case_dir)


def test_run_case_refuses_on_a_live_pidfile_even_without_registry(tmp_path, fake_allrun,
                                                                  isolated_registry):
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    process = isolated_registry[start.run_id].process
    isolated_registry.clear()                # restarted server, run still live

    with pytest.raises(mechanics.BackgroundRunError, match="live background run"):
        mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(tmp_path))

    _cue_exit(case_dir)
    process.wait(timeout=20)


def test_run_case_proceeds_once_the_background_run_is_over(tmp_path, fake_allrun,
                                                           monkeypatch):
    case_dir = _make_case(tmp_path)
    fake_allrun(GOOD_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    _cue_exit(case_dir)
    _poll_until(start.run_id, tmp_path, lambda s: s.status != "running")

    # The blocking path with its usual stubbed subprocess boundary
    # (prior art: test_ledger_lifecycle.py) — the guard must not fire.
    def _fake_run_sourced(command, cwd, timeout):
        (Path(cwd) / "log.icoFoam").write_text(GOOD_LOG)
        return 0, "", "", False

    monkeypatch.setattr(mechanics, "_run_sourced", _fake_run_sourced)
    errors = mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(tmp_path))

    assert errors == []
    row = _row(tmp_path, "cavity")
    assert (row["status"], row["result"]) == ("done", "converged")
