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
import re
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


def _wait_for_file(path, timeout=20.0):
    """Wait until the child process has written a file (startup handshake)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if Path(path).exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"{path} never appeared within {timeout}s")


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


# ---------------------------------------------------------------------------
# Pidfile identity hardening (#75): (pid, starttime) + zombie awareness.
# Harvest #73: the pid namespace resets on a container restart so pid reuse
# is real; kill(pid, 0) succeeds on a zombie and PID-1 python never reaps
# orphans. (pid, starttime from /proc/<pid>/stat field 22) is the airtight
# same-process identity; the state letter is the honest liveness signal.
# ---------------------------------------------------------------------------

# A representative /proc/<pid>/stat line (the harvest's planted sleep child):
# state is field 3, starttime field 22 (jiffies since HOST boot).
STAT_LINE = (
    "2069 (sleep) S 1 2069 2069 0 -1 4194560 145 0 0 0 0 0 0 0 20 0 1 0 "
    "987690 5619712 88 18446744073709551615 0 0 0 0 0 0 0 0 0 0 0 0 17 3 0 0 0 0 0"
)


def test_parse_proc_stat_reads_state_and_starttime():
    assert mechanics._parse_proc_stat(STAT_LINE) == ("S", 987690)


def test_parse_proc_stat_handles_parentheses_in_the_command_name():
    # comm (field 2) may itself contain spaces and parentheses — the parse
    # must anchor on the LAST ')', not the first.
    line = ("42 (a (weird) name) Z 1 42 42 0 -1 4194560 "
            "0 0 0 0 0 0 0 0 20 0 1 0 4242 0")
    assert mechanics._parse_proc_stat(line) == ("Z", 4242)


def test_parse_proc_stat_rejects_garbage():
    assert mechanics._parse_proc_stat("not a stat line") is None
    # A truncated line still yields the state; starttime is honestly unknown.
    assert mechanics._parse_proc_stat("7 (x) R 1 7") == ("R", None)


def test_start_case_records_the_process_starttime_identity(tmp_path, fake_allrun):
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)

    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))

    info = json.loads((Path(case_dir) / "Allrun.pid").read_text(encoding="utf-8"))
    assert "starttime" in info                    # pid-reuse-proof identity
    if os.path.isdir("/proc"):
        stat = Path(f"/proc/{start.pid}/stat").read_text()
        assert info["starttime"] == mechanics._parse_proc_stat(stat)[1]
    else:
        assert info["starttime"] is None          # Windows: no /proc interface
    _cue_exit(case_dir)


@pytest.mark.skipif(not os.path.isdir("/proc"),
                    reason="needs /proc for the starttime identity")
def test_pidfile_with_a_mismatched_starttime_reads_dead(tmp_path, fake_allrun,
                                                        isolated_registry):
    # Pid reuse after a container restart: same pid, different process — the
    # recorded starttime disagrees, so the run must NOT read as running.
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    _wait_for_file(Path(case_dir) / "log.icoFoam")  # the heal below reads logs
    process = isolated_registry[start.run_id].process
    isolated_registry.clear()                     # the server-restart stand-in
    pidfile = Path(case_dir) / "Allrun.pid"
    info = json.loads(pidfile.read_text(encoding="utf-8"))
    info["starttime"] = (info["starttime"] or 0) + 12345
    pidfile.write_text(json.dumps(info))

    status = mechanics.case_status(start.run_id, run_directory=str(tmp_path))

    # Not our process, so the stuck row heals through the completion path
    # (INCOMPLETE_LOG never reached End -> debugging).
    assert status.status == "debugging"
    _cue_exit(case_dir)
    process.wait(timeout=20)


def test_zombie_state_reads_dead_for_pidfile_liveness(tmp_path, fake_allrun,
                                                      isolated_registry, monkeypatch):
    # kill(pid, 0) succeeds on a zombie; the state letter must overrule it.
    # No real zombie can be staged portably, so the /proc read is substituted
    # at its seam and the crafted 'Z' must read as dead.
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    _wait_for_file(Path(case_dir) / "log.icoFoam")  # the heal below reads logs
    process = isolated_registry[start.run_id].process
    isolated_registry.clear()
    monkeypatch.setattr(mechanics, "_read_proc_stat", lambda pid: ("Z", 123))

    status = mechanics.case_status(start.run_id, run_directory=str(tmp_path))

    assert status.status == "debugging"           # healed as dead, not running
    _cue_exit(case_dir)
    process.wait(timeout=20)


def test_legacy_pidfile_without_starttime_keeps_working(tmp_path, fake_allrun,
                                                        isolated_registry):
    # A pidfile written before #75 has no starttime — behavior must be
    # exactly the old pid-alive check.
    case_dir = _make_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    process = isolated_registry[start.run_id].process
    isolated_registry.clear()
    pidfile = Path(case_dir) / "Allrun.pid"
    info = json.loads(pidfile.read_text(encoding="utf-8"))
    info.pop("starttime", None)
    pidfile.write_text(json.dumps(info))

    status = mechanics.case_status(start.run_id, run_directory=str(tmp_path))

    assert status.status == "running"
    assert status.pid == start.pid
    _cue_exit(case_dir)
    process.wait(timeout=20)


# ---------------------------------------------------------------------------
# stop_case (#75): graceful-first stop, evidence kept.
# The harvest-pinned recipe (#73): graceful requires runTimeModifiable true
# (v10's compiled default is FALSE); the stopAt-writeNow edit is invisible
# until the dict's mtime clears the fileModificationSkew gate (10 s default),
# so the edit is followed by future-stamped touches; on timeout the whole
# process group is killed; either way stopAt is restored to endTime (a
# leftover writeNow insta-stops any rerun after one iteration).
# ---------------------------------------------------------------------------

def _make_graceful_case(runs_root) -> str:
    """A case whose controlDict allows the graceful route (runTimeModifiable
    true, an explicit stopAt entry for the in-place rewrite to hit)."""
    case_dir = _make_case(runs_root)
    control_dict = Path(case_dir) / "system" / "controlDict"
    control_dict.write_text(control_dict.read_text()
                            + "\nrunTimeModifiable true;\nstopAt          endTime;\n")
    return case_dir


@pytest.fixture
def graceful_allrun(monkeypatch, tmp_path):
    """A cooperative 'solver': writes its log on startup, then honors the
    graceful cue — it exits as soon as system/controlDict says writeNow,
    recording what it saw (the external evidence that the edit landed
    before the restore)."""
    def _install(log_content):
        script = tmp_path / "graceful_allrun.py"
        script.write_text(
            "import os, time\n"
            f"with open('log.icoFoam', 'w') as fh:\n"
            f"    fh.write({log_content!r})\n"
            "cd = os.path.join('system', 'controlDict')\n"
            "deadline = time.time() + 30\n"
            "while time.time() < deadline:\n"
            "    try:\n"
            "        with open(cd) as fh:\n"
            "            text = fh.read()\n"
            "    except OSError:\n"
            "        text = ''\n"
            "    if 'writeNow' in text:\n"
            "        with open('graceful.observed', 'w') as fh:\n"
            "            fh.write(text)\n"
            "        break\n"
            "    time.sleep(0.02)\n"
        )
        monkeypatch.setattr(mechanics, "_allrun_argv",
                            lambda case_dir: [sys.executable, str(script)])

    return _install


def test_stop_case_graceful_stops_a_cooperative_run_and_keeps_evidence(
        tmp_path, graceful_allrun):
    case_dir = _make_graceful_case(tmp_path)
    graceful_allrun(GOOD_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    _wait_for_file(Path(case_dir) / "log.icoFoam")

    stop = mechanics.stop_case(start.run_id, grace_seconds=15.0,
                               run_directory=str(tmp_path))

    assert stop.method == "graceful"
    assert stop.status == "done"
    assert stop.result == "converged"        # the parser-backed Result
    assert stop.errors == []
    # The solver really saw the writeNow edit (the graceful cue)...
    observed = (Path(case_dir) / "graceful.observed").read_text()
    assert "writeNow" in observed
    # ...and the dict was restored afterwards: a rerun must not insta-stop.
    restored = (Path(case_dir) / "system" / "controlDict").read_text()
    assert "writeNow" not in restored
    assert re.search(r"stopAt\s+endTime;", restored)
    # Stamped through the same completion path; the note explains the row.
    row = _row(tmp_path, "cavity")
    assert (row["status"], row["result"]) == ("done", "converged")
    assert "stopped deliberately via stop_case (graceful)" in row["notes"]
    assert stop.note == "stopped deliberately via stop_case (graceful)"
    assert not (Path(case_dir) / "Allrun.pid").exists()


def test_stop_case_skips_graceful_without_run_time_modifiable(tmp_path, fake_allrun):
    # Harvest #73: with runTimeModifiable absent (compiled default FALSE)
    # the writeNow edit is never seen — 35 s of nothing, verified live — so
    # the stop must go straight to the kill ladder and leave the dict alone.
    case_dir = _make_case(tmp_path)                  # no runTimeModifiable
    control_dict = Path(case_dir) / "system" / "controlDict"
    before = control_dict.read_text()
    fake_allrun(INCOMPLETE_LOG)                      # ignores the graceful cue
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    _wait_for_file(Path(case_dir) / "log.icoFoam")

    stop = mechanics.stop_case(start.run_id, grace_seconds=15.0,
                               run_directory=str(tmp_path))

    assert stop.method == "killed"
    assert control_dict.read_text() == before        # never touched
    # Killed run: log has no End -> the same error gate -> debugging.
    assert stop.status == "debugging"
    assert stop.result == "pending"
    assert stop.errors and stop.errors[0]["file"] == "log.icoFoam"
    row = _row(tmp_path, "cavity")
    assert (row["status"], row["result"]) == ("debugging", "pending")
    assert "stopped deliberately via stop_case (killed)" in row["notes"]
    assert not (Path(case_dir) / "Allrun.pid").exists()


def test_stop_case_kills_a_defiant_run_after_the_grace_window(tmp_path, fake_allrun):
    # runTimeModifiable true, but the child ignores the writeNow cue: the
    # bounded grace window expires, the kill ladder fires, and stopAt is
    # restored afterwards so a rerun does not insta-stop.
    case_dir = _make_graceful_case(tmp_path)
    fake_allrun(INCOMPLETE_LOG)                      # ignores the graceful cue
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    _wait_for_file(Path(case_dir) / "log.icoFoam")

    stop = mechanics.stop_case(start.run_id, grace_seconds=0.5,
                               run_directory=str(tmp_path))

    assert stop.method == "killed"
    assert stop.note == "stopped deliberately via stop_case (killed)"
    restored = (Path(case_dir) / "system" / "controlDict").read_text()
    assert "writeNow" not in restored                # restored after the kill
    assert re.search(r"stopAt\s+endTime;", restored)
    assert stop.status == "debugging"
    assert not (Path(case_dir) / "Allrun.pid").exists()


@pytest.mark.skipif(os.name == "nt",
                    reason="SIGTERM/SIGKILL escalation is POSIX-only")
def test_stop_case_escalates_to_sigkill_for_a_term_ignoring_run(tmp_path,
                                                                monkeypatch):
    # The final rung: a child that ignores SIGTERM must still die to the
    # SIGKILL escalation (harvest kept SIGKILL as the last rung).
    case_dir = _make_case(tmp_path)
    script = tmp_path / "term_ignoring_allrun.py"
    script.write_text(
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"with open('log.icoFoam', 'w') as fh:\n"
        f"    fh.write({INCOMPLETE_LOG!r})\n"
        "time.sleep(30)\n"
    )
    monkeypatch.setattr(mechanics, "_allrun_argv",
                        lambda case_dir: [sys.executable, str(script)])
    monkeypatch.setattr(mechanics, "_KILL_ESCALATION_SECONDS", 0.5)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    _wait_for_file(Path(case_dir) / "log.icoFoam")

    stop = mechanics.stop_case(start.run_id, grace_seconds=0,
                               run_directory=str(tmp_path))

    assert stop.method == "killed"
    assert stop.status == "debugging"
    assert not mechanics._pid_alive(start.pid)


def test_stop_case_on_a_run_that_just_exited_stamps_normally(tmp_path, fake_allrun,
                                                             isolated_registry):
    # Idempotence: the run exited on its own between the caller's poll and
    # the stop call — stamped through the normal completion path, reported
    # as such, and NOT marked as a deliberate stop (nothing was stopped).
    case_dir = _make_case(tmp_path)
    fake_allrun(GOOD_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    _cue_exit(case_dir)
    isolated_registry[start.run_id].process.wait(timeout=20)

    stop = mechanics.stop_case(start.run_id, run_directory=str(tmp_path))

    assert stop.method == "already_exited"
    assert stop.note is None
    assert (stop.status, stop.result) == ("done", "converged")
    row = _row(tmp_path, "cavity")
    assert (row["status"], row["result"]) == ("done", "converged")
    assert "stop_case" not in row["notes"]           # no deliberate-stop note
    assert not (Path(case_dir) / "Allrun.pid").exists()


def test_stop_case_unknown_run_id_is_a_typed_error(tmp_path):
    _make_case(tmp_path)  # ledger exists, but not this id

    with pytest.raises(mechanics.BackgroundRunError, match="0042"):
        mechanics.stop_case("0042", run_directory=str(tmp_path))


def test_stop_case_on_a_run_that_never_ran_is_a_typed_error(tmp_path):
    _make_case(tmp_path)  # planned row: no process, no pidfile

    with pytest.raises(mechanics.BackgroundRunError, match="no live"):
        mechanics.stop_case("0001", run_directory=str(tmp_path))

    assert _row(tmp_path, "cavity")["status"] == "planned"  # row untouched


def test_stop_case_preserves_an_existing_note(tmp_path, graceful_allrun):
    # The sanctioned note write APPENDS: a hand-written note survives.
    case_dir = _make_graceful_case(tmp_path)
    graceful_allrun(GOOD_LOG)
    start = mechanics.start_case(case_dir, run_directory=str(tmp_path))
    _wait_for_file(Path(case_dir) / "log.icoFoam")
    mechanics.set_run_note(start.run_id, note="baseline mesh study",
                           run_directory=str(tmp_path))

    stop = mechanics.stop_case(start.run_id, grace_seconds=15.0,
                               run_directory=str(tmp_path))

    assert stop.note == ("baseline mesh study; "
                         "stopped deliberately via stop_case (graceful)")
    assert _row(tmp_path, "cavity")["notes"] == stop.note


# ---------------------------------------------------------------------------
# The pure stop helpers, exercised on temp files
# ---------------------------------------------------------------------------

def test_touch_past_mtime_gate_sets_a_future_mtime(tmp_path):
    # The gate: fileMonitor needs newTime > mtime-at-last-read + 10 s (the
    # fileModificationSkew default), so the touch must land safely past it.
    target = tmp_path / "controlDict"
    target.write_text("stopAt writeNow;")

    before = time.time()
    mechanics._touch_past_mtime_gate(str(target))

    assert target.stat().st_mtime > before + 10.0


def test_set_control_dict_entry_rewrites_in_place_and_appends(tmp_path):
    control_dict = tmp_path / "controlDict"
    control_dict.write_text(
        "application     icoFoam;\nstopAt          endTime;\nendTime  0.5;\n")

    # An existing entry is rewritten in place, everything else untouched.
    mechanics._set_control_dict_entry(str(control_dict), "stopAt", "writeNow")
    text = control_dict.read_text()
    assert re.search(r"^stopAt\s+writeNow;$", text, re.MULTILINE)
    assert text.count("stopAt") == 1
    assert "application     icoFoam;" in text and "endTime  0.5;" in text

    # A missing entry is appended (stopAt has a compiled default and may be
    # absent from generated dicts).
    control_dict.write_text("application     icoFoam;\n")
    mechanics._set_control_dict_entry(str(control_dict), "stopAt", "writeNow")
    assert re.search(r"^stopAt\s+writeNow;$", control_dict.read_text(),
                     re.MULTILINE)


def test_run_time_modifiable_reads_the_switch(tmp_path):
    case_dir = tmp_path / "case"
    (case_dir / "system").mkdir(parents=True)
    control_dict = case_dir / "system" / "controlDict"

    control_dict.write_text("runTimeModifiable true;\n")
    assert mechanics._run_time_modifiable(str(case_dir)) is True
    control_dict.write_text("runTimeModifiable yes;\n")
    assert mechanics._run_time_modifiable(str(case_dir)) is True
    control_dict.write_text("runTimeModifiable false;\n")
    assert mechanics._run_time_modifiable(str(case_dir)) is False
    # ABSENT entry = the compiled default = false (harvest #73, Time.C:363).
    control_dict.write_text("application icoFoam;\n")
    assert mechanics._run_time_modifiable(str(case_dir)) is False
    # No controlDict at all: graceful is impossible, never an exception.
    control_dict.unlink()
    assert mechanics._run_time_modifiable(str(case_dir)) is False
