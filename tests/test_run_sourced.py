"""Unit tests for the shared _run_sourced helper and its two callers.

These never spawn bash or OpenFOAM: subprocess.Popen (and the Unix-only
os.killpg/os.getpgid) are patched with fakes, so the tests exercise the pure
control flow — command composition, the process-group timeout path, and how
run_command / run_openfoam_command surface stdout/stderr — on any platform.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import mechanics  # noqa: E402


class _FakeProc:
    """Stand-in for a Popen object. Optionally times out on the first wait."""

    def __init__(self, out, err, returncode, timeout_first=False):
        self._out, self._err = out, err
        self.returncode = returncode
        self.pid = 4242
        self._timeout_first = timeout_first
        self._calls = 0

    def communicate(self, timeout=None):
        self._calls += 1
        if self._timeout_first and self._calls == 1:
            raise subprocess.TimeoutExpired(cmd="bash", timeout=timeout)
        return self._out, self._err


@pytest.fixture
def run_with(monkeypatch):
    """Install a fake process for Popen; return a dict capturing the Popen call."""
    monkeypatch.setattr(mechanics, "_openfoam_bashrc", lambda: "/fake/bashrc")
    # os.killpg / os.getpgid and signal.SIGKILL are Unix-only — add stand-ins so
    # the timeout path is exercisable on Windows too (the real runtime is Linux).
    monkeypatch.setattr(mechanics.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(mechanics.os, "killpg", lambda pgid, sig: None, raising=False)
    monkeypatch.setattr(mechanics.signal, "SIGKILL", 9, raising=False)

    captured = {}

    def _install(proc):
        def _popen(*args, **kwargs):
            captured["args"], captured["kwargs"] = args, kwargs
            return proc
        monkeypatch.setattr(mechanics.subprocess, "Popen", _popen)
        return captured

    return _install


# ---------------------------------------------------------------------------
# _run_sourced: composition + the (rc, out, err, timed_out) contract
# ---------------------------------------------------------------------------

def test_run_sourced_sources_bashrc_and_runs_in_new_group(run_with):
    cap = run_with(_FakeProc("o", "e", 0))
    rc, out, err, timed_out = mechanics._run_sourced("checkMesh", "/case", 5)
    assert (rc, out, err, timed_out) == (0, "o", "e", False)
    # command is prefixed with the sourced bashrc, run via bash -c
    assert cap["args"][0] == ["bash", "-c", "source /fake/bashrc && checkMesh"]
    assert cap["kwargs"]["cwd"] == "/case"
    assert cap["kwargs"]["start_new_session"] is True


def test_run_sourced_timeout_returns_minus_one_and_flag(run_with):
    run_with(_FakeProc("partial", "boom", -9, timeout_first=True))
    assert mechanics._run_sourced("icoFoam", "/case", 3) == (-1, "partial", "boom", True)


# ---------------------------------------------------------------------------
# run_openfoam_command: keeps its own timeout message
# ---------------------------------------------------------------------------

def test_run_openfoam_command_success(run_with):
    run_with(_FakeProc("OUT", "ERR", 0))
    assert mechanics.run_openfoam_command("/case", "checkMesh", timeout=5) == (0, "OUT", "ERR")


def test_run_openfoam_command_timeout_appends_message(run_with):
    run_with(_FakeProc("partial-out", "partial-err", -9, timeout_first=True))
    rc, out, err = mechanics.run_openfoam_command("/case", "icoFoam", timeout=7)
    assert rc == -1
    assert out == "partial-out"
    assert err == "partial-err\nCommand timed out after 7s"


# ---------------------------------------------------------------------------
# run_command: writes streams to files, prepends its own timeout message
# ---------------------------------------------------------------------------

def test_run_command_writes_streams(tmp_path, run_with):
    run_with(_FakeProc("solver stdout", "solver stderr", 0))
    script = tmp_path / "Allrun"
    script.write_text("#!/bin/bash\necho hi\n")
    out_file, err_file = tmp_path / "Allrun.out", tmp_path / "Allrun.err"

    mechanics.run_command(str(script), str(out_file), str(err_file), str(tmp_path), 5)

    assert out_file.read_text() == "solver stdout"
    assert err_file.read_text() == "solver stderr"


def test_run_command_timeout_prepends_message(tmp_path, run_with):
    run_with(_FakeProc("so", "se", -9, timeout_first=True))
    script = tmp_path / "Allrun"
    script.write_text("#!/bin/bash\n")
    out_file, err_file = tmp_path / "Allrun.out", tmp_path / "Allrun.err"

    mechanics.run_command(str(script), str(out_file), str(err_file), str(tmp_path), 3)

    out_txt, err_txt = out_file.read_text(), err_file.read_text()
    assert out_txt.startswith("OpenFOAM execution took too long.")
    assert out_txt.endswith("so")
    assert err_txt.startswith("OpenFOAM execution took too long.")
    assert err_txt.endswith("se")
