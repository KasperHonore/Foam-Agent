"""Unit tests for the run-lifecycle ledger transitions (issue #31).

Executing a case follows its ledger row: planned/debugging -> running when
the run starts, then done plus a Result stamped from the convergence
parser's verdict on success (issue #40), or back to debugging (Result
pending) on failure. SLURM submissions move through the same states. These tests drive the mechanics layer against a temporary
runs directory with fake run scripts and a stubbed subprocess boundary
(prior art: test_run_sourced.py) and assert on the ledger file itself —
the externally observable contract. Key-free, no OpenFOAM, no server.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import mechanics  # noqa: E402

COLUMNS = ["id", "case", "created", "solver", "mesh", "status", "result", "key_result", "notes"]

# Fake case contents: enough for the deterministic Solver/Mesh inspection.
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
# exception"/"sigFpe" words must never read as a blow-up (shakedown regression:
# the v1 verdict stamped every successful run diverged; permanent policy since).
SIGFPE_BANNER = "sigFpe : Enabling floating point exception trapping (FOAM_SIGFPE).\n"

# Parser-realistic fake solver logs: genuine banner plus minimal real v10
# residual/time lines (line formats cribbed from the captured fixtures under
# tests/fixtures/convergence/), so parse_solver_log produces the intended
# verdicts. GOOD_LOG carries realistic converged residual lines — the
# converged rule is "completed AND every field's last final residual < 1e-4",
# which a log with zero residual lines would pass vacuously.
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
"""  # no End marker — the run stopped early

DIVERGING_LOG = SIGFPE_BANNER + """\
Time = 0.01s

Courant Number mean: 45.8817 max: 1250.73
smoothSolver:  Solving for Ux, Initial residual = 2.51004e+08, Final residual = 9.23117e+07, No Iterations 1000
End
"""  # completed but blew up: residual explosion and Courant blow-up

UNCONVERGED_LOG = SIGFPE_BANNER + """\
Time = 0.5s

Courant Number mean: 0.0976825 max: 0.585607
DICPCG:  Solving for p, Initial residual = 0.428925, Final residual = 0.0103739, No Iterations 22
End
"""  # completed cleanly, but p's last final residual is above the 1e-4 threshold


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


@pytest.fixture
def fake_solver(monkeypatch):
    """Stub the subprocess boundary with a fake run that plants log files.

    Returns an installer: give it the runs root, the case key and the log
    files the 'solver' should leave behind. The stub also captures the
    case's ledger row while the run is in flight, so tests can observe the
    running state.
    """
    def _install(runs_root, case, logs):
        seen = {}

        def _fake_run_sourced(command, cwd, timeout):
            rows = _rows(Path(runs_root)) if (Path(runs_root) / "ledger.md").exists() else []
            seen["mid_run"] = next((r for r in rows if r["case"] == case), None)
            for name, content in logs.items():
                (Path(cwd) / name).write_text(content)
            return 0, "", "", False

        monkeypatch.setattr(mechanics, "_run_sourced", _fake_run_sourced)
        return seen

    return _install


# ---------------------------------------------------------------------------
# Local runs: planned -> running -> done/debugging
# ---------------------------------------------------------------------------

def test_run_start_flips_row_to_running(tmp_path, fake_solver):
    case_dir = _make_case(tmp_path)
    seen = fake_solver(tmp_path, "cavity", {"log.icoFoam": GOOD_LOG})

    mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(tmp_path))

    assert seen["mid_run"] is not None
    assert seen["mid_run"]["status"] == "running"
    assert seen["mid_run"]["result"] == "pending"


def test_successful_run_stamps_done_and_converged(tmp_path, fake_solver):
    case_dir = _make_case(tmp_path)
    fake_solver(tmp_path, "cavity", {"log.blockMesh": GOOD_LOG, "log.icoFoam": GOOD_LOG})

    errors = mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(tmp_path))

    assert errors == []
    row = _row(tmp_path, "cavity")
    assert row["status"] == "done"
    assert row["result"] == "converged"


def test_completed_run_that_blew_up_stamps_diverged(tmp_path, fake_solver):
    case_dir = _make_case(tmp_path)
    fake_solver(tmp_path, "cavity", {"log.icoFoam": DIVERGING_LOG})

    errors = mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(tmp_path))

    assert errors == []  # ran to completion — the verdict is what flags it
    row = _row(tmp_path, "cavity")
    assert row["status"] == "done"
    assert row["result"] == "diverged"


def test_completed_run_with_residuals_above_threshold_stamps_diverged(tmp_path, fake_solver):
    # The parser calls this flavor 'incomplete' (completed, but a final
    # residual is not under the converged threshold); the ledger maps every
    # non-converged verdict to diverged — it must never overstate trust.
    case_dir = _make_case(tmp_path)
    fake_solver(tmp_path, "cavity", {"log.icoFoam": UNCONVERGED_LOG})

    errors = mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(tmp_path))

    assert errors == []
    row = _row(tmp_path, "cavity")
    assert row["status"] == "done"
    assert row["result"] == "diverged"


def test_failed_run_flips_to_debugging_result_pending(tmp_path, fake_solver):
    case_dir = _make_case(tmp_path)
    fake_solver(tmp_path, "cavity", {"log.icoFoam": INCOMPLETE_LOG})

    errors = mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(tmp_path))

    assert errors  # the solver never reached End
    row = _row(tmp_path, "cavity")
    assert row["status"] == "debugging"
    assert row["result"] == "pending"


def test_rerun_updates_existing_row_and_preserves_created(tmp_path, fake_solver):
    case_dir = _make_case(tmp_path)
    fake_solver(tmp_path, "cavity", {"log.icoFoam": INCOMPLETE_LOG})
    mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(tmp_path))

    # Age the row so preservation is distinguishable from restamping today.
    ledger_file = tmp_path / "ledger.md"
    text = ledger_file.read_text(encoding="utf-8")
    ledger_file.write_text(text.replace(_row(tmp_path, "cavity")["created"], "2001-01-01"),
                           encoding="utf-8")

    fake_solver(tmp_path, "cavity", {"log.icoFoam": GOOD_LOG})
    errors = mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(tmp_path))

    assert errors == []
    rows = _rows(tmp_path)
    assert len(rows) == 1  # no duplicate row for the re-run
    assert rows[0]["id"] == "0001"
    assert rows[0]["created"] == "2001-01-01"
    assert rows[0]["status"] == "done"
    assert rows[0]["result"] == "converged"


# ---------------------------------------------------------------------------
# Solver/Mesh stamping: deterministic best-effort inspection at run time
# ---------------------------------------------------------------------------

def test_run_stamps_solver_and_mesh_from_case_contents(tmp_path, fake_solver):
    case_dir = _make_case(tmp_path)
    fake_solver(tmp_path, "cavity", {"log.icoFoam": GOOD_LOG})

    mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(tmp_path))

    row = _row(tmp_path, "cavity")
    assert row["solver"] == "icoFoam"  # controlDict application entry
    assert row["mesh"] == "blockMesh"  # mesh tool mentioned in Allrun


def test_solver_and_mesh_placeholder_when_undeterminable(tmp_path, fake_solver):
    case_dir = mechanics.resolve_case_dir("bare", run_directory=str(tmp_path))
    Path(case_dir).mkdir(parents=True)
    # Allrun without any known mesh tool, and no controlDict at all.
    (Path(case_dir) / "Allrun").write_text("#!/bin/sh\nrunApplication icoFoam\n")
    fake_solver(tmp_path, "bare", {"log.icoFoam": GOOD_LOG})

    mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(tmp_path))

    row = _row(tmp_path, "bare")
    assert row["solver"] == "-"
    assert row["mesh"] == "-"


# ---------------------------------------------------------------------------
# Direct tool usage: server-owned writes hold without resolve_case_dir
# ---------------------------------------------------------------------------

def test_running_an_unresolved_case_adopts_a_row(tmp_path, fake_solver):
    # No resolve_case_dir: the case dir appeared under the runs root by
    # other means, and run_case is called on it directly.
    case_dir = tmp_path / "adopted"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "system" / "controlDict").write_text(CONTROL_DICT)
    (case_dir / "Allrun").write_text(ALLRUN)
    fake_solver(tmp_path, "adopted", {"log.icoFoam": GOOD_LOG})

    mechanics.run_allrun_and_collect_errors(str(case_dir), run_directory=str(tmp_path))

    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["id"] == "0001"
    assert rows[0]["case"] == "adopted"
    assert rows[0]["status"] == "done"
    assert rows[0]["result"] == "converged"


def test_out_of_tree_case_dir_is_not_tracked(tmp_path, fake_solver):
    runs_root = tmp_path / "runs"
    case_dir = tmp_path / "elsewhere" / "case"
    case_dir.mkdir(parents=True)
    (case_dir / "Allrun").write_text(ALLRUN)
    fake_solver(runs_root, "case", {"log.icoFoam": GOOD_LOG})

    errors = mechanics.run_allrun_and_collect_errors(str(case_dir), run_directory=str(runs_root))

    assert errors == []  # the run itself is unaffected
    assert not (runs_root / "ledger.md").exists()
    assert not (case_dir.parent / "ledger.md").exists()


# ---------------------------------------------------------------------------
# SLURM runs: same states, driven by the submit/status pair
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_slurm(monkeypatch):
    """Stub the sbatch/squeue subprocess boundary.

    Returns the fake queue: {job_id: squeue state}. A job absent from it
    has left the queue (squeue prints nothing -> COMPLETED).
    """
    monkeypatch.setattr(mechanics, "_SLURM_JOBS", {}, raising=False)  # isolate module state
    queue = {}

    def _fake_run(cmd, **kwargs):
        if cmd[0] == "sbatch":
            return subprocess.CompletedProcess(cmd, 0, stdout="Submitted batch job 4242\n", stderr="")
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout=queue.get(cmd[2], ""), stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(mechanics.subprocess, "run", _fake_run)
    return queue


def test_slurm_submission_flips_row_to_running(tmp_path, fake_slurm):
    case_dir = _make_case(tmp_path)
    script = Path(case_dir) / "submit_job.slurm"
    script.write_text("#!/bin/sh\n./Allrun\n")

    job_id, submitted, error = mechanics.submit_slurm_job(
        str(script), case_dir=case_dir, run_directory=str(tmp_path)
    )

    assert (job_id, submitted, error) == ("4242", True, "")
    row = _row(tmp_path, "cavity")
    assert row["status"] == "running"
    assert row["result"] == "pending"
    assert row["solver"] == "icoFoam"  # stamped at submission, same inspection
    assert row["mesh"] == "blockMesh"


def test_slurm_completion_observed_via_status_stamps_done(tmp_path, fake_slurm):
    case_dir = _make_case(tmp_path)
    script = Path(case_dir) / "submit_job.slurm"
    script.write_text("#!/bin/sh\n./Allrun\n")
    job_id, _, _ = mechanics.submit_slurm_job(
        str(script), case_dir=case_dir, run_directory=str(tmp_path)
    )
    (Path(case_dir) / "log.icoFoam").write_text(GOOD_LOG)  # the job left clean logs

    fake_slurm[job_id] = "RUNNING"
    status, ok, _ = mechanics.check_job_status(job_id)
    assert (status, ok) == ("RUNNING", True)
    assert _row(tmp_path, "cavity")["status"] == "running"  # still in flight

    del fake_slurm[job_id]  # job left the queue -> COMPLETED
    status, ok, _ = mechanics.check_job_status(job_id)
    assert (status, ok) == ("COMPLETED", True)
    row = _row(tmp_path, "cavity")
    assert row["status"] == "done"
    assert row["result"] == "converged"


def test_slurm_completion_with_unconverged_residuals_stamps_diverged(tmp_path, fake_slurm):
    # SLURM completion stamps through the same parser-backed mapping as a
    # local run: completed-but-above-threshold is not converged.
    case_dir = _make_case(tmp_path)
    script = Path(case_dir) / "submit_job.slurm"
    script.write_text("#!/bin/sh\n./Allrun\n")
    job_id, _, _ = mechanics.submit_slurm_job(
        str(script), case_dir=case_dir, run_directory=str(tmp_path)
    )
    (Path(case_dir) / "log.icoFoam").write_text(UNCONVERGED_LOG)

    status, ok, _ = mechanics.check_job_status(job_id)  # left the queue

    assert (status, ok) == ("COMPLETED", True)
    row = _row(tmp_path, "cavity")
    assert row["status"] == "done"
    assert row["result"] == "diverged"


def test_slurm_job_that_left_broken_logs_flips_to_debugging(tmp_path, fake_slurm):
    case_dir = _make_case(tmp_path)
    script = Path(case_dir) / "submit_job.slurm"
    script.write_text("#!/bin/sh\n./Allrun\n")
    job_id, _, _ = mechanics.submit_slurm_job(
        str(script), case_dir=case_dir, run_directory=str(tmp_path)
    )
    (Path(case_dir) / "log.icoFoam").write_text(INCOMPLETE_LOG)  # never reached End

    status, _, _ = mechanics.check_job_status(job_id)  # left the queue

    assert status == "COMPLETED"
    row = _row(tmp_path, "cavity")
    assert row["status"] == "debugging"
    assert row["result"] == "pending"


def test_slurm_scheduler_failure_state_flips_to_debugging(tmp_path, fake_slurm):
    case_dir = _make_case(tmp_path)
    script = Path(case_dir) / "submit_job.slurm"
    script.write_text("#!/bin/sh\n./Allrun\n")
    job_id, _, _ = mechanics.submit_slurm_job(
        str(script), case_dir=case_dir, run_directory=str(tmp_path)
    )

    fake_slurm[job_id] = "FAILED"
    status, _, _ = mechanics.check_job_status(job_id)

    assert status == "FAILED"
    row = _row(tmp_path, "cavity")
    assert row["status"] == "debugging"
    assert row["result"] == "pending"
