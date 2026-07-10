"""CLI tests for scripts/ledger_check.py over fixture ledgers (issue #33).

The seam (confirmed in the parent spec #28): the script is tested as a CLI —
subprocess invocations over fixture runs directories built in tmp_path,
asserting stdout, exit codes, and resulting file contents. One fixture per
defect class (orphan, vanished, duplicate ID, illegal value) plus a clean
one. Key-free, stdlib-only, no OpenFOAM, no server.

Fixtures are built with src/ledger.py itself where a well-formed ledger is
needed (the same producer the MCP server uses), then corrupted by editing
the file text — the way real drift happens.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import ledger  # noqa: E402

SCRIPT = REPO / "scripts" / "ledger_check.py"


def run_check(runs_root, *extra):
    """Invoke the CLI as a user would; enforce ASCII-only output (issue #18)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--runs-dir", str(runs_root), *extra],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result.stdout.encode("ascii")  # raises on mojibake-prone output
    result.stderr.encode("ascii")
    return result


def make_case(runs_root, name, solver=None, mesh_dict=None, log_text=None,
              err_text=None, time_dir=None):
    """A case directory as a run would leave it, with optional artefacts."""
    case = Path(runs_root) / name
    (case / "system").mkdir(parents=True)
    if solver is not None:
        (case / "system" / "controlDict").write_text(
            f"application     {solver};\n", encoding="utf-8"
        )
    if mesh_dict is not None:
        (case / "system" / mesh_dict).write_text("// mesh\n", encoding="utf-8")
    if log_text is not None:
        (case / f"log.{solver or 'run'}").write_text(log_text, encoding="utf-8")
    if err_text is not None:
        (case / "Allrun.err").write_text(err_text, encoding="utf-8")
    if time_dir is not None:
        (case / time_dir).mkdir()
    return case


def track(runs_root, name):
    """A ledgered case: directory plus its server-written row."""
    case = make_case(runs_root, name, solver="icoFoam")
    ledger.track_planned(str(runs_root), str(case))
    return case


def ledger_text(runs_root):
    return (Path(runs_root) / "ledger.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Clean ledger
# ---------------------------------------------------------------------------

def test_clean_ledger_exits_zero_and_writes_nothing(tmp_path):
    track(tmp_path, "cavity")
    before = ledger_text(tmp_path)

    result = run_check(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no issues" in result.stdout
    assert ledger_text(tmp_path) == before
    assert not list(tmp_path.glob("ledger.md.bak*"))


def test_clean_ledger_round_trips_byte_identical_through_fix(tmp_path):
    track(tmp_path, "cavity")
    before = (tmp_path / "ledger.md").read_bytes()

    result = run_check(tmp_path, "--fix")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "ledger.md").read_bytes() == before
    assert not list(tmp_path.glob("ledger.md.bak*"))
    assert not (tmp_path / "ledger.lock").exists()  # guard released


# ---------------------------------------------------------------------------
# Defect class: orphan case directories (also the pre-ledger migration path)
# ---------------------------------------------------------------------------

def test_orphan_dry_run_reports_proposed_row_and_writes_nothing(tmp_path):
    track(tmp_path, "cavity")
    make_case(tmp_path, "orphan_case", solver="pisoFoam",
              mesh_dict="blockMeshDict", log_text="Time = 1\nEnd\n")
    before = ledger_text(tmp_path)

    result = run_check(tmp_path)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "orphan_case" in result.stdout
    # the exact proposed change: id, inspected solver/mesh, inferred state
    assert "0002" in result.stdout
    assert "pisoFoam" in result.stdout
    assert "blockMesh" in result.stdout
    assert "done" in result.stdout
    assert "converged" in result.stdout
    assert "--fix" in result.stdout
    assert ledger_text(tmp_path) == before  # dry-run wrote nothing
    assert not list(tmp_path.glob("ledger.md.bak*"))


def test_fix_adopts_orphan_and_backs_up_ledger_first(tmp_path):
    track(tmp_path, "cavity")
    make_case(tmp_path, "orphan_case", solver="pisoFoam",
              mesh_dict="blockMeshDict", log_text="Time = 1\nEnd\n")
    before = ledger_text(tmp_path)

    result = run_check(tmp_path, "--fix")

    assert result.returncode == 1, result.stdout + result.stderr
    backups = list(tmp_path.glob("ledger.md.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == before  # pre-repair state
    by_case = {r.case: r for r in ledger.read_rows(str(tmp_path))}
    adopted = by_case["orphan_case"]
    assert adopted.id == "0002"
    assert adopted.solver == "pisoFoam"
    assert adopted.mesh == "blockMesh"
    assert adopted.status == "done"
    assert adopted.result == "converged"
    assert "adopted by ledger_check" in adopted.notes
    assert by_case["cavity"].status == "planned"  # existing row untouched
    # repaired: a second run is clean
    assert run_check(tmp_path).returncode == 0


def test_orphan_without_run_artefacts_is_adopted_as_planned(tmp_path):
    make_case(tmp_path, "untouched", solver="icoFoam")

    result = run_check(tmp_path, "--fix")

    assert result.returncode == 1, result.stdout + result.stderr
    row = ledger.read_rows(str(tmp_path))[0]
    assert (row.status, row.result) == ("planned", "pending")


def test_orphan_with_error_evidence_is_adopted_as_diverged(tmp_path):
    make_case(tmp_path, "blew_up", solver="icoFoam",
              log_text="FOAM FATAL ERROR\n", err_text="boom\n")

    result = run_check(tmp_path, "--fix")

    assert result.returncode == 1, result.stdout + result.stderr
    row = ledger.read_rows(str(tmp_path))[0]
    assert (row.status, row.result) == ("done", "diverged")


def test_pre_ledger_runs_directory_yields_complete_ledger(tmp_path):
    """The migration path: cases but no ledger.md at all (spec #28)."""
    make_case(tmp_path, "dam_break", solver="interFoam",
              mesh_dict="blockMeshDict", time_dir="0.5")
    make_case(tmp_path, "cavity", solver="icoFoam")
    proj = tmp_path / "projA"
    proj.mkdir()
    make_case(proj, "cylinder", solver="simpleFoam")

    result = run_check(tmp_path, "--fix")

    assert result.returncode == 1, result.stdout + result.stderr
    rows = ledger.read_rows(str(tmp_path))
    assert {r.case for r in rows} == {"dam_break", "cavity", "projA/cylinder"}
    assert sorted(r.id for r in rows) == ["0001", "0002", "0003"]
    assert (tmp_path / "run-states.yml").exists()  # vocabulary ships alongside
    assert run_check(tmp_path).returncode == 0  # complete: nothing left to adopt


# ---------------------------------------------------------------------------
# Defect class: rows whose case directories vanished
# ---------------------------------------------------------------------------

def test_vanished_directory_row_is_archived_with_note_never_deleted(tmp_path):
    import shutil

    case = track(tmp_path, "gone_case")
    shutil.rmtree(case)

    dry = run_check(tmp_path)
    assert dry.returncode == 1, dry.stdout + dry.stderr
    assert "gone_case" in dry.stdout
    assert "archive" in dry.stdout

    result = run_check(tmp_path, "--fix")
    assert result.returncode == 1, result.stdout + result.stderr
    rows = ledger.read_rows(str(tmp_path))
    assert len(rows) == 1  # archived, not deleted
    row = rows[0]
    assert row.case == "gone_case"
    assert row.status == "archived"
    assert row.result == "abandoned"  # pending is illegal once archived
    assert "directory missing" in row.notes
    assert run_check(tmp_path).returncode == 0  # repaired state is clean


def test_vanished_row_keeps_existing_verdict_and_note(tmp_path):
    import shutil

    case = track(tmp_path, "post_mortem")
    shutil.rmtree(case)
    text = ledger_text(tmp_path)
    text = text.replace("| planned | pending | - |  |",
                        "| done | diverged | - | inlet BC typo |")
    (tmp_path / "ledger.md").write_text(text, encoding="utf-8")

    result = run_check(tmp_path, "--fix")

    assert result.returncode == 1, result.stdout + result.stderr
    row = ledger.read_rows(str(tmp_path))[0]
    assert row.status == "archived"
    assert row.result == "diverged"  # verdict preserved
    assert "inlet BC typo" in row.notes  # hand note preserved
    assert "directory missing" in row.notes


def test_already_archived_row_with_missing_directory_is_not_an_issue(tmp_path):
    import shutil

    case = track(tmp_path, "old_run")
    shutil.rmtree(case)
    text = ledger_text(tmp_path).replace("| planned | pending |",
                                         "| archived | abandoned |")
    (tmp_path / "ledger.md").write_text(text, encoding="utf-8")

    result = run_check(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Defect class: duplicate IDs
# ---------------------------------------------------------------------------

def test_duplicate_ids_are_reassigned_older_row_keeps_its_id(tmp_path):
    track(tmp_path, "first")
    track(tmp_path, "second")
    text = ledger_text(tmp_path).replace("| 0002 | second |", "| 0001 | second |")
    (tmp_path / "ledger.md").write_text(text, encoding="utf-8")

    dry = run_check(tmp_path)
    assert dry.returncode == 1, dry.stdout + dry.stderr
    assert "0001" in dry.stdout and "0002" in dry.stdout

    result = run_check(tmp_path, "--fix")
    assert result.returncode == 1, result.stdout + result.stderr
    by_case = {r.case: r for r in ledger.read_rows(str(tmp_path))}
    assert len(by_case) == 2  # both rows kept
    assert by_case["first"].id == "0001"  # the older claimant keeps the id
    assert by_case["second"].id == "0002"  # the newer gets the next free one
    assert run_check(tmp_path).returncode == 0


def test_duplicate_id_repair_never_renumbers_an_innocent_row(tmp_path):
    """Reassigned IDs must not collide with IDs already in use elsewhere."""
    track(tmp_path, "a")
    track(tmp_path, "b")
    track(tmp_path, "c")
    text = ledger_text(tmp_path)
    text = text.replace("| 0002 | b |", "| 0001 | b |")
    text = text.replace("| 0003 | c |", "| 0002 | c |")
    (tmp_path / "ledger.md").write_text(text, encoding="utf-8")

    result = run_check(tmp_path, "--fix")

    assert result.returncode == 1, result.stdout + result.stderr
    by_case = {r.case: r for r in ledger.read_rows(str(tmp_path))}
    assert by_case["a"].id == "0001"  # older claimant keeps
    assert by_case["c"].id == "0002"  # innocent row untouched
    assert by_case["b"].id == "0003"  # the duplicate goes above all used ids
    assert run_check(tmp_path).returncode == 0


# ---------------------------------------------------------------------------
# Defect class: illegal Status/Result values
# ---------------------------------------------------------------------------

def test_illegal_status_is_repaired_from_case_inspection(tmp_path):
    case = track(tmp_path, "cavity")
    (case / "log.icoFoam").write_text("Time = 1\nEnd\n", encoding="utf-8")
    text = ledger_text(tmp_path).replace("| planned |", "| in-progress |")
    (tmp_path / "ledger.md").write_text(text, encoding="utf-8")

    dry = run_check(tmp_path)
    assert dry.returncode == 1, dry.stdout + dry.stderr
    assert "in-progress" in dry.stdout

    result = run_check(tmp_path, "--fix")
    assert result.returncode == 1, result.stdout + result.stderr
    row = ledger.read_rows(str(tmp_path))[0]
    assert (row.status, row.result) == ("done", "converged")  # log evidence
    assert run_check(tmp_path).returncode == 0


def test_illegal_result_is_repaired_conservatively(tmp_path):
    track(tmp_path, "cavity")  # no run artefacts
    text = ledger_text(tmp_path).replace("| pending |", "| great |")
    (tmp_path / "ledger.md").write_text(text, encoding="utf-8")

    result = run_check(tmp_path, "--fix")

    assert result.returncode == 1, result.stdout + result.stderr
    row = ledger.read_rows(str(tmp_path))[0]
    assert (row.status, row.result) == ("planned", "pending")


def test_pending_result_on_done_row_is_repaired(tmp_path):
    """run-states.yml: pending is only legal while planned/running/debugging."""
    track(tmp_path, "cavity")  # no run artefacts
    text = ledger_text(tmp_path).replace("| planned |", "| done |")
    (tmp_path / "ledger.md").write_text(text, encoding="utf-8")

    result = run_check(tmp_path, "--fix")

    assert result.returncode == 1, result.stdout + result.stderr
    row = ledger.read_rows(str(tmp_path))[0]
    assert row.status == "done"
    assert row.result == "abandoned"  # no evidence to claim a verdict from


# ---------------------------------------------------------------------------
# Cross-process guard (issue #33 comment): repair must not race another repair
# ---------------------------------------------------------------------------

def test_fix_aborts_when_ledger_lock_is_held(tmp_path):
    track(tmp_path, "cavity")
    make_case(tmp_path, "orphan_case", solver="icoFoam")
    (tmp_path / "ledger.lock").write_text("12345", encoding="utf-8")
    before = ledger_text(tmp_path)

    result = run_check(tmp_path, "--fix")

    assert result.returncode == 2, result.stdout + result.stderr
    assert "lock" in result.stdout
    assert ledger_text(tmp_path) == before  # nothing was written
    assert not list(tmp_path.glob("ledger.md.bak*"))
    assert (tmp_path / "ledger.lock").read_text(encoding="utf-8") == "12345"

    # dry-run reporting is unaffected by the lock
    dry = run_check(tmp_path)
    assert dry.returncode == 1, dry.stdout + dry.stderr
    assert "orphan_case" in dry.stdout
