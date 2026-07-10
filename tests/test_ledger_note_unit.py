"""Unit tests for set_run_note, the one skill-side ledger write (issue #32).

set_run_note lets a skill annotate a run (replace the Notes cell) and
archive/unarchive it — nothing else. Rows are keyed by their zero-padded
ledger ID, the handle a skill reads off the table. These tests drive the
mechanics layer against a temporary runs directory and assert on the ledger
file itself — the externally observable contract. Key-free, no OpenFOAM,
no server.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import mechanics  # noqa: E402

COLUMNS = ["id", "case", "created", "solver", "mesh", "status", "result", "key_result", "notes"]


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
        if len(cells) > len(COLUMNS):  # note containing pipes
            cells[len(COLUMNS) - 1] = " | ".join(cells[len(COLUMNS) - 1:])
            cells = cells[:len(COLUMNS)]
        rows.append(dict(zip(COLUMNS, cells)))
    return rows


def _row(runs_root: Path, case: str) -> dict:
    return next(r for r in _rows(runs_root) if r["case"] == case)


def _make_done_run(runs_root, monkeypatch, name="cavity") -> None:
    """Drive a case through the real lifecycle to done/converged (prior art:
    test_ledger_lifecycle.py's fake solver at the subprocess boundary)."""
    case_dir = mechanics.resolve_case_dir(name, run_directory=str(runs_root))
    Path(case_dir).mkdir(parents=True)
    (Path(case_dir) / "Allrun").write_text("#!/bin/sh\nrunApplication icoFoam\n")

    def _fake_run_sourced(command, cwd, timeout):
        (Path(cwd) / "log.icoFoam").write_text("Time = 0.5\nEnd\n")
        return 0, "", "", False

    monkeypatch.setattr(mechanics, "_run_sourced", _fake_run_sourced)
    errors = mechanics.run_allrun_and_collect_errors(case_dir, run_directory=str(runs_root))
    assert errors == []


# ---------------------------------------------------------------------------
# Notes: the annotation half of the sanctioned surface
# ---------------------------------------------------------------------------

def test_set_note_replaces_notes_cell_all_other_cells_untouched(tmp_path):
    mechanics.resolve_case_dir("first", run_directory=str(tmp_path))
    mechanics.resolve_case_dir("second", run_directory=str(tmp_path))
    before = {r["case"]: r for r in _rows(tmp_path)}

    mechanics.set_run_note("0001", note="inlet BC typo, fixed", run_directory=str(tmp_path))

    after = {r["case"]: r for r in _rows(tmp_path)}
    assert after["first"]["notes"] == "inlet BC typo, fixed"
    assert {k: v for k, v in after["first"].items() if k != "notes"} == \
           {k: v for k, v in before["first"].items() if k != "notes"}
    assert after["second"] == before["second"]  # the other row is untouched


def test_set_note_replaces_a_previous_note(tmp_path):
    mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))
    # A hand-written note (the one cell a human may edit) is fair game too.
    ledger_file = tmp_path / "ledger.md"
    text = ledger_file.read_text(encoding="utf-8")
    ledger_file.write_text(text.replace("| - |  |", "| - | old note |"), encoding="utf-8")

    mechanics.set_run_note("0001", note="new note", run_directory=str(tmp_path))

    assert _row(tmp_path, "cavity")["notes"] == "new note"


def test_set_note_with_empty_string_clears_the_cell(tmp_path):
    mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))
    mechanics.set_run_note("0001", note="scratch this", run_directory=str(tmp_path))

    mechanics.set_run_note("0001", note="", run_directory=str(tmp_path))

    assert _row(tmp_path, "cavity")["notes"] == ""


# ---------------------------------------------------------------------------
# Archiving: the only Status transition a skill can drive
# ---------------------------------------------------------------------------

def test_archiving_a_pending_row_stamps_result_abandoned(tmp_path):
    mechanics.resolve_case_dir("stale", run_directory=str(tmp_path))
    mechanics.set_run_note("0001", note="keep this", run_directory=str(tmp_path))

    mechanics.set_run_note("0001", archive=True, run_directory=str(tmp_path))

    row = _row(tmp_path, "stale")
    assert row["status"] == "archived"
    assert row["result"] == "abandoned"  # owner stopped pursuing this
    assert row["notes"] == "keep this"  # archive-only call leaves the note alone


def test_archiving_a_done_run_keeps_its_verdict(tmp_path, monkeypatch):
    _make_done_run(tmp_path, monkeypatch)
    assert _row(tmp_path, "cavity")["result"] == "converged"

    mechanics.set_run_note("0001", archive=True, run_directory=str(tmp_path))

    row = _row(tmp_path, "cavity")
    assert row["status"] == "archived"
    assert row["result"] == "converged"  # the verdict is history, not status


def test_unarchiving_restores_done_and_leaves_result_alone(tmp_path):
    mechanics.resolve_case_dir("stale", run_directory=str(tmp_path))
    mechanics.set_run_note("0001", archive=True, run_directory=str(tmp_path))

    mechanics.set_run_note("0001", archive=False, run_directory=str(tmp_path))

    row = _row(tmp_path, "stale")
    assert row["status"] == "done"
    assert row["result"] == "abandoned"  # stays until a new run overwrites it


# ---------------------------------------------------------------------------
# Illegal writes: typed error, ledger untouched
# ---------------------------------------------------------------------------

def test_unknown_id_raises_and_leaves_ledger_unchanged(tmp_path):
    mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))
    before = (tmp_path / "ledger.md").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="0042"):
        mechanics.set_run_note("0042", note="ghost", run_directory=str(tmp_path))

    assert (tmp_path / "ledger.md").read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# Notes that fight the table format
# ---------------------------------------------------------------------------

def test_note_with_pipes_round_trips(tmp_path):
    mechanics.resolve_case_dir("first", run_directory=str(tmp_path))
    mechanics.resolve_case_dir("second", run_directory=str(tmp_path))

    mechanics.set_run_note("0001", note="dp = 12 Pa | rerun with finer mesh",
                           run_directory=str(tmp_path))

    assert _row(tmp_path, "first")["notes"] == "dp = 12 Pa | rerun with finer mesh"
    assert len(_rows(tmp_path)) == 2  # the table did not shear
    assert _row(tmp_path, "second")["notes"] == ""


def test_note_pipes_and_newlines_normalize_to_a_stable_cell(tmp_path):
    # Raw newlines would break the row; unspaced pipes would read back
    # changed. The write normalizes both to the parser's own ' | ' form,
    # so what is stored is exactly what reads back.
    mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))

    mechanics.set_run_note("0001", note="typo in 0/U|fixed\nrerun",
                           run_directory=str(tmp_path))

    stored = _row(tmp_path, "cavity")["notes"]
    assert stored == "typo in 0/U | fixed rerun"
    # Round-trip is a fixpoint: re-setting what was read changes nothing.
    mechanics.set_run_note("0001", note=stored, run_directory=str(tmp_path))
    assert _row(tmp_path, "cavity")["notes"] == stored


def test_unarchiving_a_row_that_is_not_archived_raises_unchanged(tmp_path):
    # archive=False on a planned/running/debugging row would let a skill set
    # Status done at will — outside the sanctioned surface, so it refuses.
    mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))
    before = (tmp_path / "ledger.md").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="planned"):
        mechanics.set_run_note("0001", archive=False, run_directory=str(tmp_path))

    assert (tmp_path / "ledger.md").read_text(encoding="utf-8") == before


def test_run_id_without_zero_padding_is_accepted(tmp_path):
    mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))

    mechanics.set_run_note("1", note="found it", run_directory=str(tmp_path))

    assert _row(tmp_path, "cavity")["notes"] == "found it"


def test_failed_archive_does_not_apply_a_note_passed_alongside(tmp_path):
    # One call is one write: if any part is illegal, none of it lands.
    mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))
    before = (tmp_path / "ledger.md").read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        mechanics.set_run_note("0001", note="half-applied?", archive=False,
                               run_directory=str(tmp_path))

    assert (tmp_path / "ledger.md").read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# Concurrency: notes against two rows never corrupt the table
# ---------------------------------------------------------------------------

def test_concurrent_note_setting_never_corrupts_the_table(tmp_path):
    mechanics.resolve_case_dir("alpha", run_directory=str(tmp_path))
    mechanics.resolve_case_dir("beta", run_directory=str(tmp_path))

    def annotate(i):
        run_id = "0001" if i % 2 == 0 else "0002"
        mechanics.set_run_note(run_id, note=f"note {i}", run_directory=str(tmp_path))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(annotate, range(24)))

    rows = _rows(tmp_path)
    assert len(rows) == 2  # no shorn, duplicated or vanished rows
    by_id = {r["id"]: r for r in rows}
    assert by_id["0001"]["case"] == "alpha"
    assert by_id["0002"]["case"] == "beta"
    for run_id, row in by_id.items():
        own = 0 if run_id == "0001" else 1
        assert row["notes"] in {f"note {i}" for i in range(own, 24, 2)}
        assert row["status"] == "planned"
        assert row["result"] == "pending"
