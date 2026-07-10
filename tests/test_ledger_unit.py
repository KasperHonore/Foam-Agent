"""Unit tests for the run ledger (src/ledger.py) at the mechanics seam.

The ledger is written by the MCP server as a mechanical side effect of the
run lifecycle; these tests drive it through mechanics.resolve_case_dir
against a temporary runs directory and assert on the ledger file itself —
the externally observable contract. Key-free, no OpenFOAM, no server.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

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
        rows.append(dict(zip(COLUMNS, cells)))
    return rows


# ---------------------------------------------------------------------------
# Allocation: a resolved case gets a planned row
# ---------------------------------------------------------------------------

def test_resolve_creates_ledger_with_planned_row(tmp_path):
    case_dir = mechanics.resolve_case_dir("cavity_re1000", run_directory=str(tmp_path))
    assert case_dir == str(tmp_path / "cavity_re1000")

    text = (tmp_path / "ledger.md").read_text(encoding="utf-8")
    assert "# Foam-Agent run ledger" in text
    assert "Managed by the foamagent MCP server" in text
    assert "Only the Notes column is yours to edit by hand" in text

    states = (tmp_path / "run-states.yml").read_text(encoding="utf-8")
    for value in ("planned:", "running:", "debugging:", "done:", "archived:",
                  "pending:", "converged:", "diverged:", "abandoned:"):
        assert value in states

    rows = _rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "0001"
    assert row["case"] == "cavity_re1000"
    assert row["created"] == date.today().isoformat()
    assert row["status"] == "planned"
    assert row["result"] == "pending"
    assert row["solver"] == "-"
    assert row["mesh"] == "-"
    assert row["key_result"] == "-"
    assert row["notes"] == ""


def test_ids_increment_zero_padded_newest_first(tmp_path):
    mechanics.resolve_case_dir("first", run_directory=str(tmp_path))
    mechanics.resolve_case_dir("second", run_directory=str(tmp_path))

    rows = _rows(tmp_path)
    assert [r["id"] for r in rows] == ["0002", "0001"]  # newest on top
    assert rows[0]["case"] == "second"
    assert rows[1]["case"] == "first"


def test_re_resolving_same_case_is_idempotent(tmp_path):
    a = mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))
    b = mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))

    assert a == b
    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["id"] == "0001"


def test_shared_run_directory_namespaces_projects(tmp_path):
    """Global mode: projects share one runs root and namespace their case
    keys; one ledger at the root distinguishes them (spec #28)."""
    shared = tmp_path / "central"
    a = mechanics.resolve_case_dir("projA/cylinder", run_directory=str(shared))
    b = mechanics.resolve_case_dir("projB/cylinder", run_directory=str(shared))

    assert a != b
    rows = _rows(shared)
    assert len(rows) == 2  # one ledger at the shared root, no fragmenting
    assert {r["case"] for r in rows} == {"projA/cylinder", "projB/cylinder"}
    assert not (tmp_path / "central" / "projA" / "ledger.md").exists()


def test_out_of_tree_explicit_case_dir_is_not_tracked(tmp_path, monkeypatch):
    monkeypatch.setattr(mechanics, "RUNS_DIR", tmp_path / "runs")
    outside = tmp_path / "elsewhere" / "case"

    resolved = mechanics.resolve_case_dir("x", case_dir=str(outside))

    assert resolved == str(outside)
    assert not (tmp_path / "runs" / "ledger.md").exists()
    assert not (tmp_path / "elsewhere" / "ledger.md").exists()


def test_explicit_case_dir_reuses_existing_row(tmp_path):
    first = mechanics.resolve_case_dir("cavity", run_directory=str(tmp_path))
    again = mechanics.resolve_case_dir("ignored", case_dir=first, run_directory=str(tmp_path))

    assert again == first
    assert len(_rows(tmp_path)) == 1


def test_hand_written_note_survives_new_allocations(tmp_path):
    mechanics.resolve_case_dir("first", run_directory=str(tmp_path))
    ledger_file = tmp_path / "ledger.md"
    text = ledger_file.read_text(encoding="utf-8")
    assert "| - |  |" in text
    ledger_file.write_text(
        text.replace("| - |  |", "| - | inlet BC typo, fixed |"), encoding="utf-8"
    )

    mechanics.resolve_case_dir("second", run_directory=str(tmp_path))

    by_case = {r["case"]: r for r in _rows(tmp_path)}
    assert by_case["first"]["notes"] == "inlet BC typo, fixed"
    assert by_case["second"]["notes"] == ""


def test_concurrent_resolutions_allocate_unique_ids(tmp_path):
    names = [f"case_{i:02d}" for i in range(12)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(
            lambda n: mechanics.resolve_case_dir(n, run_directory=str(tmp_path)), names
        ))

    rows = _rows(tmp_path)
    assert len(rows) == 12
    assert sorted(r["id"] for r in rows) == [f"{i:04d}" for i in range(1, 13)]
    assert {r["case"] for r in rows} == set(names)
