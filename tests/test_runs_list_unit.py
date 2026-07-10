"""CLI tests for the zero-token run lister (scripts/runs.py).

The lister is tested as a CLI over a populated fixture ledger — subprocess
invocations asserting stdout content and exit codes, per the testing decision
in the run-ledger spec (issue #28). Key-free, stdlib-only, no OpenFOAM, no
server: CI runs this with nothing but pytest installed.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "runs.py"

# A populated ledger in the exact on-disk format written by src/ledger.py
# (newest row first). The note on row 0001 deliberately contains non-ASCII:
# printing it must not crash a Windows console (issues #18/#20).
FIXTURE = """\
# Foam-Agent run ledger

<!-- Managed by the foamagent MCP server. Verify/repair: python scripts/ledger_check.py -->
<!-- Allowed Status/Result values: see run-states.yml. Only the Notes column is yours to edit by hand. -->
<!-- Lives at runs/ledger.md — gitignored, bind-mounted, survives container recreation (update contract). -->

| ID | Case | Created | Solver | Mesh | Status | Result | Key result | Notes |
|----|------|---------|--------|------|--------|--------|------------|-------|
| 0005 | damBreak | 2026-07-09 | interFoam | blockMesh | running | pending | - |  |
| 0004 | cavity_re1000 | 2026-07-08 | icoFoam | blockMesh | done | converged | Cd=1.234 |  |
| 0003 | cylinder | 2026-07-07 | pimpleFoam | snappyHexMesh | done | diverged | - | blew up at t=0.4 |
| 0002 | old_case | 2026-06-01 | icoFoam | blockMesh | archived | converged | - |  |
| 0001 | ancient | 2026-05-20 | icoFoam | blockMesh | archived | abandoned | - | café — retired |
"""


def _populate(runs_root: Path) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    ledger = runs_root / "ledger.md"
    ledger.write_text(FIXTURE, encoding="utf-8")
    return ledger


def _list(runs_root: Path, *args: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--runs-dir", str(runs_root), *args],
        capture_output=True,
        text=True,
        env=env,
    )


# ---------------------------------------------------------------------------
# Default invocation: aligned table, archived rows hidden
# ---------------------------------------------------------------------------

def test_default_lists_aligned_table_hiding_archived(tmp_path):
    _populate(tmp_path)

    result = _list(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    for case in ("damBreak", "cavity_re1000", "cylinder"):
        assert case in out
    for archived in ("old_case", "ancient"):
        assert archived not in out

    # Aligned table: every row's Status cell starts at the header's column.
    lines = [l for l in out.splitlines() if l.strip()]
    header = next(l for l in lines if "ID" in l and "Status" in l)
    col = header.index("Status")
    data = [l for l in lines if l.split() and l.split()[0].isdigit()]
    assert len(data) == 3
    assert data[0].split()[0] == "0005"  # ledger order preserved: newest first
    for line in data:
        status = line[col:].split()[0]
        assert status in ("running", "done")


def test_default_footer_counts_hidden_archived_rows(tmp_path):
    _populate(tmp_path)

    default = _list(tmp_path)
    with_all = _list(tmp_path, "--all")

    assert "2 archived" in default.stdout
    assert "--all" in default.stdout
    assert "hidden" not in with_all.stdout


def test_list_flag_is_an_alias_for_the_default(tmp_path):
    _populate(tmp_path)

    default = _list(tmp_path)
    explicit = _list(tmp_path, "--list")

    assert explicit.returncode == 0
    assert explicit.stdout == default.stdout


# ---------------------------------------------------------------------------
# --all: archived rows included
# ---------------------------------------------------------------------------

def test_all_flag_includes_archived_rows(tmp_path):
    _populate(tmp_path)

    result = _list(tmp_path, "--all")

    assert result.returncode == 0, result.stdout + result.stderr
    for case in ("damBreak", "cavity_re1000", "cylinder", "old_case", "ancient"):
        assert case in result.stdout


# ---------------------------------------------------------------------------
# Filters: select by Status or Result value
# ---------------------------------------------------------------------------

def test_status_filter_selects_matching_rows(tmp_path):
    _populate(tmp_path)

    result = _list(tmp_path, "--status", "running")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "damBreak" in result.stdout
    for other in ("cavity_re1000", "cylinder", "old_case", "ancient"):
        assert other not in result.stdout


def test_status_archived_shows_archived_without_all_flag(tmp_path):
    """Asking for archived rows by status is an explicit request; the
    default hiding must not turn it into an empty answer."""
    _populate(tmp_path)

    result = _list(tmp_path, "--status", "archived")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "old_case" in result.stdout
    assert "ancient" in result.stdout
    assert "damBreak" not in result.stdout


def test_result_filter_selects_matching_rows(tmp_path):
    _populate(tmp_path)

    result = _list(tmp_path, "--result", "diverged")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "cylinder" in result.stdout
    for other in ("damBreak", "cavity_re1000", "old_case", "ancient"):
        assert other not in result.stdout


def test_result_filter_still_hides_archived_unless_all(tmp_path):
    _populate(tmp_path)

    default = _list(tmp_path, "--result", "converged")
    with_all = _list(tmp_path, "--result", "converged", "--all")

    assert "cavity_re1000" in default.stdout
    assert "old_case" not in default.stdout
    assert "cavity_re1000" in with_all.stdout
    assert "old_case" in with_all.stdout


def test_filter_matching_nothing_says_so_and_exits_zero(tmp_path):
    """An empty answer to a valid question is not an error."""
    _populate(tmp_path)

    result = _list(tmp_path, "--status", "debugging")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no runs match" in result.stdout.lower()


def test_everything_archived_points_at_all_flag(tmp_path):
    """A bare invocation over an all-archived ledger must say why the table
    is empty, not pretend a filter matched nothing."""
    active_rows = ("| 0005 |", "| 0004 |", "| 0003 |")
    archived_only = "\n".join(
        l for l in FIXTURE.splitlines() if not l.startswith(active_rows)
    ) + "\n"
    tmp_path.joinpath("ledger.md").write_text(archived_only, encoding="utf-8")

    result = _list(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 archived" in result.stdout
    assert "--all" in result.stdout


# ---------------------------------------------------------------------------
# Missing / empty ledger: friendly message, exit code 1
# ---------------------------------------------------------------------------

def test_missing_ledger_points_at_ledger_check_and_exits_one(tmp_path):
    result = _list(tmp_path)  # no ledger.md here

    assert result.returncode == 1
    assert result.stderr == ""  # a friendly message, not a traceback
    assert "no ledger" in result.stdout.lower()
    assert "scripts/ledger_check.py" in result.stdout


def test_empty_ledger_points_at_ledger_check_and_exits_one(tmp_path):
    header_only = FIXTURE.split("|----")[0] + "|----|------|---------|--------|------|--------|--------|------------|-------|\n"
    tmp_path.joinpath("ledger.md").write_text(header_only, encoding="utf-8")

    result = _list(tmp_path)

    assert result.returncode == 1
    assert result.stderr == ""
    assert "no runs" in result.stdout.lower()
    assert "scripts/ledger_check.py" in result.stdout


# ---------------------------------------------------------------------------
# Read-only; ASCII-safe output on Windows consoles
# ---------------------------------------------------------------------------

def test_listing_never_modifies_the_ledger(tmp_path):
    ledger = _populate(tmp_path)
    before = ledger.read_bytes()

    for args in ((), ("--all",), ("--status", "done"), ("--result", "pending")):
        _list(tmp_path, *args)

    assert ledger.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ledger.md"]


def test_output_is_ascii_even_on_a_strict_console(tmp_path):
    """The ledger file is UTF-8 (the fixture note has an em-dash and an
    accent), but console output must never crash a Windows console
    (issues #18/#20). ascii:strict emulates the harshest console."""
    _populate(tmp_path)
    env = dict(os.environ, PYTHONIOENCODING="ascii:strict")

    result = _list(tmp_path, "--all", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.isascii()
    assert "ancient" in result.stdout  # the non-ASCII-note row still prints
