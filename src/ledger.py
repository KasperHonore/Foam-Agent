# ledger.py
"""Run ledger: server-owned record of every simulation run.

The ledger is a human-readable markdown table at the root of the runs
directory (runs/ledger.md), written by the MCP server as a mechanical side
effect of the run lifecycle — never by skills and never by hand, except the
Notes column. run-states.yml beside it holds the only legal Status/Result
values. Design locked in the run-ledger spec (issue #28).

Like the rest of the mechanical layer this module is key-free, and it must
stay stdlib-only: CI runs the unit suite with nothing but pytest installed.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import PurePath

LEDGER_BASENAME = "ledger.md"
STATES_BASENAME = "run-states.yml"
PLACEHOLDER = "-"
_NUM_COLUMNS = 9

_HEADER = """# Foam-Agent run ledger

<!-- Managed by the foamagent MCP server. Verify/repair: python scripts/ledger_check.py -->
<!-- Allowed Status/Result values: see run-states.yml. Only the Notes column is yours to edit by hand. -->
<!-- Lives at runs/ledger.md — gitignored, bind-mounted, survives container recreation (update contract). -->

| ID | Case | Created | Solver | Mesh | Status | Result | Key result | Notes |
|----|------|---------|--------|------|--------|--------|------------|-------|
"""

RUN_STATES_YML = """\
# Canonical run states — the only values allowed in the ledger columns.

status:
  planned:   case directory allocated, files being generated, not yet run
  running:   Allrun in progress (or SLURM job queued/running)
  debugging: failed at least once; the foam-debugger loop is active
  done:      run finished — see result for the verdict
  archived:  kept for reference; hidden from default listings

result:
  pending:   no verdict yet (only legal while status is planned/running/debugging)
  converged: completed with residual targets met / physically sane
  diverged:  blew up; kept for post-mortem
  abandoned: owner stopped pursuing this case
"""

# One process (the MCP server) is the single allocator; this lock makes the
# read-modify-write cycle atomic across its worker threads.
_LOCK = threading.Lock()


@dataclass
class Row:
    id: str
    case: str
    created: str
    solver: str = PLACEHOLDER
    mesh: str = PLACEHOLDER
    status: str = "planned"
    result: str = "pending"
    key_result: str = PLACEHOLDER
    notes: str = ""

    def cells(self) -> list[str]:
        return [self.id, self.case, self.created, self.solver, self.mesh,
                self.status, self.result, self.key_result, self.notes]


def track_planned(runs_root: str, case_dir: str) -> Row | None:
    """Record case_dir as a planned run (idempotent); return its row.

    Returns None when case_dir is not under runs_root — out-of-tree case
    directories are not the ledger's to track.
    """
    case = _case_key(runs_root, case_dir)
    if case is None:
        return None
    with _LOCK:
        rows = read_rows(runs_root)
        for row in rows:
            if row.case == case:
                return row
        next_id = max((int(r.id) for r in rows if r.id.isdigit()), default=0) + 1
        row = Row(id=f"{next_id:04d}", case=case, created=date.today().isoformat())
        rows.insert(0, row)  # newest first
        _write(runs_root, rows)
    return row


def track_running(runs_root: str, case_dir: str,
                  solver: str = PLACEHOLDER, mesh: str = PLACEHOLDER) -> Row | None:
    """Run start: flip case_dir's row to running, Result back to pending.

    Re-runs pass through here too, so the verdict of a previous run is
    cleared while the new one is in flight. Solver/Mesh are stamped from
    the caller's best-effort inspection of the case; a placeholder never
    overwrites a previously determined value.
    """
    return _transition(runs_root, case_dir, status="running", result="pending",
                       solver=solver, mesh=mesh)


def track_done(runs_root: str, case_dir: str, result: str) -> Row | None:
    """Run completed: stamp done plus the Result verdict (converged/diverged)."""
    return _transition(runs_root, case_dir, status="done", result=result)


def track_failed(runs_root: str, case_dir: str) -> Row | None:
    """Run failed: flip to debugging, Result stays pending — an in-flight
    rescue by the debugger loop is visible as such."""
    return _transition(runs_root, case_dir, status="debugging", result="pending")


def _transition(runs_root: str, case_dir: str, *, status: str,
                result: str | None = None, solver: str = PLACEHOLDER,
                mesh: str = PLACEHOLDER) -> Row | None:
    """Server-owned lifecycle write: update case_dir's row in place.

    A case that was never resolved through resolve_case_dir (direct tool
    usage) is adopted with a fresh row, so the record stays complete no
    matter who drives the run. Returns None for out-of-tree case dirs,
    like track_planned.
    """
    case = _case_key(runs_root, case_dir)
    if case is None:
        return None
    with _LOCK:
        rows = read_rows(runs_root)
        row = next((r for r in rows if r.case == case), None)
        if row is None:
            next_id = max((int(r.id) for r in rows if r.id.isdigit()), default=0) + 1
            row = Row(id=f"{next_id:04d}", case=case, created=date.today().isoformat())
            rows.insert(0, row)  # newest first
        row.status = status
        if result is not None:
            row.result = result
        if solver != PLACEHOLDER:
            row.solver = solver
        if mesh != PLACEHOLDER:
            row.mesh = mesh
        _write(runs_root, rows)
    return row


def set_note(runs_root: str, run_id: str, *, note: str | None = None,
             archive: bool | None = None) -> Row:
    """The one skill-side ledger write: Notes plus archive/unarchive.

    Rows are keyed by their zero-padded ID — the handle a skill reads off
    the ledger (digit-only IDs are normalized, so '3' finds '0003').
    The sanctioned surface is exactly this:

    - note (None = leave alone) replaces the Notes cell, normalized so it
      round-trips through the table format; every other cell is untouched.
    - archive=True flips Status to archived; a still-pending Result is
      stamped abandoned ("owner stopped pursuing this").
    - archive=False restores Status to done and leaves Result as it is —
      abandoned stays abandoned until a new run overwrites it via the
      lifecycle. Only archived (or already-done) rows may be unarchived:
      anything else would let a skill set Status at will.

    Raises ValueError — with the ledger left untouched — for an unknown ID
    or an illegal unarchive.
    """
    key = run_id.strip()
    if key.isdigit():
        key = f"{int(key):04d}"
    with _LOCK:
        rows = read_rows(runs_root)
        row = next((r for r in rows if r.id == key), None)
        if row is None:
            raise ValueError(f"No ledger row with ID '{key}' in "
                             f"{os.path.join(runs_root, LEDGER_BASENAME)}")
        if archive is True:
            row.status = "archived"
            if row.result == "pending":
                row.result = "abandoned"  # owner stopped pursuing this
        elif archive is False:
            if row.status not in ("archived", "done"):
                raise ValueError(
                    f"Run {key} is not archived (status: {row.status}); "
                    "unarchiving cannot be used to set a status"
                )
            row.status = "done"  # Result stays — abandoned until a new run
        if note is not None:
            row.notes = _normalize_note(note)
        _write(runs_root, rows)
    return row


def set_key_result(runs_root: str, case_dir: str, key_result: str) -> Row | None:
    """Server-owned machine write: fill a case's Key result cell.

    The Key result cell is machine-owned like the lifecycle columns — this
    is NOT part of set_note's sanctioned skill surface. The force-coefficient
    parser (issue #55) stamps its compact summary here as a side effect of
    parsing. Rows are keyed by the case key (like the lifecycle writes);
    a case without a row — out-of-tree, or never tracked — returns None and
    the ledger is left untouched: stamping never adopts. Re-stamping
    overwrites idempotently. The stored cell is folded like a note so the
    table format survives whatever the summary carries.
    """
    case = _case_key(runs_root, case_dir)
    if case is None:
        return None
    with _LOCK:
        rows = read_rows(runs_root)
        row = next((r for r in rows if r.case == case), None)
        if row is None:
            return None
        row.key_result = _normalize_note(key_result) or PLACEHOLDER
        _write(runs_root, rows)
    return row


def _normalize_note(note: str) -> str:
    """Fold a note into the form the table parser reads back unchanged.

    Raw newlines would break the row and read_rows rejoins in-cell pipes
    with ' | ', so both are normalized to that form up front: the stored
    cell is a fixpoint of write -> read_rows.
    """
    flat = " ".join(note.strip().splitlines())
    return " | ".join(part.strip() for part in flat.split("|"))


def read_rows(runs_root: str) -> list[Row]:
    path = os.path.join(runs_root, LEDGER_BASENAME)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if line.lstrip("| ").startswith("ID") or set(line) <= {"|", "-", " "}:
            continue  # column header / separator
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > _NUM_COLUMNS:  # hand-written note containing pipes
            cells[_NUM_COLUMNS - 1] = " | ".join(cells[_NUM_COLUMNS - 1:])
            cells = cells[:_NUM_COLUMNS]
        elif len(cells) < _NUM_COLUMNS:
            cells += [""] * (_NUM_COLUMNS - len(cells))
        rows.append(Row(*cells))
    return rows


def _case_key(runs_root: str, case_dir: str) -> str | None:
    """The Case cell: case_dir relative to the runs root, posix-style.

    Nested paths keep their structure, which is what namespaces projects
    sharing one (future global) runs directory.
    """
    try:
        rel = os.path.relpath(os.path.abspath(case_dir), os.path.abspath(runs_root))
    except ValueError:  # different drive on Windows
        return None
    if rel == os.curdir or rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return None
    return PurePath(rel).as_posix()


def _write(runs_root: str, rows: list[Row]) -> None:
    os.makedirs(runs_root, exist_ok=True)
    states = os.path.join(runs_root, STATES_BASENAME)
    if not os.path.exists(states):
        _replace_atomically(states, RUN_STATES_YML)
    body = "".join("| " + " | ".join(r.cells()) + " |\n" for r in rows)
    _replace_atomically(os.path.join(runs_root, LEDGER_BASENAME), _HEADER + body)


def _replace_atomically(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    os.replace(tmp, path)
