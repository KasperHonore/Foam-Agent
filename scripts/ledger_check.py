#!/usr/bin/env python3
"""Foam-Agent ledger check: verify/repair runs/ledger.md without an AI agent.

The deterministic conscience of the run ledger (spec issue #28) — a peer of
doctor.py in manners: dry-run by default, prints exactly what it would do,
deterministic, no LLM. It detects and (with --fix) repairs:

  - orphan case directories missing from the ledger — adopted as new rows
    via best-effort inspection; this is also the migration path, so an
    existing pre-ledger runs directory yields a complete ledger
  - rows whose case directories vanished — archived with an explanatory
    note, never deleted
  - duplicate IDs — the older row keeps the ID, newer claimants are
    reassigned to the next free one
  - Status/Result values outside the run-states.yml vocabulary — repaired
    to a legal, conservative state

Repair writes a timestamped backup of ledger.md before modifying anything
and takes an exclusive lockfile (ledger.lock in the runs directory) so two
repairs cannot race. NOTE: the MCP server's own ledger writes are serialized
only in-process, so run --fix while the server is idle.

Usage:
    python scripts/ledger_check.py [--runs-dir DIR]          # dry-run report
    python scripts/ledger_check.py [--runs-dir DIR] --fix    # repair

Exit code 0 on a clean ledger, 1 when issues are found (repaired if --fix),
2 when repair could not run (lockfile held).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import ledger  # noqa: E402

# Anything with one of these directly inside is treated as an OpenFOAM case.
CASE_MARKERS = ("system", "constant", "0", "Allrun")


def _vocabulary(section: str) -> frozenset:
    """The legal values of a run-states.yml section, from the shipped constant."""
    values, current = [], None
    for line in ledger.RUN_STATES_YML.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" "):
            current = stripped.split(":")[0]
        elif current == section:
            values.append(stripped.split(":")[0])
    return frozenset(values)


LEGAL_STATUS = _vocabulary("status")
LEGAL_RESULT = _vocabulary("result")
VERDICTS = LEGAL_RESULT - {"pending"}
# run-states.yml: pending is only legal while status is one of these.
PENDING_OK_STATUS = ("planned", "running", "debugging")


def _say(text: str = "") -> None:
    """Print, coerced to ASCII — Windows consoles mangle anything else (#18)."""
    print(text.encode("ascii", "replace").decode("ascii"))


class Issue:
    def __init__(self, kind: str, detail: str, fix: str):
        self.kind, self.detail, self.fix = kind, detail, fix


# ---------------------------------------------------------------------------
# Case-directory discovery and best-effort inspection
# ---------------------------------------------------------------------------

def _is_case_dir(path: Path) -> bool:
    return any((path / marker).exists() for marker in CASE_MARKERS)


def _find_case_dirs(runs_root: Path) -> list[str]:
    """Case keys (posix-relative to runs root) for every case directory.

    Scans two levels: top-level case dirs, plus cases one level down inside
    non-case dirs — the project namespacing of a shared runs root (spec #28).
    """
    cases = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if _is_case_dir(child):
            cases.append(child.name)
            continue
        for sub in sorted(child.iterdir()):
            if sub.is_dir() and _is_case_dir(sub):
                cases.append(f"{child.name}/{sub.name}")
    return cases


def _inspect_solver(case: Path) -> str:
    control = case / "system" / "controlDict"
    if control.is_file():
        for line in control.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"\s*application\s+(\S+?)\s*;", line)
            if match:
                return match.group(1)
    return ledger.PLACEHOLDER


def _inspect_mesh(case: Path) -> str:
    if (case / "system" / "snappyHexMeshDict").is_file():
        return "snappyHexMesh"
    if (case / "system" / "blockMeshDict").is_file():
        return "blockMesh"
    return ledger.PLACEHOLDER


def _ran(case: Path) -> bool:
    """Did anything ever execute here? Logs, Allrun output, or time dirs."""
    if (case / "Allrun.out").is_file() or any(case.glob("log.*")):
        return True
    for child in case.iterdir():
        if child.is_dir():
            try:
                if float(child.name) > 0:
                    return True
            except ValueError:
                pass
    return False


def _failed(case: Path) -> bool:
    """Cheap error evidence: a non-empty Allrun.err or a FATAL in any log."""
    err = case / "Allrun.err"
    if err.is_file() and err.stat().st_size > 0:
        return True
    for log in [case / "Allrun.out", *sorted(case.glob("log.*"))]:
        if not log.is_file():
            continue
        with open(log, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "FOAM FATAL" in line:
                    return True
    return False


def _infer_status(case: Path) -> str:
    """Conservative v1 inference for adoption/repair: a case that shows run
    artefacts is done, an untouched one is planned. Never running/debugging —
    this script cannot know a run is live."""
    return "done" if _ran(case) else "planned"


def _infer_verdict(case: Path) -> str:
    """Conservative v1 verdict (the convergence parser upgrades this later):
    error evidence means diverged, a clean-looking finished run counts as
    converged, and no evidence at all means abandoned."""
    if _failed(case):
        return "diverged"
    if _ran(case):
        return "converged"
    return "abandoned"


# ---------------------------------------------------------------------------
# The check: every defect class, computing both the report and the repair
# ---------------------------------------------------------------------------

def check(runs_root: Path) -> tuple[list[Issue], list[ledger.Row]]:
    """Return (issues, repaired rows). Pure computation — writes nothing."""
    rows = [replace(row) for row in ledger.read_rows(str(runs_root))]
    issues: list[Issue] = []

    # Duplicate IDs: rows are newest-first, so walking bottom-up lets the
    # oldest claimant keep its ID and reassigns newer ones — IDs grow with
    # time, so the newer row is the one renumbered. Fresh IDs are allocated
    # above every ID in the ledger so no innocent row is ever renumbered.
    next_free = max((int(r.id) for r in rows if r.id.isdigit()), default=0) + 1
    claimed: dict[str, ledger.Row] = {}
    for row in reversed(rows):
        if row.id not in claimed:
            claimed[row.id] = row
            continue
        new_id = f"{next_free:04d}"
        next_free += 1
        issues.append(Issue(
            "DUP-ID", f"rows '{claimed[row.id].case}' and '{row.case}' both have id {row.id}",
            f"keep {row.id} on the older row '{claimed[row.id].case}', "
            f"reassign '{row.case}' to id {new_id}",
        ))
        row.id = new_id
        claimed[new_id] = row

    # Rows whose case directories vanished: archive with a note, never
    # delete — the ledger is history. Rows already archived are left alone
    # (deleting an archived run's directory is legitimate housekeeping).
    for row in rows:
        if row.status == "archived" or (runs_root / row.case).is_dir():
            continue
        note = f"directory missing, archived by ledger_check {date.today().isoformat()}"
        row.status = "archived"
        # pending is only legal while planned/running/debugging; a vanished
        # run without a verdict is conservatively abandoned.
        if row.result not in VERDICTS:
            row.result = "abandoned"
        row.notes = f"{row.notes}; {note}" if row.notes else note
        issues.append(Issue(
            "VANISHED", f"row {row.id} '{row.case}': case directory is gone",
            f"archive row (result {row.result}), note: {note}",
        ))

    # Status/Result outside the run-states.yml vocabulary (including a
    # pending result on a done/archived row): repair from case inspection,
    # never trusting the illegal value.
    for row in rows:
        case = runs_root / row.case
        if row.status not in LEGAL_STATUS:
            new_status = _infer_status(case)
            issues.append(Issue(
                "ILLEGAL", f"row {row.id} '{row.case}': status '{row.status}' "
                           "is not in run-states.yml",
                f"set status {new_status} (from case inspection)",
            ))
            row.status = new_status
        if row.result not in LEGAL_RESULT:
            new_result = "pending" if row.status in PENDING_OK_STATUS else _infer_verdict(case)
            issues.append(Issue(
                "ILLEGAL", f"row {row.id} '{row.case}': result '{row.result}' "
                           "is not in run-states.yml",
                f"set result {new_result}",
            ))
            row.result = new_result
        if row.result == "pending" and row.status not in PENDING_OK_STATUS:
            new_result = _infer_verdict(case)
            issues.append(Issue(
                "ILLEGAL", f"row {row.id} '{row.case}': result pending is only "
                           f"legal while planned/running/debugging (status is {row.status})",
                f"set result {new_result}",
            ))
            row.result = new_result

    known = {row.case for row in rows}
    next_id = max((int(r.id) for r in rows if r.id.isdigit()), default=0) + 1
    for case_key in _find_case_dirs(runs_root):
        if case_key in known:
            continue
        case = runs_root / case_key
        status = _infer_status(case)
        result = "pending" if status == "planned" else _infer_verdict(case)
        row = ledger.Row(
            id=f"{next_id:04d}",
            case=case_key,
            created=date.fromtimestamp(case.stat().st_mtime).isoformat(),
            solver=_inspect_solver(case),
            mesh=_inspect_mesh(case),
            status=status,
            result=result,
            notes="adopted by ledger_check",
        )
        next_id += 1
        rows.insert(0, row)  # newest first, like the server
        issues.append(Issue(
            "ORPHAN", f"case directory '{case_key}' has no ledger row",
            f"adopt as id {row.id}: solver {row.solver}, mesh {row.mesh}, "
            f"status {row.status}, result {row.result}",
        ))

    return issues, rows


def _report(issues: list[Issue], row_count: int, fixed: bool) -> int:
    if not issues:
        _say(f"Ledger is clean: {row_count} row(s), no issues.")
        return 0
    for issue in issues:
        _say(f"[{issue.kind}] {issue.detail}")
        _say(f"    fix: {issue.fix}")
    _say()
    if fixed:
        _say(f"{len(issues)} issue(s) found and repaired.")
    else:
        _say(f"{len(issues)} issue(s) found. Dry run: nothing written. "
             "Re-run with --fix to repair.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", default=str(REPO / "runs"),
                        help="runs directory holding ledger.md (default: repo runs/)")
    parser.add_argument("--fix", action="store_true",
                        help="repair issues (writes a timestamped ledger backup first); "
                             "default is a dry-run report")
    args = parser.parse_args()
    runs_root = Path(args.runs_dir)

    _say("Foam-Agent ledger check")
    _say("=" * 23)
    _say(f"runs root: {runs_root}")
    _say()

    if not runs_root.is_dir():
        _say("Runs directory does not exist yet: nothing to check, no issues.")
        return 0

    if not args.fix:
        issues, rows = check(runs_root)
        return _report(issues, len(rows), fixed=False)

    # Repair: hold an exclusive lockfile across read-check-write so two
    # repairs cannot race each other. The MCP server's own writes are
    # serialized only in-process — run this while the server is idle.
    lock_path = runs_root / "ledger.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        _say(f"Another repair holds the ledger lock ({lock_path}).")
        _say("If no ledger_check is running, delete the lockfile and re-run.")
        return 2
    except OSError as exc:
        _say(f"Cannot take the ledger lock ({lock_path}): {exc}")
        return 2
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        issues, rows = check(runs_root)
        if issues:
            ledger_path = runs_root / ledger.LEDGER_BASENAME
            if ledger_path.exists():
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                backup = runs_root / f"{ledger.LEDGER_BASENAME}.bak.{stamp}"
                shutil.copy2(ledger_path, backup)
                _say(f"backup written: {backup}")
                _say()
            ledger._write(str(runs_root), rows)
        return _report(issues, len(rows), fixed=True)
    finally:
        os.close(lock_fd)
        os.unlink(lock_path)


if __name__ == "__main__":
    sys.exit(main())
