#!/usr/bin/env python3
"""Foam-Agent run lister: pretty-print the run ledger without an AI agent.

Zero-token read surface for runs/ledger.md (see src/ledger.py for the file
contract). Read-only and deterministic — it never modifies the ledger. The
agent-driven equivalent is the `foam-runs` skill; both read the same file.

Usage:
    python scripts/runs.py                      # active runs (archived hidden)
    python scripts/runs.py --all                # archived rows too
    python scripts/runs.py --status running     # filter by Status value
    python scripts/runs.py --result diverged    # filter by Result value
    python scripts/runs.py --runs-dir PATH      # a runs dir other than runs/

Exit code 0 when the ledger was listed (even if no rows match the filters),
1 when there is no ledger or it has no rows yet — the message then points at
scripts/ledger_check.py, which adopts pre-ledger run directories.

Output is sanitized to ASCII so Windows consoles never crash on ledger
content (the file itself is UTF-8; non-ASCII characters print as '?').
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import ledger  # noqa: E402

HEADERS = ["ID", "Case", "Created", "Solver", "Mesh",
           "Status", "Result", "Key result", "Notes"]


def _print_ascii(line: str) -> None:
    """ASCII-only console output (issues #18/#20: em-dash mojibake)."""
    print(line.encode("ascii", "replace").decode("ascii").rstrip())


def _print_table(rows: list) -> None:
    grid = [HEADERS] + [row.cells() for row in rows]
    widths = [max(len(r[i]) for r in grid) for i in range(len(HEADERS))]
    _print_ascii("  ".join(h.ljust(w) for h, w in zip(HEADERS, widths)))
    _print_ascii("  ".join("-" * w for w in widths))
    for row in rows:
        _print_ascii("  ".join(c.ljust(w) for c, w in zip(row.cells(), widths)))


def main() -> int:
    parser = argparse.ArgumentParser(description="List the Foam-Agent run ledger.")
    parser.add_argument("--list", action="store_true",
                        help="list the ledger (the default action; accepted for clarity)")
    parser.add_argument("--all", action="store_true",
                        help="include archived rows (hidden by default)")
    parser.add_argument("--status", metavar="VALUE",
                        help="only rows whose Status is VALUE (see run-states.yml); "
                             "'--status archived' shows archived rows without --all")
    parser.add_argument("--result", metavar="VALUE",
                        help="only rows whose Result is VALUE (see run-states.yml)")
    parser.add_argument("--runs-dir", default=str(REPO / "runs"),
                        help="runs directory containing ledger.md (default: %(default)s)")
    args = parser.parse_args()

    rows = ledger.read_rows(args.runs_dir)
    if not rows:
        ledger_path = Path(args.runs_dir) / ledger.LEDGER_BASENAME
        if ledger_path.exists():
            _print_ascii(f"The ledger at {ledger_path} has no runs yet.")
        else:
            _print_ascii(f"No ledger found at {ledger_path}.")
        _print_ascii("Runs are ledgered automatically as the server executes them. If this")
        _print_ascii("runs directory predates the ledger, adopt the existing run directories:")
        _print_ascii("  python scripts/ledger_check.py")
        return 1

    visible = rows
    if args.status:
        visible = [r for r in visible if r.status == args.status.lower()]
    if args.result:
        visible = [r for r in visible if r.result == args.result.lower()]
    hidden = 0
    if not args.all and not args.status:  # an explicit --status is an explicit ask
        hidden = sum(1 for r in visible if r.status == "archived")
        visible = [r for r in visible if r.status != "archived"]

    note = f"({hidden} archived row{'s' if hidden != 1 else ''} hidden; use --all to include)"
    if not visible:
        _print_ascii(note if hidden else "No runs match the given filters.")
        return 0
    _print_table(visible)
    if hidden:
        _print_ascii(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
