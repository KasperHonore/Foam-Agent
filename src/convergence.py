# convergence.py
"""Solver-log convergence parser: typed residuals and verdicts (issue #39).

Reads a case's solver log (Foundation OpenFOAM v10 format) and returns
structured convergence facts — per-field residuals, Courant numbers,
continuity errors, time progress, completion — plus a verdict in
{converged, diverged, incomplete, error} with human-readable evidence
strings naming what drove it. Numbers are computed from the log, never
guessed; the parsing is deterministic and key-free.

Verdict rules (mechanical defaults — judgement beyond them is skill-side):

- ``error``      a FOAM FATAL (IO) ERROR is present in the log.
- ``diverged``   a residual explosion (any residual >= 1e6 or non-finite),
                 a Courant blow-up (max Courant >= 100), or a floating point
                 exception trap actually firing. Evidence is evaluated per
                 line, and the startup banner "sigFpe : Enabling floating
                 point exception trapping" is explicitly excluded — the
                 shakedown regression (PR #37), now permanent policy.
- ``incomplete`` the log ends without an ``End`` marker (partial/in-flight
                 runs parse cleanly to this plus time progress), or the run
                 completed but some field's last final residual is not under
                 the converged threshold.
- ``converged``  the run completed and every field's last final residual is
                 under 1e-4 (deliberately conservative built-in default).

Like the rest of the mechanical layer this module is key-free, and it must
stay stdlib-only: CI runs the unit suite with nothing but pytest installed.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import mechanics

# Conservative built-in thresholds (documented defaults; per-solver judgement
# lives in the foam skill's convergence reference, ticket #41).
CONVERGED_FINAL_RESIDUAL_MAX = 1e-4   # every field's last final residual must be under this
RESIDUAL_EXPLOSION_THRESHOLD = 1e6    # any residual at/above this (or NaN/inf) is a blow-up
COURANT_BLOWUP_THRESHOLD = 100.0      # any max Courant at/above this is a blow-up

# The genuine startup banner every OpenFOAM log opens with — its "sigFpe"/
# "floating point exception" words are trap SETUP, never trap evidence.
_BANNER_MARKER = "enabling floating point exception trapping"
_FPE_MARKERS = ("floating point exception", "sigfpe")

_RESIDUAL_RE = re.compile(
    r"Solving for (\S+?), Initial residual = (\S+), Final residual = (\S+),"
)
_COURANT_RE = re.compile(r"^Courant Number mean: (\S+) max: (\S+)")
_CONTINUITY_RE = re.compile(
    r"^time step continuity errors :.*\bcumulative = (\S+)\s*$"
)
_TIME_RE = re.compile(r"^Time = ([0-9eE+\-.]+)s?\s*$")
_EXEC_RE = re.compile(r"^Exec\s*:\s*(\S+)")
_FATAL_RE = re.compile(r"FOAM FATAL (?:IO )?ERROR")
_END_FATAL_RE = re.compile(r"^FOAM (?:exiting|aborting)")


@dataclass
class FieldResiduals:
    """Residual summary for one solved field (e.g. Ux, p)."""
    field: str
    first_initial: float
    last_initial: float
    last_final: float
    worst_initial: float
    worst_initial_time: float | None


@dataclass
class CourantSummary:
    """Courant number summary: the run-wide maximum plus the last reading."""
    max: float
    max_time: float | None
    last_mean: float
    last_max: float


@dataclass
class TimeProgress:
    """Solver time progress: first/latest Time lines vs the controlDict target."""
    first_time: float | None
    latest_time: float | None
    end_time: float | None


@dataclass
class SolverLogAnalysis:
    """Typed convergence facts parsed from one solver log."""
    solver: str
    log_file: str
    completed: bool
    time: TimeProgress = field(default_factory=lambda: TimeProgress(None, None, None))
    residuals: list[FieldResiduals] = field(default_factory=list)
    courant: CourantSummary | None = None
    cumulative_continuity: float | None = None
    fatal_errors: list[str] = field(default_factory=list)
    verdict: str = "incomplete"
    evidence: list[str] = field(default_factory=list)


def parse_solver_log(case_dir: str, log_file: str | None = None) -> SolverLogAnalysis:
    """Parse a case's solver log into typed convergence facts plus a verdict.

    By default the log is ``log.<application>`` with the application read
    from the case's ``system/controlDict`` (the same inspection the ledger's
    Solver stamping uses); an explicit ``log_file`` name overrides.
    """
    if not log_file:
        application = mechanics._inspect_solver(case_dir)
        if application == "-":
            raise ValueError(
                f"Cannot select a solver log for {case_dir}: no application entry "
                "in system/controlDict. Pass an explicit log_file."
            )
        log_file = f"log.{application}"

    log_path = os.path.join(case_dir, log_file)
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Solver log not found: {log_path}")

    with open(log_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()

    solver = ""
    completed = False
    first_time: float | None = None
    current_time: float | None = None
    fields: dict[str, FieldResiduals] = {}
    courant: CourantSummary | None = None
    cumulative_continuity: float | None = None

    fatal_errors: list[str] = []
    fatal_block: list[str] | None = None

    # Divergence evidence, evaluated per line (first occurrence per trigger
    # so a thousand exploded steps stay one evidence string each).
    divergence: list[str] = []
    exploded_fields: set[str] = set()
    courant_blew_up = False
    fpe_fired = False

    def _at(time: float | None) -> str:
        return "before the first time step" if time is None else f"at Time = {_fmt(time)}"

    for line in lines:
        stripped = line.strip()

        if fatal_block is not None:
            fatal_block.append(line)
            if _END_FATAL_RE.match(stripped):
                fatal_errors.append("\n".join(fatal_block).strip())
                fatal_block = None
            continue

        if not solver:
            exec_match = _EXEC_RE.match(stripped)
            if exec_match:
                solver = exec_match.group(1)
                continue

        if stripped == "End":
            completed = True
            continue

        # FPE trap actually firing — per line, with the startup banner
        # ("Enabling floating point exception trapping") excluded (PR #37).
        lowered = line.lower()
        if (not fpe_fired and _BANNER_MARKER not in lowered
                and any(marker in lowered for marker in _FPE_MARKERS)):
            fpe_fired = True
            divergence.append(
                f"floating point exception trap fired {_at(current_time)}: {stripped}"
            )
            continue

        time_match = _TIME_RE.match(stripped)
        if time_match:
            current_time = float(time_match.group(1))
            if first_time is None:
                first_time = current_time
            continue

        if _FATAL_RE.search(line):
            fatal_block = [line]
            continue

        courant_match = _COURANT_RE.match(stripped)
        if courant_match:
            mean = float(courant_match.group(1))
            cmax = float(courant_match.group(2))
            if courant is None:
                courant = CourantSummary(max=cmax, max_time=current_time,
                                         last_mean=mean, last_max=cmax)
            else:
                courant.last_mean = mean
                courant.last_max = cmax
                if cmax > courant.max or math.isnan(cmax):
                    courant.max = cmax
                    courant.max_time = current_time
            if not courant_blew_up and _blown(cmax, COURANT_BLOWUP_THRESHOLD):
                courant_blew_up = True
                divergence.append(
                    f"Courant number blow-up: max {_fmt(cmax)} {_at(current_time)} "
                    f"(threshold {COURANT_BLOWUP_THRESHOLD:g})"
                )
            continue

        continuity_match = _CONTINUITY_RE.match(stripped)
        if continuity_match:
            cumulative_continuity = float(continuity_match.group(1))
            continue

        residual_match = _RESIDUAL_RE.search(line)
        if residual_match:
            name = residual_match.group(1)
            initial = float(residual_match.group(2))
            final = float(residual_match.group(3))
            summary = fields.get(name)
            if summary is None:
                fields[name] = FieldResiduals(
                    field=name, first_initial=initial, last_initial=initial,
                    last_final=final, worst_initial=initial,
                    worst_initial_time=current_time,
                )
            else:
                summary.last_initial = initial
                summary.last_final = final
                if initial > summary.worst_initial or math.isnan(initial):
                    summary.worst_initial = initial
                    summary.worst_initial_time = current_time
            if name not in exploded_fields:
                blown_value = next(
                    (v for v in (initial, final)
                     if _blown(v, RESIDUAL_EXPLOSION_THRESHOLD)), None)
                if blown_value is not None:
                    exploded_fields.add(name)
                    divergence.append(
                        f"residual explosion: {name} residual {_fmt(blown_value)} "
                        f"{_at(current_time)} (threshold {RESIDUAL_EXPLOSION_THRESHOLD:g})"
                    )
            continue

    if fatal_block is not None:  # log ended inside the error block
        fatal_errors.append("\n".join(fatal_block).strip())

    analysis = SolverLogAnalysis(
        solver=solver,
        log_file=os.path.basename(log_file),
        completed=completed,
        time=TimeProgress(first_time=first_time, latest_time=current_time,
                          end_time=_control_dict_end_time(case_dir)),
        residuals=list(fields.values()),
        courant=courant,
        cumulative_continuity=cumulative_continuity,
        fatal_errors=fatal_errors,
    )
    analysis.verdict, analysis.evidence = _verdict(analysis, divergence)
    return analysis


def _blown(value: float, threshold: float) -> bool:
    """A value counts as blown up when non-finite or at/above the threshold."""
    return not math.isfinite(value) or value >= threshold


def _control_dict_end_time(case_dir: str) -> float | None:
    """The endTime target from system/controlDict (None when unreadable)."""
    content = mechanics.read_file(os.path.join(case_dir, "system", "controlDict"))
    match = re.search(r"^\s*endTime\s+(\S+?)\s*;", content, re.MULTILINE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _verdict(analysis: SolverLogAnalysis,
             divergence: list[str]) -> tuple[str, list[str]]:
    """Apply the verdict rules (module docstring) to the parsed facts."""
    if analysis.fatal_errors:
        first_line = analysis.fatal_errors[0].splitlines()[0].strip()
        return "error", [f"FOAM FATAL error in {analysis.log_file}: {first_line}"]

    if divergence:
        return "diverged", list(divergence)

    if not analysis.completed:
        progress = ""
        if analysis.time.latest_time is not None:
            progress = f" at Time = {_fmt(analysis.time.latest_time)}"
            if analysis.time.end_time is not None:
                progress += f" of endTime {_fmt(analysis.time.end_time)}"
        return "incomplete", [
            f"no 'End' marker — {analysis.log_file} ends{progress}"
        ]

    over = [r for r in analysis.residuals
            if not r.last_final < CONVERGED_FINAL_RESIDUAL_MAX]
    if over:
        return "incomplete", [
            f"completed, but last final residual for {r.field} ({_fmt(r.last_final)}) "
            f"is not under the converged threshold ({CONVERGED_FINAL_RESIDUAL_MAX:g})"
            for r in over
        ]

    evidence = ["'End' marker found — run completed"]
    if analysis.residuals:
        finals = ", ".join(f"{r.field}={_fmt(r.last_final)}" for r in analysis.residuals)
        evidence.append(
            f"final residuals all under {CONVERGED_FINAL_RESIDUAL_MAX:g}: {finals}"
        )
    else:
        evidence.append("no residual lines found in the log")
    return "converged", evidence
