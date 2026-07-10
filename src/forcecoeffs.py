# forcecoeffs.py
"""Force-coefficient parser: typed Cd/Cl/Cm, Key result filled (issue #55).

Reads a case's forceCoeffs function-object output (Foundation OpenFOAM v10
format: a ``#`` header block carrying the reference values — magUInf, lRef,
Aref, CofR, lift/drag/pitch directions — then tab-separated columns
``Time Cm Cd Cl Cl(f) Cl(r)``) and returns structured coefficient facts:
the reference metadata, sample count and time span, and per-coefficient
first/final values plus tail-window statistics (mean/min/max). Numbers are
computed from the file, never guessed; the parsing is deterministic and
key-free.

Tail window (documented default): the last ``TAIL_WINDOW_FRACTION`` (20%)
of samples, rounded up, floored at ``TAIL_WINDOW_MIN_SAMPLES`` (10) and
capped at all samples when fewer exist. The window actually used (sample
count and time span) is reported in the output.

Key result: when the case has a run-ledger row, ``parse_force_coefficients``
stamps a compact summary into its Key result cell as a server-owned side
effect (machine-owned cell, NOT part of set_run_note's sanctioned skill
surface). The format is ``Cd=<tail mean:.4g> Cl=<tail mean:.4g> (tail mean)``
— e.g. ``Cd=-2.501 Cl=0.1275 (tail mean)`` — short enough for a table cell.
Rowless (out-of-tree or never-tracked) cases return the analysis unstamped;
re-parsing re-stamps idempotently.

Honest failure: no postProcessing output, an ambiguous choice between several
force function objects, a headerless/truncated file, or zero data rows each
raise :class:`ForceCoefficientsError` pointing at the forces recipe (run the
solver with a forceCoeffs function object; see the foam skill's forces
reference) — never fabricated statistics.

Like the rest of the mechanical layer this module is key-free, and it must
stay stdlib-only: CI runs the unit suite with nothing but pytest installed.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import PurePath

import ledger
import mechanics

# Documented defaults for the tail window (see module docstring).
TAIL_WINDOW_FRACTION = 0.2     # last 20% of samples, rounded up ...
TAIL_WINDOW_MIN_SAMPLES = 10   # ... but never fewer than 10 (or all when fewer)

# The forceCoeffs function object's output file name (Foundation v10).
DAT_BASENAME = "forceCoeffs.dat"

_RECIPE = ("run the solver with a forceCoeffs function object — see the foam "
           "skill's forces reference for the working Foundation v10 recipe")

# Header metadata lines, e.g. "# magUInf     : 1.000000e+00" and
# "# liftDir     : (0.000000e+00 1.000000e+00 0.000000e+00)".
_META_RE = re.compile(r"^#\s*(\w+)\s*:\s*(.+?)\s*$")
_META_FIELDS = {
    "liftDir": "lift_dir",
    "dragDir": "drag_dir",
    "pitchAxis": "pitch_axis",
    "magUInf": "mag_u_inf",
    "lRef": "l_ref",
    "Aref": "a_ref",
    "CofR": "cofr",
}


class ForceCoefficientsError(RuntimeError):
    """No usable forceCoeffs output (missing, ambiguous, headerless or empty).
    Never fabricated statistics."""


@dataclass
class ForceReference:
    """Reference values from the dat header (None when the line is absent) —
    report these alongside the coefficients so a wrong normalization
    (Aref/lRef/magUInf) is visible before a number is trusted."""
    lift_dir: list[float] | None = None
    drag_dir: list[float] | None = None
    pitch_axis: list[float] | None = None
    mag_u_inf: float | None = None
    l_ref: float | None = None
    a_ref: float | None = None
    cofr: list[float] | None = None


@dataclass
class TailWindow:
    """The tail window the statistics were computed over (documented default:
    last 20% of samples, rounded up, floored at 10, capped at all)."""
    samples: int
    fraction: float
    min_samples: int
    start_time: float
    end_time: float


@dataclass
class CoefficientSeries:
    """One coefficient column's summary: first/final values plus tail-window
    statistics over the reported window."""
    name: str
    first: float
    final: float
    tail_mean: float
    tail_min: float
    tail_max: float


@dataclass
class ForceCoefficientsAnalysis:
    """Typed force-coefficient facts parsed from one forceCoeffs.dat."""
    function_name: str
    dat_file: str
    reference: ForceReference
    samples: int
    start_time: float
    end_time: float
    window: TailWindow
    coefficients: list[CoefficientSeries] = field(default_factory=list)
    key_result: str = ""
    stamped: bool = False


def _parse_meta_value(name: str, raw: str) -> float | list[float] | None:
    """A header metadata value: '(x y z)' vectors as float lists, scalars as
    floats; None when the value does not parse (never a guessed number)."""
    try:
        vector = re.fullmatch(r"\(([^)]*)\)", raw)
        if vector:
            return [float(tok) for tok in vector.group(1).split()]
        return float(raw)
    except ValueError:
        return None


def parse_forcecoeffs_dat(text: str, function_name: str = "",
                          dat_file: str = "") -> ForceCoefficientsAnalysis:
    """Parse one forceCoeffs.dat text (Foundation v10 format) into a typed
    :class:`ForceCoefficientsAnalysis`. Pure function.

    Raises ``ValueError`` for a headerless/truncated file (no ``# Time ...``
    column header) or zero data rows — statistics are never computed over
    nothing. Data rows that do not parse (e.g. a partial final line while the
    solver is still writing) are skipped.
    """
    reference = ForceReference()
    columns: list[str] | None = None
    times: list[float] = []
    series: list[list[float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            cells = [c.strip() for c in line.lstrip("#").split("\t")]
            if cells and cells[0] == "Time":
                columns = [c for c in cells if c]
                series = [[] for _ in columns[1:]]
                continue
            meta = _META_RE.match(line)
            if meta and meta.group(1) in _META_FIELDS:
                setattr(reference, _META_FIELDS[meta.group(1)],
                        _parse_meta_value(meta.group(1), meta.group(2)))
            continue
        if columns is None:
            continue  # data-looking line before any column header: not ours
        tokens = [t.strip() for t in line.split("\t")]
        if len(tokens) < len(columns):
            continue  # partial row (in-flight write): skip, never guess
        try:
            values = [float(t) for t in tokens[: len(columns)]]
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in values):
            continue
        times.append(values[0])
        for i, value in enumerate(values[1:]):
            series[i].append(value)

    if columns is None:
        raise ValueError(
            "no column header ('# Time\\t...') found — headerless or truncated "
            f"forceCoeffs file; {_RECIPE}."
        )
    if not times:
        raise ValueError(
            f"zero data rows under the column header — statistics over nothing "
            f"are never computed; {_RECIPE}."
        )

    n = len(times)
    w = min(n, max(TAIL_WINDOW_MIN_SAMPLES, math.ceil(n * TAIL_WINDOW_FRACTION)))
    window = TailWindow(
        samples=w, fraction=TAIL_WINDOW_FRACTION,
        min_samples=TAIL_WINDOW_MIN_SAMPLES,
        start_time=times[-w], end_time=times[-1],
    )
    coefficients = []
    for name, values in zip(columns[1:], series):
        tail = values[-w:]
        coefficients.append(CoefficientSeries(
            name=name, first=values[0], final=values[-1],
            tail_mean=sum(tail) / len(tail), tail_min=min(tail),
            tail_max=max(tail),
        ))

    return ForceCoefficientsAnalysis(
        function_name=function_name, dat_file=dat_file, reference=reference,
        samples=n, start_time=times[0], end_time=times[-1], window=window,
        coefficients=coefficients,
        key_result=_key_result(coefficients),
    )


def _key_result(coefficients: list[CoefficientSeries]) -> str:
    """The compact Key-result summary: ``Cd=<tail mean:.4g> Cl=<tail
    mean:.4g> (tail mean)`` from whichever of Cd/Cl the file carries
    ('' when neither does — nothing worth stamping)."""
    by_name = {c.name: c for c in coefficients}
    parts = [f"{name}={by_name[name].tail_mean:.4g}"
             for name in ("Cd", "Cl") if name in by_name]
    if not parts:
        return ""
    return " ".join(parts) + " (tail mean)"


# ---------------------------------------------------------------------------
# Discovery: the case's forceCoeffs output under its postProcessing tree
# ---------------------------------------------------------------------------

def _newest_dat(function_dir: str) -> str | None:
    """The forceCoeffs.dat in the function dir's newest time directory
    (numerically largest name), or None when no time dir holds one."""
    best_path: str | None = None
    best_time: float | None = None
    for entry in os.listdir(function_dir):
        try:
            time = float(entry)
        except ValueError:
            continue  # not a time directory
        path = os.path.join(function_dir, entry, DAT_BASENAME)
        if os.path.isfile(path) and (best_time is None or time > best_time):
            best_path, best_time = path, time
    return best_path


def _discover_dat(case_dir: str, function_name: str | None) -> tuple[str, str]:
    """Locate the case's forceCoeffs.dat: (function name, absolute path).

    Candidates are postProcessing subdirectories holding a forceCoeffs.dat
    in at least one time directory; within one, the newest time directory
    wins. Several candidates without an explicit function_name raise the
    typed error naming them — ambiguity is surfaced, never silently
    resolved.
    """
    post_dir = os.path.join(case_dir, "postProcessing")
    if not os.path.isdir(post_dir):
        raise ForceCoefficientsError(
            f"No postProcessing directory in {case_dir} — the case has no "
            f"forceCoeffs output yet; {_RECIPE}."
        )
    candidates: dict[str, str] = {}
    for entry in sorted(os.listdir(post_dir)):
        function_dir = os.path.join(post_dir, entry)
        if not os.path.isdir(function_dir):
            continue
        newest = _newest_dat(function_dir)
        if newest is not None:
            candidates[entry] = newest
    if function_name:
        if function_name not in candidates:
            available = ", ".join(candidates) if candidates else "none"
            raise ForceCoefficientsError(
                f"No {DAT_BASENAME} under postProcessing/{function_name} in "
                f"{case_dir}. Function objects with forceCoeffs output: "
                f"{available}."
            )
        return function_name, candidates[function_name]
    if not candidates:
        raise ForceCoefficientsError(
            f"No {DAT_BASENAME} anywhere under {post_dir} — the case has no "
            f"forceCoeffs output; {_RECIPE}."
        )
    if len(candidates) > 1:
        raise ForceCoefficientsError(
            f"Several force function objects have output in {case_dir}: "
            f"{', '.join(candidates)}. Pass function_name to choose one."
        )
    ((name, path),) = candidates.items()
    return name, path


def parse_force_coefficients(
    case_dir: str,
    function_name: str | None = None,
    run_directory: str | None = None,
) -> ForceCoefficientsAnalysis:
    """Locate, parse and summarize a case's forceCoeffs output; stamp the
    ledger's Key result cell when the case has a row.

    Discovery: the newest time directory under postProcessing/<function>/;
    an explicit function_name disambiguates when several force function
    objects have output (ambiguity is a typed error naming the candidates).
    Stamping is a server-owned side effect against the runs root (default:
    the repo runs/ directory, same convention as the run lifecycle); a case
    without a ledger row returns the analysis unstamped. Raises
    :class:`ForceCoefficientsError` for missing/headerless/empty output —
    never fabricated statistics.
    """
    if not os.path.isdir(case_dir):
        raise ForceCoefficientsError(f"Case directory does not exist: {case_dir}")
    name, dat_path = _discover_dat(case_dir, function_name)
    with open(dat_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    rel_dat = PurePath(os.path.relpath(dat_path, case_dir)).as_posix()
    try:
        analysis = parse_forcecoeffs_dat(text, function_name=name, dat_file=rel_dat)
    except ValueError as exc:
        raise ForceCoefficientsError(f"{dat_path}: {exc}") from exc
    if analysis.key_result:
        row = ledger.set_key_result(mechanics._runs_root(run_directory),
                                    case_dir, analysis.key_result)
        analysis.stamped = row is not None
    return analysis
