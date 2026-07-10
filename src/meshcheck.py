# meshcheck.py
"""Structured checkMesh assessment: typed mesh quality facts (issue #44).

``assess_mesh(case_dir)`` runs ``checkMesh -allTopology -allGeometry`` on a
case through the existing sourced-environment execution machinery and parses
the output with a separate pure function, ``parse_checkmesh_log(text)``.
The result is typed: mesh census (points/faces/cells and the cell-type
breakdown), per-metric entries (name, value where checkMesh reports one,
checkMesh's own ok/failed mark, a classification in {pass, warn, fail},
topology vs geometry distinguished), the flags used, checkMesh's own failure
count, and an overall verdict in {ok, warnings, failed} with the offending
metric names as evidence. Numbers are computed from the output, never
guessed; parsing targets Foundation OpenFOAM v10 checkMesh format (the pin).

Classification rules (mechanical defaults — per-application judgement is
skill-side):

- checkMesh's own failure marks are ground truth: every ``***``-marked check
  is ``fail``, and the overall verdict is never better than checkMesh's own
  conclusion ("Mesh OK." vs "Failed N mesh checks.").
- A conservative built-in warn band flags marginal-but-legal values under
  the hard limits (thresholds below). checkMesh's own single-``*`` notices
  (e.g. multiple regions, severely non-orthogonal faces) also read as warn.
- An output with zero parsed quality checks NEVER classifies as ok (the
  vacuous-pass trap): it is ``failed`` with evidence saying why.
- No mesh / checkMesh unable to run raises ``MeshAssessmentError`` — a typed
  error, never a fabricated assessment.

Warn-band defaults (values at/above these classify ``warn`` even when
checkMesh itself says OK; deliberately conservative):

- non-orthogonality: warn from 65 degrees. checkMesh v10 marks faces above
  70 as severe (a ``*`` notice) and only fails the check above 90.
- skewness: warn from 2. checkMesh v10 fails internal faces above 4.
- aspect ratio: warn from 100. checkMesh v10 fails above 1000.

Like the rest of the mechanical layer this module is key-free and stdlib-only
at import (CI runs the unit suite with nothing but pytest installed).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import mechanics

# The flags every assessment runs with (recorded in the output for the
# conversation, so a stricter/looser future variant is distinguishable).
CHECKMESH_FLAGS = "-allTopology -allGeometry"

# Conservative built-in warn thresholds (documented defaults, see module
# docstring; per-application judgement lives in the foam skill, ticket #45).
NON_ORTHOGONALITY_WARN = 65.0
SKEWNESS_WARN = 2.0
ASPECT_RATIO_WARN = 100.0

_WARN_BANDS = {
    "max_non_orthogonality": NON_ORTHOGONALITY_WARN,
    "max_skewness": SKEWNESS_WARN,
    "max_aspect_ratio": ASPECT_RATIO_WARN,
}


class MeshAssessmentError(RuntimeError):
    """checkMesh could not produce an assessable output (no mesh, failed
    invocation, truncated/unrecognizable output). Never a fabricated verdict."""


@dataclass
class MeshCensus:
    """Mesh size facts from checkMesh's 'Mesh stats' block (None when absent)."""
    points: int | None = None
    faces: int | None = None
    internal_faces: int | None = None
    cells: int | None = None
    boundary_patches: int | None = None
    cell_types: dict[str, int] = field(default_factory=dict)


@dataclass
class MeshMetric:
    """One parsed checkMesh check: its value (where reported), checkMesh's own
    mark, and the conservative classification on top."""
    name: str
    value: float | None
    check: str            # "topology" or "geometry"
    checkmesh_ok: bool    # checkMesh's own mark (False for ***-failed checks)
    classification: str   # "pass", "warn" or "fail"


@dataclass
class MeshAssessment:
    """Typed mesh assessment parsed from one checkMesh output."""
    flags: str
    census: MeshCensus
    metrics: list[MeshMetric]
    failed_checks: int    # checkMesh's own reported count (0 for "Mesh OK.")
    mesh_ok: bool         # True iff checkMesh concluded "Mesh OK."
    verdict: str          # "ok", "warnings" or "failed"
    evidence: list[str] = field(default_factory=list)


# --- line patterns (Foundation v10 checkMesh output) -----------------------

_EXEC_RE = re.compile(r"^Exec\s*:\s*checkMesh\s*(.*)$")
_MESH_OK_RE = re.compile(r"^Mesh OK\.$")
_FAILED_SUMMARY_RE = re.compile(r"^Failed (\d+) mesh checks\.$")

_CENSUS_RES = {
    "points": re.compile(r"^points:\s+(\d+)$"),
    "faces": re.compile(r"^faces:\s+(\d+)$"),
    "internal_faces": re.compile(r"^internal faces:\s+(\d+)$"),
    "cells": re.compile(r"^cells:\s+(\d+)$"),
    "boundary_patches": re.compile(r"^boundary patches:\s+(\d+)$"),
}
_CELL_TYPE_RE = re.compile(
    r"^(hexahedra|prisms|wedges|pyramids|tet wedges|tetrahedra|polyhedra):\s+(\d+)$"
)

# Value-bearing checks with a stable metric name. Passing shapes are genuine
# v10 lines (harvested fixture); failing shapes follow the v10 checkMesh
# source conventions (" ***..." with the value inline).
_NAMED_CHECKS: list[tuple[re.Pattern, str, bool]] = [
    (re.compile(r"^Max cell openness = (\S+) OK\.$"), "max_cell_openness", True),
    (re.compile(r"^Max aspect ratio = (\S+) OK\.$"), "max_aspect_ratio", True),
    (re.compile(r"^Minimum face area = (\S+?)\. Maximum face area = \S+?\.\s+Face area magnitudes OK\.$"),
     "min_face_area", True),
    (re.compile(r"^Min volume = (\S+?)\. Max volume = .*Cell volumes OK\.$"),
     "min_volume", True),
    (re.compile(r"^Max skewness = (\S+) OK\.$"), "max_skewness", True),
    (re.compile(r"^Min/max edge length = (\S+) \S+ OK\.$"), "min_edge_length", True),
    (re.compile(r"^Number of regions: (\d+) \(OK\)\.$"), "number_of_regions", True),
    (re.compile(r"^\*\*\*Max skewness = (\S+?),"), "max_skewness", False),
    (re.compile(r"^\*\*\*High aspect ratio cells found, Max aspect ratio: (\S+?),"),
     "max_aspect_ratio", False),
    (re.compile(r"^\*\*\*Open cells found, max cell openness: (\S+?),"),
     "max_cell_openness", False),
    (re.compile(r"^\*\*\*Zero or negative face area detected\.\s*Minimum area: (\S+?)\.?$"),
     "min_face_area", False),
    (re.compile(r"^\*\*\*Zero or negative cell volume detected\..*volume: (\S+?),"),
     "min_volume", False),
    (re.compile(r"^\*\*\*Edges too small, min/max edge length = (\S+) "),
     "min_edge_length", False),
    (re.compile(r"^\*\*\*Unused points found in the mesh, number unused: (\d+)$"),
     "point_usage", False),
]

# Non-orthogonality is reported on two lines: the value line, then the
# check's own verdict line. The value is stashed and attached to the verdict.
_NON_ORTHO_VALUE_RE = re.compile(r"^Mesh non-orthogonality Max: (\S+) average: \S+$")
_NON_ORTHO_OK_RE = re.compile(r"^Non-orthogonality check OK\.$")
_NON_ORTHO_FAIL_RE = re.compile(r"^\*\*\*Number of non-orthogonality errors: \d+\.$")


def _metric_name(text: str) -> str:
    """A stable snake_case metric name from a checkMesh line's check text."""
    text = re.sub(r"\([^)]*\)", " ", text)          # drop parenthesized asides
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _to_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _classify(name: str, value: float | None, checkmesh_ok: bool) -> str:
    """pass/warn/fail for one metric: checkMesh's mark is ground truth, then
    the conservative warn band on top of values checkMesh itself passed."""
    if not checkmesh_ok:
        return "fail"
    band = _WARN_BANDS.get(name)
    if band is not None and value is not None and value >= band:
        return "warn"
    return "pass"


def parse_checkmesh_log(text: str) -> MeshAssessment:
    """Parse one ``checkMesh -allTopology -allGeometry`` output (Foundation
    v10 format) into a typed :class:`MeshAssessment`. Pure function.

    Raises ``ValueError`` when the output carries no "Mesh OK." /
    "Failed N mesh checks." summary — a truncated or failed invocation is
    never turned into an assessment.
    """
    flags = ""
    census = MeshCensus()
    metrics: list[MeshMetric] = []
    mesh_ok = False
    failed_checks: int | None = None
    summary_seen = False
    section = ""                       # "topology" / "geometry" once entered
    pending_non_ortho: float | None = None

    def _add(name: str, value: float | None, checkmesh_ok: bool,
             classification: str | None = None) -> None:
        metrics.append(MeshMetric(
            name=name, value=value, check=section, checkmesh_ok=checkmesh_ok,
            classification=classification or _classify(name, value, checkmesh_ok),
        ))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("<<"):
            continue

        exec_match = _EXEC_RE.match(line)
        if exec_match:
            flags = exec_match.group(1).strip()
            continue

        if line == "Checking topology..." or line.startswith("Checking patch topology"):
            section = "topology"
            continue
        if line == "Checking geometry...":
            section = "geometry"
            continue

        matched_census = False
        for census_field, census_re in _CENSUS_RES.items():
            census_match = census_re.match(line)
            if census_match:
                setattr(census, census_field, int(census_match.group(1)))
                matched_census = True
                break
        if matched_census:
            continue
        cell_type_match = _CELL_TYPE_RE.match(line)
        if cell_type_match:
            census.cell_types[cell_type_match.group(1)] = int(cell_type_match.group(2))
            continue

        if _MESH_OK_RE.match(line):
            mesh_ok, failed_checks, summary_seen = True, 0, True
            continue
        failed_match = _FAILED_SUMMARY_RE.match(line)
        if failed_match:
            mesh_ok, failed_checks, summary_seen = False, int(failed_match.group(1)), True
            continue

        non_ortho_value = _NON_ORTHO_VALUE_RE.match(line)
        if non_ortho_value:
            pending_non_ortho = _to_float(non_ortho_value.group(1))
            continue
        if _NON_ORTHO_OK_RE.match(line):
            _add("max_non_orthogonality", pending_non_ortho, checkmesh_ok=True)
            pending_non_ortho = None
            continue
        if _NON_ORTHO_FAIL_RE.match(line):
            _add("max_non_orthogonality", pending_non_ortho, checkmesh_ok=False)
            pending_non_ortho = None
            continue

        matched_named = False
        for check_re, name, checkmesh_ok in _NAMED_CHECKS:
            named_match = check_re.match(line)
            if named_match:
                _add(name, _to_float(named_match.group(1)), checkmesh_ok)
                matched_named = True
                break
        if matched_named:
            continue

        if line.startswith("***"):
            # Any other ***-marked line is a failed check (checkMesh counts
            # exactly these); name it from the text before the first detail.
            _add(_metric_name(re.split(r"[:,]", line[3:], maxsplit=1)[0]),
                 None, checkmesh_ok=False)
            continue
        if line.startswith("*"):
            # Single-star notices (multiple regions, severely non-orthogonal
            # faces) are checkMesh's own attention marks, not failed checks.
            _add(_metric_name(re.split(r"[:,]", line[1:], maxsplit=1)[0]),
                 None, checkmesh_ok=True, classification="warn")
            continue
        if line.endswith(" OK."):
            _add(_metric_name(line[: -len(" OK.")]), None, checkmesh_ok=True)
            continue

    if not summary_seen:
        raise ValueError(
            "checkMesh output carries no 'Mesh OK.' or 'Failed N mesh checks.' "
            "summary — truncated or failed invocation; refusing to assess."
        )

    verdict, evidence = _verdict(mesh_ok, failed_checks or 0, metrics)
    return MeshAssessment(
        flags=flags, census=census, metrics=metrics,
        failed_checks=failed_checks or 0, mesh_ok=mesh_ok,
        verdict=verdict, evidence=evidence,
    )


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _verdict(mesh_ok: bool, failed_checks: int,
             metrics: list[MeshMetric]) -> tuple[str, list[str]]:
    """Apply the classification rules (module docstring) to the parsed facts."""
    if not metrics:
        return "failed", [
            "no mesh quality checks could be parsed from the checkMesh output "
            "— refusing to classify an unverified mesh as ok"
        ]

    fails = [m for m in metrics if m.classification == "fail"]
    warns = [m for m in metrics if m.classification == "warn"]

    if failed_checks > 0 or fails or not mesh_ok:
        evidence = []
        if not mesh_ok:
            evidence.append(f"checkMesh itself concluded: Failed {failed_checks} mesh checks.")
        for m in fails:
            detail = f" = {_fmt(m.value)}" if m.value is not None else ""
            evidence.append(f"{m.name}{detail} marked failed by checkMesh ({m.check} check)")
        if failed_checks > len(fails):
            evidence.append(
                f"checkMesh reported {failed_checks} failed checks but only "
                f"{len(fails)} failure lines were parsed — read the raw log"
            )
        return "failed", evidence

    if warns:
        evidence = []
        for m in warns:
            band = _WARN_BANDS.get(m.name)
            if band is not None and m.value is not None:
                evidence.append(
                    f"{m.name} = {_fmt(m.value)} is at/above the warn threshold "
                    f"{_fmt(band)} (legal but marginal)"
                )
            else:
                evidence.append(f"{m.name} flagged by checkMesh with a '*' notice")
        return "warnings", evidence

    return "ok", [
        "checkMesh concluded: Mesh OK.",
        f"all {len(metrics)} parsed checks pass",
    ]


def _fatal_excerpt(text: str, max_lines: int = 5) -> str:
    """The first FOAM FATAL block's opening lines, if any."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "FOAM FATAL" in line:
            return "\n".join(lines[i:i + max_lines]).strip()
    return ""


def assess_mesh(case_dir: str, timeout: int = 600) -> MeshAssessment:
    """Run ``checkMesh -allTopology -allGeometry`` on a case (through the
    sourced-environment execution machinery) and return the typed assessment.

    Raises :class:`MeshAssessmentError` when the case directory is missing or
    checkMesh cannot produce an assessable output (no mesh, failed or
    timed-out invocation) — never a fabricated assessment.
    """
    if not os.path.isdir(case_dir):
        raise MeshAssessmentError(f"Case directory does not exist: {case_dir}")

    returncode, stdout, stderr = mechanics.run_openfoam_command(
        case_dir, f"checkMesh {CHECKMESH_FLAGS}", timeout
    )
    try:
        return parse_checkmesh_log(stdout)
    except ValueError as exc:
        detail = _fatal_excerpt(stdout) or _fatal_excerpt(stderr) or stderr.strip()
        raise MeshAssessmentError(
            f"checkMesh did not produce an assessable output for {case_dir} "
            f"(exit code {returncode})."
            + (f"\n{detail}" if detail else "")
        ) from exc
