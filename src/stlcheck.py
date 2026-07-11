# stlcheck.py
"""Structured surfaceCheck inspection of an STL surface (issue #60).

``inspect_stl(path)`` runs a plain ``surfaceCheck <file>`` on an STL surface
through the existing sourced-environment execution machinery and parses the
output with a separate pure function, ``parse_surfacecheck_log(text)``.
The result is typed: closed/watertight status with the defective-edge counts
(edges connected to one face / to more than two faces), triangle and vertex
counts, bounding box (min/max/extents), unconnected-parts and zones counts,
a units-suspicion flag derived from the extents, and an overall verdict in
{ok, warnings, failed} with evidence lines naming each problem. Numbers are
computed from the output, never guessed; parsing targets Foundation OpenFOAM
v10 surfaceCheck format (the pin).

Classification rules (mechanical defaults — per-application judgement is
skill-side):

- surfaceCheck exits 0 even for defective surfaces (verified live, #59
  harvest), so the TEXT is the only ground truth: every verdict below comes
  from parsed lines, never from the exit code.
- An open surface ("Surface is not closed since not all edges connected to
  two faces:") is ``failed`` — snappyHexMesh needs a watertight surface. The
  evidence names both counts: edges connected to one face (holes / open
  boundaries) and edges connected to >2 faces (non-manifold).
- A closed surface still ``warns`` when: it carries more than one unconnected
  part (multiple shells in one file — legitimate for multiple bodies, wrong
  for a single part); its normal-orientation zone count exceeds the part
  count (normals flip within a part); or its extents look like a millimetre
  export (below).
- Units suspicion: STL carries no unit metadata, so the flag can only come
  from bounding-box extents. The documented, deliberately conservative
  threshold is ``UNITS_SUSPICION_EXTENT = 1000.0``: suspicion is raised only
  when the LARGEST extent is at/above 1000 — i.e. a part that would be a
  kilometre across if the file really were in metres, which is far more
  plausibly a >=1 m part exported in millimetres. Smaller mm exports (a
  100 mm part reads as extent 100) pass unflagged — that is the price of
  never flagging legitimate metre-scale geometry. A suspected-mm export
  WARNS, it never fails.
- An output with no closed/not-closed status line (or no statistics at all)
  raises ``ValueError`` from the parser / ``SurfaceInspectionError`` from the
  runner — a truncated or crashed surfaceCheck is never turned into a report.

Working-directory choice (documented, deliberate): surfaceCheck dumps files
into its cwd on defective surfaces — ``problemFaces`` (with a "Dumping
conflicting face labels to \"problemFaces\"" line), plus zoning ``.vtk`` and
per-part ``.obj`` files for multi-part surfaces. ``inspect_stl`` therefore
runs ``surfaceCheck`` with cwd set to the STL's OWN directory and passes just
the basename: the droppings are diagnostic artifacts (problemFaces feeds
surfaceSubset) and belong next to the inspected surface, where the caller can
find them — not in the server's working directory.

Like the rest of the mechanical layer this module is key-free and stdlib-only
at import (CI runs the unit suite with nothing but pytest installed).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import mechanics

# Conservative built-in units-suspicion threshold (documented default, see
# module docstring): suspect a millimetre export only when the largest
# bounding-box extent is at/above this value.
UNITS_SUSPICION_EXTENT = 1000.0


class SurfaceInspectionError(RuntimeError):
    """surfaceCheck could not produce an inspectable output (missing file,
    failed invocation, truncated/unrecognizable output). Never a fabricated
    report."""


@dataclass
class BoundingBox:
    """The surface's axis-aligned bounding box from the Statistics block."""
    min: tuple[float, float, float]
    max: tuple[float, float, float]
    extents: tuple[float, float, float]   # max - min, per axis


@dataclass
class SurfaceReport:
    """Typed STL surface report parsed from one surfaceCheck output."""
    surface_file: str                     # from the run's own Exec line
    triangles: int
    vertices: int | None
    bounding_box: BoundingBox | None
    closed: bool                          # watertight: all edges on two faces
    edges_connected_to_one_face: int | None       # holes / open boundaries
    edges_connected_to_more_than_two_faces: int | None  # non-manifold edges
    unconnected_parts: int | None
    zones: int | None    # connected areas with consistent normal orientation
    units_suspicious: bool                # extents look like a mm export
    verdict: str                          # "ok", "warnings" or "failed"
    evidence: list[str] = field(default_factory=list)


# --- line patterns (Foundation v10 surfaceCheck output) ---------------------

_EXEC_RE = re.compile(r"^Exec\s*:\s*surfaceCheck\s+(.*)$")
_TRIANGLES_RE = re.compile(r"^Triangles\s*:\s*(\d+)$")
_VERTICES_RE = re.compile(r"^Vertices\s*:\s*(\d+)$")
_BBOX_RE = re.compile(r"^Bounding Box\s*:\s*\(([^)]*)\)\s*\(([^)]*)\)$")
_CLOSED_RE = re.compile(r"^Surface is closed\. All edges connected to two faces\.$")
_NOT_CLOSED_RE = re.compile(
    r"^Surface is not closed since not all edges connected to two faces:$"
)
_ONE_FACE_RE = re.compile(r"^connected to one face\s*:\s*(\d+)$")
_OVER_TWO_FACES_RE = re.compile(r"^connected to >2 faces\s*:\s*(\d+)$")
_PARTS_RE = re.compile(r"^Number of unconnected parts\s*:\s*(\d+)$")
_ZONES_RE = re.compile(
    r"^Number of zones \(connected area with consistent normal\)\s*:\s*(\d+)$"
)
_PROBLEM_DUMP_RE = re.compile(r'^Dumping conflicting face labels to "([^"]+)"')


def _vector(token: str) -> tuple[float, float, float] | None:
    parts = token.split()
    if len(parts) != 3:
        return None
    try:
        x, y, z = (float(p) for p in parts)
    except ValueError:
        return None
    return (x, y, z)


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def parse_surfacecheck_log(text: str) -> SurfaceReport:
    """Parse one plain ``surfaceCheck <file>`` output (Foundation v10 format)
    into a typed :class:`SurfaceReport`. Pure function.

    Raises ``ValueError`` when the output carries no closed/not-closed status
    line or no triangle statistics — a truncated or failed invocation is
    never turned into a report.
    """
    surface_file = ""
    triangles: int | None = None
    vertices: int | None = None
    bounding_box: BoundingBox | None = None
    closed: bool | None = None
    edges_one_face: int | None = None
    edges_over_two: int | None = None
    unconnected_parts: int | None = None
    zones: int | None = None
    problem_dump: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        exec_match = _EXEC_RE.match(line)
        if exec_match:
            surface_file = exec_match.group(1).strip().strip('"')
            continue

        triangles_match = _TRIANGLES_RE.match(line)
        if triangles_match:
            triangles = int(triangles_match.group(1))
            continue
        vertices_match = _VERTICES_RE.match(line)
        if vertices_match:
            vertices = int(vertices_match.group(1))
            continue

        bbox_match = _BBOX_RE.match(line)
        if bbox_match:
            bbox_min = _vector(bbox_match.group(1))
            bbox_max = _vector(bbox_match.group(2))
            if bbox_min is not None and bbox_max is not None:
                bounding_box = BoundingBox(
                    min=bbox_min, max=bbox_max,
                    extents=tuple(hi - lo for hi, lo in zip(bbox_max, bbox_min)),
                )
            continue

        if _CLOSED_RE.match(line):
            # "All edges connected to two faces." — both defective-edge
            # counts are zero by surfaceCheck's own statement.
            closed, edges_one_face, edges_over_two = True, 0, 0
            continue
        if _NOT_CLOSED_RE.match(line):
            closed = False
            continue
        one_face_match = _ONE_FACE_RE.match(line)
        if one_face_match:
            edges_one_face = int(one_face_match.group(1))
            continue
        over_two_match = _OVER_TWO_FACES_RE.match(line)
        if over_two_match:
            edges_over_two = int(over_two_match.group(1))
            continue

        parts_match = _PARTS_RE.match(line)
        if parts_match:
            unconnected_parts = int(parts_match.group(1))
            continue
        zones_match = _ZONES_RE.match(line)
        if zones_match:
            zones = int(zones_match.group(1))
            continue

        dump_match = _PROBLEM_DUMP_RE.match(line)
        if dump_match:
            problem_dump = dump_match.group(1)
            continue

    if triangles is None:
        raise ValueError(
            "surfaceCheck output carries no 'Triangles : N' statistics — "
            "truncated or failed invocation; refusing to report."
        )
    if closed is None:
        raise ValueError(
            "surfaceCheck output carries no closed/not-closed status line — "
            "truncated or failed invocation; refusing to report."
        )

    units_suspicious = (
        bounding_box is not None
        and max(bounding_box.extents) >= UNITS_SUSPICION_EXTENT
    )

    verdict, evidence = _verdict(
        closed, edges_one_face, edges_over_two, unconnected_parts, zones,
        bounding_box, units_suspicious, problem_dump,
    )
    return SurfaceReport(
        surface_file=surface_file, triangles=triangles, vertices=vertices,
        bounding_box=bounding_box, closed=closed,
        edges_connected_to_one_face=edges_one_face,
        edges_connected_to_more_than_two_faces=edges_over_two,
        unconnected_parts=unconnected_parts, zones=zones,
        units_suspicious=units_suspicious, verdict=verdict, evidence=evidence,
    )


def _verdict(closed: bool, edges_one_face: int | None,
             edges_over_two: int | None, unconnected_parts: int | None,
             zones: int | None, bounding_box: BoundingBox | None,
             units_suspicious: bool,
             problem_dump: str | None) -> tuple[str, list[str]]:
    """Apply the classification rules (module docstring) to the parsed facts."""
    if not closed:
        evidence = ["surface is not closed — not watertight"]
        if edges_one_face:
            evidence.append(
                f"{edges_one_face} edge(s) connected to only one face "
                "(holes / open boundaries)"
            )
        if edges_over_two:
            evidence.append(
                f"{edges_over_two} edge(s) connected to more than two faces "
                "(non-manifold)"
            )
        if not edges_one_face and not edges_over_two:
            evidence.append(
                "defective-edge counts could not be parsed — read the raw log"
            )
        if problem_dump:
            evidence.append(
                f'surfaceCheck dumped the conflicting face labels to '
                f'"{problem_dump}" in its working directory '
                "(input for surfaceSubset)"
            )
        return "failed", evidence

    warnings: list[str] = []
    if unconnected_parts is not None and unconnected_parts > 1:
        warnings.append(
            f"{unconnected_parts} unconnected parts — the file carries "
            "multiple shells (legitimate for multiple bodies, wrong for a "
            "single part)"
        )
    if (zones is not None and unconnected_parts is not None
            and zones > unconnected_parts):
        warnings.append(
            f"{zones} normal-orientation zones across {unconnected_parts} "
            "part(s) — the normals flip orientation within a part"
        )
    if units_suspicious and bounding_box is not None:
        warnings.append(
            f"largest bounding-box extent {_fmt(max(bounding_box.extents))} "
            f"is at/above {_fmt(UNITS_SUSPICION_EXTENT)} — suspected "
            "millimetre export of a metre-scale part (OpenFOAM expects "
            "metres; STL carries no unit metadata, so judge from the "
            "geometry's real size)"
        )
    if warnings:
        return "warnings", warnings

    evidence = ["surface is closed (watertight): all edges connected to two faces"]
    if unconnected_parts is not None:
        evidence.append(f"{unconnected_parts} connected part(s)")
    if bounding_box is not None:
        evidence.append(
            "bounding-box extents ("
            + " ".join(_fmt(e) for e in bounding_box.extents)
            + ") are plausible for metres"
        )
    return "ok", evidence


def _fatal_excerpt(text: str, max_lines: int = 5) -> str:
    """The first FOAM FATAL block's opening lines, if any."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "FOAM FATAL" in line:
            return "\n".join(lines[i:i + max_lines]).strip()
    return ""


def inspect_stl(path: str, timeout: int = 600) -> SurfaceReport:
    """Run a plain ``surfaceCheck`` on an STL surface (through the
    sourced-environment execution machinery) and return the typed report.

    Runs with cwd set to the surface's own directory (documented choice, see
    module docstring: surfaceCheck's problemFaces/zoning/part dump files
    belong next to the inspected surface).

    Raises :class:`SurfaceInspectionError` when the file is missing or
    surfaceCheck cannot produce an inspectable output (unreadable surface,
    failed or timed-out invocation) — never a fabricated report.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SurfaceInspectionError(f"STL file does not exist: {path}")

    surface_dir = os.path.dirname(path)
    basename = os.path.basename(path)
    returncode, stdout, stderr = mechanics.run_openfoam_command(
        surface_dir, f'surfaceCheck "{basename}"', timeout
    )
    try:
        return parse_surfacecheck_log(stdout)
    except ValueError as exc:
        detail = _fatal_excerpt(stdout) or _fatal_excerpt(stderr) or stderr.strip()
        raise SurfaceInspectionError(
            f"surfaceCheck did not produce an inspectable output for {path} "
            f"(exit code {returncode})."
            + (f"\n{detail}" if detail else "")
        ) from exc
