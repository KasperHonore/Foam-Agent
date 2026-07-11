"""Unit tests for the structured surfaceCheck STL inspection (issue #60).

Two seams, both key-free (CI runs these with nothing but pytest):

- The PURE parser (src/stlcheck.py, parse_surfacecheck_log) over committed
  fixture logs, asserting on the typed report's fields — statistics, edge
  counts, bounding box, units suspicion, verdict, evidence — with expected
  values read from the fixtures BY EYE, never recomputed the parser's way.
- The execution wiring (inspect_stl) at the stubbed subprocess boundary:
  mechanics._run_sourced is monkeypatched (prior art: test_run_sourced.py),
  so command assembly, the cwd choice and error surfacing are exercised
  without OpenFOAM.

MCP registration for the tool lives in test_mcp_helpers.py (importorskip
pattern — fastmcp is not installed in CI).

Fixture provenance (tests/fixtures/surfacecheck/, surfaces in
tests/fixtures/stl/): harvested live for ticket #59 on 2026-07-11 inside the
foamagent container (OpenFOAM v10, Build 10-c4cf895ad8fa). Each
log.surfaceCheck is the complete output of a plain `surfaceCheck
<surface>.stl` on the matching generated STL, bytes kept exactly as captured.
The authoritative provenance tables (variant -> surface -> signal it
carries, regeneration commands) live in the fixture-dir READMEs:
tests/fixtures/surfacecheck/README.md and tests/fixtures/stl/README.md.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import mechanics  # noqa: E402
import stlcheck  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "surfacecheck"


def _fixture_text(variant: str) -> str:
    return (FIXTURES / variant / "log.surfaceCheck").read_text()


# ---------------------------------------------------------------------------
# watertight: the healthy closed 1 m cube — statistics and ok verdict
# ---------------------------------------------------------------------------

def test_watertight_statistics_read_by_eye():
    # All values read from the fixture's "Statistics:" block by eye.
    report = stlcheck.parse_surfacecheck_log(_fixture_text("watertight"))

    assert report.surface_file == "watertight_cube.stl"
    assert report.triangles == 12
    assert report.vertices == 8
    assert report.bounding_box.min == pytest.approx((0.0, 0.0, 0.0))
    assert report.bounding_box.max == pytest.approx((1.0, 1.0, 1.0))
    assert report.bounding_box.extents == pytest.approx((1.0, 1.0, 1.0))


def test_watertight_is_closed_with_zero_defective_edge_counts():
    # "Surface is closed. All edges connected to two faces." — so both
    # defective-edge counts are zero by definition.
    report = stlcheck.parse_surfacecheck_log(_fixture_text("watertight"))

    assert report.closed is True
    assert report.edges_connected_to_one_face == 0
    assert report.edges_connected_to_more_than_two_faces == 0
    assert report.unconnected_parts == 1
    assert report.zones == 1
    assert report.units_suspicious is False


def test_watertight_gets_ok_verdict_with_evidence():
    report = stlcheck.parse_surfacecheck_log(_fixture_text("watertight"))

    assert report.verdict == "ok"
    assert report.evidence                     # names what drove the verdict
    assert "closed" in "\n".join(report.evidence)


# ---------------------------------------------------------------------------
# open: the cube missing its +z face — open edges are a failed verdict
# ---------------------------------------------------------------------------

def test_open_surface_is_not_closed_with_edge_counts_read_by_eye():
    # By eye: "connected to one face : 4" and "connected to >2 faces : 0".
    report = stlcheck.parse_surfacecheck_log(_fixture_text("open"))

    assert report.closed is False
    assert report.edges_connected_to_one_face == 4
    assert report.edges_connected_to_more_than_two_faces == 0
    assert report.triangles == 10
    assert report.vertices == 8
    assert report.unconnected_parts == 1
    assert report.zones == 1


def test_open_surface_fails_with_named_evidence():
    report = stlcheck.parse_surfacecheck_log(_fixture_text("open"))

    assert report.verdict == "failed"
    evidence = "\n".join(report.evidence)
    assert "not closed" in evidence
    assert "4" in evidence and "one face" in evidence
    # the fixture's "Dumping conflicting face labels to \"problemFaces\""
    # side effect is surfaced so the caller knows where to look
    assert "problemFaces" in evidence


# ---------------------------------------------------------------------------
# multi_shell: two disjoint closed cubes — closed, but a warning
# ---------------------------------------------------------------------------

def test_multi_shell_statistics_read_by_eye():
    report = stlcheck.parse_surfacecheck_log(_fixture_text("multi_shell"))

    assert report.triangles == 24
    assert report.vertices == 16
    assert report.bounding_box.max == pytest.approx((4.0, 1.0, 1.0))
    assert report.bounding_box.extents == pytest.approx((4.0, 1.0, 1.0))
    assert report.closed is True
    assert report.unconnected_parts == 2
    assert report.zones == 2


def test_multi_shell_warns_naming_the_part_count():
    report = stlcheck.parse_surfacecheck_log(_fixture_text("multi_shell"))

    assert report.verdict == "warnings"
    assert report.units_suspicious is False    # extent 4 is a plausible metre
    evidence = "\n".join(report.evidence)
    assert "2 unconnected parts" in evidence


# ---------------------------------------------------------------------------
# mm_scaled: a 1 m part exported in millimetres — warns, does NOT fail
# ---------------------------------------------------------------------------

def test_mm_scaled_bounding_box_read_by_eye():
    report = stlcheck.parse_surfacecheck_log(_fixture_text("mm_scaled"))

    assert report.bounding_box.min == pytest.approx((0.0, 0.0, 0.0))
    assert report.bounding_box.max == pytest.approx((1000.0, 1000.0, 1000.0))
    assert report.bounding_box.extents == pytest.approx((1000.0, 1000.0, 1000.0))
    assert report.closed is True
    assert report.unconnected_parts == 1


def test_mm_scaled_raises_units_suspicion_as_warning_not_failure():
    # STL carries no unit metadata; suspicion can only come from extents
    # (#59 harvest). A suspected-mm export warns — it never fails.
    report = stlcheck.parse_surfacecheck_log(_fixture_text("mm_scaled"))

    assert report.units_suspicious is True
    assert report.verdict == "warnings"
    assert report.verdict != "failed"
    evidence = "\n".join(report.evidence)
    assert "1000" in evidence and "millimetre" in evidence


def test_units_suspicion_threshold_is_conservative():
    # The documented threshold: largest extent >= 1000 (a >=1 m part exported
    # in mm). The 1 m watertight cube (extent 1) must never be flagged.
    assert stlcheck.UNITS_SUSPICION_EXTENT == 1000.0
    report = stlcheck.parse_surfacecheck_log(_fixture_text("watertight"))
    assert report.units_suspicious is False


# ---------------------------------------------------------------------------
# Derived variants: orientation flips warn; truncated output never reports
# ---------------------------------------------------------------------------

def test_more_zones_than_parts_reads_as_inconsistent_orientation_warning():
    # Derived from the watertight log: one part, but two normal-orientation
    # zones — normals flip somewhere within the part.
    text = _fixture_text("watertight").replace(
        "Number of zones (connected area with consistent normal) : 1",
        "Number of zones (connected area with consistent normal) : 2",
    )
    report = stlcheck.parse_surfacecheck_log(text)

    assert report.closed is True
    assert report.unconnected_parts == 1
    assert report.zones == 2
    assert report.verdict == "warnings"
    assert "orientation" in "\n".join(report.evidence)


def test_output_truncated_before_closed_status_raises_instead_of_reporting():
    # Derived from the watertight log: cut before the closed/not-closed line
    # — the shape of a crashed or timed-out surfaceCheck. No fabricated report.
    text = _fixture_text("watertight").split("Surface is closed.")[0]
    with pytest.raises(ValueError, match="closed"):
        stlcheck.parse_surfacecheck_log(text)


def test_output_without_statistics_raises_instead_of_reporting():
    # Derived: header only, no "Triangles" statistics at all.
    text = _fixture_text("watertight").split("Statistics:")[0]
    with pytest.raises(ValueError):
        stlcheck.parse_surfacecheck_log(text)


# ---------------------------------------------------------------------------
# inspect_stl: execution wiring at the stubbed subprocess boundary
# (prior art: test_run_sourced.py — no bash, no OpenFOAM)
# ---------------------------------------------------------------------------

# The v10 FATAL block surfaceCheck emits when the surface file cannot be read
# (shape follows the fatal fixture precedent in tests/fixtures/convergence/).
NO_SURFACE_FATAL = """\
--> FOAM FATAL ERROR:\x20
cannot find file "missing.stl"

    From function virtual Foam::autoPtr<Foam::ISstream> Foam::fileOperations::uncollatedFileOperation::readStream(Foam::regIOobject&, const Foam::fileName&, const Foam::word&, bool) const
    in file global/fileOperations/uncollatedFileOperation/uncollatedFileOperation.C at line 538.

FOAM exiting
"""


@pytest.fixture
def stub_surfacecheck(monkeypatch):
    """Install a fake mechanics._run_sourced; returns the captured call."""
    captured = {}

    def _install(returncode, stdout, stderr="", timed_out=False):
        def fake_run_sourced(command, cwd, timeout):
            captured["command"], captured["cwd"] = command, cwd
            return returncode, stdout, stderr, timed_out
        monkeypatch.setattr(mechanics, "_run_sourced", fake_run_sourced)
        return captured

    return _install


def test_inspect_stl_runs_surfacecheck_in_the_surface_own_directory(
        tmp_path, stub_surfacecheck):
    # cwd is the STL's parent directory (documented choice: surfaceCheck
    # dumps problemFaces/zoning files into its cwd on defective surfaces —
    # they belong next to the inspected surface, not in the server root).
    stl = tmp_path / "watertight_cube.stl"
    stl.write_text("solid cube\nendsolid cube\n")
    captured = stub_surfacecheck(0, _fixture_text("watertight"))

    report = stlcheck.inspect_stl(str(stl))

    assert captured["command"] == 'surfaceCheck "watertight_cube.stl"'
    assert captured["cwd"] == str(tmp_path)
    assert report.verdict == "ok"
    assert report.closed is True
    assert report.triangles == 12


def test_inspect_stl_missing_file_raises_typed_error(tmp_path):
    with pytest.raises(stlcheck.SurfaceInspectionError, match="does not exist"):
        stlcheck.inspect_stl(str(tmp_path / "nope.stl"))


def test_inspect_stl_unreadable_surface_raises_typed_error_with_fatal(
        tmp_path, stub_surfacecheck):
    # surfaceCheck dying in a FOAM FATAL ERROR yields no closed-status line —
    # a typed error carrying the FATAL excerpt, never a fabricated report.
    stl = tmp_path / "broken.stl"
    stl.write_text("not really an stl\n")
    header = _fixture_text("watertight").split("Statistics:")[0]
    stub_surfacecheck(1, header + NO_SURFACE_FATAL)

    with pytest.raises(stlcheck.SurfaceInspectionError) as excinfo:
        stlcheck.inspect_stl(str(stl))
    message = str(excinfo.value)
    assert "FOAM FATAL ERROR" in message
    assert "missing.stl" in message


def test_inspect_stl_timeout_raises_typed_error(tmp_path, stub_surfacecheck):
    # A timed-out surfaceCheck yields partial output without a closed-status
    # line; the timeout note run_openfoam_command appends must reach the error.
    stl = tmp_path / "big.stl"
    stl.write_text("solid big\nendsolid big\n")
    partial = _fixture_text("watertight").split("Surface is closed.")[0]
    stub_surfacecheck(-1, partial, stderr="", timed_out=True)

    with pytest.raises(stlcheck.SurfaceInspectionError, match="timed out"):
        stlcheck.inspect_stl(str(stl), timeout=60)
