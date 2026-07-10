"""Unit tests for the structured checkMesh assessment (issue #44).

Two seams, both key-free (CI runs these with nothing but pytest):

- The PURE parser (src/meshcheck.py, parse_checkmesh_log) over committed
  fixture logs, asserting on the typed output's fields — census, metric
  values, classifications, verdict, evidence — with expected values read
  from the fixtures BY EYE, never recomputed the parser's way.
- The execution wiring (assess_mesh) at the stubbed subprocess boundary:
  mechanics._run_sourced is monkeypatched (prior art: test_run_sourced.py),
  so flag assembly and error surfacing are exercised without OpenFOAM.

MCP registration for the tool lives in test_mcp_helpers.py (importorskip
pattern — fastmcp is not installed in CI).

Fixture provenance (tests/fixtures/checkmesh/):
- ok/ is the REAL Foundation v10 `checkMesh -allTopology -allGeometry` log
  from the live lid-driven-cavity shakedown mesh ("Mesh OK."), byte-for-byte.
- geometry_failed/, warnings/, topology_failed/ are synthetic variants
  derived from it: genuine header/stats/section shapes kept, failure and
  warning lines written to Foundation v10 checkMesh source conventions
  (" ***..." failed-check marks, "Failed N mesh checks." summaries).
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import mechanics  # noqa: E402
import meshcheck  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "checkmesh"


def _fixture_text(variant: str) -> str:
    return (FIXTURES / variant / "log.checkMesh").read_text()


# ---------------------------------------------------------------------------
# The real "Mesh OK." cavity log: census and verdict
# ---------------------------------------------------------------------------

def test_ok_log_census_read_by_eye():
    # All values read from the fixture's "Mesh stats" block by eye.
    assessment = meshcheck.parse_checkmesh_log(_fixture_text("ok"))

    assert assessment.census.points == 882
    assert assessment.census.faces == 1640
    assert assessment.census.internal_faces == 760
    assert assessment.census.cells == 400
    assert assessment.census.boundary_patches == 3
    assert assessment.census.cell_types["hexahedra"] == 400
    assert assessment.census.cell_types["prisms"] == 0
    assert assessment.census.cell_types["tetrahedra"] == 0
    assert assessment.census.cell_types["polyhedra"] == 0


def test_ok_log_gets_ok_verdict_with_checkmesh_ground_truth():
    assessment = meshcheck.parse_checkmesh_log(_fixture_text("ok"))

    assert assessment.mesh_ok is True          # "Mesh OK." in the log
    assert assessment.failed_checks == 0
    assert assessment.verdict == "ok"
    assert assessment.evidence                 # names what drove the verdict
    assert all(m.classification == "pass" for m in assessment.metrics)


def test_ok_log_flags_come_from_the_exec_line():
    # The fixture's Exec line reads "checkMesh -allTopology -allGeometry".
    assessment = meshcheck.parse_checkmesh_log(_fixture_text("ok"))
    assert assessment.flags == "-allTopology -allGeometry"


def test_ok_log_metric_values_read_by_eye():
    # All values read from the fixture's "Checking geometry..." block by eye.
    assessment = meshcheck.parse_checkmesh_log(_fixture_text("ok"))
    by_name = {m.name: m for m in assessment.metrics}

    assert by_name["max_skewness"].value == pytest.approx(1.66533e-14)
    assert by_name["max_non_orthogonality"].value == pytest.approx(0.0)
    assert by_name["max_aspect_ratio"].value == pytest.approx(1.0)
    assert by_name["min_face_area"].value == pytest.approx(2.5e-05)
    assert by_name["min_volume"].value == pytest.approx(2.5e-07)
    assert by_name["min_edge_length"].value == pytest.approx(0.005)
    assert by_name["max_cell_openness"].value == pytest.approx(1.35525e-16)
    assert by_name["number_of_regions"].value == pytest.approx(1.0)
    # every one of them carries checkMesh's own ok mark
    assert all(by_name[n].checkmesh_ok for n in by_name)


def test_ok_log_distinguishes_topology_from_geometry():
    assessment = meshcheck.parse_checkmesh_log(_fixture_text("ok"))
    by_name = {m.name: m for m in assessment.metrics}

    # topology block (by eye): boundary definition, point usage, regions, ...
    assert by_name["boundary_definition"].check == "topology"
    assert by_name["point_usage"].check == "topology"
    assert by_name["number_of_regions"].check == "topology"
    assert by_name["face_face_connectivity"].check == "topology"
    # geometry block (by eye): skewness, non-orthogonality, pyramids, ...
    assert by_name["max_skewness"].check == "geometry"
    assert by_name["max_non_orthogonality"].check == "geometry"
    assert by_name["face_pyramids"].check == "geometry"
    assert by_name["boundary_openness"].check == "geometry"


# ---------------------------------------------------------------------------
# Geometry failures: checkMesh's *** marks are ground truth
# ---------------------------------------------------------------------------

def test_geometry_failures_drive_failed_verdict_with_named_evidence():
    # Synthetic fixture, by eye: " ***Max skewness = 8.5385, ..." and
    # " ***Number of non-orthogonality errors: 12." under a Max of 105.674;
    # summary "Failed 2 mesh checks."
    assessment = meshcheck.parse_checkmesh_log(_fixture_text("geometry_failed"))

    assert assessment.mesh_ok is False
    assert assessment.failed_checks == 2
    assert assessment.verdict == "failed"
    evidence = "\n".join(assessment.evidence)
    assert "max_skewness" in evidence and "max_non_orthogonality" in evidence
    assert "Failed 2 mesh checks" in evidence


def test_geometry_failure_metrics_carry_value_mark_and_classification():
    assessment = meshcheck.parse_checkmesh_log(_fixture_text("geometry_failed"))
    by_name = {m.name: m for m in assessment.metrics}

    skew = by_name["max_skewness"]
    assert skew.value == pytest.approx(8.5385)
    assert skew.checkmesh_ok is False
    assert skew.classification == "fail"
    assert skew.check == "geometry"

    non_ortho = by_name["max_non_orthogonality"]
    assert non_ortho.value == pytest.approx(105.674)
    assert non_ortho.checkmesh_ok is False
    assert non_ortho.classification == "fail"

    # checks checkMesh itself passed stay pass (e.g. aspect ratio = 1)
    assert by_name["max_aspect_ratio"].classification == "pass"
    # the census still parses on a failing mesh
    assert assessment.census.cells == 400


# ---------------------------------------------------------------------------
# The warn band: marginal-but-legal values under checkMesh's hard limits
# ---------------------------------------------------------------------------

def test_marginal_values_get_warnings_verdict_despite_mesh_ok():
    # Synthetic fixture, by eye: non-orthogonality Max 67.5 (warn band starts
    # at 65), Max skewness = 2.84 OK (band starts at 2), Max aspect ratio =
    # 142.7 OK (band starts at 100); checkMesh itself says "Mesh OK."
    assessment = meshcheck.parse_checkmesh_log(_fixture_text("warnings"))

    assert assessment.mesh_ok is True
    assert assessment.failed_checks == 0
    assert assessment.verdict == "warnings"
    by_name = {m.name: m for m in assessment.metrics}

    non_ortho = by_name["max_non_orthogonality"]
    assert non_ortho.value == pytest.approx(67.5)
    assert non_ortho.checkmesh_ok is True     # checkMesh's mark is honored
    assert non_ortho.classification == "warn"  # the band flags it anyway

    assert by_name["max_skewness"].value == pytest.approx(2.84)
    assert by_name["max_skewness"].classification == "warn"
    assert by_name["max_aspect_ratio"].value == pytest.approx(142.7)
    assert by_name["max_aspect_ratio"].classification == "warn"
    # unrelated checks stay pass
    assert by_name["face_pyramids"].classification == "pass"

    evidence = "\n".join(assessment.evidence)
    assert "max_non_orthogonality" in evidence and "67.5" in evidence
    assert "max_skewness" in evidence and "max_aspect_ratio" in evidence


# ---------------------------------------------------------------------------
# Topology failures are reported distinctly from geometry ones
# ---------------------------------------------------------------------------

def test_topology_failure_is_failed_and_marked_as_topology():
    # Synthetic fixture, by eye: " ***Unused points found in the mesh,
    # number unused: 42" in the topology block; geometry is all OK;
    # summary "Failed 1 mesh checks."
    assessment = meshcheck.parse_checkmesh_log(_fixture_text("topology_failed"))

    assert assessment.verdict == "failed"
    assert assessment.failed_checks == 1
    by_name = {m.name: m for m in assessment.metrics}

    usage = by_name["point_usage"]
    assert usage.check == "topology"
    assert usage.checkmesh_ok is False
    assert usage.classification == "fail"
    assert usage.value == pytest.approx(42)

    # geometry checks all still pass — the failure is topological
    assert all(m.classification == "pass"
               for m in assessment.metrics if m.check == "geometry")
    assert "point_usage" in "\n".join(assessment.evidence)


# ---------------------------------------------------------------------------
# Hard rules: checkMesh's conclusion is a floor, vacuous pass is a trap,
# a summary-less output is never assessed
# ---------------------------------------------------------------------------

def test_verdict_is_never_better_than_checkmesh_own_conclusion():
    # Derived from the real ok log: same checks (none of which we parse as
    # failed), but the summary says "Failed 1 mesh checks." — e.g. a failure
    # shape the parser does not recognize. The verdict must still be failed.
    text = _fixture_text("ok").replace("Mesh OK.", "Failed 1 mesh checks.")
    assessment = meshcheck.parse_checkmesh_log(text)

    assert assessment.mesh_ok is False
    assert assessment.failed_checks == 1
    assert assessment.verdict == "failed"
    evidence = "\n".join(assessment.evidence)
    assert "Failed 1 mesh checks" in evidence


def test_zero_parsed_metrics_never_classifies_ok():
    # Derived from the real ok log: header and summary kept, every check
    # line dropped — "Mesh OK." with nothing verified must not read as ok.
    lines = _fixture_text("ok").splitlines()
    text = "\n".join(
        line for line in lines
        if not line.strip().endswith(" OK.")
        and not line.strip().startswith(("Number of regions", "Mesh non-orthogonality"))
        or line.strip() == "Mesh OK."
    )
    assessment = meshcheck.parse_checkmesh_log(text)

    assert assessment.metrics == []
    assert assessment.mesh_ok is True      # checkMesh's own summary, honored
    assert assessment.verdict != "ok"      # ...but never a vacuous pass
    assert assessment.verdict == "failed"
    assert assessment.evidence


def test_output_without_summary_raises_instead_of_assessing():
    # Derived from the real ok log: truncated before the summary — the shape
    # of a crashed or timed-out checkMesh. No fabricated assessment.
    text = _fixture_text("ok").split("Mesh OK.")[0]
    with pytest.raises(ValueError, match="summary"):
        meshcheck.parse_checkmesh_log(text)


# ---------------------------------------------------------------------------
# assess_mesh: execution wiring at the stubbed subprocess boundary
# (prior art: test_run_sourced.py — no bash, no OpenFOAM)
# ---------------------------------------------------------------------------

# The v10 FATAL block checkMesh emits when a case has no mesh (shape follows
# the fatal fixture precedent in tests/fixtures/convergence/fatal/).
NO_MESH_FATAL = """\
--> FOAM FATAL ERROR:\x20
cannot find file "constant/polyMesh/points"

    From function virtual Foam::autoPtr<Foam::ISstream> Foam::fileOperations::uncollatedFileOperation::readStream(Foam::regIOobject&, const Foam::fileName&, const Foam::word&, bool) const
    in file global/fileOperations/uncollatedFileOperation/uncollatedFileOperation.C at line 538.

FOAM exiting
"""


@pytest.fixture
def stub_checkmesh(monkeypatch):
    """Install a fake mechanics._run_sourced; returns the captured call."""
    captured = {}

    def _install(returncode, stdout, stderr="", timed_out=False):
        def fake_run_sourced(command, cwd, timeout):
            captured["command"], captured["cwd"] = command, cwd
            return returncode, stdout, stderr, timed_out
        monkeypatch.setattr(mechanics, "_run_sourced", fake_run_sourced)
        return captured

    return _install


def test_assess_mesh_runs_checkmesh_with_both_flags_in_the_case_dir(
        tmp_path, stub_checkmesh):
    captured = stub_checkmesh(0, _fixture_text("ok"))

    assessment = meshcheck.assess_mesh(str(tmp_path))

    assert captured["command"] == "checkMesh -allTopology -allGeometry"
    assert captured["cwd"] == str(tmp_path)
    assert assessment.verdict == "ok"
    assert assessment.flags == "-allTopology -allGeometry"
    assert assessment.census.cells == 400


def test_assess_mesh_missing_case_dir_raises_typed_error(tmp_path):
    with pytest.raises(meshcheck.MeshAssessmentError, match="does not exist"):
        meshcheck.assess_mesh(str(tmp_path / "nope"))


def test_assess_mesh_no_mesh_fatal_raises_typed_error(tmp_path, stub_checkmesh):
    # checkMesh on a mesh-less case dies in a FOAM FATAL ERROR before any
    # summary line — a typed error carrying the FATAL excerpt, never a
    # fabricated assessment.
    header = _fixture_text("ok").split("Mesh stats")[0]
    stub_checkmesh(1, header + NO_MESH_FATAL)

    with pytest.raises(meshcheck.MeshAssessmentError) as excinfo:
        meshcheck.assess_mesh(str(tmp_path))
    message = str(excinfo.value)
    assert "FOAM FATAL ERROR" in message
    assert "constant/polyMesh/points" in message


def test_assess_mesh_timeout_raises_typed_error(tmp_path, stub_checkmesh):
    # A timed-out checkMesh yields partial output without a summary; the
    # timeout note run_openfoam_command appends must reach the error.
    partial = _fixture_text("ok").split("Checking geometry...")[0]
    stub_checkmesh(-1, partial, stderr="", timed_out=True)

    with pytest.raises(meshcheck.MeshAssessmentError, match="timed out"):
        meshcheck.assess_mesh(str(tmp_path), timeout=600)
