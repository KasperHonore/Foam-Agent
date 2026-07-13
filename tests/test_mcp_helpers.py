"""Unit tests for the shared helpers in the FastMCP server.

The CI unit job installs only pytest (no project deps), so this module skips
there via importorskip. Where fastmcp IS installed it covers the extracted
_require_case_dir guard and the _truncate_head/_truncate_tail helpers.
"""

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Import fastmcp BEFORE putting src/mcp on sys.path, so a transitive `import mcp`
# resolves to the pip package rather than this repo's src/mcp/ directory.
pytest.importorskip("fastmcp")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "mcp"))

import fastmcp_server as fs  # noqa: E402


# ---------------------------------------------------------------------------
# _truncate_head / _truncate_tail
# ---------------------------------------------------------------------------

def test_truncate_head_under_limit_is_unchanged():
    assert fs._truncate_head("abc", 10) == "abc"


def test_truncate_head_marks_the_cut_tail():
    assert fs._truncate_head("abcdef", 3) == "abc\n... [truncated]"


def test_truncate_tail_under_limit_is_unchanged():
    assert fs._truncate_tail("abc", 10) == "abc"


def test_truncate_tail_marks_the_cut_head():
    assert fs._truncate_tail("abcdef", 3) == "... [truncated] ...\ndef"


# ---------------------------------------------------------------------------
# _require_case_dir
# ---------------------------------------------------------------------------

def test_require_case_dir_returns_abspath_when_exists(tmp_path):
    assert fs._require_case_dir(str(tmp_path)) == os.path.abspath(str(tmp_path))


def test_require_case_dir_raises_when_missing(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        fs._require_case_dir(str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# search_tutorials
# ---------------------------------------------------------------------------

def test_search_tutorials_index_defaults_to_tutorials_details():
    # The onboarding skills warm the embedding model with a bare
    # search_tutorials call, so `index` must not be a required argument (#15).
    import inspect

    # Depending on the fastmcp version, @mcp.tool returns either the plain
    # function or a FunctionTool wrapper exposing it as .fn.
    fn = getattr(fs.search_tutorials, "fn", fs.search_tutorials)
    param = inspect.signature(fn).parameters["index"]
    assert param.default.default == "openfoam_tutorials_details"


# ---------------------------------------------------------------------------
# write_case_file
# ---------------------------------------------------------------------------

def test_write_case_file_permission_error_is_actionable(tmp_path, monkeypatch):
    # Stale root-owned case dirs (left by pre-non-root images) surface as bare
    # EACCES; the tool must translate that into an actionable message (#16).
    import asyncio

    def deny(path, content):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(fs.mechanics, "save_file", deny)
    fn = getattr(fs.write_case_file, "fn", fs.write_case_file)
    with pytest.raises(PermissionError, match="chown -R openfoam:openfoam"):
        asyncio.run(fn(
            case_dir=str(tmp_path),
            relative_path="system/controlDict",
            content="x",
        ))


# ---------------------------------------------------------------------------
# set_run_note (#32): the one skill-side ledger write
# ---------------------------------------------------------------------------

def test_set_run_note_is_registered():
    import asyncio

    names = {t.name for t in asyncio.run(fs.mcp.list_tools())}
    assert "set_run_note" in names
    # The tool count is pinned by the newest tool's registration test below.


def test_set_run_note_returns_the_updated_row(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setattr(fs.mechanics, "RUNS_DIR", tmp_path)
    fs.mechanics.resolve_case_dir("cavity")

    fn = getattr(fs.set_run_note, "fn", fs.set_run_note)
    resp = asyncio.run(fn(id="0001", note="looks good", archive=None))

    assert (resp.id, resp.case, resp.status) == ("0001", "cavity", "planned")
    assert resp.notes == "looks good"


def test_set_run_note_surfaces_unknown_id_as_typed_error(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setattr(fs.mechanics, "RUNS_DIR", tmp_path)

    fn = getattr(fs.set_run_note, "fn", fs.set_run_note)
    with pytest.raises(ValueError, match="0042"):
        asyncio.run(fn(id="0042", note="ghost", archive=None))


# ---------------------------------------------------------------------------
# parse_solver_log (#39): typed convergence facts, callable via MCP
# ---------------------------------------------------------------------------

FIXTURES = REPO / "tests" / "fixtures" / "convergence"


def test_parse_solver_log_is_registered():
    import asyncio

    names = {t.name for t in asyncio.run(fs.mcp.list_tools())}
    assert "parse_solver_log" in names
    # The tool count is pinned by the newest tool's registration test below.


def test_parse_solver_log_returns_the_typed_verdict():
    import asyncio

    fn = getattr(fs.parse_solver_log, "fn", fs.parse_solver_log)
    resp = asyncio.run(fn(case_dir=str(FIXTURES / "converged"), log_file=""))

    assert resp.solver == "icoFoam"
    assert resp.log_file == "log.icoFoam"  # picked from controlDict application
    assert resp.completed is True
    assert resp.verdict == "converged"
    assert resp.evidence
    assert resp.time.latest_time == pytest.approx(0.5)
    assert [r.field for r in resp.residuals] == ["Ux", "Uy", "p"]
    assert resp.courant.last_max == pytest.approx(0.852134)
    assert resp.fatal_errors == []


def test_parse_solver_log_explicit_log_file_overrides():
    import asyncio

    fn = getattr(fs.parse_solver_log, "fn", fs.parse_solver_log)
    resp = asyncio.run(fn(case_dir=str(FIXTURES / "converged"),
                          log_file="log.blockMesh"))

    assert resp.solver == "blockMesh"
    assert resp.residuals == []


# ---------------------------------------------------------------------------
# assess_mesh (#44): structured checkMesh assessment, callable via MCP
# ---------------------------------------------------------------------------

CHECKMESH_FIXTURES = REPO / "tests" / "fixtures" / "checkmesh"


def test_assess_mesh_is_registered():
    import asyncio

    names = {t.name for t in asyncio.run(fs.mcp.list_tools())}
    assert "assess_mesh" in names
    # The tool count is pinned by the newest tool's registration test below.


def test_assess_mesh_returns_the_typed_assessment(tmp_path, monkeypatch):
    # Stubbed subprocess boundary (prior art: test_run_sourced.py): the tool
    # sees the REAL harvested "Mesh OK." checkMesh output, no OpenFOAM needed.
    import asyncio

    ok_text = (CHECKMESH_FIXTURES / "ok" / "log.checkMesh").read_text()
    monkeypatch.setattr(fs.mechanics, "_run_sourced",
                        lambda command, cwd, timeout: (0, ok_text, "", False))

    fn = getattr(fs.assess_mesh, "fn", fs.assess_mesh)
    resp = asyncio.run(fn(case_dir=str(tmp_path)))

    assert resp.verdict == "ok"
    assert resp.mesh_ok is True
    assert resp.failed_checks == 0
    assert resp.flags == "-allTopology -allGeometry"
    assert resp.census.cells == 400          # by eye from the fixture
    assert resp.census.cell_types["hexahedra"] == 400
    by_name = {m.name: m for m in resp.metrics}
    assert by_name["max_skewness"].classification == "pass"
    assert by_name["max_skewness"].check == "geometry"
    assert resp.evidence


def test_assess_mesh_surfaces_the_typed_error(tmp_path, monkeypatch):
    # A summary-less output (crashed checkMesh) must surface as the typed
    # MeshAssessmentError, never a fabricated assessment.
    import asyncio

    import meshcheck

    monkeypatch.setattr(fs.mechanics, "_run_sourced",
                        lambda command, cwd, timeout: (1, "no mesh here", "", False))

    fn = getattr(fs.assess_mesh, "fn", fs.assess_mesh)
    with pytest.raises(meshcheck.MeshAssessmentError):
        asyncio.run(fn(case_dir=str(tmp_path)))


# ---------------------------------------------------------------------------
# parse_force_coefficients (#55): typed Cd/Cl/Cm, callable via MCP
# ---------------------------------------------------------------------------

FORCECOEFFS_DAT = (REPO / "tests" / "fixtures" / "forcecoeffs" / "cavity" /
                   "postProcessing" / "forceCoeffs1" / "0" / "forceCoeffs.dat")


def test_parse_force_coefficients_is_registered():
    import asyncio

    names = {t.name for t in asyncio.run(fs.mcp.list_tools())}
    assert "parse_force_coefficients" in names
    # The tool count is pinned by the newest tool's registration test below.


def test_parse_force_coefficients_returns_the_typed_analysis_and_stamps(
        tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setattr(fs.mechanics, "RUNS_DIR", tmp_path)
    case_dir = fs.mechanics.resolve_case_dir("cavity")
    time_dir = Path(case_dir) / "postProcessing" / "forceCoeffs1" / "0"
    time_dir.mkdir(parents=True)
    (time_dir / "forceCoeffs.dat").write_text(FORCECOEFFS_DAT.read_text())

    fn = getattr(fs.parse_force_coefficients, "fn", fs.parse_force_coefficients)
    resp = asyncio.run(fn(case_dir=case_dir, function_name=""))

    assert resp.function_name == "forceCoeffs1"
    assert resp.samples == 101                       # by eye from the fixture
    assert resp.reference.mag_u_inf == pytest.approx(1.0)
    assert resp.window.samples == 21
    assert [c.name for c in resp.coefficients] == \
        ["Cm", "Cd", "Cl", "Cl(f)", "Cl(r)"]
    assert resp.key_result == "Cd=-2.501 Cl=0.1275 (tail mean)"
    assert resp.stamped is True                      # the row's cell is filled
    ledger_text = (tmp_path / "ledger.md").read_text(encoding="utf-8")
    assert "Cd=-2.501 Cl=0.1275 (tail mean)" in ledger_text


def test_parse_force_coefficients_surfaces_the_typed_error(tmp_path):
    import asyncio

    import forcecoeffs

    with pytest.raises(forcecoeffs.ForceCoefficientsError, match="forces reference"):
        fn = getattr(fs.parse_force_coefficients, "fn", fs.parse_force_coefficients)
        asyncio.run(fn(case_dir=str(tmp_path), function_name=""))


# ---------------------------------------------------------------------------
# inspect_stl (#60): structured surfaceCheck STL report, callable via MCP
# ---------------------------------------------------------------------------

SURFACECHECK_FIXTURES = REPO / "tests" / "fixtures" / "surfacecheck"


def test_inspect_stl_is_registered():
    import asyncio

    names = {t.name for t in asyncio.run(fs.mcp.list_tools())}
    assert "inspect_stl" in names
    # The tool count is pinned by the newest tool's registration test below.


def test_inspect_stl_returns_the_typed_report(tmp_path, monkeypatch):
    # Stubbed subprocess boundary (prior art: test_run_sourced.py): the tool
    # sees the REAL harvested watertight surfaceCheck output, no OpenFOAM.
    import asyncio

    watertight = (SURFACECHECK_FIXTURES / "watertight" /
                  "log.surfaceCheck").read_text()
    monkeypatch.setattr(fs.mechanics, "_run_sourced",
                        lambda command, cwd, timeout: (0, watertight, "", False))
    stl = tmp_path / "watertight_cube.stl"
    stl.write_text("solid cube\nendsolid cube\n")

    fn = getattr(fs.inspect_stl, "fn", fs.inspect_stl)
    resp = asyncio.run(fn(path=str(stl)))

    assert resp.verdict == "ok"
    assert resp.closed is True
    assert resp.triangles == 12               # by eye from the fixture
    assert resp.vertices == 8
    assert resp.bounding_box.extents == pytest.approx([1.0, 1.0, 1.0])
    assert resp.units_suspicious is False
    assert resp.evidence


def test_inspect_stl_surfaces_the_typed_error(tmp_path, monkeypatch):
    # An output without surfaceCheck's status lines (crashed invocation) must
    # surface as the typed SurfaceInspectionError, never a fabricated report.
    import asyncio

    import stlcheck

    monkeypatch.setattr(fs.mechanics, "_run_sourced",
                        lambda command, cwd, timeout: (1, "no surface here", "", False))
    stl = tmp_path / "broken.stl"
    stl.write_text("not an stl\n")

    fn = getattr(fs.inspect_stl, "fn", fs.inspect_stl)
    with pytest.raises(stlcheck.SurfaceInspectionError):
        asyncio.run(fn(path=str(stl)))


# ---------------------------------------------------------------------------
# import_geometry (#61): STL into constant/triSurface, callable via MCP
# ---------------------------------------------------------------------------

STL_FIXTURE = REPO / "tests" / "fixtures" / "stl" / "watertight_cube.stl"


def test_import_geometry_is_registered():
    import asyncio

    names = {t.name for t in asyncio.run(fs.mcp.list_tools())}
    assert "import_geometry" in names
    # The tool count is pinned by the newest tool's registration test below.


def test_import_geometry_returns_the_typed_result(tmp_path):
    import asyncio

    fn = getattr(fs.import_geometry, "fn", fs.import_geometry)
    resp = asyncio.run(fn(case_dir=str(tmp_path), src_path=str(STL_FIXTURE),
                          scale=None, timeout=600))

    assert resp.dest_path == "constant/triSurface/watertight_cube.stl"
    assert resp.scale is None
    assert resp.size_bytes == STL_FIXTURE.stat().st_size
    assert resp.overwrote is False
    assert (tmp_path / "constant" / "triSurface" / "watertight_cube.stl").is_file()


def test_import_geometry_surfaces_the_typed_error(tmp_path):
    import asyncio

    import mechanics

    fn = getattr(fs.import_geometry, "fn", fs.import_geometry)
    with pytest.raises(mechanics.GeometryImportError, match="does not exist"):
        asyncio.run(fn(case_dir=str(tmp_path), src_path=str(tmp_path / "ghost.stl"),
                       scale=None, timeout=600))


# ---------------------------------------------------------------------------
# estimate_wall_spacing (#67): pure wall-spacing calculator, callable via MCP
# ---------------------------------------------------------------------------

def test_estimate_wall_spacing_is_registered():
    import asyncio

    names = {t.name for t in asyncio.run(fs.mcp.list_tools())}
    assert "estimate_wall_spacing" in names
    # The tool count is pinned by the newest tool's registration test below.


def test_estimate_wall_spacing_returns_the_typed_estimate():
    # The first purely computational tool: no case dir, no filesystem, no
    # subprocess stub — the wrapper passes scalars through and returns the
    # typed result unchanged. Expected values are the frozen known-value
    # literals from test_wall_spacing_unit.py (hand-derived, cited there).
    import asyncio

    fn = getattr(fs.estimate_wall_spacing, "fn", fs.estimate_wall_spacing)
    resp = asyncio.run(fn(velocity=50.0, characteristic_length=1.0,
                          kinematic_viscosity=1.5e-5, target_y_plus=1.0,
                          flow_type="external", expansion_ratio=1.2))

    assert resp.regime == "turbulent"
    assert resp.reynolds_number.value == pytest.approx(3.33333e6, rel=1e-4)
    assert "Schlichting" in resp.skin_friction_coefficient.formula
    assert resp.first_cell_centre_distance.value == pytest.approx(
        7.67185e-6, rel=1e-4)
    assert resp.first_cell_height.value == pytest.approx(
        1.53437e-5, rel=1e-4)
    assert resp.suggested_layer_count.value == 31
    assert resp.evidence


def test_estimate_wall_spacing_surfaces_the_typed_error():
    import asyncio

    import wallspacing

    fn = getattr(fs.estimate_wall_spacing, "fn", fs.estimate_wall_spacing)
    with pytest.raises(wallspacing.WallSpacingError, match="velocity"):
        asyncio.run(fn(velocity=-1.0, characteristic_length=1.0,
                       kinematic_viscosity=1.5e-5, target_y_plus=1.0,
                       flow_type="external", expansion_ratio=1.2))


# ---------------------------------------------------------------------------
# estimate_turbulence_inlet (#68): inlet k/epsilon/omega, callable via MCP
# ---------------------------------------------------------------------------

def test_estimate_turbulence_inlet_is_registered():
    import asyncio

    names = {t.name for t in asyncio.run(fs.mcp.list_tools())}
    assert "estimate_turbulence_inlet" in names
    # The tool count is pinned by the newest tool's registration test below.


def test_estimate_turbulence_inlet_returns_the_typed_estimate():
    # The published pitzDaily inlet (see test_turbulence_inlet_unit.py for
    # the provenance): U=10, I=5%, l=0.00254 m, nu=1e-5 — the wrapper passes
    # scalars through and returns the typed result unchanged.
    import asyncio

    fn = getattr(fs.estimate_turbulence_inlet, "fn", fs.estimate_turbulence_inlet)
    resp = asyncio.run(fn(velocity=10.0, intensity=0.05, length_scale=0.00254,
                          hydraulic_diameter=None, kinematic_viscosity=1e-5))

    assert resp.k.value == pytest.approx(0.375)        # published 0/k value
    assert resp.epsilon.value == pytest.approx(14.855, rel=1e-3)  # 0/epsilon
    assert resp.k.formula == "k = 3/2*(U*I)^2"
    assert resp.c_mu == 0.09
    assert resp.viscosity_ratio.value == pytest.approx(85.199, rel=1e-3)
    assert resp.assumptions == []


def test_estimate_turbulence_inlet_surfaces_the_typed_error():
    import asyncio

    import turbinlet

    fn = getattr(fs.estimate_turbulence_inlet, "fn", fs.estimate_turbulence_inlet)
    with pytest.raises(turbinlet.TurbulenceInletError, match="exactly one"):
        asyncio.run(fn(velocity=10.0, intensity=None, length_scale=None,
                       hydraulic_diameter=None, kinematic_viscosity=None))


# ---------------------------------------------------------------------------
# start_case / case_status (#74): the background run loop, callable via MCP
# ---------------------------------------------------------------------------

CONTROL_DICT = """FoamFile
{
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     icoFoam;
endTime         0.5;
"""

GOOD_LOG = (
    "sigFpe : Enabling floating point exception trapping (FOAM_SIGFPE).\n"
    "Time = 0.5s\n\n"
    "Courant Number mean: 0.222158 max: 0.852134\n"
    "smoothSolver:  Solving for Ux, Initial residual = 2.3091e-07, "
    "Final residual = 2.3091e-07, No Iterations 0\n"
    "End\n"
)


@pytest.fixture
def background_case(tmp_path, monkeypatch):
    """A ledgered temp case with the Allrun seam replaced by a cued child
    process (the mechanics-seam prior art: tests/test_background_run.py)."""
    monkeypatch.setattr(fs.mechanics, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(fs.mechanics, "_BACKGROUND_RUNS", {}, raising=False)
    case_dir = fs.mechanics.resolve_case_dir("cavity")
    (Path(case_dir) / "system").mkdir(parents=True)
    (Path(case_dir) / "system" / "controlDict").write_text(CONTROL_DICT)
    (Path(case_dir) / "Allrun").write_text("#!/bin/sh\nrunApplication icoFoam\n")

    script = tmp_path / "fake_allrun.py"
    script.write_text(
        "import os, time\n"
        f"with open('log.icoFoam', 'w') as fh:\n"
        f"    fh.write({GOOD_LOG!r})\n"
        "deadline = time.time() + 30\n"
        "while time.time() < deadline and not os.path.exists('exit.cue'):\n"
        "    time.sleep(0.05)\n"
    )
    monkeypatch.setattr(fs.mechanics, "_allrun_argv",
                        lambda case_dir: [sys.executable, str(script)])
    yield case_dir
    (Path(case_dir) / "exit.cue").write_text("")
    for run in list(fs.mechanics._BACKGROUND_RUNS.values()):
        if run.process.poll() is None:
            run.process.kill()
            run.process.wait(timeout=10)


def test_start_case_is_registered():
    import asyncio

    names = {t.name for t in asyncio.run(fs.mcp.list_tools())}
    assert "start_case" in names
    # The tool count is pinned by the newest tool's registration test below.


def test_case_status_is_registered_as_tool_twenty_five():
    import asyncio

    names = {t.name for t in asyncio.run(fs.mcp.list_tools())}
    assert "case_status" in names
    assert len(names) == 25  # the background run pair joins the 23 existing tools


def test_start_case_then_case_status_returns_the_typed_lifecycle(background_case):
    import asyncio
    import time

    start_fn = getattr(fs.start_case, "fn", fs.start_case)
    start = asyncio.run(start_fn(case_dir=background_case))

    assert start.run_id == "0001"
    assert start.status == "running"
    assert start.pid > 0

    status_fn = getattr(fs.case_status, "fn", fs.case_status)
    running = asyncio.run(status_fn(run_id=start.run_id))
    assert running.status == "running"
    assert running.elapsed_seconds >= 0

    (Path(background_case) / "exit.cue").write_text("")
    deadline = time.time() + 20
    final = running
    while time.time() < deadline and final.status == "running":
        final = asyncio.run(status_fn(run_id=start.run_id))
        time.sleep(0.05)

    assert final.status == "done"
    assert final.result == "converged"
    assert final.errors == []


def test_case_status_mid_run_progress_is_the_solver_log_response(background_case):
    import asyncio
    import time

    start_fn = getattr(fs.start_case, "fn", fs.start_case)
    start = asyncio.run(start_fn(case_dir=background_case))

    status_fn = getattr(fs.case_status, "fn", fs.case_status)
    deadline = time.time() + 20
    status = asyncio.run(status_fn(run_id=start.run_id))
    while time.time() < deadline and status.progress is None:
        status = asyncio.run(status_fn(run_id=start.run_id))
        time.sleep(0.05)

    # The progress payload is the same typed model parse_solver_log returns.
    assert isinstance(status.progress, fs.SolverLogResponse)
    assert status.progress.time.latest_time == pytest.approx(0.5)


def test_start_case_surfaces_the_typed_error(tmp_path, monkeypatch):
    import asyncio

    import mechanics

    monkeypatch.setattr(fs.mechanics, "RUNS_DIR", tmp_path)
    case_dir = tmp_path / "bare"
    case_dir.mkdir()

    fn = getattr(fs.start_case, "fn", fs.start_case)
    with pytest.raises(mechanics.BackgroundRunError, match="Allrun"):
        asyncio.run(fn(case_dir=str(case_dir)))


def test_case_status_surfaces_the_typed_error(tmp_path, monkeypatch):
    import asyncio

    import mechanics

    monkeypatch.setattr(fs.mechanics, "RUNS_DIR", tmp_path)

    fn = getattr(fs.case_status, "fn", fs.case_status)
    with pytest.raises(mechanics.BackgroundRunError, match="0042"):
        asyncio.run(fn(run_id="0042"))
