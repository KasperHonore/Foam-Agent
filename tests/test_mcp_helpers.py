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


def test_assess_mesh_is_registered_as_tool_eighteen():
    import asyncio

    names = {t.name for t in asyncio.run(fs.mcp.list_tools())}
    assert "assess_mesh" in names
    assert len(names) == 18  # the mesh assessment joins the 17 existing tools


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
