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

def test_set_run_note_is_registered_as_tool_sixteen():
    import asyncio

    names = {t.name for t in asyncio.run(fs.mcp.list_tools())}
    assert "set_run_note" in names
    assert len(names) == 16  # the ledger write joins the 15 existing tools


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
