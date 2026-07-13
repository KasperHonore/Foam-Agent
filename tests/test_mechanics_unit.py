"""Unit tests for the key-free mechanical layer (src/mechanics.py).

These run without OpenFOAM, FAISS indices, or any LLM/API key — they cover
the pure-Python parts: file I/O, log parsing, path resolution and the
agent-asset sync script.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import mechanics  # noqa: E402


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def test_save_scan_read_roundtrip(tmp_path):
    case = tmp_path / "case"
    mechanics.save_file(str(case / "system" / "controlDict"), "FoamFile {}\n")
    mechanics.save_file(str(case / "0" / "U"), "dimensions [0 1 -1 0 0 0 0];\n")

    structure = mechanics.scan_case_directory(str(case))
    assert structure == {"system": ["controlDict"], "0": ["U"]}

    files = mechanics.read_case_files(str(case))
    by_name = {f["file_name"]: f for f in files}
    assert by_name["controlDict"]["folder_name"] == "system"
    assert "dimensions" in by_name["U"]["content"]


def test_scan_case_directory_deep_surfaces_polymesh(tmp_path):
    case = tmp_path / "case"
    mechanics.save_file(str(case / "system" / "controlDict"), "FoamFile {}\n")
    mechanics.save_file(str(case / "constant" / "physicalProperties"), "nu 1e-05;\n")
    mechanics.save_file(str(case / "constant" / "polyMesh" / "boundary"), "3 ()\n")
    mechanics.save_file(str(case / "constant" / "polyMesh" / "points"), "0 ()\n")

    shallow = mechanics.scan_case_directory(str(case))
    assert "constant/polyMesh" not in shallow  # default stays shallow: read_case_files must not ingest mesh data

    deep = mechanics.scan_case_directory(str(case), deep=True)
    assert deep["constant"] == ["physicalProperties"]
    assert sorted(deep["constant/polyMesh"]) == ["boundary", "points"]
    assert deep["system"] == ["controlDict"]


def test_remove_numeric_folders_keeps_zero(tmp_path):
    for name in ("0", "0.5", "1", "constant"):
        (tmp_path / name).mkdir()
    mechanics.remove_numeric_folders(str(tmp_path))
    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {"0", "constant"}


def test_resolve_case_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mechanics, "RUNS_DIR", tmp_path / "runs")
    explicit = mechanics.resolve_case_dir("x", case_dir="/tmp/somewhere")
    assert explicit == "/tmp/somewhere"
    default = mechanics.resolve_case_dir("my_case")
    assert default.endswith(os.path.join("runs", "my_case"))


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def test_check_foam_errors_explicit_error(tmp_path):
    (tmp_path / "log.blockMesh").write_text("stuff\nERROR: something broke\nmore\n")
    errors = mechanics.check_foam_errors(str(tmp_path))
    assert len(errors) == 1
    assert errors[0]["file"] == "log.blockMesh"
    assert "something broke" in errors[0]["error_content"]


def test_check_foam_errors_missing_end_marker(tmp_path):
    (tmp_path / "log.blockMesh").write_text("fine\nEnd\n")
    (tmp_path / "log.icoFoam").write_text("Time = 1\nCourant blew up\n")
    errors = mechanics.check_foam_errors(str(tmp_path))
    assert len(errors) == 1
    assert errors[0]["file"] == "log.icoFoam"
    assert "no 'End' marker" in errors[0]["error_content"]


def test_check_foam_errors_success(tmp_path):
    (tmp_path / "log.icoFoam").write_text("Time = 10\nEnd\n")
    assert mechanics.check_foam_errors(str(tmp_path)) == []


def test_check_foam_errors_vanished_case_dir(tmp_path):
    # Issue #78: a case directory deleted mid-run (external cleanup, changed
    # bind mount) must come back as an actionable infrastructure error, not
    # an uncaught FileNotFoundError — case_status/stop_case/run_case all
    # route their completion gate through this function.
    gone = tmp_path / "was_here"
    gone.mkdir()
    gone.rmdir()
    errors = mechanics.check_foam_errors(str(gone))
    assert len(errors) == 1
    assert errors[0]["file"] == "case_dir"
    assert "vanished" in errors[0]["error_content"]
    assert "bind mount" in errors[0]["error_content"]
    assert str(gone) in errors[0]["error_content"]


def test_extract_commands_from_allrun_out(tmp_path):
    out = tmp_path / "Allrun.out"
    out.write_text("Running blockMesh on case\nnoise\nRunning icoFoam on case\n")
    assert mechanics.extract_commands_from_allrun_out(str(out)) == ["blockMesh", "icoFoam"]


def test_parse_directory_structure():
    data = (
        "<dir>directory name: system. File names in this directory: [controlDict, fvSchemes, fvSolution]</dir>"
        "<dir>directory name: 0. File names in this directory: [U, p]</dir>"
    )
    assert mechanics.parse_directory_structure(data) == {"system": 3, "0": 2}


# ---------------------------------------------------------------------------
# Similar-case retrieval (FAISS calls stubbed out)
# ---------------------------------------------------------------------------

_DIR_STRUCTURE = (
    "<dir>directory name: system. File names in this directory: [controlDict]</dir>\n"
    "<dir>directory name: 0. File names in this directory: [U, p]</dir>"
)


def _fake_retrieve_faiss(calls):
    """Stub returning one matching tutorial; records (database_name, topk)."""
    candidate = {
        "case_name": "cavity",
        "case_domain": "incompressible",
        "case_category": "cavity",
        "case_solver": "icoFoam",
        "score": 0.1,
        "full_content": (
            "<index>\ncase name: cavity\n</index>\n"
            f"<directory_structure>\n{_DIR_STRUCTURE}\n</directory_structure>"
        ),
    }

    def fake(database_name, query, topk=1):
        calls.append((database_name, topk))
        if database_name == "openfoam_allrun_scripts":
            return [{"full_content": "<allrun_script>blockMesh</allrun_script>"}]
        return [candidate]

    return fake


def test_find_similar_case_mirrors_dir_structure_into_selected_case(monkeypatch):
    calls = []
    monkeypatch.setattr(mechanics, "retrieve_faiss", _fake_retrieve_faiss(calls))

    result = mechanics.find_similar_case("my_cavity", "icoFoam", "incompressible", "cavity", 2)

    assert result["found"] is True
    assert result["dir_structure"] == _DIR_STRUCTURE
    assert result["selected_case"]["dir_structure"] == _DIR_STRUCTURE
    assert "<allrun_script>" in result["allrun_reference"]
    assert ("openfoam_allrun_scripts", 2) in calls
    # candidates stay lightweight summaries
    assert "dir_structure" not in result["candidates"][0]


def test_find_similar_case_searchdocs_zero_skips_allrun_retrieval(monkeypatch):
    calls = []
    monkeypatch.setattr(mechanics, "retrieve_faiss", _fake_retrieve_faiss(calls))

    result = mechanics.find_similar_case("my_cavity", "icoFoam", "incompressible", "cavity", 0)

    assert result["found"] is True
    assert result["selected_case"]["dir_structure"] == _DIR_STRUCTURE
    assert result["allrun_reference"] == ""
    assert [db for db, _ in calls] == ["openfoam_tutorials_structure"]


def test_find_similar_case_domain_mismatch_returns_not_found(monkeypatch):
    calls = []
    monkeypatch.setattr(mechanics, "retrieve_faiss", _fake_retrieve_faiss(calls))

    result = mechanics.find_similar_case("my_case", "icoFoam", "multiphase", "cavity", 2)

    assert result["found"] is False
    assert result["selected_case"] is None
    assert result["dir_structure"] == ""


# ---------------------------------------------------------------------------
# Mesh boundary parsing
# ---------------------------------------------------------------------------

def test_read_mesh_boundaries(tmp_path):
    boundary = tmp_path / "constant" / "polyMesh" / "boundary"
    boundary.parent.mkdir(parents=True)
    boundary.write_text(
        """
3
(
    movingWall
    {
        type            wall;
        nFaces          20;
    }
    fixedWalls
    {
        type            wall;
    }
    frontAndBack
    {
        type            empty;
    }
)
"""
    )
    result = mechanics.read_mesh_boundaries(str(tmp_path))
    assert result["exists"] is True
    assert result["boundary_names"] == ["movingWall", "fixedWalls", "frontAndBack"]


def test_read_mesh_boundaries_missing(tmp_path):
    result = mechanics.read_mesh_boundaries(str(tmp_path))
    assert result == {"exists": False, "boundary_names": [], "content": ""}


# ---------------------------------------------------------------------------
# Python script execution (uses the host interpreter, no OpenFOAM)
# ---------------------------------------------------------------------------

def test_run_python_script_expected_output(tmp_path):
    ok, artifact, errors, stdout = mechanics.run_python_script(
        str(tmp_path),
        "print('working'); open('result.txt', 'w').write('hi')",
        filename="make.py",
        expected_output="result.txt",
        timeout_s=30,
    )
    assert ok, errors
    assert artifact.endswith("result.txt")
    assert "working" in stdout


def test_run_python_script_failure(tmp_path):
    ok, artifact, errors, stdout = mechanics.run_python_script(
        str(tmp_path),
        "print('before crash'); raise RuntimeError('boom')",
        filename="fail.py",
        timeout_s=30,
    )
    assert not ok
    assert any("boom" in e for e in errors)
    assert "before crash" in stdout


# ---------------------------------------------------------------------------
# Geometry import (#61): STL into constant/triSurface, optional scaling
# ---------------------------------------------------------------------------

STL_FIXTURES = REPO / "tests" / "fixtures" / "stl"


def _fake_surface_transform(calls, write_dest=True, returncode=0):
    """Stub for mechanics.run_openfoam_command: records the invocation and
    (like the real surfaceTransformPoints) writes the destination file."""
    import shlex

    def fake(case_dir, command, timeout=600):
        calls.append((case_dir, command, timeout))
        if write_dest:
            dest = shlex.split(command)[-1]
            Path(dest).write_bytes(b"scaled surface bytes")
        return returncode, "surfaceTransformPoints stdout", "" if returncode == 0 else "boom"

    return fake


def test_import_geometry_copies_into_trisurface(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    src = STL_FIXTURES / "watertight_cube.stl"

    result = mechanics.import_geometry(str(case), str(src))

    assert result.dest_path == "constant/triSurface/watertight_cube.stl"
    assert result.scale is None
    assert result.overwrote is False
    dest = case / "constant" / "triSurface" / "watertight_cube.stl"
    assert dest.read_bytes() == src.read_bytes()
    assert result.size_bytes == src.stat().st_size


def test_import_geometry_normalizes_the_name(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    src = tmp_path / "My Part (v2).STL"
    src.write_bytes((STL_FIXTURES / "watertight_cube.stl").read_bytes())

    result = mechanics.import_geometry(str(case), str(src))

    assert result.dest_path == "constant/triSurface/My_Part_v2.stl"
    assert (case / "constant" / "triSurface" / "My_Part_v2.stl").is_file()


def test_import_geometry_reports_overwrite(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    src = STL_FIXTURES / "watertight_cube.stl"

    first = mechanics.import_geometry(str(case), str(src))
    second = mechanics.import_geometry(str(case), str(src))

    assert first.overwrote is False
    assert second.overwrote is True
    assert second.dest_path == first.dest_path


def test_import_geometry_missing_source_leaves_case_untouched(tmp_path):
    case = tmp_path / "case"
    case.mkdir()

    with pytest.raises(mechanics.GeometryImportError, match="does not exist"):
        mechanics.import_geometry(str(case), str(tmp_path / "missing.stl"))

    assert list(case.iterdir()) == []  # no constant/, no droppings


def test_import_geometry_scale_runs_the_verified_invocation(tmp_path, monkeypatch):
    # Harvest-verified v10 form (#59): a transformations STRING, not flags.
    import shlex

    case = tmp_path / "case"
    case.mkdir()
    src = STL_FIXTURES / "cube_mm.stl"
    calls = []
    monkeypatch.setattr(mechanics, "run_openfoam_command",
                        _fake_surface_transform(calls))

    result = mechanics.import_geometry(str(case), str(src), scale=0.001)

    assert len(calls) == 1
    case_dir, command, _timeout = calls[0]
    assert case_dir == str(case)
    assert command.startswith('surfaceTransformPoints "scale=(0.001 0.001 0.001)" ')
    tokens = shlex.split(command)
    assert tokens[1] == "scale=(0.001 0.001 0.001)"
    assert tokens[2] == str(src)
    # The utility writes inside triSurface to an .stl name (format inference).
    assert Path(tokens[3]).parent == case / "constant" / "triSurface"
    assert tokens[3].endswith(".stl")

    assert result.scale == 0.001
    assert result.dest_path == "constant/triSurface/cube_mm.stl"
    dest = case / "constant" / "triSurface" / "cube_mm.stl"
    assert dest.read_bytes() == b"scaled surface bytes"
    assert result.size_bytes == len(b"scaled surface bytes")


def test_import_geometry_scale_failure_preserves_existing_surface(tmp_path, monkeypatch):
    case = tmp_path / "case"
    tri = case / "constant" / "triSurface"
    tri.mkdir(parents=True)
    (tri / "cube_mm.stl").write_bytes(b"the original surface")
    src = STL_FIXTURES / "cube_mm.stl"
    calls = []
    monkeypatch.setattr(mechanics, "run_openfoam_command",
                        _fake_surface_transform(calls, write_dest=False, returncode=1))

    with pytest.raises(mechanics.GeometryImportError, match="surfaceTransformPoints"):
        mechanics.import_geometry(str(case), str(src), scale=0.001)

    assert (tri / "cube_mm.stl").read_bytes() == b"the original surface"
    assert [p.name for p in tri.iterdir()] == ["cube_mm.stl"]  # no temp droppings


# ---------------------------------------------------------------------------
# Agent asset sync
# ---------------------------------------------------------------------------

def test_agent_assets_in_sync():
    """Generated per-tool copies must match the canonical agents/ sources."""
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "sync_agent_assets.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
