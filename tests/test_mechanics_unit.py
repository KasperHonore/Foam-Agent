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


def test_remove_numeric_folders_keeps_zero(tmp_path):
    for name in ("0", "0.5", "1", "constant"):
        (tmp_path / name).mkdir()
    mechanics.remove_numeric_folders(str(tmp_path))
    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {"0", "constant"}


def test_resolve_case_dir():
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
