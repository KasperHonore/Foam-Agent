# mechanics.py
"""Mechanical (non-LLM) capabilities of Foam-Agent.

Everything in this module runs WITHOUT any LLM provider or API key:
file I/O, OpenFOAM execution (blocking and detached background runs),
log parsing, FAISS tutorial retrieval (local HuggingFace embeddings by
default), Python script execution (GMSH meshing / PyVista visualization)
and SLURM job management.

The MCP server (src/mcp/fastmcp_server.py) exposes these functions as
tools. The reasoning that used to be LLM calls inside this repo now
lives in the portable skills/subagents under agents/ and runs on
whatever agent harness the user already has (Claude Code, Cursor,
Codex, OpenCode, ...).

Diagnostics are printed to stderr so the module is safe to import from
a stdio MCP server (stdout is the protocol channel).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import ledger
from text_utils import tokenize

if TYPE_CHECKING:  # runtime import would be a cycle (convergence imports mechanics)
    import convergence

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
DATABASE_DIR = REPO_DIR / "database"
RUNS_DIR = REPO_DIR / "runs"

FAISS_INDEX_NAMES = [
    "openfoam_allrun_scripts",
    "openfoam_tutorials_structure",
    "openfoam_tutorials_details",
    "openfoam_command_help",
]


def _log(message: str) -> None:
    print(message, file=sys.stderr)


# ============================================================================
# Text helpers
# ============================================================================
# `tokenize` is imported from text_utils so the query-time normalization here
# matches the index-time normalization used by the FAISS builders exactly.


def parse_directory_structure(data: str) -> dict:
    """Parse a <dir>...</dir> structure string into {directory: file_count}."""
    directory_file_counts = {}
    dir_blocks = re.findall(r'<dir>(.*?)</dir>', data, re.DOTALL)
    for block in dir_blocks:
        dir_name_match = re.search(r'directory name:\s*(.*?)\.', block)
        files_match = re.search(r'File names in this directory:\s*\[(.*?)\]', block)
        if dir_name_match and files_match:
            dir_name = dir_name_match.group(1).strip()
            files_str = files_match.group(1)
            file_list = [filename.strip() for filename in files_str.split(',')]
            directory_file_counts[dir_name] = len(file_list)
    return directory_file_counts


# ============================================================================
# File I/O
# ============================================================================

def save_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    _log(f"Saved file at {path}")


def read_file(path: str) -> str:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return ""


def list_case_files(case_dir: str) -> str:
    files = [f for f in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, f))]
    return ", ".join(files)


def remove_files(directory: str, prefix: str) -> None:
    for file in os.listdir(directory):
        if file.startswith(prefix):
            os.remove(os.path.join(directory, file))
    _log(f"Removed files with prefix '{prefix}' in {directory}")


def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        _log(f"Removed file {path}")


def remove_numeric_folders(case_dir: str) -> None:
    """Remove time-step folders (numeric names) except '0'."""
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if os.path.isdir(item_path) and item != "0":
            try:
                float(item)
                try:
                    shutil.rmtree(item_path)
                    _log(f"Removed numeric folder: {item_path}")
                except Exception as e:
                    _log(f"Error removing folder {item_path}: {str(e)}")
            except ValueError:
                pass


def scan_case_directory(case_dir: str, deep: bool = False) -> Dict[str, List[str]]:
    """Scan a case directory: {folder: [files]}.

    One level deep by default — callers like read_case_files load every listed
    file into LLM context, and recursing would pull in constant/polyMesh mesh
    data. deep=True recurses, with relative paths ("constant/polyMesh") as keys.
    """
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")

    dir_structure = {}
    base_depth = case_dir.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(case_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        if root == case_dir:
            continue
        current_depth = root.rstrip(os.sep).count(os.sep)
        if not deep and current_depth != base_depth + 1:
            continue
        folder_name = os.path.relpath(root, case_dir).replace(os.sep, "/")
        regular_files = [f for f in files if not f.startswith('.')]
        if regular_files:
            dir_structure[folder_name] = regular_files
    return dir_structure


def read_case_files(case_dir: str, dir_structure: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, str]]:
    """Read all case files into [{file_name, folder_name, content}]."""
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")

    if dir_structure is None:
        dir_structure = scan_case_directory(case_dir)

    result = []
    for folder_name, file_names in dir_structure.items():
        for file_name in file_names:
            file_path = os.path.join(case_dir, folder_name, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                result.append({
                    "file_name": file_name,
                    "folder_name": folder_name,
                    "content": content,
                })
            except UnicodeDecodeError:
                _log(f"Warning: Skipping file due to encoding error: {file_path}")
            except Exception as e:
                _log(f"Warning: Error reading file {file_path}: {e}")
    return result


def _runs_root(run_directory: Optional[str]) -> str:
    """The runs directory whose root owns the ledger (default: repo runs/)."""
    return str(run_directory) if run_directory else str(RUNS_DIR)


def resolve_case_dir(
    case_name: str,
    case_dir: str = "",
    run_times: int = 1,
    run_directory: Optional[str] = None,
) -> str:
    """Resolve the output directory for a case and ledger it as planned.

    Side effect: inserts (idempotently) a planned row for the case into the
    run ledger at the runs root — server-owned tracking, so every run is on
    the record no matter which skill or harness drove it. Out-of-tree
    explicit case_dir values are left untracked.
    """
    base_dir = _runs_root(run_directory)
    if case_dir:
        resolved = case_dir
    elif run_times > 1:
        resolved = os.path.join(base_dir, f"{case_name}_{run_times}")
    else:
        resolved = os.path.join(base_dir, case_name)
    ledger.track_planned(base_dir, resolved)
    return resolved


def set_run_note(run_id: str, note: Optional[str] = None,
                 archive: Optional[bool] = None,
                 run_directory: Optional[str] = None) -> ledger.Row:
    """The one sanctioned skill-side ledger write (spec #28, issue #32).

    Sets a row's Notes cell and/or archives/unarchives it, keyed by the
    row's ledger ID. Everything else stays machine-owned. Raises ValueError
    (ledger untouched) for an unknown ID or an illegal transition.
    """
    return ledger.set_note(_runs_root(run_directory), run_id,
                           note=note, archive=archive)


# ============================================================================
# OpenFOAM execution and log parsing
# ============================================================================

def _openfoam_bashrc() -> str:
    openfoam_dir = os.getenv("WM_PROJECT_DIR")
    if not openfoam_dir:
        raise RuntimeError(
            "WM_PROJECT_DIR is not set. Please source OpenFOAM environment before running Foam-Agent "
            "(e.g., source env/common.sh and env/foamagent.sh)."
        )
    bashrc_path = os.path.join(openfoam_dir, "etc", "bashrc")
    if not os.path.exists(bashrc_path):
        raise RuntimeError(f"OpenFOAM bashrc not found at: {bashrc_path}")
    return bashrc_path


def _run_sourced(command: str, cwd: str, timeout: int) -> Tuple[int, str, str, bool]:
    """Run ``command`` inside the sourced OpenFOAM environment.

    Shared subprocess handling for run_command and run_openfoam_command: source
    the OpenFOAM bashrc, run in a new process group (so a timeout SIGKILLs the
    whole solver/mpirun tree, not just the bash wrapper), and capture output.

    Returns (returncode, stdout, stderr, timed_out). On timeout the process
    group is killed, returncode is -1 and timed_out is True; callers add their
    own timeout message.
    """
    bashrc_path = _openfoam_bashrc()
    full_command = f"source {bashrc_path} && {command}"

    process = subprocess.Popen(
        ['bash', '-c', full_command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout or "", stderr or "", False
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        stdout, stderr = process.communicate()
        return -1, stdout or "", stderr or "", True


def run_command(script_path: str, out_file: str, err_file: str, working_dir: str, max_time_limit: int) -> None:
    """Execute a shell script inside the sourced OpenFOAM environment."""
    _log(f"Executing script {script_path} in {working_dir}")
    os.chmod(script_path, 0o777)

    returncode, stdout, stderr, timed_out = _run_sourced(
        f"bash {os.path.abspath(script_path)}", working_dir, max_time_limit
    )
    if timed_out:
        timeout_message = (
            "OpenFOAM execution took too long. "
            "This case, if set up right, does not require such large execution times.\n"
        )
        stdout = timeout_message + stdout
        stderr = timeout_message + stderr
        _log(f"Execution timed out: {script_path}")

    with open(out_file, 'w') as out, open(err_file, 'w') as err:
        out.write(stdout)
        err.write(stderr)

    _log(f"Executed script {script_path}")


def run_openfoam_command(
    case_dir: str,
    command: str,
    timeout: int = 600,
) -> Tuple[int, str, str]:
    """Run a single OpenFOAM command (e.g. 'checkMesh', 'gmshToFoam mesh.msh')
    inside the sourced OpenFOAM environment, cwd=case_dir.

    Returns (returncode, stdout, stderr).
    """
    returncode, stdout, stderr, timed_out = _run_sourced(command, case_dir, timeout)
    if timed_out:
        stderr = stderr + f"\nCommand timed out after {timeout}s"
    return returncode, stdout, stderr


# ============================================================================
# Geometry import (STL -> constant/triSurface), issue #61
# ============================================================================

class GeometryImportError(RuntimeError):
    """The surface could not be imported (missing source, degenerate name,
    or a failed surfaceTransformPoints run) — the case is left untouched."""


@dataclass
class GeometryImportResult:
    """Typed outcome of one import_geometry call."""
    dest_path: str            # relative to the case, posix-style
    scale: Optional[float]    # None when the surface was copied unscaled
    size_bytes: int           # byte size of the imported surface file
    overwrote: bool           # a surface of this name already existed


_UNSAFE_SURFACE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _normalize_surface_name(basename: str) -> str:
    """Normalize a source basename into the triSurface file name.

    Rule: a case-insensitive trailing '.stl' becomes lowercase '.stl' (appended
    when absent); in the stem, every run of characters outside [A-Za-z0-9._-]
    (whitespace included) becomes a single '_', and leading/trailing '_'/'.'
    are stripped. 'My Part (v2).STL' -> 'My_Part_v2.stl'. A name with nothing
    left after sanitizing raises :class:`GeometryImportError`.
    """
    stem = basename[:-4] if basename.lower().endswith(".stl") else basename
    stem = _UNSAFE_SURFACE_CHARS.sub("_", stem).strip("_.")
    if not stem:
        raise GeometryImportError(
            f"Source basename normalizes to an empty surface name: {basename!r}")
    return stem + ".stl"


def import_geometry(
    case_dir: str,
    src_path: str,
    scale: Optional[float] = None,
    timeout: int = 600,
) -> GeometryImportResult:
    """Import a server-visible STL into <case_dir>/constant/triSurface/.

    The surface lands under its normalized source basename (see
    :func:`_normalize_surface_name`); the triSurface directory is created when
    absent. Without ``scale`` the file is copied byte-for-byte. With ``scale``
    (e.g. 0.001 for mm->m) the copy is produced by Foundation v10's
    surfaceTransformPoints using the harvest-verified transformations-string
    form (#59) — ``surfaceTransformPoints "scale=(s s s)" src dst`` — so unit
    correction is deterministic, never a hand-edited dict.

    Both paths write to a temporary name inside triSurface and rename into
    place, so a failure never leaves a partial surface behind: a missing
    source or a failed surfaceTransformPoints run raises
    :class:`GeometryImportError` with the case untouched. An existing surface
    of the same name is replaced and reported via ``overwrote`` — never
    silently.
    """
    if not os.path.isfile(src_path):
        raise GeometryImportError(f"Source STL does not exist: {src_path}")
    name = _normalize_surface_name(os.path.basename(src_path))

    tri_dir = os.path.join(case_dir, "constant", "triSurface")
    dest = os.path.join(tri_dir, name)
    tmp_dest = os.path.join(tri_dir, f".{name}.import.stl")
    overwrote = os.path.isfile(dest)
    os.makedirs(tri_dir, exist_ok=True)

    try:
        if scale is None:
            shutil.copyfile(src_path, tmp_dest)
        else:
            s = format(scale, "g")
            command = (f'surfaceTransformPoints "scale=({s} {s} {s})" '
                       f"'{os.path.abspath(src_path)}' '{tmp_dest}'")
            returncode, stdout, stderr = run_openfoam_command(
                case_dir, command, timeout)
            if returncode != 0 or not os.path.isfile(tmp_dest):
                tail = (stderr or stdout).strip()[-2000:]
                raise GeometryImportError(
                    f"surfaceTransformPoints failed (exit {returncode}) "
                    f"scaling {src_path}: {tail}")
        os.replace(tmp_dest, dest)
    finally:
        if os.path.isfile(tmp_dest):
            os.remove(tmp_dest)

    _log(f"Imported surface {src_path} -> {dest}"
         + (f" (scale {scale})" if scale is not None else ""))
    return GeometryImportResult(
        dest_path=f"constant/triSurface/{name}",
        scale=scale,
        size_bytes=os.path.getsize(dest),
        overwrote=overwrote,
    )


def check_foam_errors(directory: str) -> list:
    """Check OpenFOAM log files for errors.

    Tier 1: explicit ``ERROR:`` lines. Tier 2 (safety-net): a log file
    missing the ``End`` marker is reported with its last 30 lines.
    """
    error_logs = []
    log_contents = {}

    pattern = re.compile(r"ERROR:(.*)", re.DOTALL)

    for file in os.listdir(directory):
        if file.startswith("log"):
            filepath = os.path.join(directory, file)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
            except (IOError, OSError):
                error_logs.append({"file": file, "error_content": f"Could not read log file: {filepath}"})
                continue

            log_contents[file] = content

            match = pattern.search(content)
            if match:
                error_content = match.group(0).strip()
                error_logs.append({"file": file, "error_content": error_content})
            elif "error" in content.lower():
                _log(f"Warning: file {file} contains 'error' but does not match expected format.")

    if not error_logs and log_contents:
        end_pattern = re.compile(r"^\s*End\s*$", re.MULTILINE)
        for file, content in log_contents.items():
            if not end_pattern.search(content):
                last_lines = "\n".join(content.strip().split("\n")[-30:])
                error_logs.append({
                    "file": file,
                    "error_content": (
                        f"Solver did not complete (no 'End' marker found). "
                        f"Last 30 lines:\n{last_lines}"
                    ),
                })

    return error_logs


def extract_commands_from_allrun_out(out_file: str) -> list:
    commands = []
    if not os.path.exists(out_file):
        return commands
    with open(out_file, 'r') as f:
        for line in f:
            if line.startswith("Running "):
                parts = line.split(" ")
                if len(parts) > 1:
                    commands.append(parts[1].strip())
    return commands


# Mesh tools looked for in Allrun, in stamping order.
_MESH_TOOLS = ("blockMesh", "gmshToFoam", "snappyHexMesh")


def _inspect_solver(case_dir: str) -> str:
    """Best-effort Solver for the ledger: the controlDict application entry.

    Deterministic inspection of case contents; the placeholder when there is
    no controlDict or no application line to read.
    """
    content = read_file(os.path.join(case_dir, "system", "controlDict"))
    match = re.search(r"^\s*application\s+(\S+?)\s*;", content, re.MULTILINE)
    return match.group(1) if match else ledger.PLACEHOLDER


def _inspect_mesh(case_dir: str) -> str:
    """Best-effort Mesh for the ledger: known mesh tools mentioned in Allrun.

    Several tools chain with '+' (e.g. blockMesh+snappyHexMesh); the
    placeholder when Allrun mentions none of them.
    """
    content = read_file(os.path.join(case_dir, "Allrun"))
    found = [tool for tool in _MESH_TOOLS if tool in content]
    return "+".join(found) if found else ledger.PLACEHOLDER


def _parsed_result(case_dir: str) -> str:
    """Result verdict for the done path, from the convergence parser (#40).

    Verdict -> Result mapping (decided in spec #38): Result is 'converged'
    if and only if the parser verdict is 'converged'; ANY other verdict maps
    to 'diverged' — that covers 'diverged' itself and the 'incomplete'
    flavor meaning "completed but final residuals above threshold".
    Conservative by design: the ledger must never overstate trust, and the
    run's typed evidence stays one parse_solver_log call away for anyone
    asking why. ('error' and the no-End 'incomplete' cannot reach the done
    path: check_foam_errors already gates it to the debugging path.)

    A log the parser cannot locate or read (no controlDict application
    entry, missing log file) gets the same conservative 'diverged' — an
    unverifiable run is never presented as trusted.
    """
    # Lazy import: convergence imports mechanics at module top (it reuses
    # _inspect_solver), so a top-level import here would be a cycle — same
    # precedent as the lazy FAISS imports below.
    import convergence
    try:
        return "converged" if convergence.parse_solver_log(case_dir).verdict == "converged" else "diverged"
    except (OSError, ValueError):
        return "diverged"


def _stamp_running(runs_root: str, case_dir: str) -> Optional[ledger.Row]:
    """The run-start stamp shared by blocking and background runs: flip the
    case's row to running (Result back to pending) with the best-effort
    Solver/Mesh inspection. Returns None for out-of-tree case dirs."""
    return ledger.track_running(runs_root, case_dir,
                                solver=_inspect_solver(case_dir),
                                mesh=_inspect_mesh(case_dir))


def _run_output_files(case_dir: str) -> Tuple[str, str]:
    """The Allrun stdout/stderr capture files, shared by both run modes."""
    return os.path.join(case_dir, "Allrun.out"), os.path.join(case_dir, "Allrun.err")


def _sweep_run_artifacts(case_dir: str) -> None:
    """The clean-rerun sweep shared by blocking and background runs:
    previous logs, the Allrun output captures and old time-step folders."""
    out_file, err_file = _run_output_files(case_dir)
    remove_files(case_dir, prefix="log")
    remove_file(err_file)
    remove_file(out_file)
    remove_numeric_folders(case_dir)


def _finish_run(runs_root: str, case_dir: str, error_logs: List[dict]) -> List[dict]:
    """The one completion path every run takes (blocking, background and the
    lazy background heal): the check_foam_errors output gates the row to done
    with the parser-backed Result (#40), or to debugging with Result pending.
    Returns error_logs unchanged so callers can hand them straight back."""
    if not error_logs:
        ledger.track_done(runs_root, case_dir, _parsed_result(case_dir))
    else:
        ledger.track_failed(runs_root, case_dir)
    return error_logs


def run_allrun_and_collect_errors(
    case_dir: str,
    timeout: int = 3600,
    max_retries: int = 1,
    run_directory: Optional[str] = None,
) -> List[dict]:
    """Execute the Allrun script and return any error logs (empty = success).

    Ledger side effect (server-owned, spec #28): the case's row follows the
    run — running while Allrun is in flight, then done plus a Result stamped
    from the convergence parser's verdict (#40) on success, or debugging
    (Result pending) on failure. A missing Allrun script never started a
    run, so the row is left untouched.

    Guard (spec #72): a case with a LIVE background run (start_case) refuses
    with a typed :class:`BackgroundRunError` — never two solvers against one
    case directory. Everything else about the blocking contract is unchanged.
    """
    live = _live_background_run(case_dir)
    if live is not None:
        live_run_id, live_pid = live
        raise BackgroundRunError(
            f"Case already has a live background run "
            f"(run id {live_run_id or 'unknown'}, pid {live_pid}): poll "
            f"case_status until it finishes, or stop it, before run_case."
        )

    allrun_file_path = os.path.join(case_dir, "Allrun")
    if not os.path.exists(allrun_file_path):
        return [{"file": "Allrun", "error_content": f"Allrun script not found at {allrun_file_path}"}]

    runs_root = _runs_root(run_directory)
    _stamp_running(runs_root, case_dir)

    out_file, err_file = _run_output_files(case_dir)
    _sweep_run_artifacts(case_dir)

    last_error_logs: List[dict] = []
    for attempt in range(1, max_retries + 1):
        _log(f"Running Allrun (attempt {attempt}/{max_retries})")
        run_command(allrun_file_path, out_file, err_file, case_dir, timeout)

        error_logs = check_foam_errors(case_dir)
        if len(error_logs) == 0:
            return _finish_run(runs_root, case_dir, [])

        last_error_logs = error_logs
        if attempt < max_retries:
            _log("Allrun reported errors; retrying after cleanup...")
            _sweep_run_artifacts(case_dir)

    return _finish_run(runs_root, case_dir, last_error_logs)


# ============================================================================
# FAISS tutorial retrieval (lazy; local embeddings by default, no API key)
# ============================================================================

_FAISS_CACHE: dict = {}


def _embedding_settings() -> Tuple[str, str]:
    provider = (os.getenv("FOAMAGENT_EMBEDDING_PROVIDER") or "huggingface").strip().lower() or "huggingface"
    model = (os.getenv("FOAMAGENT_EMBEDDING_MODEL") or "Qwen/Qwen3-Embedding-0.6B").strip()
    return provider, model


def get_embedding_model():
    provider, model = _embedding_settings()
    if provider == "openai":
        from langchain_openai.embeddings import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def load_faiss_dbs() -> dict:
    from langchain_community.vectorstores import FAISS

    embedding_model = get_embedding_model()
    _, model_name = _embedding_settings()
    model_dir_name = model_name.replace("/", "_").replace(":", "_")
    db_path = DATABASE_DIR / "faiss" / model_dir_name

    _log(f"Loading FAISS indices from: {db_path} with model: {model_name}")

    dbs = {}
    for index in FAISS_INDEX_NAMES:
        index_path = db_path / index
        if index_path.exists():
            try:
                dbs[index] = FAISS.load_local(
                    str(index_path), embedding_model, allow_dangerous_deserialization=True
                )
            except Exception as e:
                _log(f"Failed to load index {index}: {e}")
        else:
            _log(f"Warning: Index path does not exist: {index_path}")
    return dbs


def _get_faiss_db(database_name: str):
    if database_name not in _FAISS_CACHE:
        _FAISS_CACHE.update(load_faiss_dbs())
    if database_name not in _FAISS_CACHE:
        raise ValueError(f"Database '{database_name}' is not loaded.")
    return _FAISS_CACHE[database_name]


def retrieve_faiss(database_name: str, query: str, topk: int = 1) -> List[dict]:
    """Similarity-search a FAISS index and return formatted results."""
    vectordb = _get_faiss_db(database_name)

    query = tokenize(query)
    try:
        docs_and_scores = vectordb.similarity_search_with_score(query, k=topk)
        docs = [d for d, _ in docs_and_scores]
        scores = [s for _, s in docs_and_scores]
    except Exception:
        docs = vectordb.similarity_search(query, k=topk)
        scores = [None] * len(docs)

    if not docs:
        raise ValueError(f"No documents found for query: {query}")

    formatted_results = []
    for doc, score in zip(docs, scores):
        metadata = doc.metadata or {}
        # FAISS returns numpy.float32 scores, which are not JSON-serializable
        score = float(score) if score is not None else None

        if database_name == "openfoam_allrun_scripts":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "allrun_script": metadata.get("allrun_script", "N/A"),
                "score": score,
            })
        elif database_name == "openfoam_command_help":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "command": metadata.get("command", "unknown"),
                "help_text": metadata.get("help_text", "unknown"),
                "score": score,
            })
        elif database_name == "openfoam_tutorials_structure":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "score": score,
            })
        elif database_name == "openfoam_tutorials_details":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "tutorials": metadata.get("tutorials", "N/A"),
                "score": score,
            })
        else:
            raise ValueError(f"Unknown database name: {database_name}")

    return formatted_results


def find_similar_case(
    case_name: str,
    case_solver: str,
    case_domain: str,
    case_category: str,
    searchdocs: int = 5,
) -> dict:
    """Retrieve the most similar tutorial case plus Allrun references.

    This is the mechanical part of the old plan-service retrieval: recall by
    semantic search, hard-filter on domain, rerank by solver match, and
    collect the matching Allrun scripts. Judging HOW to use the reference
    is the calling agent's job.

    Every call pays two query-embedding passes on the local CPU model; the
    Allrun one embeds the selected case's directory structure and dominates
    the latency. searchdocs=0 skips it.
    """
    case_info = (
        f"case name: {case_name}\ncase domain: {case_domain}\n"
        f"case category: {case_category}\ncase solver: {case_solver}"
    )

    recall_k = max(10, int(searchdocs))
    candidates = retrieve_faiss("openfoam_tutorials_structure", case_info, topk=recall_k)

    def _summary(item: dict) -> dict:
        return {
            "case_name": item.get("case_name"),
            "case_domain": item.get("case_domain"),
            "case_category": item.get("case_category"),
            "case_solver": item.get("case_solver"),
            "score": item.get("score"),
        }

    domain_matched = [c for c in candidates if c.get("case_domain") == case_domain]
    if not domain_matched:
        return {
            "found": False,
            "selected_case": None,
            "tutorial_reference": "",
            "dir_structure": "",
            "allrun_reference": "",
            "candidates": [_summary(c) for c in candidates[:5]],
        }

    def _rank_key(item: dict) -> tuple:
        solver_match = 1 if item.get("case_solver") == case_solver else 0
        score = item.get("score")
        score_val = 0.0 if score is None else float(score)
        return (-solver_match, score_val)

    ranked = sorted(domain_matched, key=_rank_key)
    selected = ranked[0]

    tutorial_reference = re.sub(r"\n{3}", "\n", selected.get("full_content", ""))

    dir_structure = ""
    allrun_reference = ""
    m = re.search(r"<directory_structure>(.*?)</directory_structure>", tutorial_reference, re.DOTALL)
    if m:
        dir_structure = m.group(1).strip()
    if dir_structure and searchdocs > 0:
        index_content = (
            f"<index>\ncase name: {selected.get('case_name')}\ncase solver: {selected.get('case_solver')}\n</index>\n"
            f"<directory_structure>\n{dir_structure}\n</directory_structure>"
        )
        faiss_allrun = retrieve_faiss("openfoam_allrun_scripts", index_content, topk=searchdocs)
        allrun_reference = (
            "Similar cases are ordered, with smaller numbers indicating greater similarity.\n"
        )
        for idx, item in enumerate(faiss_allrun):
            allrun_reference += f"<similar_case_{idx + 1}>{item['full_content']}</similar_case_{idx + 1}>\n\n\n"

    return {
        "found": True,
        "selected_case": dict(_summary(selected), dir_structure=dir_structure),
        "tutorial_reference": tutorial_reference,
        "dir_structure": dir_structure,
        "allrun_reference": allrun_reference,
        "candidates": [_summary(c) for c in ranked[:5]],
    }


# ============================================================================
# Python script execution (GMSH meshing, PyVista visualization)
# ============================================================================

def ensure_foam_file(case_dir: str) -> str:
    """Ensure a .foam marker file exists for visualization tools."""
    case_dir = os.path.abspath(case_dir)
    foam = f"{os.path.basename(case_dir)}.foam"
    foam_path = os.path.join(case_dir, foam)
    if not os.path.exists(foam_path):
        with open(foam_path, 'w'):
            pass
    else:
        os.utime(foam_path, None)
    return foam


def _as_text(stream) -> str:
    """Decode a subprocess stream (bytes or str) to text."""
    return stream.decode() if isinstance(stream, bytes) else str(stream)


def run_python_script(
    case_dir: str,
    script: str,
    *,
    filename: str = "script.py",
    expected_output: Optional[str] = None,
    timeout_s: int = 180,
) -> Tuple[bool, str, List[str], str]:
    """Write and run a Python script with cwd = the case directory.

    Used for GMSH mesh generation and PyVista visualization scripts.
    If expected_output is given, success requires that file to exist and
    be non-empty after execution. Returns (ok, artifact_path, errors, stdout).
    """
    case_dir = os.path.abspath(case_dir)
    script_path = os.path.join(case_dir, filename)
    save_file(script_path, script)

    expected_abs = os.path.abspath(os.path.join(case_dir, expected_output)) if expected_output else None

    try:
        completed = subprocess.run(
            [sys.executable, script_path],
            cwd=case_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
        stdout = _as_text(completed.stdout)

        if expected_abs:
            if os.path.exists(expected_abs) and os.path.getsize(expected_abs) > 0:
                return True, expected_abs, [], stdout
            return False, "", [
                "Script executed but expected output was not created",
                f"expected_output={expected_abs}",
            ], stdout
        return True, "", [], stdout

    except subprocess.TimeoutExpired as e:
        out = _as_text(e.stdout)
        err = _as_text(e.stderr)
        return False, "", [
            f"Script timed out after {timeout_s}s",
            f"STDERR:\n{err}",
        ], out
    except subprocess.CalledProcessError as e:
        err = _as_text(e.stderr)
        out = _as_text(e.stdout)
        return False, "", [
            f"Script execution failed (exit code {e.returncode})\nSTDERR:\n{err}"
        ], out
    except FileNotFoundError:
        return False, "", [f"Python interpreter not found: {sys.executable}"], ""
    except Exception as e:
        return False, "", [f"Unexpected error running script: {str(e)}"], ""


def read_mesh_boundaries(case_dir: str) -> dict:
    """Read constant/polyMesh/boundary and extract patch names."""
    boundary_file = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    if not os.path.exists(boundary_file):
        return {"exists": False, "boundary_names": [], "content": ""}
    content = read_file(boundary_file)
    names = re.findall(r'^\s*([A-Za-z_][\w\-]*)\s*\n\s*\{', content, re.MULTILINE)
    names = [n for n in names if n != "FoamFile"]
    return {"exists": True, "boundary_names": names, "content": content}


# ============================================================================
# SLURM / HPC job management
# ============================================================================

# Jobs whose ledger row this server owns: job_id -> (case_dir, runs_root).
# In-memory is enough — the MCP server that submitted a job is the one whose
# status tool gets polled for it (server-owned writes, spec #28).
_SLURM_JOBS: Dict[str, Tuple[str, str]] = {}


def submit_slurm_job(
    script_path: str,
    case_dir: Optional[str] = None,
    run_directory: Optional[str] = None,
) -> Tuple[Optional[str], bool, str]:
    """Submit a SLURM batch script with sbatch.

    Ledger side effect (server-owned, spec #28): when case_dir is given, a
    successful submission flips the case's row to running — queued-or-running
    is one state to the ledger — and registers the job so check_job_status
    can stamp the outcome later.
    """
    try:
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        job_id_match = re.search(r'Submitted batch job (\d+)', output)
        if job_id_match:
            job_id = job_id_match.group(1)
            if case_dir:
                runs_root = _runs_root(run_directory)
                ledger.track_running(runs_root, case_dir,
                                     solver=_inspect_solver(case_dir), mesh=_inspect_mesh(case_dir))
                _SLURM_JOBS[job_id] = (case_dir, runs_root)
            return job_id, True, ""
        return None, False, f"Could not extract job ID from output: {output}"
    except subprocess.CalledProcessError as e:
        return None, False, f"Failed to submit job: {e.stderr}"
    except Exception as e:
        return None, False, f"Unexpected error: {str(e)}"


# squeue states that mean the job is over without a normal completion.
_SLURM_FAILED_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                        "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}


def check_job_status(job_id: str) -> Tuple[Optional[str], bool, str]:
    """Check a SLURM job via squeue ('COMPLETED' once it left the queue).

    Ledger side effect (server-owned, spec #28): observing a terminal state
    of a job submitted through submit_slurm_job stamps its row exactly like
    a local run — done plus the parser-backed Result (#40) when the case
    logs are clean, debugging when they are not or the scheduler reports a
    failure state.
    """
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "-o", "%T"],
            capture_output=True, text=True, check=True,
        )
        status = result.stdout.strip()
        if not status:
            status = "COMPLETED"
        _stamp_slurm_outcome(job_id, status)
        return status, True, ""
    except subprocess.CalledProcessError as e:
        return None, False, f"Failed to check job status: {e.stderr}"
    except Exception as e:
        return None, False, f"Unexpected error: {str(e)}"


def _stamp_slurm_outcome(job_id: str, status: str) -> None:
    """Move a tracked job's ledger row when its SLURM state is terminal."""
    tracked = _SLURM_JOBS.get(job_id)
    if tracked is None:
        return
    case_dir, runs_root = tracked
    if status == "COMPLETED":
        if check_foam_errors(case_dir):
            ledger.track_failed(runs_root, case_dir)
        else:
            ledger.track_done(runs_root, case_dir, _parsed_result(case_dir))
    elif status in _SLURM_FAILED_STATES:
        ledger.track_failed(runs_root, case_dir)
    else:
        return  # still queued/running — nothing to stamp
    _SLURM_JOBS.pop(job_id, None)


# ============================================================================
# Background (detached) runs: start_case / case_status (issue #74)
#
# Mirrors the SLURM pair's shape: an in-memory registry plus lazy outcome
# stamping at the status poll. The registry is a CACHE — the pidfile in the
# case directory plus the ledger are the truth, so a server restart mid-run
# reconciles through them.
# ============================================================================

PIDFILE_BASENAME = "Allrun.pid"

# Whether this platform exposes the procfs the (pid, starttime) identity and
# the zombie check read. False on Windows (and macOS), where the platform
# branch of _pid_alive is the whole liveness verdict.
_HAS_PROC = os.path.isdir("/proc")


class BackgroundRunError(RuntimeError):
    """A background run could not be started, stopped or resolved: missing
    Allrun, a case that already has a live run, an out-of-tree case
    directory, an unknown run id, or a stop against a run that is not
    live."""


@dataclass
class BackgroundRunStart:
    """Typed outcome of one start_case call."""
    run_id: str        # the LEDGER run id (zero-padded, e.g. '0005')
    case: str          # case key (case dir relative to the runs root)
    case_dir: str      # absolute case directory
    pid: int           # pid of the detached Allrun process
    status: str        # ledger Status after the start ('running')
    solver: str        # Solver stamped on the row ('-' when unreadable)
    mesh: str          # Mesh stamped on the row ('-' when unreadable)


@dataclass
class BackgroundRunStatus:
    """Typed outcome of one case_status poll."""
    run_id: str
    case: str
    status: str                        # ledger Status after this poll
    result: str                        # ledger Result after this poll
    pid: Optional[int]                 # pid while running, None otherwise
    elapsed_seconds: Optional[float]   # wall seconds since start, while running
    progress: Optional[convergence.SolverLogAnalysis]  # typed progress mid-run
    errors: List[dict]                 # extracted errors when this poll gated to debugging


@dataclass
class _BackgroundRun:
    """Registry entry for a run this server process launched."""
    case_dir: str
    runs_root: str
    process: subprocess.Popen
    started: float


# Runs whose Popen handle this server owns, keyed by the LEDGER run id.
_BACKGROUND_RUNS: Dict[str, _BackgroundRun] = {}


def _pid_alive(pid: Optional[int]) -> bool:
    """True when a process with this pid is currently alive.

    POSIX (the server's runtime): signal 0 probes without touching the
    process; our own unreaped child is reaped first (waitpid WNOHANG) so an
    exited-but-unreaped child reads as dead. Windows (test platform only):
    os.kill(pid, 0) would TERMINATE the process there, so liveness comes
    from OpenProcess + GetExitCodeProcess instead.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.waitpid(pid, os.WNOHANG)  # reap our own zombie child, if it is one
    except (ChildProcessError, OSError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive, owned by someone else
    return True


def _parse_proc_stat(text: str) -> Optional[Tuple[str, Optional[int]]]:
    """(state letter, starttime) out of a /proc/<pid>/stat line, else None.

    Pure parse, unit-testable without a live /proc. The comm field (field 2)
    is parenthesized and may itself contain spaces and parentheses, so the
    split anchors on the LAST ')'. State is field 3 ('Z' = zombie); starttime
    is field 22, jiffies since HOST boot — boot-monotonic across container
    restarts, so (pid, starttime) is an airtight same-process identity
    (harvest #73: the pid namespace resets on restart and reuse is real).
    """
    _, sep, tail = text.rpartition(")")
    if not sep:
        return None
    fields = tail.split()
    if not fields:
        return None
    state = fields[0]
    starttime: Optional[int] = None
    if len(fields) >= 20:  # fields[0] is field 3, so field 22 is fields[19]
        try:
            starttime = int(fields[19])
        except ValueError:
            starttime = None
    return state, starttime


def _read_proc_stat(pid: int) -> Optional[Tuple[str, Optional[int]]]:
    """(state, starttime) from /proc/<pid>/stat, or None where that interface
    is unavailable (Windows/macOS) or the process entry is gone."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as fh:
            return _parse_proc_stat(fh.read())
    except OSError:
        return None


def _proc_starttime(pid: int) -> Optional[int]:
    """The process's boot-monotonic starttime, or None without /proc."""
    stat = _read_proc_stat(pid)
    return stat[1] if stat is not None else None


def _pidfile_path(case_dir: str) -> str:
    return os.path.join(case_dir, PIDFILE_BASENAME)


def _write_pidfile(case_dir: str, run_id: str, pid: int, started: float) -> None:
    """Record the detached run in the case directory — the recovery truth.

    Carries the (pid, starttime) identity (starttime null where /proc is
    unavailable) so a later reader can tell our process from an unrelated
    one that reused the pid after a container restart (#73).
    """
    pgid = None
    if hasattr(os, "getpgid"):
        try:
            pgid = os.getpgid(pid)
        except OSError:
            pgid = None  # the child is already gone; liveness will say so
    payload = {"run_id": run_id, "pid": pid, "pgid": pgid, "started": started,
               "starttime": _proc_starttime(pid)}
    with open(_pidfile_path(case_dir), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _read_pidfile(case_dir: str) -> Optional[dict]:
    """The pidfile's payload, or None when absent/unreadable (never raises)."""
    path = _pidfile_path(case_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("pid"), int):
        return None
    return data


def _pidfile_alive(info: dict) -> bool:
    """Liveness for a pidfile-recorded run: pid alive AND the same process
    AND not a zombie.

    Harvest #73 hardening: kill(pid, 0) succeeds on a zombie (PID-1 python
    never reaps orphans), so the /proc state letter overrules it; and the
    pid namespace resets on a container restart, so a recorded starttime
    that disagrees with the live process means the pid was REUSED by an
    unrelated process. A pidfile without starttime (pre-#75) and platforms
    without /proc (Windows: _pid_alive's handle check) keep the old verdict.
    """
    pid = info["pid"]
    if not _pid_alive(pid):
        return False
    stat = _read_proc_stat(pid)
    if stat is None:
        # No /proc on this platform -> the platform check above is final.
        # /proc exists but the entry vanished -> it died in the gap.
        return not _HAS_PROC
    state, starttime = stat
    if state == "Z":
        return False  # an unreaped corpse is not a live run
    recorded = info.get("starttime")
    if isinstance(recorded, int) and isinstance(starttime, int) \
            and recorded != starttime:
        return False  # pid reused by an unrelated process after a restart
    return True


def _live_background_run(case_dir: str) -> Optional[Tuple[Optional[str], int]]:
    """(run_id, pid) of the case's live background run, or None.

    Registry first (the cheap cache), then the pidfile (the truth a server
    restart leaves behind). A dead pid never counts as live, so a stale
    pidfile or a row stuck 'running' never blocks a new run.
    """
    case_dir = os.path.abspath(case_dir)
    for run_id, run in _BACKGROUND_RUNS.items():
        if os.path.abspath(run.case_dir) == case_dir and run.process.poll() is None:
            return run_id, run.process.pid
    info = _read_pidfile(case_dir)
    if info is not None and _pidfile_alive(info):
        return info.get("run_id"), info["pid"]
    return None


def _allrun_argv(case_dir: str) -> List[str]:
    """The detached counterpart of run_command's invocation: the case's
    Allrun inside the sourced OpenFOAM environment. Tests substitute this
    seam with a test-controlled child process."""
    allrun = os.path.abspath(os.path.join(case_dir, "Allrun"))
    os.chmod(allrun, 0o777)
    return ["bash", "-c", f"source {_openfoam_bashrc()} && bash {allrun}"]


def start_case(case_dir: str, run_directory: Optional[str] = None) -> BackgroundRunStart:
    """Start the case's Allrun detached; return its ledger run id immediately.

    Performs exactly the blocking run's preparation and ledger side effects —
    the running stamp with the inspected solver/mesh, then the clean-rerun
    sweep — and launches Allrun in its own session/process group with
    stdout/stderr captured to Allrun.out/Allrun.err, the same files run_case
    writes. A pidfile (Allrun.pid: run id, pid, process group, wall start
    time, plus the /proc starttime that makes the pid-reuse-proof process
    identity — null where /proc is unavailable) lands in the case directory
    and the run is registered in-memory keyed by
    the ledger run id; poll case_status with that id to observe progress and
    have the outcome stamped. No watchdog and no timeout: the case's own
    endTime bounds the run.

    Typed errors (:class:`BackgroundRunError`): missing Allrun; a case that
    already has a live run (live registry entry, live pidfile process, or a
    'running' row with a live pid); a case directory outside the runs root
    (no ledger row means no run id to hand back).
    """
    case_dir = os.path.abspath(case_dir)

    # Same guard order as run_allrun_and_collect_errors: the already-running
    # refusal wins over every other error when both apply.
    live = _live_background_run(case_dir)
    if live is not None:
        live_run_id, live_pid = live
        raise BackgroundRunError(
            f"Case already has a live background run "
            f"(run id {live_run_id or 'unknown'}, pid {live_pid}): poll "
            f"case_status until it finishes before starting another."
        )

    allrun_file_path = os.path.join(case_dir, "Allrun")
    if not os.path.exists(allrun_file_path):
        raise BackgroundRunError(f"Allrun script not found at {allrun_file_path}")

    runs_root = _runs_root(run_directory)
    row = _stamp_running(runs_root, case_dir)
    if row is None:
        raise BackgroundRunError(
            f"Case directory {case_dir} is outside the runs root ({runs_root}): "
            "background runs are keyed by their ledger run id, so the case "
            "must live under the runs directory."
        )

    _sweep_run_artifacts(case_dir)

    out_file, err_file = _run_output_files(case_dir)
    started = time.time()
    with open(out_file, "w") as out, open(err_file, "w") as err:
        process = subprocess.Popen(
            _allrun_argv(case_dir),
            cwd=case_dir,
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # its own session/process group (POSIX; ignored on Windows)
        )
    _write_pidfile(case_dir, row.id, process.pid, started)
    _BACKGROUND_RUNS[row.id] = _BackgroundRun(
        case_dir=case_dir, runs_root=runs_root, process=process, started=started)
    _log(f"Started background Allrun for {case_dir} (run {row.id}, pid {process.pid})")
    return BackgroundRunStart(run_id=row.id, case=row.case, case_dir=case_dir,
                              pid=process.pid, status=row.status,
                              solver=row.solver, mesh=row.mesh)


def _in_flight_progress(case_dir: str):
    """Typed progress on the partial solver log, or None before it exists.

    parse_solver_log handles partial logs by design (verdict 'incomplete'
    mid-run); early in a run the solver log may not exist yet, and a case
    without a readable controlDict application entry cannot name its log —
    both simply mean 'no progress to report yet'.
    """
    # Lazy import: convergence imports mechanics at module top (same
    # precedent as _parsed_result).
    import convergence
    try:
        return convergence.parse_solver_log(case_dir)
    except (OSError, ValueError):
        return None


def _finish_background_run(run_id: str, runs_root: str, case_dir: str) -> List[dict]:
    """Stamp an exited background run through the shared completion path,
    then deregister it and drop its pidfile (a dead pid is stale truth)."""
    errors = _finish_run(runs_root, case_dir, check_foam_errors(case_dir))
    _BACKGROUND_RUNS.pop(run_id, None)
    remove_file(_pidfile_path(case_dir))
    return errors


def _running_status(run_id: str, row: ledger.Row, case_dir: str,
                    pid: int, elapsed: Optional[float]) -> BackgroundRunStatus:
    return BackgroundRunStatus(run_id=run_id, case=row.case, status=row.status,
                               result=row.result, pid=pid,
                               elapsed_seconds=elapsed,
                               progress=_in_flight_progress(case_dir), errors=[])


def _ledger_row(runs_root: str, run_id: str) -> ledger.Row:
    """The row for an id known to exist (resolved earlier in the call)."""
    return next(r for r in ledger.read_rows(runs_root) if r.id == run_id)


def _final_status(run_id: str, runs_root: str, errors: List[dict]) -> BackgroundRunStatus:
    row = _ledger_row(runs_root, run_id)
    return BackgroundRunStatus(run_id=run_id, case=row.case, status=row.status,
                               result=row.result, pid=None, elapsed_seconds=None,
                               progress=None, errors=errors)


def _resolve_run_row(runs_root: str, run_id: str) -> Tuple[str, ledger.Row]:
    """(normalized key, row) for a run id, shared by case_status/stop_case.

    Typed error (:class:`BackgroundRunError`): unknown run id (no row).
    """
    key = run_id.strip()
    if key.isdigit():
        key = f"{int(key):04d}"  # same normalization as ledger.set_note
    row = next((r for r in ledger.read_rows(runs_root) if r.id == key), None)
    if row is None:
        raise BackgroundRunError(
            f"No ledger row with ID '{key}' in "
            f"{os.path.join(runs_root, ledger.LEDGER_BASENAME)}")
    return key, row


def case_status(run_id: str, run_directory: Optional[str] = None) -> BackgroundRunStatus:
    """Poll a background run by its ledger run id.

    Process alive -> status 'running' plus typed in-flight progress: the
    convergence parser on the partial solver log (verdict 'incomplete' by
    construction mid-run) and elapsed wall seconds. Process exited -> the
    IDENTICAL completion path the blocking run takes (check_foam_errors
    gates to done with the parser-backed Result, or to debugging with the
    extracted errors), then the run is deregistered and the final ledger
    state returned. A registry miss (server restarted mid-run) reconciles
    through the pidfile: live pid -> running; dead pid with a row stuck
    'running' -> stamped lazily through the same completion path. The
    registry is a cache; the pidfile plus the ledger are the truth.

    Typed error (:class:`BackgroundRunError`): unknown run id (no row).
    """
    runs_root = _runs_root(run_directory)
    key, row = _resolve_run_row(runs_root, run_id)
    case_dir = os.path.abspath(os.path.join(runs_root, row.case))

    run = _BACKGROUND_RUNS.get(key)
    if run is not None:
        if run.process.poll() is None:
            return _running_status(key, row, run.case_dir, run.process.pid,
                                   time.time() - run.started)
        errors = _finish_background_run(key, run.runs_root, run.case_dir)
        return _final_status(key, run.runs_root, errors)

    info = _read_pidfile(case_dir)
    if info is not None and _pidfile_alive(info):
        started = info.get("started")
        elapsed = time.time() - started if isinstance(started, (int, float)) else None
        return _running_status(key, row, case_dir, info["pid"], elapsed)

    if row.status == "running":
        # A row stuck running with no live process (server restarted mid-run,
        # or the run exited unpolled): heal through the same completion path.
        errors = _finish_background_run(key, runs_root, case_dir)
        return _final_status(key, runs_root, errors)

    return BackgroundRunStatus(run_id=key, case=row.case, status=row.status,
                               result=row.result, pid=None, elapsed_seconds=None,
                               progress=None, errors=[])


# ----------------------------------------------------------------------------
# stop_case (issue #75): graceful-first stop, evidence kept.
#
# Every number below is a live-harvested v10 fact (issue #73), not folklore:
# the graceful route needs runTimeModifiable true (compiled default FALSE,
# Time.C:363); the stopAt-writeNow edit stays invisible until the dict's
# mtime clears fileModificationSkew (10 s default) past the solver's last
# read (fileMonitor.C:411) — a future-stamped touch opens the gate at once;
# a group SIGTERM takes the whole detached tree out in <1 s but SIGTERM is
# never graceful (writeNow signals compiled off, exit 143, no fields); and a
# leftover stopAt writeNow insta-stops any rerun after one iteration.
# ----------------------------------------------------------------------------

_STOP_POLL_SECONDS = 0.05          # process poll cadence inside the stop loops
_STOP_TOUCH_INTERVAL_SECONDS = 2.0  # re-touch cadence while waiting (harvest #73)
_MTIME_GATE_MARGIN_SECONDS = 30.0  # comfortably past the 10 s skew default
_KILL_ESCALATION_SECONDS = 5.0     # per-rung wait before/after SIGKILL

# OpenFOAM Switch spellings that read as true (Switch.C word variants).
_TRUE_SWITCH = frozenset({"true", "on", "yes", "y", "t", "1"})


@dataclass
class BackgroundRunStop:
    """Typed outcome of one stop_case call."""
    run_id: str
    case: str
    status: str          # ledger Status after the stop ('done'/'debugging')
    result: str          # ledger Result after the stop (parser-backed)
    method: str          # 'graceful' | 'killed' | 'already_exited'
    note: Optional[str]  # the deliberate-stop note recorded on the row
    #                      (None when the run had already exited on its own)
    errors: List[dict]   # extracted errors when the gate landed on debugging


def _control_dict_path(case_dir: str) -> str:
    return os.path.join(case_dir, "system", "controlDict")


def _run_time_modifiable(case_dir: str) -> bool:
    """True when the case's controlDict enables runtime dict re-reads.

    v10's COMPILED default is false (harvest #73: Time.C:363, verified live
    — 35 s of edits ignored without the entry), so an absent entry means the
    writeNow route can never be seen and the stop goes straight to the kill
    ladder. Tutorials often ship 'true' explicitly, masking the default.
    """
    try:
        with open(_control_dict_path(case_dir), encoding="utf-8",
                  errors="replace") as fh:
            text = fh.read()
    except OSError:
        return False
    match = re.search(r"^\s*runTimeModifiable\s+(\w+)\s*;", text, re.MULTILINE)
    return match is not None and match.group(1).lower() in _TRUE_SWITCH


def _set_control_dict_entry(path: str, key: str, value: str) -> None:
    """Rewrite (or append) one top-level controlDict entry in place.

    The sed-equivalent the harvest verified: the edit mechanism is
    irrelevant (foamDictionary and sed both work) — only the mtime gate
    decides whether the solver sees it.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    pattern = re.compile(rf"^(\s*{key}\s+)[^;]*;", re.MULTILINE)
    new_text, count = pattern.subn(rf"\g<1>{value};", text, count=1)
    if count == 0:
        new_text = text.rstrip("\n") + f"\n\n{key}         {value};\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new_text)


def _touch_past_mtime_gate(path: str) -> None:
    """Stamp the dict's mtime past v10's fileModificationSkew gate.

    fileMonitor fires only when the file's mtime exceeds its
    mtime-at-last-read by MORE than fileModificationSkew (10 s default), so
    a single edit landing shortly after the solver last read the dict is
    invisible FOREVER (harvest #73, observed twice). A future mtime
    (now + margin > skew) opens the gate immediately.
    """
    future = time.time() + _MTIME_GATE_MARGIN_SECONDS
    try:
        os.utime(path, (future, future))
    except OSError:
        pass  # the dict may be mid-rewrite; the next touch retries


def _stop_target_alive(process: Optional[subprocess.Popen],
                       info: Optional[dict]) -> bool:
    """Liveness for the run being stopped: our own Popen (poll() reaps) when
    the registry has it, else the pidfile identity check."""
    if process is not None:
        return process.poll() is None
    return info is not None and _pidfile_alive(info)


def _graceful_stop(case_dir: str, process: Optional[subprocess.Popen],
                   info: Optional[dict], grace_seconds: float) -> Tuple[bool, bool]:
    """The harvest-pinned graceful recipe. Returns (attempted, exited).

    Precondition gate first: without runTimeModifiable true the edit can
    never be seen — not attempted, straight to the kill ladder. Otherwise:
    write stopAt writeNow, then defeat the mtime-skew gate with a
    future-stamped touch, re-stamped every ~2 s while polling the process
    for the length of the grace window. On a graceful exit the solver exits
    0 at the next time-step boundary, writes the current time directory and
    the log ends 'End'.
    """
    if grace_seconds <= 0 or not _run_time_modifiable(case_dir):
        return False, False
    path = _control_dict_path(case_dir)
    try:
        _set_control_dict_entry(path, "stopAt", "writeNow")
    except OSError:
        return False, False
    deadline = time.time() + grace_seconds
    next_touch = 0.0
    while True:
        now = time.time()
        if now >= next_touch:
            _touch_past_mtime_gate(path)
            next_touch = now + _STOP_TOUCH_INTERVAL_SECONDS
        if not _stop_target_alive(process, info):
            return True, True
        if now >= deadline:
            return True, False
        time.sleep(min(_STOP_POLL_SECONDS, deadline - now))


def _signal_stop_target(pid: int, pgid: Optional[int],
                        process: Optional[subprocess.Popen], *, force: bool) -> None:
    """One kill rung. POSIX: signal the whole process group (with
    start_new_session the child's pid IS the pgid) — TERM first, KILL when
    forced. Windows (test platform): TerminateProcess via the Popen handle
    or os.kill; there is no group to address and no graceful signal anyway."""
    if os.name == "nt":
        try:
            if process is not None:
                process.kill()
            else:
                os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows
        except OSError:
            pass
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    target_pgid = pgid if isinstance(pgid, int) and pgid > 0 else None
    if target_pgid is None:
        try:
            target_pgid = os.getpgid(pid)
        except OSError:
            target_pgid = None
    try:
        if target_pgid is not None:
            os.killpg(target_pgid, sig)
        else:
            os.kill(pid, sig)
    except OSError:
        pass  # already gone (ProcessLookupError) or not ours to signal


def _kill_stop_target(pid: int, pgid: Optional[int],
                      process: Optional[subprocess.Popen],
                      info: Optional[dict]) -> bool:
    """The kill ladder: group TERM, short wait, SIGKILL escalation.

    Harvest #73: the group TERM took out every member in <1 s live; the
    corpses may linger as zombies (PID-1 python never reaps orphans), which
    the liveness check reads as dead via the /proc state letter. Returns
    True when the run is down.
    """
    for force in (False, True):
        _signal_stop_target(pid, pgid, process, force=force)
        deadline = time.time() + _KILL_ESCALATION_SECONDS
        while time.time() < deadline:
            if not _stop_target_alive(process, info):
                return True
            time.sleep(_STOP_POLL_SECONDS)
    return not _stop_target_alive(process, info)


def _record_stop_note(runs_root: str, run_id: str, method: str) -> str:
    """Record the deliberate stop through the ONE sanctioned note write
    (ledger.set_note — the same call set_run_note makes), appending to any
    existing note instead of clobbering it. The note is what keeps a stopped
    run from reading like a crash next week."""
    row = _ledger_row(runs_root, run_id)
    text = f"stopped deliberately via stop_case ({method})"
    existing = row.notes.strip()
    note = f"{existing}; {text}" if existing and existing != ledger.PLACEHOLDER else text
    ledger.set_note(runs_root, run_id, note=note)
    return note


def _stopped_status(run_id: str, runs_root: str, *, method: str,
                    note: Optional[str], errors: List[dict]) -> BackgroundRunStop:
    row = _ledger_row(runs_root, run_id)
    return BackgroundRunStop(run_id=run_id, case=row.case, status=row.status,
                             result=row.result, method=method, note=note,
                             errors=errors)


def stop_case(run_id: str, grace_seconds: float = 30.0,
              run_directory: Optional[str] = None) -> BackgroundRunStop:
    """Stop a background run — gracefully first, keeping the evidence.

    Graceful (requires runTimeModifiable true in the case's controlDict —
    v10's compiled default is false, so the precondition is read first and
    the kill ladder is taken directly when it fails): stopAt writeNow is
    written into system/controlDict and the dict's mtime is stamped past the
    fileModificationSkew gate (re-stamped every ~2 s), letting the solver
    finish the current time step, WRITE the current fields and exit 0 within
    the bounded grace window. On timeout — or when graceful was impossible —
    the whole process group is SIGTERMed with a SIGKILL escalation. Either
    way stopAt is restored to endTime afterwards (a leftover writeNow makes
    any rerun insta-stop after one iteration), the row is stamped through
    the IDENTICAL completion path every run takes (check_foam_errors gate ->
    done with the parser-backed Result, or debugging with extracted errors),
    the run is deregistered, the pidfile dropped, and the deliberate stop is
    recorded on the row via the sanctioned note write — no new ledger
    states; a stopped run explains itself instead of reading like a crash.

    Idempotence: a run that exited between the caller's last poll and this
    call is stamped normally (method 'already_exited', no stop note) — not
    an error. Typed errors (:class:`BackgroundRunError`): unknown run id; a
    run that is not live and left nothing to stamp.
    """
    runs_root = _runs_root(run_directory)
    key, row = _resolve_run_row(runs_root, run_id)
    case_dir = os.path.abspath(os.path.join(runs_root, row.case))

    run = _BACKGROUND_RUNS.get(key)
    process = run.process if run is not None else None
    if run is not None:
        case_dir, runs_root = run.case_dir, run.runs_root
    info = _read_pidfile(case_dir)

    if not _stop_target_alive(process, info):
        if run is None and info is None and row.status != "running":
            raise BackgroundRunError(
                f"Run {key} has no live background process to stop "
                f"(status: {row.status}, result: {row.result}).")
        # The run exited on its own between the caller's poll and this call
        # (or the row is stuck 'running' after a server restart): stamp it
        # normally through the completion path — nothing was stopped, so no
        # deliberate-stop note.
        errors = _finish_background_run(key, runs_root, case_dir)
        return _stopped_status(key, runs_root, method="already_exited",
                               note=None, errors=errors)

    pid = process.pid if process is not None else info["pid"]
    pgid = info.get("pgid") if info is not None else None

    attempted, exited = _graceful_stop(case_dir, process, info, grace_seconds)
    method = "graceful"
    if not exited:
        method = "killed"
        exited = _kill_stop_target(pid, pgid, process, info)
    if attempted:
        # Restore AFTER the process is down, whichever rung stopped it.
        try:
            _set_control_dict_entry(_control_dict_path(case_dir),
                                    "stopAt", "endTime")
        except OSError:
            _log(f"Could not restore 'stopAt endTime;' in {case_dir}")
    if not exited:
        raise BackgroundRunError(
            f"Run {key} (pid {pid}) survived SIGTERM and SIGKILL; "
            "row left running for a later poll.")

    errors = _finish_background_run(key, runs_root, case_dir)
    note = _record_stop_note(runs_root, key, method)
    _log(f"Stopped background run {key} ({method}) in {case_dir}")
    return _stopped_status(key, runs_root, method=method, note=note,
                           errors=errors)
