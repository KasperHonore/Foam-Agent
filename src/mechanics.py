# mechanics.py
"""Mechanical (non-LLM) capabilities of Foam-Agent.

Everything in this module runs WITHOUT any LLM provider or API key:
file I/O, OpenFOAM execution, log parsing, FAISS tutorial retrieval
(local HuggingFace embeddings by default), Python script execution
(GMSH meshing / PyVista visualization) and SLURM job management.

The MCP server (src/mcp/fastmcp_server.py) exposes these functions as
tools. The reasoning that used to be LLM calls inside this repo now
lives in the portable skills/subagents under agents/ and runs on
whatever agent harness the user already has (Claude Code, Cursor,
Codex, OpenCode, pi, ...).

Diagnostics are printed to stderr so the module is safe to import from
a stdio MCP server (stdout is the protocol channel).
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

def tokenize(text: str) -> str:
    text = text.replace('_', ' ')
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    return text.lower()


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


def resolve_case_dir(
    case_name: str,
    case_dir: str = "",
    run_times: int = 1,
    run_directory: Optional[str] = None,
) -> str:
    """Resolve the output directory for a case."""
    if case_dir:
        return case_dir
    base_dir = str(run_directory) if run_directory else str(RUNS_DIR)
    if run_times > 1:
        return os.path.join(base_dir, f"{case_name}_{run_times}")
    return os.path.join(base_dir, case_name)


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


def run_command(script_path: str, out_file: str, err_file: str, working_dir: str, max_time_limit: int) -> None:
    """Execute a shell script inside the sourced OpenFOAM environment."""
    _log(f"Executing script {script_path} in {working_dir}")
    os.chmod(script_path, 0o777)
    bashrc_path = _openfoam_bashrc()

    command = f"source {bashrc_path} && bash {os.path.abspath(script_path)}"

    with open(out_file, 'w') as out, open(err_file, 'w') as err:
        process = subprocess.Popen(
            ['bash', "-c", command],
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=max_time_limit)
            out.write(stdout)
            err.write(stderr)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            stdout, stderr = process.communicate()
            timeout_message = (
                "OpenFOAM execution took too long. "
                "This case, if set up right, does not require such large execution times.\n"
            )
            out.write(timeout_message + stdout)
            err.write(timeout_message + stderr)
            _log(f"Execution timed out: {script_path}")

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
    bashrc_path = _openfoam_bashrc()
    full_command = f"source {bashrc_path} && {command}"

    process = subprocess.Popen(
        ['bash', '-c', full_command],
        cwd=case_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        stdout, stderr = process.communicate()
        return -1, stdout or "", (stderr or "") + f"\nCommand timed out after {timeout}s"


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


def run_allrun_and_collect_errors(
    case_dir: str,
    timeout: int = 3600,
    max_retries: int = 1,
) -> List[dict]:
    """Execute the Allrun script and return any error logs (empty = success)."""
    allrun_file_path = os.path.join(case_dir, "Allrun")
    if not os.path.exists(allrun_file_path):
        return [{"file": "Allrun", "error_content": f"Allrun script not found at {allrun_file_path}"}]

    out_file = os.path.join(case_dir, "Allrun.out")
    err_file = os.path.join(case_dir, "Allrun.err")

    remove_files(case_dir, prefix="log")
    remove_file(err_file)
    remove_file(out_file)
    remove_numeric_folders(case_dir)

    last_error_logs = []
    for attempt in range(1, max_retries + 1):
        _log(f"Running Allrun (attempt {attempt}/{max_retries})")
        run_command(allrun_file_path, out_file, err_file, case_dir, timeout)

        error_logs = check_foam_errors(case_dir)
        if len(error_logs) == 0:
            return []

        last_error_logs = error_logs
        if attempt < max_retries:
            _log("Allrun reported errors; retrying after cleanup...")
            remove_files(case_dir, prefix="log")
            remove_file(err_file)
            remove_file(out_file)
            remove_numeric_folders(case_dir)

    return last_error_logs


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
        "selected_case": _summary(selected),
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
        stdout = completed.stdout.decode() if isinstance(completed.stdout, bytes) else str(completed.stdout)

        if expected_abs:
            if os.path.exists(expected_abs) and os.path.getsize(expected_abs) > 0:
                return True, expected_abs, [], stdout
            return False, "", [
                "Script executed but expected output was not created",
                f"expected_output={expected_abs}",
            ], stdout
        return True, "", [], stdout

    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout)
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
        return False, "", [
            f"Script timed out after {timeout_s}s",
            f"STDERR:\n{err}",
        ], out
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout)
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

def submit_slurm_job(script_path: str) -> Tuple[Optional[str], bool, str]:
    try:
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        job_id_match = re.search(r'Submitted batch job (\d+)', output)
        if job_id_match:
            return job_id_match.group(1), True, ""
        return None, False, f"Could not extract job ID from output: {output}"
    except subprocess.CalledProcessError as e:
        return None, False, f"Failed to submit job: {e.stderr}"
    except Exception as e:
        return None, False, f"Unexpected error: {str(e)}"


def check_job_status(job_id: str) -> Tuple[Optional[str], bool, str]:
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "-o", "%T"],
            capture_output=True, text=True, check=True,
        )
        status = result.stdout.strip()
        if status:
            return status, True, ""
        return "COMPLETED", True, ""
    except subprocess.CalledProcessError as e:
        return None, False, f"Failed to check job status: {e.stderr}"
    except Exception as e:
        return None, False, f"Unexpected error: {str(e)}"
