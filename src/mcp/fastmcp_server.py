"""FastMCP-based OpenFOAM Agent Server (key-free).

This server exposes ONLY mechanical capabilities: FAISS tutorial retrieval
(local embeddings), case file I/O, OpenFOAM execution, log parsing, Python
script execution (GMSH meshing / PyVista visualization) and SLURM job
management. It requires NO LLM provider and NO API key.

All CFD reasoning (planning a case, writing OpenFOAM dictionaries, diagnosing
errors, generating mesh/visualization scripts) is done by the calling agent —
guided by the portable skills/subagents in agents/ at the repo root.
"""

import asyncio
import dataclasses
import json
import os
import sys
import contextlib
from typing import Dict, List, Optional

from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

# Make src/ importable (mechanics.py, convergence.py, translation/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import convergence
import forcecoeffs
import mechanics
import meshcheck
import stlcheck
import turbinlet
import wallspacing
from translation.esi_translator import ESITranslator


mcp = FastMCP(
    name="Foam-Agent",
    version="3.0.0",
    instructions="""
Foam-Agent exposes the mechanical side of OpenFOAM CFD simulation as tools:
tutorial retrieval (RAG over Foundation OpenFOAM v10 tutorials), case file
I/O, simulation execution, error extraction, Python script execution
(GMSH meshing, PyVista visualization) and SLURM job management.

YOU (the calling agent) are the brain: you plan the case, write the OpenFOAM
dictionaries, diagnose errors and decide fixes. The tools are your hands.

Recommended workflow:
1. get_case_stats — see valid domains/categories/solvers.
2. find_similar_case — retrieve the closest tutorial as reference.
3. resolve_case_dir + write_case_file — create all case files (0/, system/,
   constant/) and an Allrun script, following Foundation OpenFOAM v10
   conventions (momentumTransport, physicalProperties, ...).
4. run_case — execute Allrun; returns extracted errors on failure.
5. On errors: read_case_file / search_tutorials to diagnose, rewrite files
   via write_case_file, run_case again (iterate).
6. Optional: run_python_script for PyVista visualization (ensure_foam_file
   first) or GMSH mesh generation.

After (or during) a run, parse_solver_log turns the solver log into typed
convergence facts — per-field residuals, Courant numbers, continuity errors,
completion, and a verdict with evidence. Prefer it over reading raw logs:
the numbers are computed, never guessed, and cost the same on any log size.

After meshing, assess_mesh runs `checkMesh -allTopology -allGeometry` and
returns the mesh census plus per-metric pass/warn/fail classifications and
a verdict with evidence — prefer it over eyeballing raw checkMesh text.

Before meshing against an STL surface, inspect_stl runs surfaceCheck on it
and returns a typed report — watertightness with defective-edge counts,
triangle/vertex counts, bounding box, parts/zones, a units-suspicion flag —
plus a verdict with evidence. surfaceCheck exits 0 even for defective
surfaces, so never judge an STL by exit code; use this tool.

When a case has forceCoeffs output, parse_force_coefficients returns the
typed coefficient summary (Cd/Cl/Cm with first/final values and tail-window
statistics, plus the reference values used for normalization) and stamps a
compact summary into the run ledger's Key result cell — prefer it over
reading dat files and averaging by eye.

Before sizing a boundary-layer mesh for a turbulent case, estimate_wall_spacing
turns the flow conditions and a target y+ into computed numbers — Reynolds
number with a regime verdict, the named skin-friction correlation, friction
velocity, the first-cell CENTRE distance and the first-cell HEIGHT as two
separately labelled fields, a boundary-layer thickness estimate and a
suggested layer count. Pure math, no case directory: never do this
arithmetic from memory.

For turbulent inlet conditions, estimate_turbulence_inlet turns velocity,
intensity and a length scale (or hydraulic diameter) into k/epsilon/omega,
each carrying the formula that produced it — pure server-side math, no case
directory needed. Use it instead of recalling formulas: the constants are
pinned server-side and every silent default is echoed in the output.

The run ledger (runs/ledger.md) tracks every run automatically as a side
effect of the tools above. set_run_note is the ONLY sanctioned skill-side
ledger write (annotate a run, archive/unarchive it) — never edit the file.

IMPORTANT: always create and edit case files through write_case_file (not
your local file tools) — the server may run in a container whose filesystem
is where the simulation actually executes.
""",
)


def _abs_case_dir(case_dir: str) -> str:
    return os.path.abspath(case_dir)


def _safe_join(case_dir: str, relative_path: str) -> str:
    """Join and refuse paths that escape the case directory."""
    base = os.path.abspath(case_dir)
    full = os.path.abspath(os.path.join(base, relative_path))
    if not full.startswith(base + os.sep) and full != base:
        raise ValueError(f"relative_path escapes the case directory: {relative_path}")
    return full


def _require_case_dir(case_dir: str) -> str:
    """Normalize to an absolute path and require the case directory to exist.

    Only for tools that operate on an existing case (run/command/translate);
    write_case_file and read_case_file intentionally do their own thing.
    """
    case_dir = _abs_case_dir(case_dir)
    if not os.path.exists(case_dir):
        raise ValueError(f"Case directory does not exist: {case_dir}")
    return case_dir


def _truncate_head(text: str, max_chars: int) -> str:
    """Keep the head, mark the cut tail — for content whose start matters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _truncate_tail(text: str, max_chars: int) -> str:
    """Keep the tail, mark the cut head — for logs/output where errors are last."""
    if len(text) <= max_chars:
        return text
    return "... [truncated] ...\n" + text[-max_chars:]


# ============================================================================
# Discovery / retrieval tools
# ============================================================================

@mcp.tool(name="get_case_stats")
async def get_case_stats(ctx: Context) -> Dict[str, List[str]]:
    """List the valid case domains, categories and solvers (Foundation OpenFOAM v10).

    Use these values when choosing a solver and when calling find_similar_case.
    """
    case_stats_path = os.path.join(str(mechanics.DATABASE_DIR), "raw", "openfoam_case_stats.json")
    with open(case_stats_path, "r") as f:
        return json.load(f)


@mcp.tool(name="search_tutorials")
async def search_tutorials(
    query: str = Field(description="Semantic search query (e.g. 'incompressible lid driven cavity icoFoam' or a command name like 'blockMesh')"),
    index: str = Field(default="openfoam_tutorials_details", description="FAISS index to search: 'openfoam_tutorials_structure', 'openfoam_tutorials_details' (default), 'openfoam_allrun_scripts' or 'openfoam_command_help'"),
    topk: int = Field(default=3, description="Number of results to return"),
    max_chars_per_result: int = Field(default=20000, description="Truncate each result's full_content to this many characters"),
    ctx: Context = None,
) -> List[dict]:
    """Semantic search over the Foundation OpenFOAM v10 tutorial database.

    Returns tutorial cases, directory structures, Allrun scripts or command
    help texts depending on the chosen index. No API key needed (local
    embeddings).
    """
    if index not in mechanics.FAISS_INDEX_NAMES:
        raise ValueError(f"Unknown index '{index}'. Valid: {mechanics.FAISS_INDEX_NAMES}")

    results = await asyncio.to_thread(mechanics.retrieve_faiss, index, query, topk)
    for r in results:
        content = r.get("full_content")
        if isinstance(content, str):
            r["full_content"] = _truncate_head(content, max_chars_per_result)
    return results


@mcp.tool(name="find_similar_case")
async def find_similar_case(
    case_name: str = Field(description="Short name for the case being built (e.g. 'lid_driven_cavity')"),
    case_solver: str = Field(description="Chosen OpenFOAM solver (must be one of get_case_stats case_solver values)"),
    case_domain: str = Field(description="Case domain (one of get_case_stats case_domain values)"),
    case_category: str = Field(description="Case category (one of get_case_stats case_category values)"),
    searchdocs: int = Field(default=5, description="How many similar Allrun references to collect; 0 skips the Allrun retrieval entirely (much faster)"),
    ctx: Context = None,
) -> dict:
    """Find the most similar Foundation v10 tutorial case to use as a reference.

    Recall is semantic, then hard-filtered on domain and reranked by solver
    match. Returns the selected tutorial's full content, its directory
    structure (top-level dir_structure, also mirrored into selected_case) and
    similar Allrun scripts. YOU judge how closely to follow the reference
    (check the returned selected_case metadata against your target).

    Latency: each call embeds two queries with the local CPU embedding model
    and can take tens of seconds even when the model is warm — it is working,
    not hanging. The Allrun retrieval is the expensive half; pass searchdocs=0
    to skip it when you only need the tutorial reference.

    Note: case_category is a free-text retrieval hint and is NOT validated
    against get_case_stats values — an unknown category is accepted silently
    and only weakens the semantic match. Domain and solver matter most.
    """
    return await asyncio.to_thread(
        mechanics.find_similar_case,
        case_name, case_solver, case_domain, case_category, searchdocs,
    )


# ============================================================================
# Case file tools
# ============================================================================

class WriteFileResponse(BaseModel):
    path: str = Field(description="Absolute path of the written file")
    bytes_written: int


@mcp.tool(name="resolve_case_dir")
async def resolve_case_dir(
    case_name: str = Field(description="Case name; the case directory is derived from it"),
    case_dir: str = Field(default="", description="Optional explicit directory; returned as-is if set"),
    ctx: Context = None,
) -> str:
    """Resolve the directory where a new case should be created (under runs/).

    Also records the case as a planned run in runs/ledger.md (server-owned
    run ledger; see run-states.yml beside it for the state vocabulary).
    """
    return mechanics.resolve_case_dir(case_name=case_name, case_dir=case_dir)


@mcp.tool(name="write_case_file")
async def write_case_file(
    case_dir: str = Field(description="Case directory (from resolve_case_dir)"),
    relative_path: str = Field(description="Path inside the case, e.g. 'system/controlDict', '0/U', 'Allrun'"),
    content: str = Field(description="Full file content (OpenFOAM dictionary format, or shell script for Allrun)"),
    executable: bool = Field(default=False, description="Set true for scripts like Allrun"),
    ctx: Context = None,
) -> WriteFileResponse:
    """Create or overwrite a file in the case directory.

    Always use this (not local file tools) so files land on the filesystem
    where the simulation runs.
    """
    case_dir = _abs_case_dir(case_dir)
    path = _safe_join(case_dir, relative_path)
    try:
        mechanics.save_file(path, content)
        if executable:
            os.chmod(path, 0o777)
    except PermissionError as exc:
        raise PermissionError(
            f"Permission denied writing {path}. The case directory likely contains "
            "files owned by another user — typically left by a pre-non-root (root) "
            "Foam-Agent image. Fix: recreate the container from the latest image "
            "(its entrypoint repairs runs/ ownership at startup), or run "
            "'docker exec -u root foamagent-mcp chown -R openfoam:openfoam "
            "/home/openfoam/Foam-Agent/runs', or use a fresh case name."
        ) from exc
    return WriteFileResponse(path=path, bytes_written=len(content.encode("utf-8")))


@mcp.tool(name="read_case_file")
async def read_case_file(
    case_dir: str = Field(description="Case directory"),
    relative_path: str = Field(description="Path inside the case, e.g. 'system/fvSolution' or 'log.blockMesh'"),
    max_chars: int = Field(default=50000, description="Truncate content to this many characters (tail is kept for log files)"),
    ctx: Context = None,
) -> str:
    """Read a file from the case directory (case files, logs, Allrun output)."""
    case_dir = _abs_case_dir(case_dir)
    path = _safe_join(case_dir, relative_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    content = mechanics.read_file(path)
    # Keep the tail for logs (errors are at the end), the head otherwise.
    name = os.path.basename(relative_path)
    if name.startswith("log") or name.endswith((".out", ".err")):
        return _truncate_tail(content, max_chars)
    return _truncate_head(content, max_chars)


@mcp.tool(name="list_case_files")
async def list_case_files(
    case_dir: str = Field(description="Case directory"),
    ctx: Context = None,
) -> Dict[str, List[str]]:
    """List the full case directory structure (recursive): {folder: [files]} with
    relative-path keys like 'constant/polyMesh', plus top-level files under '.'."""
    case_dir = _abs_case_dir(case_dir)
    structure = mechanics.scan_case_directory(case_dir, deep=True)
    top_level = [
        f for f in os.listdir(case_dir)
        if os.path.isfile(os.path.join(case_dir, f)) and not f.startswith(".")
    ]
    if top_level:
        structure["."] = top_level
    return structure


# ============================================================================
# Execution tools
# ============================================================================

class RunCaseResponse(BaseModel):
    status: str = Field(description="'success' or 'failed'")
    errors: List[dict] = Field(description="Extracted errors: [{file, error_content}]")
    log_files: List[str] = Field(description="Names of log files produced (read them with read_case_file)")


@mcp.tool(name="run_case")
async def run_case(
    case_dir: str = Field(description="Case directory containing an Allrun script"),
    timeout: int = Field(default=3600, description="Max execution time in seconds"),
    ctx: Context = None,
) -> RunCaseResponse:
    """Execute the case's Allrun script and extract any errors from the logs.

    Cleans old logs and time-step folders first. On failure, errors contains
    the relevant log excerpts — diagnose them, fix the case files with
    write_case_file, and call run_case again.

    The run ledger (runs/ledger.md) follows the run automatically: the row
    is running while Allrun executes, then done with a Result verdict on
    success or debugging on failure. No bookkeeping needed on your side.
    """
    case_dir = _require_case_dir(case_dir)

    await ctx.info(f"Running Allrun in {case_dir} (timeout {timeout}s)")
    errors = await asyncio.to_thread(
        mechanics.run_allrun_and_collect_errors, case_dir, timeout, 1
    )

    log_files = sorted(
        f for f in os.listdir(case_dir)
        if f.startswith("log") or f in ("Allrun.out", "Allrun.err")
    )
    status = "success" if not errors else "failed"
    await ctx.info(f"Simulation {status} with {len(errors)} error(s)")
    return RunCaseResponse(status=status, errors=errors, log_files=log_files)


class CommandResponse(BaseModel):
    returncode: int
    stdout: str
    stderr: str


@mcp.tool(name="run_openfoam_command")
async def run_openfoam_command(
    case_dir: str = Field(description="Case directory (used as working directory)"),
    command: str = Field(description="Single OpenFOAM command line, e.g. 'checkMesh', 'gmshToFoam mesh.msh', 'postProcess -func sampleDict'"),
    timeout: int = Field(default=600, description="Max execution time in seconds"),
    max_chars: int = Field(default=20000, description="Truncate stdout/stderr to this many characters (tail kept)"),
    ctx: Context = None,
) -> CommandResponse:
    """Run one OpenFOAM utility command inside the sourced OpenFOAM environment.

    Useful for mesh conversion (gmshToFoam), mesh quality checks (checkMesh),
    decomposePar, postProcess, etc. For full simulations use run_case instead.
    """
    case_dir = _require_case_dir(case_dir)

    returncode, stdout, stderr = await asyncio.to_thread(
        mechanics.run_openfoam_command, case_dir, command, timeout
    )

    return CommandResponse(
        returncode=returncode,
        stdout=_truncate_tail(stdout, max_chars),
        stderr=_truncate_tail(stderr, max_chars),
    )


class PythonScriptResponse(BaseModel):
    success: bool
    artifact: str = Field(description="Absolute path of expected_output if produced, else ''")
    errors: List[str]
    stdout: str = Field(description="Captured stdout of the script (truncated)")


@mcp.tool(name="run_python_script")
async def run_python_script(
    case_dir: str = Field(description="Case directory (used as working directory)"),
    script: str = Field(description="Full Python script content to execute"),
    filename: str = Field(default="script.py", description="Filename to save the script as inside the case"),
    expected_output: str = Field(default="", description="Relative path of a file the script must produce (e.g. 'velocity.png' or 'geometry.msh'); success requires it to exist"),
    timeout: int = Field(default=300, description="Max execution time in seconds"),
    max_chars: int = Field(default=20000, description="Truncate captured stdout to this many characters (tail kept)"),
    ctx: Context = None,
) -> PythonScriptResponse:
    """Execute a Python script in the case directory (server-side Python).

    The script runs with cwd = case_dir, so relative paths in the script
    (input files like the .foam marker, and expected_output) resolve inside
    the case. Use for PyVista visualization (pyvista is installed; set
    expected_output to the PNG path) and GMSH mesh generation (set
    expected_output to the .msh file). On failure, fix the script based on
    the returned errors/stdout and retry.
    """
    case_dir = _abs_case_dir(case_dir)
    ok, artifact, errors, stdout = await asyncio.to_thread(
        lambda: mechanics.run_python_script(
            case_dir, script,
            filename=filename,
            expected_output=expected_output or None,
            timeout_s=timeout,
        )
    )
    return PythonScriptResponse(
        success=ok, artifact=artifact, errors=errors, stdout=_truncate_tail(stdout, max_chars)
    )


@mcp.tool(name="ensure_foam_file")
async def ensure_foam_file(
    case_dir: str = Field(description="Case directory"),
    ctx: Context = None,
) -> str:
    """Create/refresh the .foam marker file needed by PyVista's OpenFOAM reader.

    Returns the filename RELATIVE to case_dir (e.g. 'my_case.foam') — use it
    as-is inside run_python_script scripts, which execute with cwd=case_dir.
    """
    return mechanics.ensure_foam_file(_abs_case_dir(case_dir))


@mcp.tool(name="read_mesh_boundaries")
async def read_mesh_boundaries(
    case_dir: str = Field(description="Case directory"),
    ctx: Context = None,
) -> dict:
    """Read constant/polyMesh/boundary: returns {exists, boundary_names, content}.

    Use after mesh conversion to verify patch names match the boundary
    conditions you plan to write in 0/.
    """
    return mechanics.read_mesh_boundaries(_abs_case_dir(case_dir))


# ============================================================================
# Solver-log convergence parsing (deterministic, key-free)
# ============================================================================

class TimeProgressModel(BaseModel):
    first_time: Optional[float] = Field(description="First 'Time =' value in the log (None before the first step)")
    latest_time: Optional[float] = Field(description="Latest 'Time =' value reached")
    end_time: Optional[float] = Field(description="endTime target from system/controlDict (None when unreadable)")


class FieldResidualModel(BaseModel):
    field: str = Field(description="Field name as the solver reports it (e.g. 'Ux', 'p')")
    first_initial: float = Field(description="Initial residual of the field's first solve")
    last_initial: float = Field(description="Initial residual of the field's last solve")
    last_final: float = Field(description="Final residual of the field's last solve")
    worst_initial: float = Field(description="Largest initial residual seen for the field")
    worst_initial_time: Optional[float] = Field(description="Solver time of that worst initial residual (None before the first step)")


class CourantModel(BaseModel):
    max: float = Field(description="Largest 'max' Courant number in the log")
    max_time: Optional[float] = Field(description="Solver time where that maximum was first reached")
    last_mean: float = Field(description="'mean' value of the last Courant line")
    last_max: float = Field(description="'max' value of the last Courant line")


class SolverLogResponse(BaseModel):
    solver: str = Field(description="Solver that wrote the log (the log's Exec entry)")
    log_file: str = Field(description="Name of the log file that was parsed")
    completed: bool = Field(description="True when the log reached its 'End' marker")
    time: TimeProgressModel = Field(description="Time progress vs the controlDict endTime target")
    residuals: List[FieldResidualModel] = Field(description="Per-field residual summary, in first-appearance order")
    courant: Optional[CourantModel] = Field(description="Courant number summary (None when the log has no Courant lines)")
    cumulative_continuity: Optional[float] = Field(description="Last cumulative time-step continuity error")
    fatal_errors: List[str] = Field(description="Extracted FOAM FATAL (IO) ERROR blocks, verbatim")
    verdict: str = Field(description="'converged', 'diverged', 'incomplete' or 'error'")
    evidence: List[str] = Field(description="Human-readable strings naming what drove the verdict")


@mcp.tool(name="parse_solver_log")
async def parse_solver_log(
    case_dir: str = Field(description="Case directory containing the solver log"),
    log_file: str = Field(default="", description="Explicit log file name (e.g. 'log.icoFoam'); default: log.<application> with the application read from system/controlDict"),
    ctx: Context = None,
) -> SolverLogResponse:
    """Parse a case's solver log into typed convergence facts plus a verdict.

    Deterministic and key-free: per-field residuals, Courant numbers,
    continuity errors, time progress and completion are computed from the
    log — never guessed — and the verdict (converged / diverged /
    incomplete / error) comes with evidence strings naming what drove it.
    Works on partial, in-flight logs too (verdict 'incomplete' plus time
    progress). Built-in thresholds are deliberately conservative defaults:
    converged needs every field's last final residual under 1e-4; residuals
    at/above 1e6 or a max Courant at/above 100 read as divergence. Judge
    anything near those lines yourself before trusting a run.
    """
    case_dir = _require_case_dir(case_dir)
    analysis = await asyncio.to_thread(
        convergence.parse_solver_log, case_dir, log_file or None
    )
    return SolverLogResponse(**dataclasses.asdict(analysis))


# ============================================================================
# Structured mesh assessment (deterministic, key-free)
# ============================================================================

class MeshCensusModel(BaseModel):
    points: Optional[int] = Field(description="Point count from checkMesh's 'Mesh stats' block (None when absent)")
    faces: Optional[int] = Field(description="Face count")
    internal_faces: Optional[int] = Field(description="Internal face count")
    cells: Optional[int] = Field(description="Cell count")
    boundary_patches: Optional[int] = Field(description="Boundary patch count")
    cell_types: Dict[str, int] = Field(description="Cell counts by type, keyed as checkMesh prints them (hexahedra, prisms, wedges, pyramids, 'tet wedges', tetrahedra, polyhedra)")


class MeshMetricModel(BaseModel):
    name: str = Field(description="Stable snake_case metric name (e.g. 'max_skewness', 'max_non_orthogonality', 'point_usage')")
    value: Optional[float] = Field(description="Numeric value where checkMesh reports one (None for value-less checks)")
    check: str = Field(description="'topology' or 'geometry' — which checkMesh block the metric came from")
    checkmesh_ok: bool = Field(description="checkMesh's own mark: False exactly for its ***-failed checks")
    classification: str = Field(description="'pass', 'warn' (marginal but legal, or a checkMesh '*' notice) or 'fail'")


class MeshAssessmentResponse(BaseModel):
    flags: str = Field(description="checkMesh flags used, read from the run's own Exec line")
    census: MeshCensusModel = Field(description="Mesh size facts: points/faces/cells and the cell-type breakdown")
    metrics: List[MeshMetricModel] = Field(description="Per-check entries with checkMesh's own mark and the pass/warn/fail classification")
    failed_checks: int = Field(description="checkMesh's own reported failure count (0 for 'Mesh OK.')")
    mesh_ok: bool = Field(description="True iff checkMesh itself concluded 'Mesh OK.'")
    verdict: str = Field(description="'ok', 'warnings' or 'failed' — never better than checkMesh's own conclusion")
    evidence: List[str] = Field(description="Human-readable strings naming the offending (or marginal) metrics")


@mcp.tool(name="assess_mesh")
async def assess_mesh(
    case_dir: str = Field(description="Case directory whose mesh should be assessed"),
    timeout: int = Field(default=600, description="Max checkMesh execution time in seconds"),
    ctx: Context = None,
) -> MeshAssessmentResponse:
    """Run `checkMesh -allTopology -allGeometry` on the case and return a typed
    mesh quality assessment.

    Deterministic and key-free on the parsing side: the mesh census
    (points/faces/cells, cell types), every quality metric with its value and
    checkMesh's own ok/failed mark (topology vs geometry distinguished), and
    a verdict (ok / warnings / failed) with evidence naming the offending
    metrics. checkMesh's failure marks are ground truth — the verdict is
    never better than checkMesh's own conclusion. On top, a conservative
    built-in warn band flags marginal-but-legal meshes: non-orthogonality
    from 65 degrees (checkMesh fails at 90), skewness from 2 (fails at 4),
    aspect ratio from 100 (fails at 1000). Judge warn-band values against
    your application yourself — a marginal mesh may be fine for a steady
    RANS estimate and unusable for an LES. A case with no mesh (or a
    checkMesh crash) raises a typed error, never a fabricated assessment.
    """
    case_dir = _require_case_dir(case_dir)
    assessment = await asyncio.to_thread(meshcheck.assess_mesh, case_dir, timeout)
    return MeshAssessmentResponse(**dataclasses.asdict(assessment))


# ============================================================================
# Force-coefficient parsing (deterministic, key-free)
# ============================================================================

class ForceReferenceModel(BaseModel):
    lift_dir: Optional[List[float]] = Field(description="Lift direction vector from the dat header (None when the line is absent)")
    drag_dir: Optional[List[float]] = Field(description="Drag direction vector")
    pitch_axis: Optional[List[float]] = Field(description="Pitch axis for the moment coefficient")
    mag_u_inf: Optional[float] = Field(description="Free-stream velocity magnitude the coefficients were normalized with")
    l_ref: Optional[float] = Field(description="Reference length (moment normalization)")
    a_ref: Optional[float] = Field(description="Reference area (force normalization)")
    cofr: Optional[List[float]] = Field(description="Centre of rotation for moments")


class TailWindowModel(BaseModel):
    samples: int = Field(description="Number of samples the tail statistics were computed over")
    fraction: float = Field(description="The documented default window fraction (0.2 = last 20% of samples, rounded up)")
    min_samples: int = Field(description="The documented window floor (at least this many samples, or all when fewer exist)")
    start_time: float = Field(description="Solver time of the window's first sample")
    end_time: float = Field(description="Solver time of the window's last sample")


class CoefficientSeriesModel(BaseModel):
    name: str = Field(description="Coefficient name as the dat header carries it ('Cm', 'Cd', 'Cl', 'Cl(f)', 'Cl(r)')")
    first: float = Field(description="Value at the first sample")
    final: float = Field(description="Value at the last sample")
    tail_mean: float = Field(description="Mean over the reported tail window")
    tail_min: float = Field(description="Minimum over the reported tail window")
    tail_max: float = Field(description="Maximum over the reported tail window")


class ForceCoefficientsResponse(BaseModel):
    function_name: str = Field(description="The force function object whose output was parsed (e.g. 'forceCoeffs1')")
    dat_file: str = Field(description="The dat file parsed, relative to the case directory (posix-style)")
    reference: ForceReferenceModel = Field(description="Reference values from the header — check these before trusting a number (wrong Aref/lRef/magUInf means wrong normalization)")
    samples: int = Field(description="Number of data samples parsed")
    start_time: float = Field(description="Solver time of the first sample")
    end_time: float = Field(description="Solver time of the last sample")
    window: TailWindowModel = Field(description="The tail window the statistics were computed over")
    coefficients: List[CoefficientSeriesModel] = Field(description="Per-coefficient summary, in the dat header's column order")
    key_result: str = Field(description="The compact summary stamped into the ledger: 'Cd=<tail mean:.4g> Cl=<tail mean:.4g> (tail mean)'")
    stamped: bool = Field(description="True when the case's run-ledger row existed and its Key result cell was filled")


@mcp.tool(name="parse_force_coefficients")
async def parse_force_coefficients(
    case_dir: str = Field(description="Case directory whose forceCoeffs output should be parsed"),
    function_name: str = Field(default="", description="Force function object to read (e.g. 'forceCoeffs1'); required only when several force function objects have output — the error names the candidates"),
    ctx: Context = None,
) -> ForceCoefficientsResponse:
    """Parse the case's forceCoeffs output into typed coefficient facts and
    fill the run ledger's Key result cell.

    Deterministic and key-free: locates the newest time directory under
    postProcessing/<function>/, parses the Foundation v10 forceCoeffs.dat
    (header reference values plus the Time/Cm/Cd/Cl/Cl(f)/Cl(r) series) and
    returns per-coefficient first/final values with tail-window statistics
    (mean/min/max over the last 20% of samples, at least 10, window reported)
    — numbers computed from the file, never averaged by eye. When the case
    has a run-ledger row, a compact summary (tail-mean Cd and Cl) is stamped
    into its Key result cell as a server-owned side effect; re-parsing
    re-stamps idempotently and rowless cases still get the full analysis.
    A case with no forceCoeffs output, several candidate function objects
    and no explicit function_name, or an empty/headerless dat file raises a
    typed error — never fabricated statistics.
    """
    case_dir = _require_case_dir(case_dir)
    analysis = await asyncio.to_thread(
        forcecoeffs.parse_force_coefficients, case_dir, function_name or None
    )
    return ForceCoefficientsResponse(**dataclasses.asdict(analysis))


# ============================================================================
# Structured STL surface inspection (deterministic, key-free)
# ============================================================================

class BoundingBoxModel(BaseModel):
    min: List[float] = Field(description="Bounding-box minimum corner (x y z)")
    max: List[float] = Field(description="Bounding-box maximum corner (x y z)")
    extents: List[float] = Field(description="Per-axis extents (max - min)")


class SurfaceReportResponse(BaseModel):
    surface_file: str = Field(description="Surface file surfaceCheck ran on, read from the run's own Exec line")
    triangles: int = Field(description="Triangle count from the Statistics block")
    vertices: Optional[int] = Field(description="Vertex count (None when absent)")
    bounding_box: Optional[BoundingBoxModel] = Field(description="Axis-aligned bounding box with per-axis extents (None when absent)")
    closed: bool = Field(description="True iff surfaceCheck reported the surface closed (watertight: all edges connected to two faces)")
    edges_connected_to_one_face: Optional[int] = Field(description="Edges connected to only one face — holes / open boundaries (0 for a closed surface)")
    edges_connected_to_more_than_two_faces: Optional[int] = Field(description="Edges connected to more than two faces — non-manifold (0 for a closed surface)")
    unconnected_parts: Optional[int] = Field(description="Number of unconnected parts (>1 means multiple shells in one file)")
    zones: Optional[int] = Field(description="Number of zones (connected areas with consistent normal orientation)")
    units_suspicious: bool = Field(description="True when the largest bounding-box extent is at/above 1000 — plausibly a millimetre export of a metre-scale part (warns, never fails)")
    verdict: str = Field(description="'ok', 'warnings' or 'failed' — computed from the surfaceCheck TEXT (its exit code is 0 even for defective surfaces)")
    evidence: List[str] = Field(description="Human-readable strings naming each problem (or what makes the surface ok)")


@mcp.tool(name="inspect_stl")
async def inspect_stl(
    path: str = Field(description="Path to the STL surface — absolute (on the server's filesystem) or relative to the server's working directory (e.g. 'runs/<case>/constant/triSurface/part.stl')"),
    timeout: int = Field(default=600, description="Max surfaceCheck execution time in seconds"),
    ctx: Context = None,
) -> SurfaceReportResponse:
    """Run `surfaceCheck` on an STL surface and return a typed geometry report.

    Deterministic and key-free on the parsing side: watertightness with the
    defective-edge counts (connected to one face = holes, to >2 faces =
    non-manifold), triangle/vertex counts, the bounding box with per-axis
    extents, unconnected-parts and normal-orientation-zone counts, a
    units-suspicion flag, and a verdict (ok / warnings / failed) with
    evidence naming each problem. surfaceCheck exits 0 even for defective
    surfaces, so every verdict comes from the output text. An open surface
    is failed (snappyHexMesh needs a watertight surface); multiple shells,
    normals flipping within a part, or a suspected millimetre export
    (largest extent at/above 1000 — STL carries no unit metadata) warn
    without failing. surfaceCheck runs in the STL's own directory, so its
    diagnostic dump files on defective surfaces (problemFaces, zoning .vtk,
    per-part .obj) land next to the inspected surface. A missing file or an
    unreadable surface raises a typed error, never a fabricated report.
    """
    # Same path convention as _abs_case_dir: relative paths resolve against
    # the server's working directory, absolute in-container paths pass through.
    path = os.path.abspath(path)
    report = await asyncio.to_thread(stlcheck.inspect_stl, path, timeout)
    return SurfaceReportResponse(**dataclasses.asdict(report))


# ============================================================================
# Geometry import: STL into constant/triSurface, deterministic scaling
# ============================================================================

class GeometryImportResponse(BaseModel):
    dest_path: str = Field(description="Imported surface path relative to the case (posix-style), e.g. 'constant/triSurface/cube.stl' — reference this name in snappyHexMeshDict/surfaceFeaturesDict")
    scale: Optional[float] = Field(description="Uniform scale factor applied via surfaceTransformPoints (null when the surface was copied unscaled)")
    size_bytes: int = Field(description="Byte size of the imported surface file")
    overwrote: bool = Field(description="True when a surface of the same normalized name already existed and was replaced")


@mcp.tool(name="import_geometry")
async def import_geometry(
    case_dir: str = Field(description="Case directory to import the surface into"),
    src_path: str = Field(description="Server-visible path of the source STL (e.g. inside the runs bind mount — copy the user's file there host-side first)"),
    scale: Optional[float] = Field(default=None, description="Optional uniform scale factor, e.g. 0.001 for mm->m; applied deterministically with Foundation v10 surfaceTransformPoints"),
    timeout: int = Field(default=600, description="Max scaling time in seconds (ignored for plain copies)"),
    ctx: Context = None,
) -> GeometryImportResponse:
    """Import an STL surface into the case's constant/triSurface/ directory.

    Copies the surface under a normalized name (whitespace and unsafe
    characters become underscores; lowercase '.stl' extension), creating
    constant/triSurface/ when absent. With ``scale`` the copy is produced by
    surfaceTransformPoints ("scale=(s s s)" form) so mm->m unit correction is
    computed, never a hand-edited dict. The typed result names the
    case-relative destination, the applied scale, the imported byte size and
    whether an existing surface was overwritten (never silent). A missing
    source raises a typed error with the case left untouched.
    """
    case_dir = _require_case_dir(case_dir)
    result = await asyncio.to_thread(
        mechanics.import_geometry, case_dir, src_path, scale, timeout
    )
    return GeometryImportResponse(**dataclasses.asdict(result))


# ============================================================================
# Wall-spacing calculator (pure math, deterministic, key-free)
# ============================================================================

class QuantityModel(BaseModel):
    value: float = Field(description="Computed value (SI units; the field description names them)")
    formula: str = Field(description="The named correlation/formula that produced the value")


class LayerCountModel(BaseModel):
    value: int = Field(description="Suggested number of boundary layers")
    formula: str = Field(description="The covering rule that chose the count")


class WallSpacingResponse(BaseModel):
    flow_type: str = Field(description="'external' (flat-plate correlations) or 'internal' (pipe correlations) — echoed input")
    velocity: float = Field(description="Freestream/bulk velocity U [m/s] — echoed input")
    characteristic_length: float = Field(description="Characteristic length [m] — echoed input (plate/body length for external, hydraulic diameter for internal)")
    kinematic_viscosity: float = Field(description="Kinematic viscosity nu [m^2/s] — echoed input")
    target_y_plus: float = Field(description="Target y+ for the first cell CENTRE — echoed input")
    expansion_ratio: float = Field(description="Layer expansion ratio used for the layer-count suggestion — echoed input (default 1.2)")
    reynolds_number: QuantityModel = Field(description="Reynolds number on the characteristic length (Re_x external, Re_D internal)")
    regime: str = Field(description="'laminar', 'transitional' or 'turbulent' — laminar means wall functions and turbulence models are inapplicable (the numbers still come back, from the laminar correlation)")
    skin_friction_coefficient: QuantityModel = Field(description="Skin-friction coefficient Cf, Fanning convention (tau_w = Cf/2*rho*U^2); the formula names the pinned correlation")
    kinematic_wall_shear_stress: QuantityModel = Field(description="KINEMATIC wall shear stress tau_w/rho [m^2/s^2] — the tool takes nu only, density cancels in the spacing chain")
    friction_velocity: QuantityModel = Field(description="Friction velocity u_tau [m/s]")
    first_cell_centre_distance: QuantityModel = Field(description="Distance of the first cell CENTRE from the wall, y1 = y+*nu/u_tau [m] — where y+ is evaluated")
    first_cell_height: QuantityModel = Field(description="First-cell HEIGHT, 2*y1 [m] — the layer thickness a mesher needs; NOT the centre distance")
    boundary_layer_thickness: QuantityModel = Field(description="Boundary-layer thickness estimate delta [m] the layer stack should cover")
    suggested_layer_count: LayerCountModel = Field(description="Smallest layer count whose geometric stack (first layer = first-cell height, growth = expansion_ratio) covers delta")
    evidence: List[str] = Field(description="Human-readable strings naming the regime thresholds, correlation choices and any extrapolation beyond a correlation's stated validity")


@mcp.tool(name="estimate_wall_spacing")
async def estimate_wall_spacing(
    velocity: float = Field(description="Freestream/bulk velocity U in m/s (positive)"),
    characteristic_length: float = Field(description="Characteristic length in m (positive): plate/body length for external flow, hydraulic diameter for internal flow"),
    kinematic_viscosity: float = Field(description="Kinematic viscosity nu in m^2/s (positive), e.g. 1.5e-5 for air, 1e-6 for water"),
    target_y_plus: float = Field(description="Target y+ for the first cell CENTRE (positive): ~1 for low-Re wall treatment, 30-300 for wall functions"),
    flow_type: str = Field(default="external", description="'external' (flat-plate correlation family, default) or 'internal' (pipe family)"),
    expansion_ratio: float = Field(default=wallspacing.DEFAULT_EXPANSION_RATIO, description="Layer expansion ratio for the layer-count suggestion (>= 1; default 1.2, the snappy reference's conservative default)"),
    ctx: Context = None,
) -> WallSpacingResponse:
    """Compute the wall-normal mesh spacing for a target y+ from flow
    conditions. Pure math — no case directory, no filesystem.

    Deterministic and key-free: Reynolds number with a regime verdict
    (laminar / transitional / turbulent, thresholds named in the evidence),
    the skin-friction coefficient from the pinned correlation for the flow
    type (Schlichting/Blasius flat plate external, Blasius/Hagen-Poiseuille
    pipe internal — every numeric field carries the name of the correlation
    or formula that produced it), kinematic wall shear stress, friction
    velocity, the first-cell CENTRE distance (where y+ is evaluated) and the
    first-cell HEIGHT (what a mesher needs) as two separately labelled
    fields, a boundary-layer thickness estimate, and the smallest layer
    count covering it at the given expansion ratio. A laminar-regime result
    still returns every number; the verdict tells you wall functions and
    turbulence models are inapplicable. Non-physical inputs (non-positive
    velocity, length, viscosity or y+ target) raise a typed error — never
    plausible garbage numbers. Numbers are computed, never recalled: do not
    do this arithmetic from memory.
    """
    estimate = wallspacing.estimate_wall_spacing(
        velocity=velocity,
        characteristic_length=characteristic_length,
        kinematic_viscosity=kinematic_viscosity,
        target_y_plus=target_y_plus,
        flow_type=flow_type,
        expansion_ratio=expansion_ratio,
    )
    return WallSpacingResponse(**dataclasses.asdict(estimate))


# ============================================================================
# Inlet turbulence calculator (pure math, deterministic, key-free)
# ============================================================================

class InletQuantityModel(BaseModel):
    value: float = Field(description="The computed value")
    units: str = Field(description="SI units of the value ('-' for dimensionless)")
    formula: str = Field(description="The formula that produced the value, with the pinned model constant echoed where it enters")


class TurbulenceInletResponse(BaseModel):
    velocity: float = Field(description="Mean inlet velocity U [m/s] the estimate was computed for (echoed input)")
    intensity: float = Field(description="Turbulence intensity I as applied (fraction: 0.05 = 5%)")
    intensity_source: str = Field(description="'caller-supplied', or the documented medium default named as an applied assumption")
    length_scale: float = Field(description="Turbulence length scale l [m] as applied")
    length_scale_source: str = Field(description="'caller-supplied', or the named l = 0.07*D_h conversion from the given hydraulic diameter")
    c_mu: float = Field(description="The model constant C_mu, pinned server-side (the turbulence reference cites this tool as the constants' source)")
    k: InletQuantityModel = Field(description="Turbulence kinetic energy [m^2/s^2]")
    epsilon: InletQuantityModel = Field(description="Turbulence dissipation rate [m^2/s^3]")
    omega: InletQuantityModel = Field(description="Specific dissipation rate [1/s]")
    nu_t: InletQuantityModel = Field(description="Turbulent (eddy) viscosity [m^2/s]")
    viscosity_ratio: Optional[InletQuantityModel] = Field(description="The nu_t/nu sanity figure — included only when kinematic_viscosity is supplied (a healthy RAS inlet sits well under ~1000; None when nu was not given)")
    assumptions: List[str] = Field(description="Every applied default or conversion, stated — silence never hides an assumption")


@mcp.tool(name="estimate_turbulence_inlet")
async def estimate_turbulence_inlet(
    velocity: float = Field(description="Mean inlet velocity magnitude U [m/s], > 0"),
    intensity: Optional[float] = Field(default=None, description="Turbulence intensity I as a fraction (0.05 = 5%); omitted, the documented medium default 0.05 is applied and echoed in the output"),
    length_scale: Optional[float] = Field(default=None, description="Turbulence length scale l [m]; give exactly ONE of length_scale / hydraulic_diameter"),
    hydraulic_diameter: Optional[float] = Field(default=None, description="Hydraulic diameter D_h [m]; converted via the standard l = 0.07*D_h rule, named in the output"),
    kinematic_viscosity: Optional[float] = Field(default=None, description="Optional kinematic viscosity nu [m^2/s]; when given, the turbulent-viscosity ratio nu_t/nu is included as the sanity figure"),
    ctx: Context = None,
) -> TurbulenceInletResponse:
    """Compute inlet turbulence quantities k, epsilon and omega from velocity,
    intensity and a length scale — computed, never recalled.

    Pure server-side math (no case directory, no filesystem): k =
    3/2*(U*I)^2, epsilon = C_mu^(3/4)*k^(3/2)/l, omega =
    sqrt(k)/(C_mu^(1/4)*l), nu_t = C_mu*k^2/epsilon — each result carries
    the formula that produced it, with the pinned constant (C_mu = 0.09)
    echoed where it enters. An omitted intensity applies the documented
    medium default (0.05) and states it; a hydraulic diameter converts via
    the named l = 0.07*D_h rule. Supply kinematic_viscosity to get the
    nu_t/nu ratio as the sanity figure for spotting pathological
    combinations. Exactly one of length_scale / hydraulic_diameter is
    required (neither or both is a typed error), and non-physical inputs
    (non-positive or non-finite) raise typed errors — never plausible
    garbage numbers.
    """
    estimate = turbinlet.estimate_turbulence_inlet(
        velocity=velocity,
        intensity=intensity,
        length_scale=length_scale,
        hydraulic_diameter=hydraulic_diameter,
        kinematic_viscosity=kinematic_viscosity,
    )
    return TurbulenceInletResponse(**dataclasses.asdict(estimate))


# ============================================================================
# ESI translation (mechanical, rules-based)
# ============================================================================

@mcp.tool(name="translate_case_to_esi")
async def translate_case_to_esi(
    case_dir: str = Field(description="Case directory generated with Foundation v10 conventions"),
    ctx: Context = None,
) -> str:
    """Translate a Foundation OpenFOAM v10 case to ESI OpenFOAM (openfoam.com) conventions.

    Rules-based best-effort translation (dictionary names, keywords, solver
    names). Verify the translated case against your ESI installation.
    """
    case_dir = _require_case_dir(case_dir)

    def _translate():
        # The translator prints progress to stdout; keep stdio transport clean.
        with contextlib.redirect_stdout(sys.stderr):
            ESITranslator(case_dir).run_translation_pipeline()

    await asyncio.to_thread(_translate)
    return f"ESI translation complete for {case_dir}"


# ============================================================================
# SLURM / HPC tools
# ============================================================================

class SlurmSubmitResponse(BaseModel):
    job_id: Optional[str]
    submitted: bool
    error: str


@mcp.tool(name="submit_slurm_job")
async def submit_slurm_job(
    case_dir: str = Field(description="Case directory"),
    script_content: str = Field(description="Full SLURM batch script content (you write it for the user's cluster)"),
    ctx: Context = None,
) -> SlurmSubmitResponse:
    """Save a SLURM script into the case directory and submit it with sbatch.

    A successful submission flips the case's run-ledger row to running;
    poll slurm_job_status to have the outcome stamped when the job ends.
    """
    case_dir = _abs_case_dir(case_dir)
    script_path = os.path.join(case_dir, "submit_job.slurm")
    mechanics.save_file(script_path, script_content)
    job_id, submitted, error = await asyncio.to_thread(
        mechanics.submit_slurm_job, script_path, case_dir
    )
    return SlurmSubmitResponse(job_id=job_id, submitted=submitted, error=error)


@mcp.tool(name="slurm_job_status")
async def slurm_job_status(
    job_id: str = Field(description="SLURM job id returned by submit_slurm_job"),
    ctx: Context = None,
) -> str:
    """Check a SLURM job's status via squeue ('COMPLETED' when no longer queued).

    Observing a terminal state stamps the case's run-ledger row like a local
    run: done plus a Result verdict, or debugging when the job failed.
    """
    status, ok, error = await asyncio.to_thread(mechanics.check_job_status, job_id)
    if not ok:
        raise RuntimeError(error)
    return status


# ============================================================================
# Run ledger
# ============================================================================

class RunNoteResponse(BaseModel):
    id: str = Field(description="Zero-padded ledger ID of the updated row")
    case: str = Field(description="Case key (case directory relative to the runs root)")
    status: str = Field(description="Row Status after the write")
    result: str = Field(description="Row Result after the write")
    notes: str = Field(description="Notes cell after the write (as stored)")


@mcp.tool(name="set_run_note")
async def set_run_note(
    id: str = Field(description="Run ID from the runs/ledger.md table (zero-padded, e.g. '0003'; unpadded '3' is accepted)"),
    note: Optional[str] = Field(default=None, description="Replaces the row's Notes cell ('' clears it); omit to leave the note unchanged"),
    archive: Optional[bool] = Field(default=None, description="true archives the run (Status archived; a pending Result is stamped abandoned), false unarchives it (Status back to done); omit to leave Status unchanged"),
    ctx: Context = None,
) -> RunNoteResponse:
    """Annotate a run in the ledger and/or archive/unarchive it.

    This is the ONLY sanctioned skill-side write to runs/ledger.md — never
    edit the file directly. Notes plus archive/unarchive are the whole
    surface: every other cell (Status, Result, Solver, ...) stays owned by
    the server's run lifecycle. Unarchiving restores Status to done and
    leaves Result as it is. An unknown ID, or unarchiving a run that is not
    archived, fails with the ledger left untouched.
    """
    row = await asyncio.to_thread(mechanics.set_run_note, id, note, archive)
    return RunNoteResponse(id=row.id, case=row.case, status=row.status,
                           result=row.result, notes=row.notes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FastMCP OpenFOAM Agent Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="http")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        uvicorn_config = {"ws": "websockets"}
        mcp.run("http", host=args.host, port=args.port, uvicorn_config=uvicorn_config)


# run the server:
# python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
