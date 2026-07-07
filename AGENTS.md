# AGENTS.md

> This file helps AI agents (Codex, Cursor, Claude Code, OpenCode, pi, Copilot, etc.) understand and work with this codebase.

## What is Foam-Agent?

Foam-Agent automates CFD (Computational Fluid Dynamics) simulations in **Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) from natural language prompts. The architecture is "brain out, hands in":

- **The `foamagent` MCP server** (this repo) is the *hands*: key-free mechanical tools — FAISS tutorial retrieval (local embeddings), case file I/O, OpenFOAM execution, error extraction, PyVista/GMSH script execution, SLURM. No LLM provider needed.
- **Skills and subagents under `agents/`** are the *brain-guidance*: they run on YOUR agent harness's model (Claude Code, Cursor, Codex, OpenCode, pi, ...) and orchestrate the MCP tools.
- A legacy self-contained LangGraph pipeline (`foambench_main.py`, `pip install -e .[pipeline]`) still exists for harness-less batch runs; only it needs `FOAMAGENT_MODEL_*` API keys.

> **Important:** All generated case files, dictionary names, and solver binaries follow Foundation OpenFOAM v10 conventions. ESI OpenFOAM (openfoam.com, e.g., v2312, v2406, v2512) is supported only via best-effort translation (`translate_case_to_esi`).

## Skills, subagents, and MCP registration

- **Canonical sources (edit these):** `agents/skills/<name>/SKILL.md` (+ `references/`) and `agents/subagents/<name>.md`.
- **Generated copies (do NOT edit):** `.claude/skills`, `.claude/agents`, `.opencode/skill`, `.opencode/agent`, `.codex/skills`, `.pi/skills`, `.cursor/skills`, `.cursor/rules/foamagent-skills.mdc`. Regenerate with `python scripts/sync_agent_assets.py` (CI runs `--check`).
- **MCP registration is committed per tool:** `.mcp.json` (Claude Code), `.cursor/mcp.json`, `opencode.json`, `.codex/config.toml` — all pointing at `http://localhost:7860/mcp` (start it with Docker, see `src/mcp/README.md`).
- **Universal fallback:** if your tool auto-discovers none of the above, read `agents/skills/foam/SKILL.md` when the user asks for a CFD simulation and follow it; subagent roles are in `agents/subagents/` — follow them inline.
- **Server not responding?** Follow `agents/skills/foam-setup/SKILL.md` — it diagnoses and starts the Dockerized MCP server (no API key needed). `python scripts/doctor.py` runs the same checks deterministically (read-only, prints fix commands).

## First run? Onboard the user

If the user seems new here — says "get me set up", "onboard me", "what is this repo?", "getting started", or asks for a simulation on a machine where the `foamagent` MCP tools don't respond — follow `agents/skills/foam-onboard/SKILL.md`. It walks them conversationally through: health check (`python scripts/doctor.py`) → fixing anything broken (via foam-setup) → one-time embedding warm-up → an optional demo simulation (lid-driven cavity) → a short tour. Never make the user edit files by hand during setup.

## Build and Run

```bash
# Environment setup (bare-metal; the Docker image installs the same manifest with uv)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[all]"

# Run a simulation (sample prompts live in examples/; user copies at the repo
# root are gitignored, so edit those rather than the examples)
python foambench_main.py --output ./output --prompt_path ./examples/user_requirement.txt

# Run with custom mesh
python foambench_main.py --output ./output --prompt_path ./examples/user_req_tandem_wing.txt --custom_mesh_path ./examples/tandem_wing.msh

# Run tests (key-free unit tests + agent-asset drift check)
pytest tests/test_mechanics_unit.py -v

# End-to-end test (needs the MCP server running with OpenFOAM)
python tests/test_lid_driven_cavity_mcp.py

# Start MCP server (key-free)
python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
```

Requires **Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) at runtime (`$WM_PROJECT_DIR` must be set). ESI OpenFOAM (openfoam.com) is not compatible. Python 3.12; all dependencies live in `pyproject.toml` (the Docker image installs them CPU-only with uv).

## Architecture

### Legacy pipeline workflow (LangGraph StateGraph)

The sections below describe the **legacy self-contained pipeline** (`foambench_main.py`). The MCP server does not run this pipeline — it exposes only the mechanical layer (`src/mechanics.py`) as tools; the calling agent harness does the reasoning.

Defined in `src/main.py`:

```
PLANNER -> [mesh routing] -> MESHING (if needed) -> INPUT_WRITER -> [HPC/local routing]
-> RUNNER -> [error check] -> REVIEWER -> INPUT_WRITER (retry loop, max 25 iterations)
-> VISUALIZATION (if requested) -> END
```

All routing decisions (mesh type, HPC vs local, visualization) are LLM calls in `src/router_func.py`.

### Directory Structure

```
src/
  main.py              # LangGraph workflow definition and entry point
  config.py            # Config dataclass with env var overrides
  utils.py             # GraphState (TypedDict), LLMService (unified LLM interface)
  models.py            # I/O models for the legacy pipeline's run services
  router_func.py       # LLM-based routing decisions
  logger.py            # Structured XML-tagged logging
  nodes/               # LangGraph node functions (thin wrappers calling services)
    planner_node.py
    input_writer_node.py
    meshing_node.py
    local_runner_node.py
    hpc_runner_node.py
    reviewer_node.py
    visualization_node.py
  services/            # Business logic (where the real work happens)
    plan.py            # Case planning and analysis
    input_writer.py    # OpenFOAM file generation via LLM + RAG
    mesh.py            # Mesh generation (blockMesh / Gmsh conversion)
    run_local.py       # Local OpenFOAM execution
    run_hpc.py         # HPC job submission
    review.py          # Error diagnosis and fix planning
    visualization.py   # PyVista-based post-processing
  mcp/                 # key-free FastMCP server (mechanical tools; see src/mcp/README.md)
database/
  faiss/               # Pre-built FAISS vector indices (do NOT regenerate unless necessary)
  raw/                 # Raw OpenFOAM tutorial data
tests/                 # pytest tests (service layer + MCP integration)
docker/                # Dockerfile for containerized deployment
```

### Key Abstractions

- **`GraphState`** (`src/utils.py`): TypedDict threaded through all workflow nodes. Contains user requirement, case metadata, generated files, error logs, loop count.
- **`LLMService`** (`src/utils.py`): Unified LLM interface supporting OpenAI, Anthropic, Bedrock, Ollama. Provides `invoke()` and `structure_output()` (Pydantic-validated).
- **`Config`** (`src/config.py`): Global config dataclass. Every field can be overridden via `FOAMAGENT_*` env vars.
- **Pydantic models**: `FoamPydantic`/`FoamfilePydantic` (`src/utils.py`) for generated files, `RewritePlan` (`src/services/review.py`) for error fixes, `CaseSummaryModel` (`src/services/plan.py`) for case metadata.

### Design Patterns

1. **Service-oriented**: Nodes in `src/nodes/` are thin orchestration wrappers. All logic lives in `src/services/`.
2. **Error correction loop**: Runner detects errors -> Reviewer diagnoses via LLM -> Input Writer rewrites targeted files -> re-run (up to `max_loop` iterations).
3. **RAG retrieval**: FAISS indices built from OpenFOAM tutorials provide reference cases to the input writer.
4. **Two generation modes** (`config.input_writer_generation_mode`):
   - `sequential_dependency` (default): Files generated in order with cross-file context.
   - `parallel_no_context`: All files generated independently (faster, relies on retry loop).

## Environment Variables

| Variable | Purpose | Needed by |
|----------|---------|-----------|
| `WM_PROJECT_DIR` | OpenFOAM installation path (required to execute simulations) | MCP server + pipeline |
| `FOAMAGENT_EMBEDDING_PROVIDER` | Embedding backend: `huggingface` (default, local/key-free), `openai`, `ollama` | MCP server + pipeline |
| `FOAMAGENT_EMBEDDING_MODEL` | Embedding model (default: `Qwen/Qwen3-Embedding-0.6B`) | MCP server + pipeline |
| `FOAMAGENT_MODEL_PROVIDER` | LLM provider: `openai`, `openai-codex`, `anthropic`, `bedrock`, `ollama` | legacy pipeline only |
| `FOAMAGENT_MODEL_VERSION` | Model identifier (e.g., `claude-opus-4-6`, `gpt-5.3-codex`) | legacy pipeline only |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | LLM API keys | legacy pipeline only |

## Common Tasks

### Adding a new LLM provider
Extend `LLMService` in `src/utils.py`. Follow the pattern of existing providers (each has an `if` branch in the constructor).

### Adding a new workflow node
1. Create service logic in `src/services/`.
2. Create a thin node wrapper in `src/nodes/`.
3. Wire it into the StateGraph in `src/main.py`.

### Modifying file generation
The input writer logic is in `src/services/input_writer.py`. It uses RAG context from FAISS indices and LLM calls to generate OpenFOAM configuration files.

### Rebuilding FAISS indices
Only needed if OpenFOAM tutorials change:
```bash
python init_database.py --openfoam_path $WM_PROJECT_DIR --force
```

## Things to Watch Out For

- **Do not regenerate FAISS indices** unless you have a specific reason. The pre-built indices in `database/faiss/` are correct and ready to use.
- **Foundation OpenFOAM v10 must be sourced** for any simulation execution. Without `$WM_PROJECT_DIR`, the runner nodes will fail. ESI OpenFOAM is not compatible.
- **The error correction loop** can run up to 25 iterations. When modifying the reviewer or input writer, consider the impact on convergence.
- **`GraphState` is mutable** and passed by reference through the entire pipeline. Be careful about unintended side effects when modifying state fields.
