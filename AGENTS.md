# AGENTS.md

> This file helps AI agents (Codex, Cursor, Claude Code, OpenCode, pi, Copilot, etc.) understand and work with this codebase.

## What is Foam-Agent?

Foam-Agent automates CFD (Computational Fluid Dynamics) simulations in **Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) from natural language prompts. The architecture is "brain out, hands in":

- **The `foamagent` MCP server** (this repo) is the *hands*: key-free mechanical tools — FAISS tutorial retrieval (local embeddings), case file I/O, OpenFOAM execution, error extraction, PyVista/GMSH script execution, SLURM. No LLM provider needed.
- **Skills and subagents under `agents/`** are the *brain-guidance*: they run on YOUR agent harness's model (Claude Code, Cursor, Codex, OpenCode, pi, ...) and orchestrate the MCP tools.

Nothing in this repo needs an LLM API key. (The original self-contained LangGraph pipeline was removed from `main`; it is preserved at the `legacy-pipeline` git tag for benchmarking.)

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

# Run tests (key-free unit tests + agent-asset drift check)
pytest tests/test_mechanics_unit.py -v

# End-to-end test (needs the MCP server running with OpenFOAM)
python tests/test_lid_driven_cavity_mcp.py

# Start MCP server (key-free)
python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
```

Requires **Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) at runtime (`$WM_PROJECT_DIR` must be set). ESI OpenFOAM (openfoam.com) is not compatible. Python 3.12; all dependencies live in `pyproject.toml` (the Docker image installs them CPU-only with uv).

## Architecture

The simulation loop is driven by the agent harness following the `foam` skill: plan the case → retrieve a similar tutorial (`find_similar_case`) → write all case files (`write_case_file`) → run (`run_case`) → diagnose/rewrite/rerun on errors (foam-debugger subagent) → optionally visualize (foam-visualizer subagent). The server never decides anything; it executes.

### Directory Structure

```
agents/                # CANONICAL skills + subagents (the "brain-guidance")
src/
  mechanics.py         # key-free mechanical layer: file I/O, execution, log
                       # parsing, lazy FAISS retrieval (local HF embeddings),
                       # Python script execution, SLURM
  ledger.py            # run ledger (runs/ledger.md): server-owned record of
                       # every run, written as run-lifecycle side effects
  mcp/                 # FastMCP server exposing mechanics as tools (see src/mcp/README.md)
  translation/         # rules-based Foundation→ESI case translation
database/
  faiss/               # Pre-built FAISS vector indices (do NOT regenerate unless necessary)
  raw/                 # Raw OpenFOAM tutorial data
tests/                 # key-free unit tests + e2e vs a running server
scripts/               # doctor.py, sync_agent_assets.py, ...
docker/                # Dockerfile for containerized deployment
examples/              # sample prompts + meshes
```

## Environment Variables

| Variable | Purpose | Needed by |
|----------|---------|-----------|
| `WM_PROJECT_DIR` | OpenFOAM installation path (required to execute simulations) | MCP server |
| `FOAMAGENT_EMBEDDING_PROVIDER` | Embedding backend: `huggingface` (default, local/key-free), `openai`, `ollama` | MCP server |
| `FOAMAGENT_EMBEDDING_MODEL` | Embedding model (default: `Qwen/Qwen3-Embedding-0.6B`) | MCP server |

## Common Tasks

### Adding a new MCP tool
Implement the mechanics in `src/mechanics.py`, expose it as a tool in `src/mcp/fastmcp_server.py`, and cover it in `tests/test_mechanics_unit.py`. If skills should use it, document it in `agents/skills/foam/` and re-run `python scripts/sync_agent_assets.py`.

### Changing how cases are generated or debugged
That logic lives in the skills, not the server: edit `agents/skills/foam/` (and `agents/subagents/`), then regenerate the per-tool copies with `python scripts/sync_agent_assets.py`.

### Rebuilding FAISS indices
Only needed if OpenFOAM tutorials change:
```bash
python init_database.py --openfoam_path $WM_PROJECT_DIR --force
```

## Things to Watch Out For

- **Do not regenerate FAISS indices** unless you have a specific reason. The pre-built indices in `database/faiss/` are correct and ready to use.
- **Foundation OpenFOAM v10 must be sourced** for any simulation execution. Without `$WM_PROJECT_DIR`, execution tools will fail. ESI OpenFOAM is not compatible.
- **Never edit generated skill copies** (`.claude/`, `.cursor/`, `.codex/`, `.opencode/`, `.pi/`) — edit `agents/` and run `python scripts/sync_agent_assets.py`; CI fails on drift.

## Agent skills

### Issue tracker

Hybrid: drafts and research live as local markdown under `.scratch/<feature>/` at the workspace root (one level above this repo checkout); actionable specs are published to GitHub Issues on KasperHonore/Foam-Agent. External PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical defaults for all five roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`); `ready-for-agent` and `wontfix` already exist on GitHub. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
