# Foam-Agent MCP Server (key-free)

Exposes the **mechanical** side of OpenFOAM CFD simulation as [MCP](https://modelcontextprotocol.io/) tools: tutorial retrieval (RAG over Foundation OpenFOAM v10 tutorials), case file I/O, simulation execution, error extraction, Python script execution (GMSH meshing, PyVista visualization) and SLURM job management.

**No LLM provider, no API key.** All CFD *reasoning* — planning the case, writing OpenFOAM dictionaries, diagnosing errors — is done by the AI agent that calls these tools, guided by the portable skills/subagents in [`agents/`](../../agents) at the repo root. Whatever agent harness you already use (Claude Code, Cursor, Codex, OpenCode, ...) supplies the intelligence; this server supplies the hands.

```
Your agent harness  (the brain — its model does the CFD reasoning)
    │  guided by agents/skills/foam + agents/subagents/*
    │  MCP protocol (stdio or HTTP)
    ▼
foamagent-mcp  (the hands — no API key)
    ├─ FAISS tutorial retrieval (local embeddings)
    ├─ case file read/write
    ├─ OpenFOAM execution + error extraction
    ├─ Python script execution (GMSH / PyVista)
    └─ SLURM submission
```

> **OpenFOAM version:** Foundation OpenFOAM v10 (openfoam.org). The `translate_case_to_esi` tool converts a generated case to ESI (openfoam.com) conventions on a best-effort basis.

## Quick start

### Docker (recommended — includes OpenFOAM v10)

```bash
# Prebuilt image from GHCR (published by .github/workflows/docker-publish.yml)
docker pull ghcr.io/kasperhonore/foamagent:latest
docker tag ghcr.io/kasperhonore/foamagent:latest foamagent:latest

# or build from source (~10 min, ~10 GB; needs `git lfs pull` first)
docker build -f docker/Dockerfile -t foamagent:latest .

docker run -it -p 7860:7860 -v "$(pwd)/runs:/home/openfoam/Foam-Agent/runs" \
  foamagent:latest foamagent-mcp --transport http
```

The MCP registration is **committed in this repo** and points at `http://localhost:7860/mcp`:

| Tool | Config file |
|------|-------------|
| Claude Code | `.mcp.json` |
| Cursor | `.cursor/mcp.json` |
| OpenCode | `opencode.json` |
| Codex | `.codex/config.toml` (copy to `~/.codex/config.toml` if project config is unsupported) |

Open the repo in your tool and the `foamagent` server is available — nothing else to configure.

### Local install (Linux/macOS with OpenFOAM v10)

```bash
pip install -e .          # key-free core only
foamagent-mcp             # stdio transport
```

Switch the config files above to stdio (`"command": "foamagent-mcp"`) if you prefer this mode. `WM_PROJECT_DIR` must point at your OpenFOAM v10 installation for execution tools to work; retrieval tools work without it.

## Tools

| Tool | What it does |
|------|--------------|
| `get_case_stats` | Valid case domains/categories/solvers |
| `search_tutorials` | Semantic search over v10 tutorials, Allrun scripts, command help |
| `find_similar_case` | Best-matching tutorial + directory structure + Allrun references |
| `resolve_case_dir` | Where a new case should live (under `runs/`) |
| `write_case_file` / `read_case_file` / `list_case_files` | Case file I/O on the server's filesystem |
| `run_case` | Execute Allrun, extract errors from logs |
| `start_case` | Start Allrun detached and return the case's ledger run id immediately — identical clean-rerun sweep and ledger stamping as `run_case`, for runs long enough to outlive a blocking call (transient, fine-mesh, parallel); pidfile in the case directory, no watchdog |
| `case_status` | Poll a background run by run id: typed convergence-parser progress on the partial log (latest time, residuals, Courant, verdict-so-far) plus elapsed wall time while it lives; the poll that observes the exit stamps the row through the identical completion path a blocking run takes (done with parser-backed Result, or debugging with extracted errors), and a stale `running` row heals on the next poll |
| `stop_case` | Stop a background run, evidence kept: graceful first (`stopAt writeNow` with the mtime handling that makes the running solver actually see it — requires `runTimeModifiable true`, which v10 does **not** default to — so the solver writes the current fields and exits cleanly within a bounded grace window), then a process-group kill; `stopAt` restored afterwards, the row stamped through the identical completion path plus a Notes entry naming the deliberate stop; a run that exited on its own is stamped normally (`already_exited`) |
| `run_openfoam_command` | One-off utilities: `checkMesh`, `gmshToFoam`, `decomposePar`, ... |
| `run_python_script` | Server-side Python (PyVista rendering, GMSH mesh generation) |
| `ensure_foam_file` | `.foam` marker for the PyVista OpenFOAM reader |
| `read_mesh_boundaries` | Patch names/types from `constant/polyMesh/boundary` |
| `parse_solver_log` | Typed convergence facts from a solver log — residuals, Courant, continuity, completion — plus a verdict with evidence |
| `assess_mesh` | Structured `checkMesh -allTopology -allGeometry`: mesh census, per-metric pass/warn/fail (topology vs geometry), verdict with evidence |
| `parse_force_coefficients` | Typed Cd/Cl/Cm from a case's forceCoeffs output — first/final values, tail-window mean/min/max, reference values — and the ledger's Key result cell filled |
| `inspect_stl` | Structured `surfaceCheck` on an STL surface: watertightness with defective-edge counts, triangles/vertices, bounding box, parts/zones, units suspicion, verdict with evidence (surfaceCheck's exit code is 0 even for defective surfaces — the text is the truth) |
| `import_geometry` | STL into the case's `constant/triSurface/` under a normalized name; optional deterministic mm→m scaling via `surfaceTransformPoints`; typed result with case-relative destination, applied scale, byte size and an explicit overwrote flag |
| `estimate_wall_spacing` | Pure-math wall-spacing calculator: flow conditions + target y⁺ → Reynolds number with regime verdict, named skin-friction correlation (flat-plate family external, pipe family internal), friction velocity, first-cell **centre** distance and first-cell **height** as two separately labelled fields, boundary-layer thickness estimate, suggested layer count — every number carries the formula that produced it |
| `estimate_turbulence_inlet` | Inlet k/ε/ω from velocity, intensity and a length scale (or hydraulic diameter via the named `l = 0.07·D_h` rule) — pure server-side math, each value carrying its formula with C_μ pinned server-side; omitted intensity applies the stated medium default (0.05); optional ν gives the ν_t/ν sanity ratio |
| `translate_case_to_esi` | Rules-based Foundation→ESI translation |
| `submit_slurm_job` / `slurm_job_status` | HPC job submission and polling |
| `set_run_note` | Annotate/archive a run in `runs/ledger.md` — the only skill-side ledger write |

## Configuration

| Environment variable | Purpose | Default |
|---------------------|---------|---------|
| `FOAMAGENT_EMBEDDING_PROVIDER` | Embedding backend for retrieval: `huggingface`, `openai`, `ollama` | `huggingface` (local, key-free) |
| `FOAMAGENT_EMBEDDING_MODEL` | Embedding model | `Qwen/Qwen3-Embedding-0.6B` |
| `WM_PROJECT_DIR` | OpenFOAM v10 installation (required for execution tools) | — |

No `FOAMAGENT_MODEL_*` variables, no API keys — the server makes no LLM calls at all.

## Troubleshooting

- **Import errors:** run `pip install -e .` from the repo root.
- **Missing FAISS indices:** they ship pre-built in `database/faiss/`; rebuild with `python init_database.py --openfoam_path $WM_PROJECT_DIR --force` only if tutorials changed.
- **`WM_PROJECT_DIR is not set`:** source OpenFOAM v10 before starting the server (the Docker image does this automatically).
