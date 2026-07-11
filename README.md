# Foam-Agent    <a href="https://arxiv.org/abs/2505.04997"><img src="https://img.shields.io/badge/arXiv-2505.04997-b31b1b.svg" alt="Paper"></a>

<p align="center">
    <em>OpenFOAM CFD simulation driven by whatever AI agent you already use — no API key in the box.</em>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Claude_Code-supported-6B4FBB" alt="Claude Code">
    <img src="https://img.shields.io/badge/Cursor-supported-000000" alt="Cursor">
    <img src="https://img.shields.io/badge/Codex-supported-10A37F" alt="Codex">
    <img src="https://img.shields.io/badge/OpenCode-supported-F5A623" alt="OpenCode">
    <img src="https://img.shields.io/badge/API_keys-none_required-2EA44F" alt="No API keys">
</p>

**Foam-Agent** automates the entire CFD workflow — meshing, case setup, execution, error correction, post-processing — from a single natural language prompt.

## The version pin is the feature

OpenFOAM is two forks (Foundation at openfoam.org, ESI at openfoam.com) and many versions, and case dictionaries are not portable across them: keywords, file layouts, and solver names all drift. A case written for the wrong version fails with cryptic dictionary errors — the top documented error family in the community, and exactly the failure class Foam-Agent exists to prevent. Everything in the box is pinned to **Foundation OpenFOAM v10** and tested as one unit:

- the tutorial database the agent retrieves from is built from the v10 tutorial suite;
- the skill references encode v10 conventions (dictionary formats, solver names, boundary conditions);
- the container ships the pinned v10 toolchain, and the run/debug pipeline is validated end to end against it;
- the doctor keeps the pin honest: it detects an ESI install on your machine and names the mismatch up front, and its lockstep check fails when your skills and container image drift out of sync after a partial update.

ESI OpenFOAM (v2312, v2406, ...) is supported via best-effort translation (`translate_case_to_esi`) — cases are still generated and debugged on v10 first. The pin is the safety guarantee, not a limitation being hidden.

## Architecture: brain out, hands in

This fork of [csml-rpi/Foam-Agent](https://github.com/csml-rpi/Foam-Agent) restructures the original around a **"brain out, hands in"** split:

```
Your agent harness            Claude Code / Cursor / Codex / OpenCode
(the BRAIN — its model        guided by portable skills + subagents in agents/
 does the CFD reasoning)
        │  MCP (HTTP)
        ▼
foamagent-mcp in Docker       20 mechanical tools, ZERO API keys:
(the HANDS)                   RAG over v10 tutorials (local embeddings),
                              case file I/O, OpenFOAM execution, error
                              extraction, GMSH/PyVista scripts, SLURM
```

The intelligence comes from the AI subscription you already pay for. The container only needs OpenFOAM, the tutorial database, and local embeddings.

## Features

| | Capability |
|---|---|
| 📌 | **Version safety** — everything pinned to Foundation v10; the doctor names ESI mismatches up front and catches skills↔image drift after partial updates |
| 🗣️ | **Prompt → simulation** — describe any CFD problem in plain language; the agent plans, writes all case files, runs, and reports |
| 📚 | **Tutorial RAG** — semantic retrieval over all Foundation v10 tutorials with local embeddings (key-free) |
| 🔁 | **Auto error correction** — failed runs are diagnosed and fixed in a dedicated debug loop until the case converges |
| 🕸️ | **GMSH meshing** — geometry described in words becomes a validated `constant/polyMesh` |
| 🖼️ | **PyVista post-processing** — headless field rendering to PNG |
| 🖥️ | **HPC/SLURM** — job submission and polling for cluster runs |
| 🔌 | **4 CLIs, one repo** — MCP registration and skills committed for Claude Code, Cursor, Codex, and OpenCode |
| 🔒 | **Key-free server** — the Docker container needs no LLM provider; your harness brings the model |
| 🧾 | **Update contract** — `git pull` and `docker pull` never touch your simulations, prompts, meshes, or local settings |

## Quick start

### One command (recommended)

From nothing to conversational onboarding (needs [uv](https://docs.astral.sh/uv/)):

```bash
uv run https://raw.githubusercontent.com/KasperHonore/Foam-Agent/main/scripts/init.py
```

It validates prerequisites (git, git-lfs, Docker daemon) *before* cloning and prints the exact fix for anything missing, clones the latest release tag (falling back to the default branch while no releases exist — and saying so), verifies the FAISS indices are real LFS content, then prints the one line to open the clone in your AI CLI so the `foam-onboard` skill finishes setup conversationally. No uv? The same script runs on plain Python (3.10+, standard library only):

```bash
curl -sLO https://raw.githubusercontent.com/KasperHonore/Foam-Agent/main/scripts/init.py
python init.py
```

(`--target DIR` picks the clone destination, `--repo URL` a different repository.)

### The same path by hand

**1. Clone and start the server** (Docker required; FAISS indices are baked into the image):

```bash
git clone https://github.com/KasperHonore/Foam-Agent.git
cd Foam-Agent

docker pull ghcr.io/kasperhonore/foamagent:latest
docker tag ghcr.io/kasperhonore/foamagent:latest foamagent:latest
docker run -d --name foamagent-mcp --restart unless-stopped -p 7860:7860 \
  -v "$(pwd)/runs:/home/openfoam/Foam-Agent/runs" \
  foamagent:latest python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
```

**2. Open the repo in your AI CLI and let it finish the setup** — no manual file editing:

```bash
claude   # or cursor / codex / opencode
```

```
> onboard me
```

The `foam-onboard` skill health-checks the server, warms the embedding model (one-time ~1.2 GB download), captures your standing defaults in the preferences file, offers a demo simulation, and gives you a tour. MCP registration is already committed for every supported CLI, so the tools are wired the moment you open the repo.

### Simulate

```
/foam Simulate lid-driven cavity flow at Re=1000
```

Prefer to verify things yourself first? `python scripts/doctor.py` runs the same health checks without an agent (zero tokens) and prints exact fix commands.

<details>
<summary>Build the image from source instead</summary>

```bash
git lfs pull                                              # FAISS indices ship via LFS
docker build -f docker/Dockerfile -t foamagent:latest .   # first build: ~10 min, ~10 GB
```

</details>

## Advanced: bring Foam-Agent to your own project

The clone above is the beginner-friendly default — everything pre-wired, nothing to manage. If you work from your own repositories and control your own agent assets, the global path installs the product skills into *your* project instead, with no Foam-Agent clone anywhere:

```bash
npx skills add KasperHonore/Foam-Agent    # the product skills, into your project
npx add-mcp http://localhost:7860/mcp     # MCP registration for your harness
```

(Both are third-party CLIs; verify with `--help` on first use — they are not covered by this repo's CI.)

Simulations from every project land in **one central runs directory**, `~/foamagent/runs`, mounted into the container in place of a clone's `runs/`:

```bash
mkdir -p ~/foamagent/runs
docker pull ghcr.io/kasperhonore/foamagent:latest
docker tag ghcr.io/kasperhonore/foamagent:latest foamagent:latest
docker run -d --name foamagent-mcp --restart unless-stopped -p 7860:7860 \
  -v "$HOME/foamagent/runs:/home/openfoam/Foam-Agent/runs" \
  foamagent:latest python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
```

One run ledger at `~/foamagent/runs/ledger.md` then covers every project; per-project namespacing comes from nested case paths (`bike-fairing/inlet-nozzle`). Set `FOAMAGENT_HOME` to move the central root — the doctor and your `-v` mount must agree.

**The install contract is the same on both paths: a green doctor.** The doctor travels with the installed skills, so it runs from any directory with no clone:

```bash
python .claude/skills/foam-setup/references/doctor.py
```

Outside a clone it skips clone-only checks (git-lfs) with a note, expects the runs mount from the central directory, and compares the image's version stamp against the version stamped into the skills at install time. The full walkthrough — including the conversational residue no CLI covers (container start, embedding warm-up, native subagent registration) — is the global-install section of [the foam-setup skill](agents/skills/foam-setup/SKILL.md).

## Usage

Where slash commands exist (Claude Code), use them; everywhere else, just say it in plain language — the skills trigger either way.

| Command | Plain-language equivalent | What happens |
|---|---|---|
| `/foam-onboard` | "get me set up" / "onboard me" | Guided first-run: health check → warm-up → your defaults → demo → tour |
| `/foam <requirement>` | "simulate flow over a cylinder at Re=40" | Full pipeline: plan → generate case → run → debug loop → visualize |
| `/foam-setup` | "the foam server isn't responding" | Doctor: diagnoses Docker/image/container/LFS and brings the server up |
| `/foam-runs` | "list my runs" / "what happened to the dam break run?" | Run history from `runs/ledger.md`: listing, status, notes, archiving, comparisons — no log-diving |
| — | "mesh a 2D channel with a cylinder using gmsh" | foam-mesher subagent: GMSH → gmshToFoam → checkMesh |
| — | "plot the velocity field of the last run" | foam-visualizer subagent: headless PyVista → PNG |

## How it works

```
"Simulate dam break with two fluids"
        │
        ▼
  PLAN        find_similar_case → closest v10 tutorial as reference
        │
        ▼
  GENERATE    write_case_file × N → 0/, system/, constant/, Allrun
        │
        ▼
  RUN         run_case → success ─────────────► VISUALIZE (PyVista → PNG)
        │                                             ▲
        ▼ errors                                      │
  DEBUG       foam-debugger: diagnose → rewrite → rerun (until converged)
```

Every simulation lands in its own directory under `runs/`, with full logs — and gets a row in `runs/ledger.md`, the run ledger the server maintains automatically (ask "list my runs", or `python scripts/runs.py` for a zero-token check).

## What costs tokens — and what's free

The server never calls an LLM — there is no API key in the box. Tokens are spent only by your own harness's model, and only when an agent flow runs. Every routine check has a free deterministic twin:

**Zero tokens** — plain scripts, no agent involved:

| Script | What it does |
|---|---|
| `python scripts/doctor.py` | health check of the whole install, exact fix commands (`--json` for machines) |
| `python scripts/runs.py` | run history straight from the ledger |
| `python scripts/ledger_check.py` | verify/repair the ledger (dry-run by default) |
| `uv run .../scripts/init.py` | the one-command install (see Quick start) |

**Costs tokens** — agent flows on your harness's model:

| Flow | What you pay for |
|---|---|
| `/foam` simulations | planning, case-file generation, the debug loop, visualization |
| `foam-onboard` | conversational first-run setup |
| `foam-setup` | diagnosing and fixing a broken server conversationally |
| `foam-runs` | run-history questions in plain language |

Rule of thumb: when you don't need judgment, reach for the script.

## Updating

The clone is your workspace, and updates are designed to never touch your work:

```bash
git pull                                          # skills, subagents, MCP configs, server code
docker pull ghcr.io/kasperhonore/foamagent:latest # the server image (then recreate the container)
docker tag ghcr.io/kasperhonore/foamagent:latest foamagent:latest
docker rm -f foamagent-mcp && docker run -d --name foamagent-mcp --restart unless-stopped -p 7860:7860 \
  -v "$(pwd)/runs:/home/openfoam/Foam-Agent/runs" \
  foamagent:latest python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
```

**The contract:** `git pull` never touches your simulations (`runs/`, `output/`), your prompts and meshes at the repo root (`user_requirement.txt`, `user_req_*.txt`, `*.msh`), your preferences (`config/user.yml`, seeded at onboarding and read by the foam skill at case-planning time), or your local agent settings (`CLAUDE.md`, `.claude/settings.local.json`, `.claude/memory/`) — they are all gitignored. Your personal harness config is **user-owned**: the repo never ships a `CLAUDE.md` (agent guidance for this repo lives in `AGENTS.md`), and no update will ever create or overwrite yours. `runs/` is bind-mounted into the container, so simulation results live in your clone and survive container recreation too.

Skills and their matching server version update together in lockstep — `python scripts/doctor.py` verifies the lockstep and tells you when a newer release exists (notify-only: it never applies anything).

## Project structure

```
Foam-Agent/
├── agents/               # CANONICAL skills + subagents (edit here)
│   ├── skills/           #   foam, foam-setup, foam-onboard, foam-runs
│   └── subagents/        #   foam-debugger, foam-mesher, foam-visualizer
├── .claude/ .cursor/ .codex/ .opencode/        # generated per-CLI copies + MCP configs
├── src/mcp/              # the FastMCP server (the "hands")
├── src/                  # mechanics.py (mechanical layer) + ESI translation
├── database/faiss/       # pre-built tutorial indices (git-lfs)
├── docker/               # server image
├── examples/             # sample prompts + meshes (copy to root to use)
├── scripts/              # doctor.py, init.py, runs.py, ledger_check.py, sync_agent_assets.py, ...
├── tests/                # key-free unit tests + e2e vs a running server
└── runs/                 # YOUR simulations (gitignored)
```

## Skills and subagents

Canonical definitions live in [`agents/`](agents) and are fanned out to every tool's native location (`.claude/`, `.cursor/`, `.codex/`, `.opencode/`) by `python scripts/sync_agent_assets.py` — edit the canonical files, never the generated copies.

| Asset | Role |
|---|---|
| `foam` skill | End-to-end orchestration: plan → generate case → run → debug loop → visualize, with reference docs on v10 conventions, file generation, multiphase/VOF, Allrun rules, error playbook, SLURM |
| `foam-onboard` skill | Guided first-run: health check → warm-up → your defaults → demo simulation → tour |
| `foam-setup` skill | Preflight/doctor for the server, plus the global install path |
| `foam-runs` skill | Conversational run history over `runs/ledger.md`: listing, status, notes, archiving, cross-run comparison (writes only via `set_run_note`) |
| `foam-debugger` subagent | Owns the diagnose → rewrite → rerun loop |
| `foam-mesher` subagent | GMSH mesh generation → gmshToFoam → checkMesh |
| `foam-visualizer` subagent | Headless PyVista rendering |

Validated by autonomous end-to-end shakedowns: steady simpleFoam (backward-facing step, Re=800), transient multiphase interFoam (dam break), and a GMSH-meshed cylinder at Re=40 — all key-free, all physically verified.

## MCP tools

| Tool | What it does |
|------|--------------|
| `get_case_stats` | Valid case domains/categories/solvers |
| `search_tutorials` | Semantic search over v10 tutorials, Allrun scripts, command help |
| `find_similar_case` | Best-matching tutorial + directory structure + Allrun references |
| `resolve_case_dir` | Where a new case lives (under `runs/`) |
| `write_case_file` / `read_case_file` / `list_case_files` | Case file I/O on the server's filesystem |
| `run_case` | Execute Allrun, extract errors from logs |
| `run_openfoam_command` | One-off utilities: `checkMesh`, `gmshToFoam`, `decomposePar`, ... |
| `run_python_script` | Server-side Python (PyVista, GMSH) with stdout capture |
| `ensure_foam_file` / `read_mesh_boundaries` | Visualization marker; patch names/types |
| `parse_solver_log` | Typed convergence facts from a solver log — residuals, Courant, continuity, completion — plus a verdict with evidence |
| `assess_mesh` | Structured `checkMesh -allTopology -allGeometry`: mesh census, per-metric pass/warn/fail (topology vs geometry), verdict with evidence |
| `parse_force_coefficients` | Typed Cd/Cl/Cm from a case's forceCoeffs output — first/final values, tail-window mean/min/max, reference values — and the ledger's Key result cell filled |
| `translate_case_to_esi` | Rules-based Foundation→ESI translation (best-effort) |
| `submit_slurm_job` / `slurm_job_status` | HPC job submission and polling |
| `set_run_note` | Annotate/archive a run in `runs/ledger.md` — the only skill-side ledger write |

See [src/mcp/README.md](src/mcp/README.md) for details and local (non-Docker) installation.

## Configuration

| Environment variable | Purpose | Default |
|---|---|---|
| `WM_PROJECT_DIR` | OpenFOAM v10 install (execution tools) | set by the Docker image |
| `FOAMAGENT_EMBEDDING_PROVIDER` | `huggingface` (local, key-free), `openai`, `ollama` | `huggingface` |
| `FOAMAGENT_EMBEDDING_MODEL` | Embedding model for retrieval | `Qwen/Qwen3-Embedding-0.6B` |
| `FOAMAGENT_HOME` | Central root for global installs (the doctor and the runs mount must agree on it) | `~/foamagent` |

No LLM API keys are needed for the server or the skills.

## Sample prompts and meshes

Sample prompts and meshes live in [`examples/`](examples). To write your own, copy one to the repo root and edit it there — root-level `user_requirement.txt`, `user_req_*.txt` and `*.msh` are gitignored, so updates never touch them.

## Benchmarking against the original pipeline

The original self-contained LangGraph pipeline (`foambench_main.py`, made its own LLM calls, needed API keys) has been removed from `main` — this fork is key-free end to end. It is preserved at the [`legacy-pipeline`](../../tree/legacy-pipeline) git tag for a future harness-vs-harness-less comparison; check out the tag and follow its README to run it. Upstream's [FoamBench](https://arxiv.org/abs/2509.20374) evaluation of that pipeline reached 100% on 110 tasks with Claude Opus 4.6 at 25 correction loops.

## Development

```bash
python -m pytest tests/test_mechanics_unit.py       # key-free unit tests + asset drift check
python tests/test_lid_driven_cavity_mcp.py          # deterministic e2e vs a running server
python scripts/sync_agent_assets.py                 # regenerate per-tool skill/agent copies
python scripts/doctor.py                            # validate the local setup (read-only)
```

`AGENTS.md` documents the architecture for AI coding agents working on this repo.

## Troubleshooting

First stop: `python scripts/doctor.py` — it checks LFS, Docker, image, container, and the MCP endpoint, and prints exact fix commands. For everything else (API keys, disk space, custom meshes, updating, ESI vs Foundation), see the [FAQ](FAQ.md).

| Problem | Solution |
|---|---|
| MCP connection refused | Container not running — run the `foam-setup` skill or see its [SKILL.md](agents/skills/foam-setup/SKILL.md) |
| First retrieval call takes minutes | One-time ~1.2 GB embedding model download inside the container — not a hang |
| Retrieval errors / index not loaded | `git lfs pull` before building the image |
| `WM_PROJECT_DIR is not set` | Recreate the container so the entrypoint sources OpenFOAM |

## Share your simulation

Ran something cool? Open a ["Share your simulation"](https://github.com/KasperHonore/Foam-Agent/issues/new?template=share-your-simulation.yml) issue — prompt, solver, and a picture is all it takes. Real-world cases directly shape which skills and error-playbook entries get improved next.

## Citation

This fork builds on Foam-Agent by Yue et al. If you use it in research, please cite:

```bibtex
@article{yue2025foam,
  title={Foam-Agent: Towards Automated Intelligent CFD Workflows},
  author={Yue, Ling and Somasekharan, Nithin and Zhang, Tingwen and Cao, Yadi and Chen, Zhangze and Di, Shimin and Pan, Shaowu},
  journal={arXiv preprint arXiv:2505.04997},
  year={2025}
}

@article{somasekharan2026cfdllmbench,
    title={CFDLLMBench: A Benchmark Suite for Evaluating Large Language Models in Computational Fluid Dynamics},
    author={Somasekharan, Nithin and Yue, Ling and Cao, Yadi and Li, Weichao and Emami, Patrick and Bhargav, Pochinapeddi Sai and Acharya, Anurag and Xie, Xingyu and Pan, Shaowu},
    journal={Journal of Data-centric Machine Learning Research},
    year={2026},
    url={https://openreview.net/forum?id=kTcH1MnkjY}
}
```
