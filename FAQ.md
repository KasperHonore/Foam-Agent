# FAQ

### What do I need to run this?

Docker and an AI coding CLI you already use (Claude Code, Cursor, Codex, OpenCode, pi). That's it — the intelligence comes from your existing AI subscription, the container brings OpenFOAM and the tutorial database.

### Do I need an API key?

No. Nothing on `main` needs an API key: retrieval runs on local embeddings, execution is plain OpenFOAM, and all CFD reasoning is done by the AI coding tool you already pay for. (The original keyed LangGraph pipeline was removed; it survives at the `legacy-pipeline` git tag for benchmarking.)

### Which OpenFOAM version does it target?

**Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) — all generated dictionaries, file names, and solver binaries follow v10 conventions. ESI OpenFOAM (openfoam.com, v2312/v2406/...) is supported only via best-effort translation (`translate_case_to_esi`).

### The first retrieval call has been silent for minutes. Is it hung?

No. The first `search_tutorials`/`find_similar_case` call downloads a ~1.2 GB embedding model inside the container and loads 4 FAISS indices. It's a one-time wait — the `foam-onboard` skill triggers it deliberately during setup so it doesn't surprise you mid-simulation.

### Where do my simulations end up?

Each case gets its own directory under `runs/`, with all generated files and `log.*` output. `runs/` is bind-mounted from your clone into the container, so results are directly visible on your machine and survive container recreation. It is gitignored, so `git pull` never touches it either.

### How do I update, and will it break my work?

```bash
git pull
docker pull ghcr.io/kasperhonore/foamagent:latest
docker tag ghcr.io/kasperhonore/foamagent:latest foamagent:latest
docker rm -f foamagent-mcp && docker run -d --name foamagent-mcp --restart unless-stopped -p 7860:7860 \
  -v "$(pwd)/runs:/home/openfoam/Foam-Agent/runs" \
  foamagent:latest python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
```

Your simulations (`runs/`, `output/`), root-level prompts and meshes (`user_requirement.txt`, `user_req_*.txt`, `*.msh`), and local agent settings (`CLAUDE.md`, `.claude/settings.local.json`, `.claude/memory/`) are all gitignored — updates never touch them.

### Can I use my own mesh?

Yes, two ways: drop a GMSH `.msh` file at the repo root and mention it in your prompt, or describe the geometry in words and the foam-mesher subagent generates it with GMSH (then converts and validates it with `gmshToFoam` + `checkMesh`).

### Which AI CLIs are supported?

MCP registration and skills are committed for **Claude Code** (`.mcp.json`, `.claude/`), **Cursor** (`.cursor/`), **Codex** (`.codex/`), **OpenCode** (`opencode.json`, `.opencode/`), and **pi** (`.pi/`). Any other MCP-capable agent works too: point it at `http://localhost:7860/mcp` and it will find the skills via `AGENTS.md`.

### Do I need a GPU?

No. Embeddings run fine on CPU, and OpenFOAM is CPU-based. For big cases there's SLURM support (`submit_slurm_job`) to push runs to a cluster.

### How much disk space does it take?

The Docker image is ~10 GB — mostly the OpenFOAM + ParaView base, plus a CPU-only Python environment and the baked-in FAISS indices. Add ~1.2 GB for the embedding model downloaded on first use, plus whatever your simulations produce.

### Can I run it without Docker?

On Linux/macOS with a local Foundation OpenFOAM v10 install: `pip install -e .` then `foamagent-mcp` (stdio transport), and switch the committed MCP configs from the HTTP URL to `"command": "foamagent-mcp"`. Retrieval works without OpenFOAM; execution tools need `WM_PROJECT_DIR`. Details in [src/mcp/README.md](src/mcp/README.md).

### Something's broken — where do I start?

`python scripts/doctor.py` — read-only, checks LFS/Docker/image/container/endpoint and prints exact fix commands. Or open the repo in your agent and say "the foam server isn't responding" (the `foam-setup` skill works through the same checks interactively). Failed *simulations* are handled automatically by the foam-debugger loop.

### A run failed and the agent couldn't fix it. What now?

Look at the `log.*` files in the case directory under `runs/` — the last `FOAM FATAL ERROR` block is usually the story. The error playbook the agent uses is human-readable too: [agents/skills/foam/references/error-playbook.md](agents/skills/foam/references/error-playbook.md). If it looks like a bug or a missing pattern, please open an issue with the prompt and the log excerpt.

### I ran something cool. Can I share it?

Yes please — use the ["Share your simulation" issue template](https://github.com/KasperHonore/Foam-Agent/issues/new?template=share-your-simulation.yml). Prompt + solver + a picture is all it takes; real-world cases directly shape which skills and error-playbook entries get improved next.
