# Foam-Agent    <a href="https://arxiv.org/abs/2505.04997"><img src="https://img.shields.io/badge/arXiv-2505.04997-b31b1b.svg" alt="Paper"></a>

<p align="center">
    <em>OpenFOAM CFD simulation driven by whatever AI agent you already use — no API key in the box.</em>
</p>

**Foam-Agent** automates the entire **Foundation OpenFOAM v10** CFD workflow — meshing, case setup, execution, error correction, post-processing — from a single natural language prompt. This fork of [csml-rpi/Foam-Agent](https://github.com/csml-rpi/Foam-Agent) restructures it around a **"brain out, hands in"** architecture:

```
Your agent harness            Claude Code / Cursor / Codex / OpenCode / pi
(the BRAIN — its model        guided by portable skills + subagents in agents/
 does the CFD reasoning)
        │  MCP (HTTP)
        ▼
foamagent-mcp in Docker       15 mechanical tools, ZERO API keys:
(the HANDS)                   RAG over v10 tutorials (local embeddings),
                              case file I/O, OpenFOAM execution, error
                              extraction, GMSH/PyVista scripts, SLURM
```

The intelligence comes from the AI subscription you already pay for. The container only needs OpenFOAM, the tutorial database, and local embeddings.

## Quick start

```bash
git clone https://github.com/KasperHonore/Foam-Agent.git
cd Foam-Agent
git lfs pull                                              # FAISS indices ship via LFS

# Build and start the server (first build: 30-45 min, ~29 GB)
docker build -f docker/Dockerfile -t foamagent:latest .
docker run -d --name foamagent-mcp --restart unless-stopped -p 7860:7860 \
  foamagent:latest python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
```

Then open the repo in your agent tool. **MCP registration is already committed** — `.mcp.json` (Claude Code), `.cursor/mcp.json`, `opencode.json`, `.codex/config.toml` all point at `http://localhost:7860/mcp`. Ask naturally, or in Claude Code:

```
/foam Simulate lid-driven cavity flow at Re=1000
```

Something not responding? The `/foam-setup` skill (or [agents/skills/foam-setup/SKILL.md](agents/skills/foam-setup/SKILL.md)) diagnoses Docker/image/container/LFS issues step by step.

## Skills and subagents

Canonical definitions live in [`agents/`](agents) and are fanned out to every tool's native location (`.claude/`, `.cursor/`, `.codex/`, `.opencode/`, `.pi/`) by `python scripts/sync_agent_assets.py` — edit the canonical files, never the generated copies.

| Asset | Role |
|---|---|
| `foam` skill | End-to-end orchestration: plan → generate case → run → debug loop → visualize, with reference docs on v10 conventions, file generation, multiphase/VOF, Allrun rules, error playbook, SLURM |
| `foam-setup` skill | Preflight/doctor for the server |
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
| `translate_case_to_esi` | Rules-based Foundation→ESI translation (best-effort) |
| `submit_slurm_job` / `slurm_job_status` | HPC job submission and polling |

See [src/mcp/README.md](src/mcp/README.md) for details and local (non-Docker) installation.

## Configuration

| Environment variable | Purpose | Default |
|---|---|---|
| `WM_PROJECT_DIR` | OpenFOAM v10 install (execution tools) | set by the Docker image |
| `FOAMAGENT_EMBEDDING_PROVIDER` | `huggingface` (local, key-free), `openai`, `ollama` | `huggingface` |
| `FOAMAGENT_EMBEDDING_MODEL` | Embedding model for retrieval | `Qwen/Qwen3-Embedding-0.6B` |
| `FOAMAGENT_OPENFOAM_FORK` | `foundation` or `esi` (best-effort translation) | `foundation` |

No LLM API keys are needed for the server or the skills.

## Legacy pipeline (self-contained, needs an LLM key)

The original LangGraph pipeline — where Foam-Agent makes its own LLM calls — is retained for harness-less batch runs and benchmarking:

```bash
pip install -e .[pipeline]
export FOAMAGENT_MODEL_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-...
export FOAMAGENT_MODEL_VERSION=claude-opus-4-6
python foambench_main.py --output ./output --prompt_path ./user_requirement.txt
# custom mesh: add --custom_mesh_path ./tandem_wing.msh
```

`FOAMAGENT_MODEL_PROVIDER` supports `openai`, `openai-codex` (ChatGPT/Codex OAuth via `~/.codex/auth.json`), `anthropic`, `bedrock`, `ollama`. Upstream's [FoamBench](https://arxiv.org/abs/2509.20374) evaluation of this pipeline reached 100% on 110 tasks with Claude Opus 4.6 at 25 correction loops.

## Development

```bash
python -m pytest tests/test_mechanics_unit.py       # key-free unit tests + asset drift check
python tests/test_lid_driven_cavity_mcp.py          # deterministic e2e vs a running server
python scripts/sync_agent_assets.py                 # regenerate per-tool skill/agent copies
```

`AGENTS.md` documents the architecture for AI coding agents working on this repo.

## Troubleshooting

| Problem | Solution |
|---|---|
| MCP connection refused | Container not running — run the `foam-setup` skill or see its [SKILL.md](agents/skills/foam-setup/SKILL.md) |
| First retrieval call takes minutes | One-time ~1.2 GB embedding model download inside the container — not a hang |
| Retrieval errors / index not loaded | `git lfs pull` before building the image |
| `WM_PROJECT_DIR is not set` | Recreate the container so the entrypoint sources OpenFOAM |

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
