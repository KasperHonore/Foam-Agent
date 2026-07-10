# Contributing to Foam-Agent

Thanks for helping improve Foam-Agent. This is a hard fork of
[csml-rpi/Foam-Agent](https://github.com/csml-rpi/Foam-Agent), developed
independently — **PRs land here, not upstream.** The architecture is
*"brain out, hands in"*: your AI harness is the brain, the Docker MCP server is
the hands. Keep that split in mind — the server stays mechanical and key-free;
the intelligence lives in the portable skills and subagents under `agents/`.

## Ways to contribute

- **🐞 Report a bug** — use the bug template; logs make or break a report.
- **💡 Request a feature** — a new tool, skill, or solver coverage.
- **🌊 Share a simulation** — the showcase template. Real cases drive what gets
  improved next.
- **🔧 Send a PR** — see the workflow below.

## Development setup

The server runs in Docker, so you don't need OpenFOAM on your host to work on
the Python layer:

```bash
git clone https://github.com/KasperHonore/Foam-Agent.git
cd Foam-Agent
python -m pip install pytest        # enough for the unit suite
```

For the full server, see the [Quick start](README.md#quick-start).

## Running the tests

The unit suite is key-free and needs no OpenFOAM, FAISS, or Docker:

```bash
pytest -m "not integration" --ignore=tests/test_lid_driven_cavity_mcp.py
```

- `tests/test_mechanics_unit.py`, `tests/test_ledger_unit.py`,
  `tests/test_esi_translator.py` — pure Python, run everywhere.
  **This is what CI gates on.**
- `tests/test_lid_driven_cavity_mcp.py` — end-to-end; needs a **running MCP
  server** with OpenFOAM v10. Run it by hand against a live server:
  `python tests/test_lid_driven_cavity_mcp.py`.

If you touch anything under `agents/`, regenerate the per-tool copies and keep
the sync check green:

```bash
python scripts/sync_agent_assets.py         # regenerate the per-tool copies
python scripts/sync_agent_assets.py --check # what CI runs — must exit 0
```

## The workflow

1. **Branch off `main`.** Direct pushes to `main` are blocked — every change
   goes through a PR. Name branches `type/short-description`, e.g.
   `fix/blockmesh-boundary-parse` or `feat/gmsh-2d-extrude`.
2. **Commit with [Conventional Commits](https://www.conventionalcommits.org/):**
   `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`, `ci:`. Add `!`
   (e.g. `refactor!:`) to flag a breaking change.
3. **Open a PR into `main`.** Fill in the template and link the issue it closes.
   CI (the unit suite on Python 3.10–3.12) must be green before merge.
4. **Keep the update contract.** `git pull` and `docker pull` must never touch a
   user's `runs/`, prompts, meshes, or local settings. If your change risks
   that, call it out in the PR.

## Scope: what belongs where

- **`src/` (the MCP server)** — mechanical only: file I/O, execution, retrieval,
  parsing. **No LLM SDKs, no API keys.**
- **`agents/` (skills + subagents)** — the reasoning: how to plan a case,
  diagnose errors, and drive the tools. This is where the "smarts" go.

When in doubt, open an issue first and we'll figure out the right home together.
