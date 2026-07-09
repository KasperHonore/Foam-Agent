## What & why

<!-- One or two sentences: what does this change, and why? -->

Closes #

## Type of change

- [ ] 🐞 Bug fix
- [ ] ✨ Feature (new tool / skill / capability)
- [ ] 📚 Docs
- [ ] 🧹 Refactor / chore (no behavior change)
- [ ] 🏗️ Infra (Docker / CI / packaging)
- [ ] ⚠️ Breaking change

## How was it tested?

<!-- Commands you ran and what you saw. e.g. the unit suite, or an end-to-end run
     against a live server with a specific prompt/solver. -->

## Checklist

- [ ] Branched off `main`; this PR targets `main`.
- [ ] `pytest -m "not integration" --ignore=tests/test_lid_driven_cavity_mcp.py` passes locally.
- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/) (`fix:`, `feat:`, `docs:`, `refactor:`, …; `!` for breaking).
- [ ] Docs updated if behavior, flags, or tools changed (README / FAQ / skill files).
- [ ] If I edited assets under `agents/`, I ran `python scripts/sync_agent_assets.py` (the CI sync check stays green).
- [ ] The **update contract** still holds: `git pull` / `docker pull` won't touch a user's `runs/`, prompts, meshes, or local settings.
