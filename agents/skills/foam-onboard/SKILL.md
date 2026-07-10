---
name: foam-onboard
description: Guided first-run onboarding for Foam-Agent. Use on a fresh clone or new machine when the user says "get me set up", "onboard me", "first time here", "getting started", or asks what this repo is / how to use it. Walks through health check, embedding warm-up, an optional demo simulation, and a tour of the workflow — conversationally, no manual file editing.
---

# Foam-Agent onboarding

Walk a new user from fresh clone to first successful simulation in one
conversation. Be brief and conversational — announce each stage in a sentence,
do the work, report the result. Never ask the user to edit files by hand.

## Stage 0 — Welcome (one short paragraph)

Tell the user what is about to happen: a health check of the Dockerized MCP
server, a one-time warm-up, an optional demo simulation, and a quick tour.
Total hands-on time is a few minutes; the warm-up download can add ~5–10
minutes of unattended waiting on first run.

## Stage 1 — Health check

Run `python scripts/doctor.py` (read-only, prints OK/FAIL per check), or if
the `foamagent` MCP tools are wired in, just call `get_case_stats`.

- **Healthy** → tell the user, move on.
- **Unhealthy** → switch to the `foam-setup` skill
  (`agents/skills/foam-setup/SKILL.md`) and work its steps until healthy,
  narrating what you fix. Come back here afterwards.

## Stage 2 — Warm-up (first run only)

The first retrieval call downloads a ~1.2 GB embedding model inside the
container and loads 4 FAISS indices. Warn the user this is a one-time wait,
then trigger it with a throwaway call:

- `search_tutorials(index="openfoam_tutorials_details", query="lid driven cavity")`
  (the `index` argument defaults to `openfoam_tutorials_details` if omitted).

If it returns results in seconds, the model was already warm — say so.

## Stage 3 — Demo simulation (offer, don't impose)

Ask whether the user wants (a) the classic demo, (b) to jump straight to
their own problem, or (c) to skip simulating for now.

- **(a)** Follow the `foam` skill with the requirement
  *"Simulate lid-driven cavity flow at Re=1000"* — a small case that runs in
  well under a minute and validates the whole chain (retrieval → file
  generation → execution).
- **(b)** Ask for their problem in plain language and follow the `foam` skill
  with it.
- **(c)** Fine — go to Stage 4.

When a run succeeds, point at the concrete artifacts: the case directory
under `runs/`, its `log.*` files, and offer a PyVista plot (foam-visualizer).

## Stage 4 — Tour (keep it to ~10 lines)

Close by telling the user, in your own words:

- **How to ask for work**: describe any CFD problem in plain language, or use
  `/foam <requirement>` where slash commands exist. Meshes: gmsh geometry
  descriptions work (foam-mesher), or drop a `.msh` file at the repo root.
- **Where things live**: simulations under `runs/`; sample prompts in
  `examples/` (copy to the repo root and edit — root copies are gitignored).
- **How updates work**: `git pull` + `docker pull ghcr.io/kasperhonore/foamagent:latest`
  (then recreate the container). Their runs, prompts, meshes and local agent
  settings are gitignored and never touched.
- **When things break**: `python scripts/doctor.py` or the `foam-setup`
  skill; failed runs are auto-debugged by the foam-debugger loop.
- **What it targets**: Foundation OpenFOAM v10 conventions (ESI only via
  best-effort translation).
