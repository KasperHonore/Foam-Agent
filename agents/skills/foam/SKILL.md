---
name: foam
description: Run a complete OpenFOAM CFD simulation (Foundation OpenFOAM v10) from a natural language prompt. You do the CFD reasoning; the Foam-Agent MCP tools are your hands (tutorial retrieval, file I/O, execution). No API key needed by the server.
---

# Foam-Agent: Agent-Driven CFD Simulation

You are the CFD engineer. The `foamagent` MCP server gives you mechanical tools:
tutorial retrieval (RAG over Foundation OpenFOAM v10 tutorials), case file I/O,
simulation execution, error extraction, Python script execution (PyVista, GMSH)
and SLURM job management. All planning, file writing and debugging reasoning is
yours.

## Input

The user provides a simulation requirement: `$ARGUMENTS`

If empty, ask the user to describe their CFD simulation (e.g. "Simulate
lid-driven cavity flow at Re=1000").

## Hard rules

- **Foundation OpenFOAM v10 conventions only** (openfoam.org). See
  [references/openfoam-v10-conventions.md](references/openfoam-v10-conventions.md).
  If the user runs ESI OpenFOAM (openfoam.com, v2312/v2406/...), generate for
  v10 first and call `translate_case_to_esi` at the end (best-effort).
- **Create and edit ALL case files with the `write_case_file` MCP tool**, never
  your local file tools — the server may run in a container whose filesystem is
  where the simulation executes. Same for reading: use `read_case_file`.
- Never modify parameters the user explicitly specified (Reynolds number,
  velocities, geometry, solver choice...). Fix errors by other means.
- Show the user your plan and get confirmation before generating files. When
  running headless/autonomously (no user to ask), proceed and include the
  plan in your final report instead.
- Report progress after each phase.

## Workflow

### 0. Preflight

If the foamagent tools don't respond (connection refused, tools missing),
run the **foam-setup** skill first — it diagnoses Docker/image/container
issues and brings the server up. Also note: the FIRST retrieval call after a
server start downloads the embedding model (~1.2 GB) and can take minutes —
that is not a hang.

### 1. Plan the case

1. Call `get_case_stats` to see the valid `case_domain`, `case_category` and
   `case_solver` values.
2. From the user requirement, decide: a short `case_name` (snake_case), and a
   domain, category and solver **chosen from those lists**.
3. Call `find_similar_case` with these values. Inspect `selected_case` and its
   `dir_structure`: judge yourself how well it matches (same solver? same
   physics?) and decide which parts of the reference to trust.
4. Decompose into a file list (the case's `dir_structure`): every file you will
   generate, as `folder/file` pairs (`system/controlDict`, `0/U`, ...). Rules:
   - Generate ALL files the solver needs (check the similar case's structure
     for completeness).
   - No gmsh files (`.geo`, `.msh`) in the file list.
   - Include `system/blockMeshDict` (or snappyHexMesh files) only when the user
     did not request a gmsh-generated or custom uploaded mesh.
5. Present the plan (case name, solver, file list, mesh strategy) and confirm.

### 2. Set up the mesh (only if custom/GMSH mesh)

- **User-provided `.msh` file**: write it into the case with `write_case_file`,
  ensure a minimal `system/controlDict` exists, run
  `run_openfoam_command(case_dir, "gmshToFoam <file>.msh")`, then
  `read_mesh_boundaries` and verify patch names/types match the boundary
  conditions you plan to write. For 2D cases set frontAndBack-type patches to
  `empty` by editing `constant/polyMesh/boundary`.
- **GMSH mesh from geometry description**: delegate to the **foam-mesher**
  subagent. If your harness has no subagents, read
  [references/subagents/foam-mesher.md](references/subagents/foam-mesher.md)
  and follow it inline.
- **blockMesh**: nothing to do here — `blockMeshDict` is generated in step 3
  and blockMesh runs inside Allrun.

### 3. Generate the case files

Call `resolve_case_dir(case_name)` to get the case directory, then generate
each file with `write_case_file`, **in dependency order**: `system/` first,
then `constant/`, then `0/`. Follow
[references/file-generation.md](references/file-generation.md) strictly — it
contains the consistency rules (cross-file coherence, dimensions, fvSolution
`*Final` entries, controlDict constraints). For free-surface/VOF cases
(interFoam family) additionally follow
[references/multiphase-vof.md](references/multiphase-vof.md) — the file
inventory and numerics differ substantially from single-phase.

Use the `tutorial_reference` from `find_similar_case` as your template where it
matches; if it is a weak match, rely on your own OpenFOAM knowledge and use
`search_tutorials(index="openfoam_tutorials_details", ...)` to find better
references for specific files.

### 4. Write the Allrun script

Follow [references/allrun-guide.md](references/allrun-guide.md). Use
`search_tutorials(index="openfoam_command_help", query="<command>")` if you are
unsure about a utility's usage. Write it with
`write_case_file(case_dir, "Allrun", content, executable=true)`.

### 5. Run and debug

1. `run_case(case_dir)` — executes Allrun and returns extracted errors.
   WARNING: every `run_case` call first deletes old logs and time-step
   folders — reruns always start from scratch; there is no warm restart.
2. `status: success` means the commands exited cleanly — it does NOT mean
   the physics is right. Always read the solver log tail
   (`read_case_file(case_dir, "log.<solver>")`) after a "successful" run:
   - STEADY solvers: confirm convergence (`SIMPLE solution converged`) or
     acceptably low final residuals. Plateaued residuals = not converged:
     see the error playbook.
   - TRANSIENT solvers: confirm the final `Time =` reached `endTime`, time
     directories were written (`list_case_files`), continuity errors stayed
     small, and — for VOF — phase fraction is conserved and alpha stays
     bounded (see references/multiphase-vof.md).
3. On failure, enter the fix loop: delegate to the **foam-debugger** subagent
   (no subagents? read
   [references/subagents/foam-debugger.md](references/subagents/foam-debugger.md)
   and [references/error-playbook.md](references/error-playbook.md), then do
   it inline). Iterate (diagnose → rewrite files → `run_case`) up to 25 times;
   keep a short history of attempts and try a *different* approach when an
   error repeats.
4. Report success/failure honestly, with the failing log excerpts if any.

### 6. HPC execution (only if the user asks for cluster/SLURM)

Follow [references/hpc-slurm.md](references/hpc-slurm.md): write a SLURM script
tailored to the user's cluster, submit with `submit_slurm_job`, poll with
`slurm_job_status`.

### 7. Visualization (only if the user asks)

Delegate to the **foam-visualizer** subagent (no subagents? follow
[references/subagents/foam-visualizer.md](references/subagents/foam-visualizer.md)
inline): `ensure_foam_file`,
then `run_python_script` with a PyVista script that loads the `.foam` file,
colors by the requested field with the `coolwarm` colormap, and saves a PNG
(pass `expected_output` so success is verified). Fix and retry on errors.

### 8. ESI translation (only if the user's OpenFOAM is the ESI fork)

Call `translate_case_to_esi(case_dir)` after generation, before `run_case`.
Warn the user this is best-effort and the run/fix loop is validated on
Foundation v10.
