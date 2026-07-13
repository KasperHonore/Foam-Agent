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
  where the simulation executes. Same for reading: use `read_case_file`. (One
  sanctioned exception: copying a user's input STL into the runs tree so the
  server can see it at all — step 2's STL bullet.)
- Never modify parameters the user explicitly specified (Reynolds number,
  velocities, geometry, solver choice...). Fix errors by other means.
- Show the user your plan and get confirmation before generating files. When
  running headless/autonomously (no user to ask), proceed and include the
  plan in your final report instead.
- Report progress after each phase.

## Context loading — what to read when

Load references on demand, not up front — only what the current situation
calls for. This table is the index; the step-local pointers in the workflow
below repeat the load-bearing ones where extra judgment guidance applies.

| Situation | Read |
|---|---|
| Writing any case file | [references/file-generation.md](references/file-generation.md) + [references/openfoam-v10-conventions.md](references/openfoam-v10-conventions.md) |
| Free-surface / VOF case (interFoam family) | [references/multiphase-vof.md](references/multiphase-vof.md) |
| Writing the Allrun script | [references/allrun-guide.md](references/allrun-guide.md) |
| Judging run outcomes (after `parse_solver_log`) | [references/convergence.md](references/convergence.md) |
| Judging mesh quality (after `assess_mesh`) | [references/mesh-quality.md](references/mesh-quality.md) |
| CAD geometry provided as an STL file (snappyHexMesh pathway) | [references/snappyhexmesh.md](references/snappyhexmesh.md) |
| Turbulent case — model choice, wall treatment / y+ target, layer sizing, inlet k/epsilon/omega | [references/turbulence.md](references/turbulence.md) |
| Force/coefficient questions (drag, lift, Cd/Cl/Cm) | [references/forces.md](references/forces.md) |
| A run failed — diagnosing errors | [references/error-playbook.md](references/error-playbook.md) |
| Cluster / SLURM execution | [references/hpc-slurm.md](references/hpc-slurm.md) |
| Harness without native subagents | [references/subagents/](references/subagents/) — inline role fallbacks (foam-mesher, foam-debugger, foam-visualizer) |

## Workflow

### 0. Preflight

If the foamagent tools don't respond (connection refused, tools missing),
run the **foam-setup** skill first — it diagnoses Docker/image/container
issues and brings the server up. Also note: the FIRST retrieval call after a
server start downloads the embedding model (~1.2 GB) and can take minutes —
that is not a hang.

### 1. Plan the case

Standing defaults first: if `config/user.yml` exists at the repo root (read
it with your local file tools — repo config, not a case file), treat its
values (units, mesh fineness bias, visualization, local-vs-HPC) as the
user's defaults for everything they don't specify in this conversation —
what they say always wins. Absent? Proceed with built-in defaults and
suggest the **foam-onboard** skill once — it seeds the file so defaults
never need restating.

1. Call `get_case_stats` to see the valid `case_domain`, `case_category` and
   `case_solver` values.
2. From the user requirement, decide: a short `case_name` (snake_case), and a
   domain, category and solver **chosen from those lists**.
3. Call `find_similar_case` with these values. It can take tens of seconds
   even when the embedding model is warm (two CPU embedding passes per call) —
   that is normal, not a hang. Inspect `selected_case` and the returned
   `dir_structure` (a top-level field, also mirrored into
   `selected_case.dir_structure`): judge yourself how well it matches (same
   solver? same physics?) and decide which parts of the reference to trust.
4. Decompose into a file list (the case's `dir_structure`): every file you will
   generate, as `folder/file` pairs (`system/controlDict`, `0/U`, ...). Rules:
   - Generate ALL files the solver needs (check the similar case's structure
     for completeness).
   - No gmsh files (`.geo`, `.msh`) and no STL surfaces in the file list —
     the foam-mesher subagent owns those.
   - Include `system/blockMeshDict` (or snappyHexMesh files) only when the user
     did not request a gmsh-generated, custom uploaded, or STL-based mesh
     (for an STL case, foam-mesher generates the whole mesh dict set).
   - Forces or force coefficients in the requirement (drag, lift, Cd/Cl/Cm)?
     Plan a `forceCoeffs` function object into `system/controlDict` now — on
     Foundation v10 the reliable path is the object active during the solver
     run, and this is the sanctioned exception to the function-object ban.
     Recipe and worked block: [references/forces.md](references/forces.md).
   - Turbulent flow expected? Decide the model and the wall-treatment
     regime (the target y+) NOW with
     [references/turbulence.md](references/turbulence.md) — the selection
     ladder and the y+ band table. Unsure it is even turbulent?
     `estimate_wall_spacing` reports the Reynolds number with a named
     regime verdict — trust it over recall. The decision adds
     `constant/momentumTransport` and the 0/ turbulence fields
     (k/epsilon/omega/nut per model) to the file list, and fixes the
     first-cell height the mesh step must deliver.
5. Present the plan (case name, solver, file list, mesh strategy) and confirm.

### 2. Set up the mesh (only if custom/GMSH/STL mesh)

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
- **User-provided STL (CAD geometry)**: first call `resolve_case_dir(case_name)`
  to fix the case directory, then get the file where the server can see it —
  copy it with your LOCAL file tools into that case directory under the runs
  tree (clone install: `<repo>/runs/<case>/`; global install: the central
  `~/foamagent/runs/<case>/` mount), creating the directory if needed. The
  runs directory is a bind mount into the server's container; there is
  deliberately NO server-side upload tool. Then delegate to the
  **foam-mesher** subagent with the case_dir and the STL's path — it runs
  the snappy pathway: `inspect_stl` (typed watertightness/units verdict) →
  `import_geometry` (into `constant/triSurface/`, deterministic mm→m scaling
  when needed) → mesh dicts per
  [references/snappyhexmesh.md](references/snappyhexmesh.md) → the v10
  meshing sequence → `read_mesh_boundaries` → `assess_mesh`. Dicts come out
  in Foundation v10 vocabulary (`surfaceFeatures`, not
  `surfaceFeatureExtract`); for ESI users nothing changes — detect, name it,
  and offer `translate_case_to_esi` at the end as usual (best-effort).
- **Turbulent case, any mesh branch**: give foam-mesher the flow conditions
  (velocity, characteristic length, kinematic viscosity) and the target y+
  from step 1 — its wall/layer sizing consumes `estimate_wall_spacing` on
  both the GMSH and snappy branches, bridged per
  [references/turbulence.md](references/turbulence.md); boundary-layer
  knobs are computed, not eyeballed.
- **blockMesh**: nothing to do here — `blockMeshDict` is generated in step 3
  and blockMesh runs inside Allrun.
- **Validate the mesh** (any source): `assess_mesh(case_dir)` — typed census,
  per-metric pass/warn/fail and a verdict with evidence — and judge marginal
  values with [references/mesh-quality.md](references/mesh-quality.md): the
  warn band is a conservative mechanical default; per-application judgement
  and the which-knob-to-turn table live there. Raw `checkMesh` via
  `run_openfoam_command` is the fallback when `assess_mesh` itself errors.
  (blockMesh cases have no mesh until Allrun runs blockMesh in step 5 —
  assess then if quality is in doubt.)

### 3. Generate the case files

Call `resolve_case_dir(case_name)` to get the case directory, then generate
each file with `write_case_file`, **in dependency order**: `system/` first,
then `constant/`, then `0/`. Follow
[references/file-generation.md](references/file-generation.md) strictly — it
contains the consistency rules (cross-file coherence, dimensions, fvSolution
`*Final` entries, controlDict constraints). For free-surface/VOF cases
(interFoam family) additionally follow
[references/multiphase-vof.md](references/multiphase-vof.md) — the file
inventory and numerics differ substantially from single-phase. For
turbulent cases, `constant/momentumTransport` and the wall BC sets on the
0/ turbulence fields follow
[references/turbulence.md](references/turbulence.md), and the inlet
k/epsilon/omega values come from `estimate_turbulence_inlet` — computed,
never recalled.

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
   the physics is right. After every "successful" run call
   `parse_solver_log(case_dir)` — typed residuals, Courant/continuity facts
   and a verdict with evidence, computed instead of read from raw text — and
   judge the numbers with
   [references/convergence.md](references/convergence.md): the verdict's
   thresholds are conservative mechanical defaults; the judgement (steady
   initial-residual trends, per-solver targets, when `converged` still
   deserves suspicion, rerun-longer calls) is yours. Also confirm time
   directories were written (`list_case_files`), and for VOF cases do the
   raw-log checks in
   [references/multiphase-vof.md](references/multiphase-vof.md) — phase
   conservation and alpha bounds are invisible to the parser. Turbulent
   case? Verify the achieved y+ with the harvested recipe in
   [references/turbulence.md](references/turbulence.md) (the solver-wrapped
   `-postProcess` form; judge the per-patch average against the target
   band) — a miss is one `estimate_wall_spacing` rescale, re-bridge and
   remesh away, per the loop in that reference.
3. Force/coefficient answers come from `parse_force_coefficients(case_dir)`
   — typed Cd/Cl/Cm with tail-window statistics and the reference values,
   and the run's ledger Key result cell filled as a side effect — never
   from reading dat files and averaging by eye. Judge the window and the
   normalization with [references/forces.md](references/forces.md). The
   tool reports "no forceCoeffs output"? The same reference carries the
   working v10 recipe (function object during the run; the post-hoc
   reality per solver family).
4. On failure, enter the fix loop: delegate to the **foam-debugger** subagent
   (no subagents? read
   [references/subagents/foam-debugger.md](references/subagents/foam-debugger.md)
   and [references/error-playbook.md](references/error-playbook.md), then do
   it inline). Iterate (diagnose → rewrite files → `run_case`) up to 25 times;
   keep a short history of attempts and try a *different* approach when an
   error repeats.
5. Report success/failure honestly, with the failing log excerpts if any.

Every run is recorded automatically in `runs/ledger.md` (the run ledger) —
no bookkeeping on your side. For run-history questions ("list my runs",
"what happened to X?") follow the **foam-runs** skill instead of re-reading
logs.

### 6. HPC execution (if the user asks for cluster/SLURM, or `execution: hpc` is their standing default)

Follow [references/hpc-slurm.md](references/hpc-slurm.md): write a SLURM script
tailored to the user's cluster, submit with `submit_slurm_job`, poll with
`slurm_job_status`.

### 7. Visualization (if the user asks — or their standing defaults say so)

The `visualization` preference governs this step after a successful run:
`auto` renders the default plot unasked, `offer` means ask first (also the
built-in behavior without a preferences file), `off` means only on explicit
request. Delegate to the **foam-visualizer** subagent (no subagents? follow
[references/subagents/foam-visualizer.md](references/subagents/foam-visualizer.md)
inline): `ensure_foam_file`,
then `run_python_script` with a PyVista script that loads the `.foam` file,
colors by the requested field (default: their `visualization_field`) with
the `coolwarm` colormap, and saves a PNG (pass `expected_output` so success
is verified). Fix and retry on errors.

### 8. ESI translation (only if the user's OpenFOAM is the ESI fork)

Call `translate_case_to_esi(case_dir)` after generation, before `run_case`.
Warn the user this is best-effort and the run/fix loop is validated on
Foundation v10.
