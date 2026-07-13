# Generating OpenFOAM case files

Generate files **in dependency order** — `system/` first, then `constant/`,
then `0/` — so each file can be checked against the ones already written.
Write every file with the `write_case_file` MCP tool. File content must be
pure OpenFOAM dictionary format: no markdown fences, no explanations, no
placeholder text.

## Cross-file consistency (check before writing each file)

- Every property defined in one file and used in another must match: if `nu`
  is defined in `constant/physicalProperties`, the fields in `0/` must be
  consistent with it (e.g. velocity chosen to hit the requested Reynolds
  number).
- Field names must match across files: every field solved by the solver needs
  a `0/<field>` file, a solver entry in `system/fvSolution`, and divergence /
  gradient schemes in `system/fvSchemes` for every term the solver uses.
- Boundary patch names in every `0/<field>` file must exactly match the
  patches defined by the mesh (`blockMeshDict` patches, or
  `constant/polyMesh/boundary` after mesh conversion — check with
  `read_mesh_boundaries`).
- Patch *types* must be compatible: a patch declared `empty` in the mesh must
  be `empty` in every field file; `wall` patches take wall conditions, etc.
- Units and dimensions must be correct on every field (`dimensions [...]`).
- Solver settings must be consistent with the user requirement (turbulence
  model, transient vs steady, compressible vs incompressible). For a
  turbulent case the model choice and the wall BC sets on the 0/
  turbulence fields come from [turbulence.md](turbulence.md) (the
  selection ladder; the per-model v10 wall-function tables), and the inlet
  k/epsilon/omega values from `estimate_turbulence_inlet` — computed,
  never recalled.

## fvSolution: `*Final` entries (transient PISO/PIMPLE solvers)

For transient pressure-velocity coupling solvers (PISO/PIMPLE), the `solvers`
dictionary MUST include matching `Final` entries for every field the solver
actually solves on the final corrector:

```
p     { solver PCG; preconditioner DIC; tolerance 1e-06; relTol 0.05; }
pFinal { $p; relTol 0; }
U     { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0; }
UFinal { $U; relTol 0; }
```

- Match the ACTUAL pressure field name: gravity/buoyancy/multiphase solvers
  (interFoam, buoyantFoam, ...) solve `p_rgh`, not `p` — they need
  `p_rgh` + `p_rghFinal` entries (and typically a `pcorr` entry for the flux
  correction in moving/initialized-field cases).
- A `UFinal` entry is only needed if the momentum equation is solved; with
  `momentumPredictor no;` (common in interFoam) there is no U solve, though a
  redundant `UFinal` is harmless.
- For grouped regex entries use a grouped Final entry:
  `"(U|k|epsilon)Final" { $U; relTol 0; }`.
- Never emit placeholder text such as `$<field>;` — `$p;` and `$U;` are real
  OpenFOAM macro references to the entries above.
- The `PIMPLE`/`PISO` sub-dictionary must match the selected solver
  (`nCorrectors`, `nNonOrthogonalCorrectors`, `nOuterCorrectors`,
  `momentumPredictor`; `pRefCell 0; pRefValue 0;` for incompressible
  closed-domain cases).

## fvSolution/fvSchemes: steady solvers (SIMPLE)

For steady solvers (`simpleFoam`, ...):

- `relaxationFactors` are required. Typical start: fields `p 0.3;`, equations
  `U 0.7;`. If residuals plateau in a limit cycle, lower them (p 0.2, U 0.5)
  and raise `endTime` (the iteration budget).
- Add `residualControl` to the `SIMPLE` sub-dictionary (e.g. `p 1e-5; U 1e-5;`)
  so the run stops at convergence and convergence is detectable in the log
  (`SIMPLE solution converged`).
- Use bounded divergence schemes: `div(phi,U) bounded Gauss linearUpwind
  grad(U);` (or `bounded Gauss upwind` for robustness).
- Set `ddtSchemes { default steadyState; }`.
- Solvers using `constant/momentumTransport` (v10) also need
  `div((nuEff*dev2(T(grad(U)))))` in `divSchemes` — **even for laminar
  cases**; omitting it gives an undefined-keyword error at startup.

## controlDict

- Include ONLY what is needed to run: `application`, time controls
  (`startTime`, `endTime`, `deltaT`), write controls.
- Transient cases (and any case meant for a background `start_case` run):
  set `runTimeModifiable true;` explicitly. v10's COMPILED default is
  false — many tutorials ship `true`, masking it — and without the entry a
  running solver never re-reads controlDict, so `stop_case`'s graceful
  stop (finish the step, write the current fields, exit cleanly) is
  impossible and a stop becomes a hard kill that writes nothing. Harmless
  on steady cases: the entry only enables mid-run dict re-reads. (Even
  with it true, a running solver only notices a controlDict edit once the
  file's mtime clears the 10 s `fileModificationSkew` gate past its last
  read — `stop_case` handles that itself; a hand edit mid-run may look
  ignored.)
- Do NOT include post-processing function objects during initial case
  generation — with ONE sanctioned exception: when the user's question is
  forces or force coefficients (drag, lift, Cd/Cl/Cm), add a `forceCoeffs`
  function object to the `functions` block, because on Foundation v10 the
  reliable path is the object being active during the solver run. The
  worked block and the recipe live in [forces.md](forces.md).
- `application` must be the chosen solver binary name.

## Using the similar-case reference

`find_similar_case` returns the closest v10 tutorial. Judge the match level
yourself:

- **Strong match** (same solver + same physics): follow its structure and
  numerical settings closely; adapt geometry/BCs/values to the requirement.
- **Weak match** (same domain only): use it for file inventory and dictionary
  skeletons, not for physics settings. Do not copy blindly.
- **No match** (`found: false`): rely on your own OpenFOAM knowledge; query
  `search_tutorials(index="openfoam_tutorials_details", query="<solver> <keywords>")`
  for specific file examples.

Whatever the source, apply your domain expertise: verify all numerical values
are consistent with the user requirement — never contradict values the user
specified.
