# Turbulence and wall treatment (on top of estimate_wall_spacing and estimate_turbulence_inlet)

The judgement layer for turbulent cases: which model to select, which
wall-treatment regime the target y+ implies, what the mesh must deliver
for it, which boundary conditions carry it, and how to verify what the
run actually achieved. The numbers come from two pure calculators —
`estimate_wall_spacing` (first-cell sizing from flow conditions and a
target y+) and `estimate_turbulence_inlet` (inlet k/epsilon/omega from
velocity, intensity and a length scale). Both name every correlation,
formula and constant they use (Cmu and friends are pinned server-side and
echoed where they enter a formula) — cite their output, never redo their
arithmetic inline: recalled formulas with silently wrong constants are
exactly the failure class the tools exist to kill. Every BC name and
every verification command below was verified live on Foundation v10
(the turbulence harvest) — follow them, not folklore.

Vocabulary pin: model selection lives in `constant/momentumTransport`
with `simulationType RAS;` and a `model` keyword; viscosity lives in
`constant/physicalProperties`. An installation expecting
`constant/turbulenceProperties` and `RASModel` is ESI — see
[openfoam-v10-conventions.md](openfoam-v10-conventions.md).

## The model-selection ladder

Climb no higher than the physics demands — each rung adds fields, mesh
requirements and divergence surface. Rung 0 is decided by evidence:
`estimate_wall_spacing` reports the Reynolds number with a named regime
verdict (laminar / transitional / turbulent) — trust that verdict over
recall.

- **Rung 0 — laminar.** The regime verdict says laminar — the Reynolds
  thresholds behind the verdict are pinned and documented in the tool
  itself; trust it over recalled transition numbers. No turbulence
  model, no wall functions, no y+ target — and the achieved-y+ verify
  step is structurally inapplicable (see the recipe section: laminar
  solvers reject it three ways). The icoFoam family is laminar by construction and reads no
  turbulence dict; simpleFoam/pisoFoam/pimpleFoam take
  `simulationType laminar;` in `constant/momentumTransport`.
- **Rung 1 — kOmegaSST.** The default turbulent rung for wall-bounded
  flow: external aero, adverse pressure gradients, separation and
  reattachment, forces on bodies. Valid both wall-resolved and with wall
  functions — v10's `omegaWallFunction` is dual-regime (its live src
  header: a wall constraint "for both low and high Reynolds number
  turbulence models").
- **Rung 2 — kEpsilon.** Free-shear and internal flows away from strong
  adverse gradients: fully developed duct/pipe flow, jets, mixing
  layers. Robust and cheap; its native habitat is wall functions in the
  log-law band. It over-diffuses in adverse pressure gradients and
  under-predicts separation — prefer kOmegaSST there. Standard kEpsilon
  assumes wall functions: a wall-resolved k-epsilon-family case needs a
  low-Re variant (e.g. LaunderSharmaKE) with the wall-resolved BC set
  below.
- **A transitional verdict has no rung** — transition modelling is out
  of scope. Bracket instead (run laminar and fully turbulent as bounds
  and report both) or state which single assumption you made.

The v10 dict, harvested shape — `incompressible/simpleFoam/pitzDaily`
ships exactly this; motorBike carries `model kOmegaSST;`:

```
simulationType RAS;

RAS
{
    model           kEpsilon;      // or kOmegaSST
    turbulence      on;
    printCoeffs     on;
    viscosityModel  Newtonian;
}
```

Solvers driven by `constant/momentumTransport` also need the
`div((nuEff*dev2(T(grad(U)))))` scheme entry — even laminar ones — and
every solved turbulence field needs its fvSolution and fvSchemes
entries: [file-generation.md](file-generation.md).

## Wall treatment: pick the y+ target before the mesh

y+ is the wall-normal distance of the FIRST CELL CENTRE in wall units.
Choosing the target is a meshing decision made before any mesh exists —
it fixes the first-cell height everything downstream consumes.

| Regime | Target y+ | Valid band | What it needs |
|---|---|---|---|
| Wall-resolved (low-Re) | ~ 1 | first centre at y+ ~ 1 | many thin layers covering the boundary layer; the wall-resolved BC sets below |
| Wall functions (high-Re) | 30–50 is a safe aim | 30 < y+ < 300 | first centre in the log layer; the wall-function BC sets below |
| Buffer layer | never | 5 < y+ < 30 | nothing — no-man's-land |

**The buffer layer (5 < y+ < 30) is no-man's-land**: too far out for the
viscous-sublayer assumption behind wall-resolved treatment, too far in
for the log-law assumption behind wall functions. Never target it, and
treat an achieved average landing in it as a miss to correct, not a
compromise to accept.

Choosing between the regimes: wall functions when the deliverable is a
global quantity (drag on a body, pressure drop through a duct) at high
Re — the mesh is dramatically cheaper. Wall-resolved when near-wall
physics IS the answer (smooth-body separation location, skin-friction
accuracy) — or when the tool's own numbers say the log layer barely
exists: if the first-cell height for y+ = 30 is not small next to the
boundary-layer thickness estimate, there is no log layer to put a wall
function in; resolve the wall instead.

## Per-model v10 wall BC sets (frozen from the live harvest)

Exact v10 names, frozen from live tutorials and src — never from memory.
Every wall-function entry carries a `value` line (`value $internalField;`
or `value uniform 0;`) — an initial/placeholder value, but it must be
present.

High-Re (wall functions, 30 < y+ < 300):

| Model | k | epsilon | omega | nut | Citing tutorial (under `$FOAM_TUTORIALS`) |
|---|---|---|---|---|---|
| kEpsilon | `kqRWallFunction` | `epsilonWallFunction` | — | `nutkWallFunction` | `incompressible/simpleFoam/pitzDaily` |
| kOmegaSST | `kqRWallFunction` | — | `omegaWallFunction` | `nutkWallFunction` | `incompressible/simpleFoam/motorBike`; identical set in `incompressible/pimpleFoam/RAS/wingMotion/wingMotion2D_simpleFoam` |

Wall-resolved (y+ ~ 1) — **there is NO matched low-Re wall-function
triplet on v10**; real v10 practice is fixedValue turbulence fields plus
a dedicated nut BC:

| Model family | k | epsilon / omega | nut |
|---|---|---|---|
| k-omega family (kOmegaSST at y+ ~ 1) | `fixedValue; value uniform 0;` | omega: `omegaWallFunction` (dual-regime in v10) | `nutLowReWallFunction` |
| Low-Re k-epsilon (e.g. LaunderSharmaKE) | `fixedValue; value uniform 1e-10;` | epsilon: `fixedValue; value uniform 1e-08;` | `nutLowReWallFunction` |

Citations from the harvest: `incompressible/boundaryFoam/
boundaryLaunderSharma` (the fixedValue k/epsilon pair with
`nutLowReWallFunction` carrying explicit `Cmu 0.09; kappa 0.41; E 9.8;`),
`incompressible/simpleFoam/T3A` (wall-resolved k-omega-family plate:
k `fixedValue; value uniform 0;` + `omegaWallFunction`), and the live
src headers. `nutLowReWallFunction` "sets nut to zero, and provides an
access function to calculate y+" — it is both the correct nut BC for a
wall-resolved mesh AND what makes the post-run y+ measurement work
there.

Anti-folklore, verified against v10 src:

- `epsilonLowReWallFunction` and `omegaLowReWallFunction` DO NOT EXIST
  in v10 — asking for them is ESI/folklore vocabulary and generating one
  is an `Unknown ... type` startup error.
- `kLowReWallFunction` exists in src but is used by ZERO v10 tutorials —
  do not generate it.
- No special low-Re epsilon/omega BC is needed because v10's
  `epsilonWallFunction` and `omegaWallFunction` are dual-regime (valid
  "for low- and high-Reynolds number turbulence models" per their src
  headers) — the wall-resolved sets above rely on that.
- Model validity is separate from BC availability: standard kEpsilon at
  y+ ~ 1 is still wrong even with legal BCs — switch to a low-Re variant
  when the k-epsilon family must be wall-resolved.

Sidenotes from the src inventory: rough walls have
`nutkRoughWallFunction` / `nutURoughWallFunction`;
`nutUSpaldingWallFunction` is the continuous all-y+ alternative (9
tutorial usages against `nutkWallFunction`'s 173). Spalart-Allmaras
`nuTilda` takes plain `zeroGradient` on walls
(`incompressible/simpleFoam/airFoil2D` convention).

Wall functions act on `type wall` patches, and the y+ report covers wall
patches only — a solid surface left as `type patch` (the gmshToFoam
default) silently gets neither. Check `read_mesh_boundaries` before
writing the 0/ fields; patch names and types must match per
[file-generation.md](file-generation.md).

## The loop: estimate → bridge → mesh → run → verify → adjust

1. **Pick the model** (ladder above) and the **wall-treatment regime**,
   which fixes the target y+ (band table above).
2. **Call `estimate_wall_spacing`** with velocity, characteristic
   length, kinematic viscosity, the target y+, the flow type
   (`external` default / `internal` — it switches the skin-friction
   correlation between the flat-plate and pipe families; sizing a duct
   with a flat-plate formula is a named failure mode), and optionally
   the layer expansion ratio (defaults to the conservative 1.2 the
   snappy reference starts with). Back come: Reynolds number with the
   regime verdict, the skin-friction coefficient with its correlation
   named, friction velocity, the **first-cell-centre distance** and the
   **first-cell height** as two separately labelled fields, a
   boundary-layer thickness estimate, and a suggested layer count for
   the expansion ratio. A laminar verdict means stop: drop to rung 0 —
   the numbers still come back, but wall functions and turbulence
   models are inappropriate.
3. **Read the right field.** y+ is defined at the cell CENTRE; mesh
   knobs consume the cell HEIGHT (twice the centre distance for the
   wall cell). The tool labels both precisely so the factor-of-2
   confusion dies in the schema — read the labelled fields, never
   derive one from the other by hand.
4. **Bridge the first-cell height into the mesh recipe** (next
   section), mesh, and judge with `assess_mesh` — layers move
   `max_aspect_ratio` first ([mesh-quality.md](mesh-quality.md),
   [snappyhexmesh.md](snappyhexmesh.md)).
5. **Run the case**, judge convergence as usual
   ([convergence.md](convergence.md)).
6. **Verify the achieved y+** with the harvested recipe (section
   below). The estimate came from a flat-plate or pipe correlation; the
   actual flow — acceleration, separation, recirculation — gets a vote,
   which is why this step exists.
7. **Adjust if outside the band.** Wall spacing is linear in the y+
   target, so the correction is one multiplication: rerun
   `estimate_wall_spacing` with the target scaled by
   target / achieved-average (keeping the arithmetic in the tool),
   re-bridge, remesh, rerun. Never tune the turbulence model to
   compensate for a mesh miss.

## The absolute-to-relative bridge into addLayersControls

snappyHexMesh's layer knobs with `relativeSizes on` (the worked pattern
in [snappyhexmesh.md](snappyhexmesh.md)) are expressed relative to the
LOCAL surface cell size — so the tool's absolute first-cell height needs
that cell size to become a knob value. What the knobs mean and what they
trade lives in the snappy reference's layers block and the
[mesh-quality.md](mesh-quality.md) knob table; this section only
supplies the arithmetic that turns metres into their relative values.

With base cell size B (blockMeshDict), surface refinement level L
(`refinementSurfaces`), first-cell height t1 (the tool's labelled
field), expansion ratio r and layer count N (the tool's suggested
count):

- local surface cell size: `dx = B / 2^L`
- last-layer thickness: `tN = t1 * r^(N-1)`
- **`finalLayerThickness = tN / dx`** — the knob names the LAST layer
- **`minThickness`** (also relative) is the collapse floor — layers
  below it are silently collapsed, not failed
  ([snappyhexmesh.md](snappyhexmesh.md)). Keep it well below the FIRST
  layer's relative thickness `t1 / dx` (a quarter of it is a safe
  start) or coverage vanishes exactly where the y+ target lives.
- **`nSurfaceLayers = N`**, `expansionRatio = r` — echo what the tool
  used.
- total stack thickness: `T = t1 * (r^N - 1) / (r - 1)`. The suggested
  N makes T cover the boundary-layer thickness estimate; if you change
  N or r, re-check coverage against that estimate — a stack stopping
  halfway through the boundary layer dumps the profile's steepest part
  into the first ambient-size cell.

Sanity-check the bridged `finalLayerThickness` against the shipped
default's magnitude (0.5): far below ~0.3 means the surface cell is too
coarse for the stack — raise the surface refinement level (dx halves
per level) rather than accepting a huge jump from the last layer to the
first ambient cell; above ~1 means the level is too fine or the stack
too thick for the cell budget.

Worked numbers — say the tool returned t1 = 1.5e-3 m (first-cell
HEIGHT), r = 1.2, N = 8, boundary-layer estimate ~ 25 mm, on the snappy
reference's 0.5 m background cells:

- `tN = 1.5e-3 * 1.2^7 ≈ 5.4e-3 m`; stack
  `T = 1.5e-3 * (1.2^8 - 1) / 0.2 ≈ 25 mm` — covers the estimate.
- Aim `finalLayerThickness ≈ 0.5` → want `dx ≈ 2 * tN ≈ 1.1e-2 m`.
  Level 5 gives dx = 0.5 / 32 = 1.56e-2 → finalLayerThickness ≈ 0.35;
  level 6 gives dx = 7.8e-3 → ≈ 0.69. Both in band — pick by cell
  budget.
- At level 6: first layer's relative thickness
  `t1 / dx = 1.5e-3 / 7.8e-3 ≈ 0.19` → `minThickness 0.05`.

**GMSH branch** (described geometry, foam-mesher Branch A): the same
absolute number anchors the size field. Set the Threshold field's wall
size so the first cell's wall-normal extent at the obstacle equals the
tool's first-cell height — replacing the ad-hoc D/20-style choice. GMSH
tets are isotropic, so the wall size IS roughly the first-cell height:
a wall-function target is usually affordable; a wall-resolved y+ ~ 1
target prices in enormous cell counts on this branch — when
wall-resolved is non-negotiable, prefer the snappy pathway with layers.
A blockMesh case can hit the same absolute height via `simpleGrading`
toward the wall. Whatever the mesher, verify the achieved y+ the same
way after the run.

## Verify achieved y+: the harvested v10 recipe

After the run finishes, in the case dir (one `run_openfoam_command`
call):

```
<solver> -postProcess -func yPlus -latestTime
```

e.g. `simpleFoam -postProcess -func yPlus -latestTime`. Facts, all
verified live:

- **Never standalone `postProcess -func yPlus`** — on v10 it exits 1
  with `Unable to find turbulence model in the database` and writes
  nothing. The standalone utility cannot construct the
  momentumTransport model — the same limitation the forces recipe
  documents ([forces.md](forces.md)). The solver-wrapped form is the
  recipe.
- Works on any solver with a momentumTransport model: verified on
  simpleFoam (steady RAS) and pisoFoam (transient RAS); pimpleFoam
  exposes the same `-postProcess` hook. `<solver> -help` says whether
  the option exists. `foamRun` is v11+ vocabulary — it does not exist
  on v10.
- **Laminar solvers are structurally inapplicable, three ways**:
  icoFoam has no `-postProcess` at all (rejected at argument parsing:
  `Invalid option: -postProcess`); standalone postProcess fails as
  above; and adding run-time `#includeFunc yPlus` to a laminar case
  ABORTS THE SOLVER itself. potentialFoam has no `-postProcess` either.
  Gate the verify step on a turbulence model being active.
- Without `-latestTime` the replay covers ALL stored time directories
  (one output dir each); with it, the final time only — use it for the
  post-run verify.

Where the output lands (real harvest excerpts, pitzDaily). Stdout:

```
yPlus yPlus write:
    writing object yPlus
    patch upperWall y+ : min = 2.82183, max = 7.24147, average = 6.08666
    patch lowerWall y+ : min = 0.336361, max = 26.5115, average = 16.065
```

and `postProcessing/yPlus/<time>/yPlus.dat` — tab-separated columns
`Time, patch, min, max, average`, one row per WALL patch (non-wall
patches are excluded automatically); with `-latestTime` the file
contains exactly one row per wall patch:

```
# y+ ()
# Time        	patch         	min           	max           	average
287           	upperWall	2.821833e+00	7.241466e+00	6.086655e+00
287           	lowerWall	3.363607e-01	2.651149e+01	1.606502e+01
```

A `yPlus` volScalarField is also written into the time directory
(values only on wall-patch boundaryField) — useful in ParaView, not for
parsing.

Run-time alternative: `#includeFunc yPlus` in controlDict's
`functions{}` (it resolves to the shipped caseDicts entry;
`executeControl writeTime`). Valid for monitoring a long run, but know
the differences: ALL row-sets accumulate in ONE file,
`postProcessing/yPlus/<startTime>/yPlus.dat` (one row-set per write
time); NOTHING is printed to the solver log; cadence follows the case's
`writeControl`. For a one-shot post-run verify prefer the post-hoc form
— it also keeps controlDict inside
[file-generation.md](file-generation.md)'s no-function-objects rule
(forces is the one sanctioned exception).

Judging the numbers:

- Judge the AVERAGE per patch against the target band — on the patches
  that carry the answer (the body, not the tunnel walls).
- Flag min/max excursions: pitzDaily's converged lowerWall spans
  0.34–26.5 with average 16.1 — a recirculating-flow case can straddle
  bands even when the average looks fine. Stagnation, separation and
  reattachment points always drag the local y+ down; small-fraction
  excursions are normal, a broad straddle on the patch that matters
  deserves refinement or a stated caveat in the report.
- Outside the band → the adjust step of the loop: scale the target by
  target / achieved-average, rerun the tool, re-bridge, remesh, rerun.

## Inlet values: estimate_turbulence_inlet → 0/ fields

Call `estimate_turbulence_inlet` with velocity, optionally the
turbulence intensity (omitted → the documented medium default of 0.05
is applied AND echoed in the output — silence produces a stated
assumption, not a hidden one), and exactly ONE of turbulence length
scale or hydraulic diameter (a diameter is converted via the standard
0.07 * D_h rule, applied and named; neither or both is a typed error).
Back come k, epsilon and omega — each carrying the name of the formula
that produced it — plus a turbulent-viscosity-ratio sanity figure; Cmu
is pinned server-side and echoed where it enters a formula. The gist:
k measures the fluctuation energy the intensity implies; epsilon and
omega set the rate at which eddies of the given length scale dissipate
it. The tool output is the authority for the formulas and constants —
cite it, never substitute remembered values.

Read the viscosity ratio (nut/nu) before writing anything: order 10–100
is typical of internal/industrial flows; near or below 1 is an
essentially laminar free stream (clean external aero runs intensity
well below the default — pass it explicitly); thousands and up is
suspicious — usually an overlong length scale or overstated intensity.
Recheck the inputs rather than writing pathological values into 0/.

The model decides which numbers you consume: kEpsilon → k + epsilon;
kOmegaSST → k + omega. The unused third value costs nothing. Field
shapes (walls from the BC tables above — each wall-function entry with
its `value` line; initialize internal fields with the inlet values —
a uniform start at inlet turbulence is the standard robust
initialization):

| Field | dimensions | internalField | inlet | outlet | walls |
|---|---|---|---|---|---|
| `0/k` | `[0 2 -2 0 0 0 0]` | `uniform <k>` | `fixedValue; value uniform <k>;` | `zeroGradient` (or `inletOutlet; inletValue uniform <k>;` where backflow is possible) | per BC tables |
| `0/epsilon` | `[0 2 -3 0 0 0 0]` | `uniform <epsilon>` | `fixedValue; value uniform <epsilon>;` | as k | per BC tables |
| `0/omega` | `[0 0 -1 0 0 0 0]` | `uniform <omega>` | `fixedValue; value uniform <omega>;` | as k | per BC tables |
| `0/nut` | `[0 2 -1 0 0 0 0]` | `uniform 0` | `calculated; value uniform 0;` | `calculated; value uniform 0;` | `nutkWallFunction` / `nutLowReWallFunction` per regime |

Cross-file discipline as always: every solved field needs its
fvSolution entry and div schemes (bounded upwind-family for k/epsilon/
omega on steady runs), and patch names must match the mesh —
[file-generation.md](file-generation.md).

## Common turbulence failure modes (for the debugger)

| Symptom | Likely cause / fix |
|---|---|
| `Unable to find turbulence model in the database` from a y+ attempt | standalone `postProcess -func yPlus` — use `<solver> -postProcess -func yPlus -latestTime` |
| `Invalid option: -postProcess` | icoFoam-family laminar solver: the verify step is structurally inapplicable — no turbulence model, no y+ question |
| laminar run aborts right after adding `#includeFunc yPlus` | the run-time functionObject needs a turbulence model and kills a laminar run — remove it |
| `Unknown ... type epsilonLowReWallFunction` (or `omegaLowReWallFunction`) | the type does not exist on v10 — folklore/ESI vocabulary; use the harvested wall-resolved sets (fixedValue + dual-regime wall functions + `nutLowReWallFunction`) |
| achieved average y+ in 5–30 on a wall-function case | first cell landed in the buffer-layer no-man's-land — scale the target by target/achieved, rerun the tool, re-bridge, remesh |
| a wall patch is missing from `yPlus.dat` | the patch is not `type wall` in `constant/polyMesh/boundary` — wall functions were not acting on it either; fix the patch type |
| turbulent run diverges within the first steps, k or epsilon exploding | inlet turbulence values orders of magnitude off — recompute with `estimate_turbulence_inlet`; check the viscosity-ratio sanity figure and bounded div schemes on the turbulence fields |
| `keyword div((nuEff*dev2(T(grad(U))))) is undefined` | required whenever `constant/momentumTransport` drives the solver, even laminar — [file-generation.md](file-generation.md) |
| layers collapsed exactly where the y+ target lived | `minThickness` too close to the first layer's relative thickness, or quality controls collapsed the stack — read the snappy log; [snappyhexmesh.md](snappyhexmesh.md)'s ladder |
| achieved y+ far off target although the mesh matched the plan | the flow disagrees with the sizing correlation (acceleration, separation, recirculation) — that is what the verify step is for; adjust and remesh, do not retune the model |

Downstream: convergence judgement for the run itself is
[convergence.md](convergence.md) (steady runs list the turbulence
fields in `residuals` — missing ones mean unjudged physics); mesh
quality for the layered mesh is [mesh-quality.md](mesh-quality.md); the
fix-loop discipline is [error-playbook.md](error-playbook.md).
