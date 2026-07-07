# OpenFOAM error diagnosis and fix loop

The fix loop: `run_case` → read errors → diagnose → rewrite the minimal set of
files with `write_case_file` → `run_case` again. Up to 25 iterations. Keep a
running history of (error, diagnosis, fix) per attempt; when the same error
returns, try a **different** approach — do not repeat a failed fix.

## Reading errors

`run_case` returns `errors: [{file, error_content}]` — the log file name tells
you which command failed (`log.blockMesh` → meshing; `log.simpleFoam` → the
solver). If the excerpt is not enough, `read_case_file(case_dir, "log.<cmd>")`
for full context, and read the case files involved before proposing a fix.

## Diagnosis rules

- **Take undefined-keyword errors literally.** If the log says
  `div(phi,(p|rho)) is undefined`, define exactly that keyword —
  `div(phi,(p|rho))` — in `system/fvSchemes`. Do not reinterpret the symbols
  (a `|` is NOT "or"); OpenFOAM means the literal string.
- **Never change parameters the user specified** (Re, velocities, geometry,
  end time, solver choice). Find another way to fix the error.
- Prefer the smallest change that addresses the root cause; plan the fix as an
  explicit list of `{file, changes}` before editing, then apply exactly that.
- Cross-check with the similar-case reference (`find_similar_case` /
  `search_tutorials`) — how does a working case of this solver configure the
  problematic dictionary?

## Common error patterns

| Symptom in log | Likely cause / fix |
|---|---|
| `keyword ... is undefined` in fvSchemes | Add the literal scheme entry (e.g. under `divSchemes`) |
| `Unknown ... type` for a BC | Wrong boundary condition name for the patch type or v10 fork — check v10 naming |
| `cannot find file ... 0/<field>` | Solver needs a field file you didn't generate — create it |
| `patch ... not found` / patch count mismatch | `0/` field patch names don't match `constant/polyMesh/boundary` — use `read_mesh_boundaries` and align |
| Floating point exception / diverging residuals | Time step too large, bad initial values, or wrong scheme; reduce `deltaT`, add relaxation, use more robust schemes (e.g. `upwind` div) |
| `Continuity errors` blowing up | Check pressure solver + `pRefCell`/`pRefValue` (closed incompressible domains) |
| Solver exits without `End` marker | Crash or timeout — read the log tail; often numerical instability |
| Dimension mismatch `[...]` | Fix the `dimensions` entry or the offending value's units |
| `gmshToFoam` patch types wrong (all `patch`) | Edit `constant/polyMesh/boundary`: set walls to `wall`, 2D front/back planes to `empty` |

## When stuck (error persists after several attempts)

- Re-read ALL case files (`list_case_files` + `read_case_file`) — the error's
  root cause is often in a different file than the one the log names.
- Search for the failing solver + error keyword in the tutorial database:
  `search_tutorials(index="openfoam_tutorials_details", query="...")`.
- Consider regenerating the problematic file from scratch from the closest
  tutorial instead of patching it again.
- After 25 failed iterations, stop and report honestly: show the last error,
  what was tried, and suggest manual intervention.
