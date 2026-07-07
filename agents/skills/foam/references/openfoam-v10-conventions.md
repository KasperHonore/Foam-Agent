# Foundation OpenFOAM v10 conventions

Foam-Agent targets **Foundation OpenFOAM v10** (openfoam.org). The tutorial
database, error playbook and all generated files follow v10 naming. ESI
OpenFOAM (openfoam.com, versions like v2312/v2406/v2512) uses different names —
do NOT mix the two.

## Foundation v10 vs ESI naming

| Concept | Foundation v10 (generate this) | ESI (do NOT generate) |
|---|---|---|
| Turbulence dictionary | `constant/momentumTransport` | `constant/turbulenceProperties` |
| Viscosity/thermo dictionary | `constant/physicalProperties` | `constant/transportProperties` (incompressible) / `constant/thermophysicalProperties` (thermo) |
| Turbulence model keyword | `model kEpsilon;` | `RASModel kEpsilon;` |
| Viscosity keyword | `viscosityModel` | `transportModel` |
| Buoyant solver | `buoyantFoam` | `buoyantPimpleFoam` |
| Compressible thermo | `hePsiThermo` valid | often `heRhoThermo` |

Notes:

- In v10, even simple incompressible solvers (`icoFoam`, `pisoFoam`,
  `simpleFoam`, `pimpleFoam`) read `constant/physicalProperties`:

  ```
  viscosityModel  constant;
  nu              [0 2 -1 0 0 0 0] 1e-05;
  ```

  Do NOT generate `constant/transportProperties` — that is the pre-v10/ESI
  name and v10 solvers will fail with
  `cannot find file ".../constant/physicalProperties"`.
- If the user's installation is ESI, generate a correct v10 case first and use
  the `translate_case_to_esi` tool — it applies these mappings mechanically
  (file renames, keyword swaps, `pFinal` injection, `pRefCell`/`pRefValue` in
  PISO). It is best-effort; verify solver availability (e.g.
  `adjointShapeOptimisationFoam` and `boundaryFoam` have no ESI equivalent).

## Standard case layout

```
case/
  0/            # initial & boundary conditions per field (U, p, k, epsilon, T, ...)
  constant/     # physical properties, turbulence dict, polyMesh/ after meshing
  system/       # controlDict, fvSchemes, fvSolution, blockMeshDict, decomposeParDict
  Allrun        # execution script
```

## Transient vs steady solvers

Transient solvers (need `*Final` solver entries in fvSolution, small
`deltaT`, `adjustTimeStep` where supported): `icoFoam`, `pisoFoam`,
`pimpleFoam`, `rhoPimpleFoam`, `rhoCentralFoam`, `interFoam`,
`multiphaseInterFoam`.

Steady solvers (SIMPLE loop, `relaxationFactors` matter): `simpleFoam`,
`potentialFoam` and friends.
