#!/usr/bin/env python3
"""Integration test for the key-free MCP server (manual, needs a running server).

Drives the full mechanical tool surface end-to-end with a known-good
lid-driven cavity case — deterministic, no LLM anywhere:

    get_case_stats → find_similar_case → resolve_case_dir → write_case_file
    → run_case → ensure_foam_file → run_python_script (PyVista)

Prerequisites:
    foamagent-mcp --transport http --port 7860   (e.g. inside the Docker image)
    OpenFOAM v10 sourced in the server's environment.

Run:  python tests/test_lid_driven_cavity_mcp.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import Client

SERVER_URL = "http://localhost:7860/mcp"

# --- Known-good Foundation OpenFOAM v10 cavity case (icoFoam tutorial) -----

CASE_FILES = {
    "system/controlDict": """FoamFile
{
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     icoFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.5;
deltaT          0.005;
writeControl    timeStep;
writeInterval   20;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""",
    "system/fvSchemes": """FoamFile
{
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; grad(p) Gauss linear; }
divSchemes      { default none; div(phi,U) Gauss linear; }
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes   { default orthogonal; }
""",
    "system/fvSolution": """FoamFile
{
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0.05;
    }

    pFinal
    {
        $p;
        relTol          0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0;
    }
}

PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}
""",
    "system/blockMeshDict": """FoamFile
{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

convertToMeters 0.1;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 0.1)
    (1 0 0.1)
    (1 1 0.1)
    (0 1 0.1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    movingWall
    {
        type wall;
        faces
        (
            (3 7 6 2)
        );
    }
    fixedWalls
    {
        type wall;
        faces
        (
            (0 4 7 3)
            (2 6 5 1)
            (1 5 4 0)
        );
    }
    frontAndBack
    {
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);

mergePatchPairs
(
);
""",
    "constant/physicalProperties": """FoamFile
{
    format      ascii;
    class       dictionary;
    object      physicalProperties;
}

viscosityModel  constant;

nu              [0 2 -1 0 0 0 0] 0.01;
""",
    "0/U": """FoamFile
{
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    movingWall
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
    fixedWalls
    {
        type            noSlip;
    }
    frontAndBack
    {
        type            empty;
    }
}
""",
    "0/p": """FoamFile
{
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    movingWall
    {
        type            zeroGradient;
    }
    fixedWalls
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
""",
}

ALLRUN = """#!/bin/sh
cd ${0%/*} || exit 1

. $WM_PROJECT_DIR/bin/tools/RunFunctions

runApplication blockMesh
runApplication icoFoam
"""

PYVISTA_SCRIPT = """import pyvista as pv

pv.OFF_SCREEN = True

reader = pv.OpenFOAMReader("{foam_file}")
reader.set_active_time_value(reader.time_values[-1])
mesh = reader.read()["internalMesh"]

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, scalars="U", cmap="coolwarm", show_scalar_bar=True)
plotter.view_xy()
plotter.screenshot("velocity.png")
"""


def unwrap(response):
    return response.structured_content or response.data or {}


async def main():
    print("Lid-Driven Cavity Test (key-free MCP, no LLM calls)")
    print("=" * 60)
    results = {}

    try:
        client = Client(SERVER_URL)
        async with client:
            print(f"Connected to {SERVER_URL}")

            # 1. Case stats
            stats = unwrap(await client.call_tool("get_case_stats", {}))
            assert "incompressible" in stats["case_domain"], stats
            assert "icoFoam" in stats["case_solver"], stats
            print(f"get_case_stats: {len(stats['case_solver'])} solvers")
            results["case_stats"] = True

            # 2. Similar-case retrieval (FAISS, local embeddings)
            similar = unwrap(await client.call_tool("find_similar_case", {
                "case_name": "lid_driven_cavity",
                "case_solver": "icoFoam",
                "case_domain": "incompressible",
                "case_category": "cavity",
                "searchdocs": 3,
            }))
            print(f"find_similar_case: found={similar['found']} "
                  f"selected={(similar.get('selected_case') or {}).get('case_name')}")
            results["retrieval"] = similar["found"]

            # 3. Create case files
            case_dir = unwrap(await client.call_tool(
                "resolve_case_dir", {"case_name": "lid_driven_cavity_mcp_test"}
            ))
            if isinstance(case_dir, dict):  # plain-string results may arrive wrapped
                case_dir = case_dir.get("result", case_dir)
            print(f"case_dir: {case_dir}")

            for rel_path, content in CASE_FILES.items():
                await client.call_tool("write_case_file", {
                    "case_dir": case_dir,
                    "relative_path": rel_path,
                    "content": content,
                })
            await client.call_tool("write_case_file", {
                "case_dir": case_dir,
                "relative_path": "Allrun",
                "content": ALLRUN,
                "executable": True,
            })
            listing = unwrap(await client.call_tool("list_case_files", {"case_dir": case_dir}))
            print(f"wrote case: {listing}")
            results["file_io"] = "system" in listing and "0" in listing

            # 4. Run the simulation
            run = unwrap(await client.call_tool("run_case", {
                "case_dir": case_dir,
                "timeout": 600,
            }))
            print(f"run_case: {run['status']} ({len(run['errors'])} errors)")
            for err in run["errors"][:3]:
                print(f"  - {err}")
            results["simulation_run"] = run["status"] == "success"

            # 5. Mesh boundary inspection
            boundaries = unwrap(await client.call_tool("read_mesh_boundaries", {"case_dir": case_dir}))
            print(f"boundaries: {boundaries['boundary_names']}")
            results["mesh_boundaries"] = set(boundaries["boundary_names"]) >= {
                "movingWall", "fixedWalls", "frontAndBack"
            }

            # 6. Visualization (PyVista, deterministic script)
            if results["simulation_run"]:
                foam_file = unwrap(await client.call_tool("ensure_foam_file", {"case_dir": case_dir}))
                if isinstance(foam_file, dict):
                    foam_file = foam_file.get("result", foam_file)
                viz = unwrap(await client.call_tool("run_python_script", {
                    "case_dir": case_dir,
                    "script": PYVISTA_SCRIPT.format(foam_file=foam_file),
                    "filename": "visualization.py",
                    "expected_output": "velocity.png",
                    "timeout": 180,
                }))
                print(f"visualization: success={viz['success']} artifact={viz['artifact']}")
                for err in viz.get("errors", [])[:3]:
                    print(f"  - {err}")
                results["visualization"] = viz["success"]
            else:
                results["visualization"] = False

        # Summary
        print("\n" + "=" * 60)
        passed = sum(1 for v in results.values() if v)
        for name, ok in results.items():
            print(f"{name:20} {'PASS' if ok else 'FAIL'}")
        print(f"\nOverall: {passed}/{len(results)} steps passed")
        return 0 if passed == len(results) else 1

    except Exception as e:
        print(f"\nTest failed: {e}")
        print("\nMake sure the MCP server is running with OpenFOAM available:")
        print("  foamagent-mcp --transport http --port 7860")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
