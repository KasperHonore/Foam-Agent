# surfaceCheck parser fixtures

Harvested live for ticket #59 (spec #58, STL → snappyHexMesh pathway) on
2026-07-11, inside the foamagent container, OpenFOAM v10 (Build
10-c4cf895ad8fa). Each `log.surfaceCheck` is the complete stdout+stderr of a
plain `surfaceCheck <surface>.stl` run on the matching surface from
`tests/fixtures/stl/` — no flags, bytes kept exactly as captured
(`.gitattributes -text`).

| Variant | Surface | The signal it carries |
|---|---|---|
| `watertight` | `watertight_cube.stl` | "Surface is closed. All edges connected to two faces." — 1 part, bbox (0 0 0)(1 1 1) |
| `open` | `open_box.stl` | "Surface is not closed since not all edges connected to two faces: connected to one face : 4" |
| `multi_shell` | `two_shells.stl` | closed, but "Number of unconnected parts : 2" |
| `mm_scaled` | `cube_mm.stl` | closed, 1 part, bbox (0 0 0)(1000 1000 1000) — a 1 m part exported in mm |

Regenerate (from a directory holding the STLs, inside the container):

    source /opt/openfoam10/etc/bashrc
    surfaceCheck <surface>.stl > log.surfaceCheck 2>&1
