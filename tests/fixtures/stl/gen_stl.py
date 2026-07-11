"""Generate the four tiny STL fixture surfaces for ticket #59.

ASCII STL, axis-aligned cubes with programmatically wound outward normals
(asserted via cross product) so watertightness/orientation checks see a
clean surface. Sizes are deliberate:

- watertight_cube.stl : 1 m cube at origin - closed, 1 shell, sane units
- open_box.stl        : same cube minus the +z face - open edges
- two_shells.stl      : two disjoint 1 m cubes in one solid - 2 parts
- cube_mm.stl         : 1000-unit cube - a 1 m part exported in mm
"""


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cube_facets(origin, side, skip_plus_z=False):
    """Triangles of an axis-aligned cube, outward-wound, as (normal, tri)."""
    facets = []
    for axis in range(3):
        b, c = (axis + 1) % 3, (axis + 2) % 3  # e_b x e_c = e_axis
        for sign in (-1.0, 1.0):
            if skip_plus_z and axis == 2 and sign > 0:
                continue
            normal = [0.0, 0.0, 0.0]
            normal[axis] = sign
            normal = tuple(normal)

            def corner(vb, vc):
                p = list(origin)
                p[axis] += side if sign > 0 else 0.0
                p[b] += vb * side
                p[c] += vc * side
                return tuple(p)

            p00, p10, p11, p01 = (corner(0, 0), corner(1, 0),
                                  corner(1, 1), corner(0, 1))
            quad = (p00, p10, p11, p01) if sign > 0 else (p00, p01, p11, p10)
            for tri in ((quad[0], quad[1], quad[2]),
                        (quad[0], quad[2], quad[3])):
                n = _cross(_sub(tri[1], tri[0]), _sub(tri[2], tri[0]))
                assert _dot(n, normal) > 0, (axis, sign, tri)
                facets.append((normal, tri))
    return facets


def write_stl(path, name, facets):
    lines = [f"solid {name}"]
    for normal, tri in facets:
        lines.append("  facet normal {:.6e} {:.6e} {:.6e}".format(*normal))
        lines.append("    outer loop")
        for v in tri:
            lines.append("      vertex {:.6e} {:.6e} {:.6e}".format(*v))
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(facets)} facets)")


write_stl("watertight_cube.stl", "cube",
          cube_facets((0.0, 0.0, 0.0), 1.0))
write_stl("open_box.stl", "openBox",
          cube_facets((0.0, 0.0, 0.0), 1.0, skip_plus_z=True))
write_stl("two_shells.stl", "twoShells",
          cube_facets((0.0, 0.0, 0.0), 1.0)
          + cube_facets((3.0, 0.0, 0.0), 1.0))
write_stl("cube_mm.stl", "cubeMm",
          cube_facets((0.0, 0.0, 0.0), 1000.0))
