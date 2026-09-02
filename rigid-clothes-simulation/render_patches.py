"""Colour every panel differently and paint the panel outlines black.

  python render_patches.py                          ->  result/v4d_skirt/*.ply
  python render_patches.py <garment_id> <outdir>    ->  <outdir>/*.ply

Panels are separate connected components in the raw mesh, so a mesh boundary
edge -- one belonging to a single face -- is exactly a panel outline.  Those
vertices go black; everything else takes its panel's colour.  Vertex colours
interpolate across a triangle, so the outline reads as a dark band a triangle
wide rather than a hairline, which is what makes it visible at all in a viewer.
"""

import collections
import os

import numpy as np

import gcd_io
import run_garment as RG

OUT = os.path.join(RG.RESULT, "v4d_skirt")

# 18 distinguishable hues, one per panel
PALETTE = np.array([
    [0.90, 0.24, 0.22], [0.20, 0.45, 0.80], [0.35, 0.72, 0.30], [0.95, 0.65, 0.15],
    [0.60, 0.30, 0.78], [0.20, 0.75, 0.72], [0.92, 0.50, 0.72], [0.45, 0.45, 0.45],
    [0.75, 0.78, 0.20], [0.15, 0.35, 0.48], [0.98, 0.84, 0.40], [0.50, 0.26, 0.18],
    [0.38, 0.66, 0.92], [0.80, 0.35, 0.50], [0.22, 0.58, 0.40], [0.68, 0.68, 0.90],
    [0.90, 0.45, 0.12], [0.12, 0.28, 0.24],
])


def boundary_vertices(faces, n):
    cnt = collections.Counter()
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            cnt[(min(a, b), max(a, b))] += 1
    bnd = np.zeros(n, bool)
    for (a, b), c in cnt.items():
        if c == 1:
            bnd[a] = bnd[b] = True
    return bnd


def write_ply(path, P, faces, rgb):
    P = np.asarray(P, float)
    if P.shape[1] == 2:                        # the flat pattern, drawn in z = 0
        P = np.hstack([P, np.zeros((len(P), 1))])
    c = np.clip(rgb * 255, 0, 255).astype(int)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % len(P))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("element face %d\n" % len(faces))
        f.write("property list uchar int vertex_indices\nend_header\n")
        for p, q in zip(P, c):
            f.write("%.6f %.6f %.6f %d %d %d\n" % (p[0], p[1], p[2], q[0], q[1], q[2]))
        for t in faces:
            f.write("3 %d %d %d\n" % (t[0], t[1], t[2]))


def main(gid=None, out=None):
    out = out or OUT
    os.makedirs(out, exist_ok=True)
    d = gcd_io.load(RG.garment_dir(gid) if gid else RG.GARMENT)
    F, pr = d["faces"], np.maximum(d["panel_of_raw"], 0)
    bnd = boundary_vertices(F, len(pr))
    rgb = PALETTE[pr % len(PALETTE)].copy()
    rgb[bnd] = 0.0                                       # panel outlines in black
    print("panels %d, vertices %d, on a panel outline %d (%.1f%%)"
          % (pr.max() + 1, len(pr), bnd.sum(), 100 * bnd.mean()))

    import glob
    todo = [("rest_flat", d["rest"]), ("placement", d["placed"]),
            ("drape_reference", d["drape"])]
    runs = sorted(glob.glob(os.path.join(out, "assembly_*.npy")))
    if not runs:
        runs = sorted(glob.glob(os.path.join(RG.RESULT, "assembly_sym_body_ease5_*.npy")))
    for f in runs:
        b = os.path.basename(f)[len("assembly_"):-4].replace("sym_body_ease5_", "")
        todo.append((b, np.load(f)))
    for name, P in todo:
        write_ply(os.path.join(out, "%s_patches.ply" % name), P, F, rgb)
        print("  wrote %s_patches.ply" % name)

    # a legend, so the colours can be read back to panel names
    with open(os.path.join(out, "panel_colours.txt"), "w") as f:
        f.write("panel colours (black = panel outline)\n\n")
        for i, nm in enumerate(d["panel_names"]):
            c = (PALETTE[i % len(PALETTE)] * 255).astype(int)
            f.write("  %-22s rgb(%3d,%3d,%3d)\n" % (nm, c[0], c[1], c[2]))
    print("  wrote panel_colours.txt")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:3])
