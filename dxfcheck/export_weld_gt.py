"""weld.obj + unweld.obj -> the seam ground truth, as JSON the web demo can draw.

CLO's own weld operation defines the correspondence: welding merges coincident boundary
vertices without moving them, so every group of unweld vertices sharing a position is a
sewn point.  The validation gate is that sum(len(group)-1) over the garment panels equals
len(unweld) - len(weld), i.e. exactly the number of vertices CLO merged.

Because the panels are draped, sewn vertices sit on top of each other in 3D and a seam
line would have zero length.  The viewer explodes the panels apart; this file just ships
the geometry, the panel assignment and the weld groups.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_obj(path):
    """-> (positions as raw text tuples, verts float array, triangles)"""
    txt, V, F = [], [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()[1:4]
                txt.append(tuple(p))
                V.append([float(t) for t in p])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):        # fan-triangulate
                    F.append([idx[0], idx[k], idx[k + 1]])
    return txt, np.asarray(V, float), np.asarray(F, int)


def components(nv, F):
    parent = list(range(nv))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for f in F:
        for k in (1, 2):
            ra, rb = find(f[0]), find(f[k])
            if ra != rb:
                parent[ra] = rb
    lab, out = {}, np.empty(nv, int)
    for i in range(nv):
        r = find(i)
        if r not in lab:
            lab[r] = len(lab)
        out[i] = lab[r]
    return out, len(lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unweld", required=True)
    ap.add_argument("--weld", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", type=int, default=10, help="garment panel count")
    a = ap.parse_args()

    txt, V, F = load_obj(a.unweld)
    nw = sum(1 for line in open(a.weld, "r", encoding="utf-8", errors="replace")
             if line.startswith("v "))
    merged = len(V) - nw
    comp, ncomp = components(len(V), F)
    sizes = np.bincount(comp)

    # the garment panels are the components that are welded to each other; the avatar
    # body and its parts are never welded to anything
    pos = defaultdict(list)
    for i, p in enumerate(txt):
        pos[p].append(i)
    groups = [g for g in pos.values() if len(g) > 1]
    cross = [g for g in groups if len({comp[i] for i in g}) > 1]
    garment = sorted({comp[i] for g in cross for i in g}, key=lambda c: -sizes[c])
    print(f"unweld {len(V)} v / weld {nw} v -> CLO merged {merged}")
    print(f"{ncomp} components; {len(garment)} take part in welds: "
          f"{[int(sizes[c]) for c in garment]}")

    gset = set(garment)
    gg = [g for g in cross if all(comp[i] in gset for i in g)]
    total = sum(len(g) - 1 for g in gg)
    print(f"weld groups spanning panels: {len(gg)}   sum(len-1) = {total}"
          f"   CLO merged = {merged}   {'MATCH' if total == merged else 'MISMATCH'}")
    if total != merged:
        raise SystemExit("refusing to export: the weld groups do not reproduce CLO's "
                         "merge count, so the correspondence is not trustworthy")

    keep = np.array([i for i in range(len(V)) if comp[i] in gset])
    remap = -np.ones(len(V), int)
    remap[keep] = np.arange(len(keep))
    panel_of_comp = {c: k for k, c in enumerate(garment)}

    tris = np.array([f for f in F if comp[f[0]] in gset])
    out = {
        "source": Path(a.unweld).name,
        "merged": int(merged),
        "verts": [round(float(v), 3) for v in V[keep].ravel()],
        "tris": [int(t) for t in remap[tris].ravel()],
        "panelOf": [int(panel_of_comp[comp[i]]) for i in keep],
        "panelSizes": [int(sizes[c]) for c in garment],
        "welds": [[int(remap[i]) for i in g] for g in gg],
    }
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {a.out}: {len(keep)} verts, {len(tris)} tris, "
          f"{len(gg)} weld groups, {len(garment)} panels")


if __name__ == "__main__":
    main()
