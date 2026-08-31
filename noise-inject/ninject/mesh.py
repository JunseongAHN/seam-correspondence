"""Panel surface for the viewer: decimate the interior, keep every boundary vertex.

Boundary vertices are the ones the seam decode is about, so they are never merged.
Interior vertices are snapped to a grid and collapsed, which cuts the face count
by an order of magnitude while the panel outline stays exactly as measured.
"""
import numpy as np


def decimate(V, F, keep, cell):
    """V[n,3], F[f,3], keep[n] bool (never merged), cell = grid size in the same units.

    -> (V2, F2, keep2, old_to_new)
    """
    q = np.floor(V / cell).astype(np.int64)
    key = {}
    remap = np.empty(len(V), np.int64)
    reps = []
    for i in range(len(V)):
        if keep[i]:
            remap[i] = len(reps); reps.append(i); continue
        k = (q[i, 0], q[i, 1], q[i, 2])
        j = key.get(k)
        if j is None:
            key[k] = len(reps); remap[i] = len(reps); reps.append(i)
        else:
            remap[i] = j
    reps = np.array(reps)
    V2 = V[reps].copy()
    # cluster centroid for merged interior vertices (kept ones stay put)
    keep2 = keep[reps]
    acc = np.zeros_like(V2); cnt = np.zeros(len(V2))
    np.add.at(acc, remap, V); np.add.at(cnt, remap, 1)
    m = (~keep2) & (cnt > 0)
    V2[m] = acc[m] / cnt[m, None]
    F2 = remap[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return V2, F2[ok], keep2, remap


def panel_surface(W, faces, loops_for_panel, cell):
    """Build one panel's decimated surface plus its boundary loops as vertex indices."""
    vids = np.unique(faces)
    idx = {v: i for i, v in enumerate(vids)}
    V = W[vids]
    F = np.array([[idx[a], idx[b], idx[c]] for a, b, c in faces], np.int64)
    keep = np.zeros(len(V), bool)
    loops_local = []
    for L in loops_for_panel:
        ll = [idx[v] for v in L if v in idx]
        if len(ll) >= 4:
            loops_local.append(ll)
            keep[ll] = True
    V2, F2, _, remap = decimate(V, F, keep, cell)
    loops2 = [[int(remap[i]) for i in ll] for ll in loops_local]
    return V2, F2, loops2
