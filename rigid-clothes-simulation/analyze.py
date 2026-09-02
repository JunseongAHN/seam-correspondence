"""Measurements (a)-(f) of the garment assembly spec.

Every cross-run and cross-target comparison is Procrustes aligned first: the
energy is invariant under a global rigid motion, so unaligned coordinates would
differ even for identical shapes.

Reflection is NOT allowed in the alignment.  A mirrored garment is a different
garment (left and right swap), so admitting reflections would let a mirrored
solution count as a match.
"""

import json
import os
from collections import defaultdict

import numpy as np

import gcd_io

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result")


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

def procrustes(A, B, allow_reflection=False, scale=False):
    """Rigid transform taking A onto B (Kabsch).  Returns the transformed A."""
    ca, cb = A.mean(0), B.mean(0)
    X, Y = A - ca, B - cb
    U, S, Vt = np.linalg.svd(X.T @ Y)
    d = np.sign(np.linalg.det(U @ Vt))
    if not allow_reflection and d < 0:
        S = S.copy()
        S[-1] *= -1
        U = U.copy()
        U[:, -1] *= -1
    R = U @ Vt
    s = (S.sum() / (X * X).sum()) if scale else 1.0
    return s * (X @ R) + cb


# ---------------------------------------------------------------------------
# (b) angle deficit on the welded REST mesh
# ---------------------------------------------------------------------------

def angle_deficit(d):
    """2*pi - sum(theta) at interior welded vertices, pi - sum(theta) at boundary.

    Angles are measured in the flat REST geometry, with each incident triangle
    contributing its own panel's angle.  This is the quantity that makes the
    stitched surface non-developable, so it is what forces a 3D shape.
    """
    rest_tri = d["rest"][d["faces"]]
    e0 = rest_tri[:, 1] - rest_tri[:, 0]
    e1 = rest_tri[:, 2] - rest_tri[:, 0]
    e2 = rest_tri[:, 2] - rest_tri[:, 1]
    ang = np.zeros((len(rest_tri), 3))
    for c, (u, v) in enumerate([(e0, e1), (-e0, e2), (-e1, -e2)]):
        cs = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        ang[:, c] = np.arccos(np.clip(cs, -1, 1))

    nw = d["n_welded"]
    tot = np.zeros(nw)
    np.add.at(tot, d["wid"][d["faces"]].ravel(), ang.ravel())

    # boundary: welded edges incident to exactly one face
    cnt = defaultdict(int)
    W = d["wid"][d["faces"]]
    for f in W:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            cnt[(min(a, b), max(a, b))] += 1
    is_bnd = np.zeros(nw, bool)
    for (a, b), c in cnt.items():
        if c == 1:
            is_bnd[a] = is_bnd[b] = True

    K = np.where(is_bnd, np.pi - tot, 2 * np.pi - tot)
    return K, is_bnd, tot


def seam_length_ratio(d, thresh=1.05):
    """Per welded seam edge, the ratio of its REST length as seen from the two
    panels that own a copy.

    A sewing pattern matches its seam lengths by construction, and the data
    confirms it: the ratio is 1.00000 at the median.  The exceptions are the
    gathers the design asks for -- this garment sets sleeve `top_ruffle`
    = 1.84426, and the mesh reproduces 1.8438.  At such a seam the welded 1:1
    vertex correspondence forces the longer side to compress by that ratio, so
    |sigma-1| there is a property of the design, not of the solver.  Faces
    touching one are reported separately rather than mixed into max|sigma-1|.

    -> ratio per shared edge, the edge keys, and a face mask for `> thresh`.
    """
    seen = defaultdict(dict)
    for t, f in enumerate(d["faces"]):
        p = d["panel_of_face"][t]
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            k = (min(d["wid"][a], d["wid"][b]), max(d["wid"][a], d["wid"][b]))
            seen[k][p] = float(np.linalg.norm(d["rest"][a] - d["rest"][b]))
    ratio, keys = [], []
    for k, v in seen.items():
        if len(v) >= 2:
            L = sorted(v.values())
            ratio.append(L[-1] / max(L[0], 1e-12))
            keys.append(k)
    ratio = np.array(ratio)
    bad = set()
    for r, k in zip(ratio, keys):
        if r > thresh:
            bad.add(k[0]); bad.add(k[1])
    W = d["wid"][d["faces"]]
    touch = np.array([bool(bad & set(w.tolist())) for w in W])
    return ratio, np.array(keys), touch


# ---------------------------------------------------------------------------
# per-class aggregation
# ---------------------------------------------------------------------------

def class_of_raw(d):
    cls = np.array([gcd_io.PANEL_CLASS[nm] for nm in d["panel_names"]], dtype=object)
    return cls[d["panel_of_raw"]]


def by_class(values, cls, fn=np.median):
    out = {}
    for c in sorted(set(cls)):
        m = cls == c
        if m.any():
            out[c] = float(fn(values[m]))
    return out


# ---------------------------------------------------------------------------
# (f) self-intersection count
# ---------------------------------------------------------------------------

def tri_tri_pairs(P, faces, cell=None, max_pairs=40_000_000):
    """Count intersecting triangle pairs via a uniform grid over triangle AABBs.

    Only pairs that share no vertex and whose AABBs overlap are tested exactly.
    """
    T = P[faces]
    lo, hi = T.min(1), T.max(1)
    if cell is None:
        cell = float(np.median(hi - lo)) * 2.0
    g0 = lo.min(0)
    gi = np.floor((lo - g0) / cell).astype(np.int64)
    gj = np.floor((hi - g0) / cell).astype(np.int64)

    buckets = defaultdict(list)
    for t in range(len(faces)):
        for x in range(gi[t, 0], gj[t, 0] + 1):
            for y in range(gi[t, 1], gj[t, 1] + 1):
                for z in range(gi[t, 2], gj[t, 2] + 1):
                    buckets[(x, y, z)].append(t)

    cand = set()
    for b in buckets.values():
        if len(b) < 2:
            continue
        b = np.array(b)
        for ii in range(len(b)):
            for jj in range(ii + 1, len(b)):
                a, c = b[ii], b[jj]
                cand.add((a, c) if a < c else (c, a))
        if len(cand) > max_pairs:
            break
    cand = np.array(sorted(cand), np.int64).reshape(-1, 2)
    if not len(cand):
        return 0, cand[:0]
    # drop pairs sharing a vertex, and AABB-reject
    fa, fb = faces[cand[:, 0]], faces[cand[:, 1]]
    share = (fa[:, :, None] == fb[:, None, :]).any((1, 2))
    ok = ~share
    ok &= (lo[cand[:, 0]] <= hi[cand[:, 1]]).all(1) & (lo[cand[:, 1]] <= hi[cand[:, 0]]).all(1)
    cand = cand[ok]
    if not len(cand):
        return 0, cand
    hits = _tri_tri(T[cand[:, 0]], T[cand[:, 1]])
    return int(hits.sum()), cand[hits]


def _seg_tri(P0, P1, V0, V1, V2, eps=1e-12):
    """Moller-Trumbore: does segment P0->P1 cross triangle V0V1V2?  Vectorised."""
    d = P1 - P0
    e1, e2 = V1 - V0, V2 - V0
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > eps
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    sv = P0 - V0
    u = f * np.einsum("ij,ij->i", sv, h)
    ok &= (u >= 0) & (u <= 1)
    q = np.cross(sv, e1)
    v = f * np.einsum("ij,ij->i", d, q)
    ok &= (v >= 0) & (u + v <= 1)
    t = f * np.einsum("ij,ij->i", e2, q)
    return ok & (t >= 0) & (t <= 1)


def _tri_tri(A, B):
    """Two triangles intersect iff an edge of one crosses the other.  Coplanar
    overlaps are not detected; they are measure-zero here and are noted as a
    limitation rather than special-cased."""
    hit = np.zeros(len(A), bool)
    for T, Q in ((A, B), (B, A)):
        for i in range(3):
            hit |= _seg_tri(T[:, i], T[:, (i + 1) % 3], Q[:, 0], Q[:, 1], Q[:, 2])
    return hit
