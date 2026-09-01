"""Rescale the REST METRIC near a seam so that neither side is ever compressed.

Why this is needed.  Walking every boundary edge, the waist seam of this garment
does not match: skirt_front is 42.10 cm against wb_front's 45.23 (0.931) and
skirt_back is 42.10 against wb_back's 39.16 (1.075).  The totals do agree --
84.20 cm of skirt against 84.39 of waistband, 0.2% -- so the metric is not
inconsistent overall; the front/back split is simply placed differently on the
two pieces.  Sewn, one side of each seam has to change length, and the two
directions are not equivalent:

  a side that must LENGTHEN resolves it in plane, and stays smooth;
  a side that must SHORTEN has no in-plane way to do it, so it buckles.

The buckling is the pinch.  Setting the shared rest length to the LONGER of the
two sides therefore removes it: every side is then either neutral or in tension,
and nothing is ever compressed.  Here that costs a 3.5% wider waist (84.39 ->
87.33 cm) and nothing else.

What is edited, and what is not.  ARAP reads the rest metric -- edge lengths and
areas -- never the 2D coordinates themselves, so the correction is applied to a
private copy of the rest triangles and `rest` is returned untouched.  The panel
outline that the geometry image is defined over does not move.  This is also
what a seam physically is: ease is worked into the fabric over a few centimetres,
which is exactly "the flat shape is unchanged but its metric is no longer
Euclidean".

Three details follow from that:

  * the correction is applied only along the SEAM TANGENT, so the panel's width
    across the seam is untouched;
  * it is graded to zero over a band rather than imposed on the seam line, since
    a step in the metric just moves the stress concentration to the step;
  * seams whose mismatch is a designed gather are left alone.  The cuff seam is
    1.8438 -- exactly `design.sleeve.cuff.top_ruffle` -- and lengthening the cuff
    by 84% is not what a gather is.  Those are reported separately, not eased.
"""

import collections

import numpy as np
from scipy.spatial import cKDTree

MAX_EASE = 0.25          # above this a mismatch is a designed gather, not ease


def _boundary(faces):
    cnt = collections.Counter()
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            cnt[(min(a, b), max(a, b))] += 1
    return [e for e, n in cnt.items() if n == 1]


def _partners(d):
    """raw vertex -> the raw vertices it is sewn to (soft pairs and shared welds)"""
    p = collections.defaultdict(set)
    for a, b in d["pairs"]:
        p[a].add(b)
        p[b].add(a)
    byw = collections.defaultdict(list)
    for r, w in enumerate(d["wid"]):
        byw[w].append(r)
    for rs in byw.values():
        if len(rs) > 1:
            for a in rs:
                p[a].update(x for x in rs if x != a)
    return p


def seam_scales(d, report=None):
    """{(panel, other): scale to apply to `panel`'s rest metric along that seam}"""
    R, faces = d["rest"], d["faces"]
    pid = np.maximum(d["panel_of_raw"], 0)
    names = np.array(d["panel_names"], dtype=object)
    part = _partners(d)
    bedges = _boundary(faces)

    # arc length of each panel's share of each seam
    arc = collections.Counter()
    for a, b in bedges:
        if pid[a] != pid[b]:
            continue
        pa = pid[a]
        common = {pid[x] for x in part[a]} & {pid[x] for x in part[b]} - {pa}
        for q in common:
            arc[(pa, q)] += float(np.linalg.norm(R[a] - R[b]))

    scales = {}
    for (p, q), la in arc.items():
        lb = arc.get((q, p))
        if not lb:
            continue
        tgt = max(la, lb)
        s = tgt / la
        if s - 1.0 > MAX_EASE:                     # a designed gather, left alone
            if report is not None:
                report.append((str(names[p]), str(names[q]), la, lb, s, "gather - not eased"))
            continue
        if abs(s - 1.0) > 1e-4:
            scales[(p, q)] = s
        if report is not None:
            report.append((str(names[p]), str(names[q]), la, lb, s,
                           "eased" if abs(s - 1.0) > 1e-4 else "already matched"))
    return scales


def eased_rest(d, band=5.0, report=None):
    """(T,3,2) rest triangles with the seam ease worked in.  `d["rest"]` is not
    modified: the geometry-image domain is exactly as it was."""
    R, faces = d["rest"], d["faces"]
    pid = np.maximum(d["panel_of_raw"], 0)
    part = _partners(d)
    bedges = _boundary(faces)
    scales = seam_scales(d, report)

    # per seam vertex: the scale it carries and the seam tangent in rest space
    vs = collections.defaultdict(lambda: 1.0)          # 1.0 = untouched, not 0.0
    vt = collections.defaultdict(lambda: np.zeros(2))
    for a, b in bedges:
        if pid[a] != pid[b]:
            continue
        pa = pid[a]
        common = {pid[x] for x in part[a]} & {pid[x] for x in part[b]} - {pa}
        t = R[b] - R[a]
        n = np.linalg.norm(t)
        if n < 1e-12:
            continue
        t = t / n
        # boundary edges are stored as (min, max), so their direction is
        # arbitrary; the stretch axis is a line, not a ray, so fix the sign or
        # consecutive edges cancel instead of accumulating
        if t[0] < 0.0 or (t[0] == 0.0 and t[1] < 0.0):
            t = -t
        for q in common:
            s = scales.get((pa, q))
            if s is None:
                continue
            for v in (a, b):
                if abs(s - 1.0) > abs(vs[v] - 1.0):
                    vs[v] = s
                vt[v] = vt[v] + t
    if not vs:
        return R[faces]

    idx = np.array(sorted(vs))
    sc = np.array([vs[i] for i in idx])
    tg = np.stack([vt[i] / max(np.linalg.norm(vt[i]), 1e-12) for i in idx])

    # nearest seam vertex WITHIN the same panel, so ease never leaks across a panel
    tri = R[faces].copy()
    cen = tri.mean(1)
    fpid = pid[faces[:, 0]]
    for p in np.unique(fpid):
        k = idx[pid[idx] == p]
        if not len(k):
            continue
        ft = np.where(fpid == p)[0]
        tree = cKDTree(R[k])
        dist, j = tree.query(cen[ft])
        w = np.maximum(0.0, 1.0 - dist / band)                 # graded to zero
        sub = np.searchsorted(idx, k)
        s = 1.0 + (sc[sub][j] - 1.0) * w
        u = tg[sub][j]
        # stretch each rest triangle by s along the seam tangent only
        v = tri[ft] - cen[ft][:, None, :]
        proj = np.einsum("tkd,td->tk", v, u)
        tri[ft] = cen[ft][:, None, :] + v + (s - 1.0)[:, None, None] * proj[:, :, None] * u[:, None, :]
    return tri
