"""Read a GarmentCodeData garment: welded mesh, panel membership, boundary loops.

Verified structure (2026-08-31):
  _sim.ply stores each seam vertex once per incident panel at the SAME xyz
  (only UV differs).  _sim_segmentation.txt indexes the welded list in
  first-occurrence order and labels each vertex with either a panel name or
  `stitch_k` (comma-separated at junctions).
"""
from collections import defaultdict
import os
import numpy as np

AX_X, AX_Y, AX_Z = 0, 1, 2


def load(garment_dir, name=None):
    """-> (welded_xyz[n,3], welded_faces[f,3], labels[n])"""
    g = name or os.path.basename(garment_dir.rstrip("/\\"))
    raw = open(os.path.join(garment_dir, g + "_sim.ply"), "rb").read()
    end = raw.find(b"end_header\n") + 11
    hdr = raw[:end].decode()
    nv = int([l for l in hdr.split("\n") if l.startswith("element vertex")][0].split()[-1])
    nf = int([l for l in hdr.split("\n") if l.startswith("element face")][0].split()[-1])
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("s", "<f8"), ("t", "<f8")])
    V = np.frombuffer(raw, dtype=dt, count=nv, offset=end)
    xyz = np.stack([V["x"], V["y"], V["z"]], 1).astype(np.float64)
    F = np.frombuffer(raw, dtype=np.dtype([("n", "u1"), ("v", "<i4", 3)]),
                      count=nf, offset=end + nv * dt.itemsize)["v"]
    labels = np.array([l.strip() for l in
                       open(os.path.join(garment_dir, g + "_sim_segmentation.txt"))], dtype=object)

    key = np.ascontiguousarray(xyz).view([("a", "<f8"), ("b", "<f8"), ("c", "<f8")]).ravel()
    _, first, inv, cnt = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
    order = np.argsort(first)                       # first-occurrence == segmentation order
    pos = np.empty(len(cnt), np.int64); pos[order] = np.arange(len(order))
    wid = pos[inv]
    if len(labels) != len(cnt):
        raise ValueError(f"{g}: segmentation {len(labels)} != welded {len(cnt)}")
    W = np.zeros((len(labels), 3)); W[wid] = xyz
    return W, wid[F], labels


def panel_membership(faces, labels, sweeps=2):
    """Panel names + per-welded-vertex set of panel indices.

    Non-seam vertices carry their panel name directly; seam vertices carry
    `stitch_k` instead and inherit the panel indices of their mesh neighbours.
    """
    n = len(labels)
    is_seam = np.array([str(l).startswith("stitch") for l in labels])
    panels = sorted({str(l) for l in labels[~is_seam]})
    idx = {p: k for k, p in enumerate(panels)}
    memb = [set() for _ in range(n)]
    for i in np.where(~is_seam)[0]:
        memb[i].add(idx[str(labels[i])])
    for _ in range(sweeps):
        for f in faces:
            u = set().union(*[memb[v] for v in f])
            for v in f:
                if is_seam[v]:
                    memb[v] |= u
    return panels, memb, is_seam


def boundary_loops(faces):
    """Edges used by exactly one face, chained into loops. -> list of vertex-id lists."""
    cnt = defaultdict(int)
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            cnt[(min(a, b), max(a, b))] += 1
    adj = defaultdict(list)
    for (a, b), c in cnt.items():
        if c == 1:
            adj[a].append(b); adj[b].append(a)
    seen, out = set(), []
    for start in list(adj):
        if start in seen:
            continue
        loop = [start]; seen.add(start); prev, cur = None, start
        while True:
            nxt = [v for v in adj[cur] if v != prev and v not in seen]
            if not nxt:
                break
            loop.append(nxt[0]); seen.add(nxt[0]); prev, cur = cur, nxt[0]
        if len(loop) >= 4:
            out.append(loop)
    return out


def panel_loops(W, faces, labels):
    """-> (panels, [(panel_index, [welded vertex ids]), ...])"""
    panels, memb, _ = panel_membership(faces, labels)
    loops = []
    for k in range(len(panels)):
        own = np.array([k in memb[v] for v in range(len(labels))])
        sel = own[faces].all(1)
        if sel.sum() < 4:
            continue
        for L in boundary_loops(faces[sel]):
            loops.append((k, L))
    return panels, loops


def stitch_sets(labels):
    """welded vertex id -> frozenset of stitch ids (seam vertices only)."""
    return {i: frozenset(str(labels[i]).split(","))
            for i in range(len(labels)) if str(labels[i]).startswith("stitch")}


def front_visible(W, ids, res=512):
    """Is this vertex the front-most (max z) in its pixel of an orthographic front view?"""
    lo = W[:, [AX_X, AX_Y]].min(0)
    step = (W[:, [AX_X, AX_Y]].max(0) - lo).max() / res
    ij = np.clip(np.floor((W[:, [AX_X, AX_Y]] - lo) / step).astype(np.int64), 0, res)
    pid = ij[:, 0] * (res + 1) + ij[:, 1]
    z = W[:, AX_Z]
    o = np.lexsort((-z, pid)); ps, zs = pid[o], z[o]
    first = np.r_[True, ps[1:] != ps[:-1]]
    fz = dict(zip(ps[first], zs[first]))
    front = np.array([fz[p] for p in pid])
    return (np.abs(z - front) <= 1e-9)[ids]
