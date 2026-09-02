"""Read a GarmentCodeData garment into the pieces an isometric-assembly solver needs.

What comes out of `_sim.ply`, and what does NOT:

  faces        the triangulation (topology only)
  (s,t) UV     the FLAT pattern parametrisation -> rest material coordinates
  welding      vertices sharing an xyz are the SAME seam point stored once per
               incident panel; this gives the seam CORRESPONDENCE, not a shape

  xyz          the draped shape.  NEVER an input to the solver.  It is loaded
               only so the finished assembly can be compared against it
               afterwards (spec 4(e)), and it is returned under `drape` so that
               any accidental use is visible at the call site.

The initial state for the solver is the PLACEMENT built from the specification's
per-panel rotation/translation, never the drape.
"""

import json
import os
import pickle
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# low-level readers
# ---------------------------------------------------------------------------

def _read_sim_ply(path):
    raw = open(path, "rb").read()
    end = raw.find(b"end_header\n") + 11
    hdr = raw[:end].decode()
    nv = int([l for l in hdr.split("\n") if l.startswith("element vertex")][0].split()[-1])
    nf = int([l for l in hdr.split("\n") if l.startswith("element face")][0].split()[-1])
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("s", "<f8"), ("t", "<f8")])
    V = np.frombuffer(raw, dtype=dt, count=nv, offset=end)
    xyz = np.stack([V["x"], V["y"], V["z"]], 1).astype(np.float64)
    uv = np.stack([V["s"], V["t"]], 1)
    F = np.frombuffer(raw, dtype=np.dtype([("n", "u1"), ("v", "<i4", 3)]),
                      count=nf, offset=end + nv * dt.itemsize)["v"].astype(np.int64)
    return xyz, uv, F


def _weld(xyz):
    """Vertices with byte-identical xyz are copies of one seam point.
    Returns raw->welded index and the welded count, in segmentation order."""
    key = np.ascontiguousarray(xyz).view([("a", "<f8"), ("b", "<f8"), ("c", "<f8")]).ravel()
    _, first, inv, cnt = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
    order = np.argsort(first)                       # first occurrence == segmentation order
    pos = np.empty(len(cnt), np.int64)
    pos[order] = np.arange(len(order))
    return pos[inv], len(cnt), cnt[np.argsort(order)] if False else cnt


def euler_xyz(r):
    """Specification convention: intrinsic x-y-z, degrees, no flips."""
    rx, ry, rz = np.deg2rad(np.asarray(r, float))
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# UV -> material coordinates
# ---------------------------------------------------------------------------

def calibrate_uv_scale(uv, faces, wid, orig_lens):
    """The UV is normalised independently per axis, so material coordinates are
    (s*Kx, t*Ky) with Kx != Ky.  `_orig_lens.pickle` gives the simulator's REST
    length for a set of welded edges -- that is the metric the drape was
    computed from, so it is the right calibration target.

    Fit is linear in (Kx^2, Ky^2) on squared lengths.  Returns (Kx, Ky, report).
    """
    ol = {(int(min(a, b)), int(max(a, b))): float(v) for (a, b), v in orig_lens.items()}
    du, dv, L = [], [], []
    seen = set()
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            k = (min(wid[a], wid[b]), max(wid[a], wid[b]))
            if k in ol and (min(a, b), max(a, b)) not in seen:
                seen.add((min(a, b), max(a, b)))
                du.append(uv[a, 0] - uv[b, 0])
                dv.append(uv[a, 1] - uv[b, 1])
                L.append(ol[k])
    du, dv, L = np.array(du), np.array(dv), np.array(L)
    A = np.stack([du ** 2, dv ** 2], 1)
    sol, *_ = np.linalg.lstsq(A, L ** 2, rcond=None)
    Kx, Ky = np.sqrt(np.abs(sol))
    err = np.abs(np.hypot(du * Kx, dv * Ky) / L - 1.0)
    iso, *_ = np.linalg.lstsq(np.hypot(du, dv)[:, None], L, rcond=None)
    iso_err = np.abs(np.hypot(du, dv) * iso[0] / L - 1.0)
    rep = dict(n_edges=int(len(L)), Kx=float(Kx), Ky=float(Ky),
               err_median=float(np.median(err)), err_p90=float(np.quantile(err, 0.9)),
               isotropic_K=float(iso[0]), isotropic_err_median=float(np.median(iso_err)))
    return float(Kx), float(Ky), rep


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def connected_components(faces, n):
    """In the RAW mesh each panel is its own component: seam vertices are stored
    once per incident panel, so nothing connects two panels."""
    adj = [[] for _ in range(n)]
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            adj[a].append(b)
            adj[b].append(a)
    comp = np.full(n, -1, np.int64)
    c = 0
    for s in range(n):
        if comp[s] != -1 or not adj[s]:
            continue
        comp[s] = c
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = c
                    q.append(v)
        c += 1
    return comp, c


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def load(garment_dir, name=None, rest_override=None):
    """rest_override replaces the flat pattern coordinates.  The placement is
    rebuilt from them, since a design change moves both -- see
    perturb_pattern.py, which uses this to measure whether nearby patterns give
    nearby shells."""
    g = name or os.path.basename(garment_dir.rstrip("/\\"))
    P = lambda suf: os.path.join(garment_dir, g + suf)

    xyz, uv, faces = _read_sim_ply(P("_sim.ply"))
    wid, nw, mult = _weld(xyz)
    labels = np.array([l.strip() for l in open(P("_sim_segmentation.txt"))], dtype=object)
    if len(labels) != nw:
        raise ValueError("segmentation %d != welded %d" % (len(labels), nw))
    spec = json.load(open(P("_specification.json")))["pattern"]
    orig_lens = pickle.load(open(P("_orig_lens.pickle"), "rb"))

    Kx, Ky, uv_report = calibrate_uv_scale(uv, faces, wid, orig_lens)
    rest = np.stack([uv[:, 0] * Kx, uv[:, 1] * Ky], 1)          # cm, flat material coords
    if rest_override is not None:
        rest = np.asarray(rest_override, float)
        if rest.shape != (len(xyz), 2):
            raise ValueError("rest_override must be (%d, 2)" % len(xyz))

    # --- panels: one connected component per panel in the raw mesh ----------
    comp, ncomp = connected_components(faces, len(xyz))
    lab_raw = labels[wid]
    names = spec["panel_order"]
    idx_of = {nm: i for i, nm in enumerate(names)}
    comp_to_panel = {}
    for c in range(ncomp):
        sel = comp == c
        vals = [str(l) for l in lab_raw[sel] if not str(l).startswith("stitch")]
        if not vals:
            raise ValueError("component %d has no panel-labelled vertex" % c)
        u, k = np.unique(vals, return_counts=True)
        comp_to_panel[c] = idx_of[u[k.argmax()]]
    panel_of_raw = np.array([comp_to_panel.get(c, -1) for c in comp], np.int64)
    panel_of_face = panel_of_raw[faces[:, 0]]
    if not (panel_of_raw[faces] == panel_of_face[:, None]).all():
        raise ValueError("a face spans two panels -- raw mesh is not panel-separated")

    # --- seam pairs: the duplicates the welding revealed --------------------
    ordr = np.argsort(wid, kind="stable")
    pairs = []
    i = 0
    while i < len(ordr):
        j = i
        while j + 1 < len(ordr) and wid[ordr[j + 1]] == wid[ordr[i]]:
            j += 1
        if j > i:                                    # link every copy to the first
            for k in range(i + 1, j + 1):
                pairs.append((ordr[i], ordr[k]))
        i = j + 1
    pairs = np.array(pairs, np.int64).reshape(-1, 2)

    # --- placement: the specification's per-panel rigid transform -----------
    placed = np.zeros_like(xyz)
    for pi, nm in enumerate(names):
        sel = panel_of_raw == pi
        if not sel.any():
            continue
        R = euler_xyz(spec["panels"][nm]["rotation"])
        t = np.array(spec["panels"][nm]["translation"], float)
        poly = np.array(spec["panels"][nm]["vertices"], float)
        # the UV atlas places each panel by a translation only, so matching the
        # panel's material centroid to the specification polygon's centroid
        # recovers the panel's own 2D frame up to that translation
        loc = rest[sel] - rest[sel].mean(0) + poly.mean(0)
        placed[sel] = (np.hstack([loc, np.zeros((sel.sum(), 1))]) @ R.T) + t

    return dict(
        name=g, faces=faces, rest=rest, uv=uv, wid=wid, n_welded=nw,
        labels=labels, panel_names=names, panel_of_raw=panel_of_raw,
        panel_of_face=panel_of_face, pairs=pairs, placed=placed,
        spec=spec, Kx=Kx, Ky=Ky, uv_report=uv_report, multiplicity=mult,
        drape=xyz,                     # comparison target ONLY (spec 4(e))
    )


PANEL_CLASS = {
    "wb_front": "wb_front", "wb_back": "wb_back",
    "skirt_front": "skirt_front", "skirt_back": "skirt_back",
    "left_ftorso": "ftorso", "right_ftorso": "ftorso",
    "left_btorso": "btorso", "right_btorso": "btorso",
    "left_sleeve_f": "sleeve_f", "right_sleeve_f": "sleeve_f",
    "left_sleeve_b": "sleeve_b", "right_sleeve_b": "sleeve_b",
    "sl_left_cuff_f": "cuff_f", "sl_right_cuff_f": "cuff_f",
    "sl_left_cuff_b": "cuff_b", "sl_right_cuff_b": "cuff_b",
    "left_hood": "hood", "right_hood": "hood",
}


if __name__ == "__main__":
    import sys
    d = load(sys.argv[1] if len(sys.argv) > 1
             else r"C:\Users\PC\Downloads\data\rand_00YONAPXZE")
    r = d["uv_report"]
    print("garment %s" % d["name"])
    print("  raw verts %d   welded %d   faces %d   seam pairs %d"
          % (len(d["rest"]), d["n_welded"], len(d["faces"]), len(d["pairs"])))
    print("  UV scale  Kx=%.4f  Ky=%.4f cm   (fit on %d edges vs orig_lens)"
          % (r["Kx"], r["Ky"], r["n_edges"]))
    print("  rest edge-length error vs orig_lens: median %.4f%%  p90 %.4f%%"
          % (r["err_median"] * 100, r["err_p90"] * 100))
    print("  isotropic single-scale would give median %.2f%% -- anisotropic is required"
          % (r["isotropic_err_median"] * 100))
    print("  panels %d:" % len(d["panel_names"]))
    for i, nm in enumerate(d["panel_names"]):
        n = int((d["panel_of_raw"] == i).sum())
        print("    %-16s %6d verts   class %s" % (nm, n, PANEL_CLASS[nm]))
    print("  placement bbox  x[%.1f %.1f] y[%.1f %.1f] z[%.1f %.1f] cm"
          % (d["placed"][:, 0].min(), d["placed"][:, 0].max(),
             d["placed"][:, 1].min(), d["placed"][:, 1].max(),
             d["placed"][:, 2].min(), d["placed"][:, 2].max()))
