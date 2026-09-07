"""
Oracle round-trip gate for the depth-map pi_2 design.

Question: if we represent the draped garment as front+back orthographic depth
maps and lift back to 3D, do seam vertices survive well enough to be decoded
by proximity?  This is the ZERO-training-error upper bound.  If it fails here,
no model helps.

Also emits the Level-1 margin distribution m(v) from the handoff spec.

Ground truth structure (verified 2026-08-31):
  - _sim.ply stores each seam vertex once per incident panel, at the SAME xyz
    (different UV).  Welded (first-occurrence) order indexes _sim_segmentation.txt.
  - stitch label gives the seam grouping directly; no coordinate matching needed.
"""
import numpy as np, os, sys, glob

def load(gdir):
    g = os.path.basename(gdir)
    raw = open(os.path.join(gdir, g + "_sim.ply"), "rb").read()
    end = raw.find(b"end_header\n") + 11
    hdr = raw[:end].decode()
    nv = int([l for l in hdr.split("\n") if l.startswith("element vertex")][0].split()[-1])
    dt = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("s","<f8"),("t","<f8")])
    V = np.frombuffer(raw, dtype=dt, count=nv, offset=end)
    xyz = np.stack([V["x"], V["y"], V["z"]], 1).astype(np.float64)
    lab = np.array([l.strip() for l in
                    open(os.path.join(gdir, g + "_sim_segmentation.txt"))], dtype=object)
    key = np.ascontiguousarray(xyz).view([("a","<f8"),("b","<f8"),("c","<f8")]).ravel()
    _, first, inv, cnt = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
    order = np.argsort(first)
    pos = np.empty(len(cnt), np.int64); pos[order] = np.arange(len(order))
    wid = pos[inv]                       # ply index -> welded id
    W = np.zeros((len(lab), 3))          # welded coordinates
    W[wid] = xyz
    if len(lab) != len(cnt):
        return None
    return W, lab

def seam_sets(lab):
    idx = np.array([i for i, l in enumerate(lab) if l.startswith("stitch")])
    sets = [frozenset(lab[i].split(",")) for i in idx]
    return idx, sets

def margins(P, sets):
    """m(v) = distance to nearest seam vertex whose stitch set is DISJOINT from v's."""
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    n = len(P)
    same = np.zeros((n, n), bool)
    for i in range(n):
        si = sets[i]
        same[i] = [not si.isdisjoint(sets[j]) for j in range(n)]
    D[same] = np.inf
    return D.min(1)

def roundtrip(W, seam_idx, res, axes):
    """Orthographic front+back depth at `res`, then lift. Returns per-seam-vertex
    (visible, position error, collided-with-other-seam)."""
    u, v, d = axes                      # image-x, image-y, depth axes
    P = W[:, [u, v, d]]
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    span = (hi - lo).max()
    step = span / res
    ij = np.floor((P[:, :2] - lo) / step).astype(np.int64)
    ij = np.clip(ij, 0, res)
    pid = ij[:, 0] * (res + 1) + ij[:, 1]
    z = P[:, 2]
    # per-pixel min (front) and max (back) depth
    order = np.lexsort((z, pid))
    ps, zs = pid[order], z[order]
    first = np.r_[True, ps[1:] != ps[:-1]]
    last  = np.r_[ps[1:] != ps[:-1], True]
    front = dict(zip(ps[first], zs[first]))
    back  = dict(zip(ps[last],  zs[last]))
    tol = 1e-9
    fz = np.array([front[p] for p in pid]); bz = np.array([back[p] for p in pid])
    vis = (np.abs(z - fz) <= tol) | (np.abs(z - bz) <= tol)
    # lifted position: pixel centre + own depth (only meaningful if visible)
    L = np.empty_like(P)
    L[:, 0] = lo[0] + (ij[:, 0] + 0.5) * step
    L[:, 1] = lo[1] + (ij[:, 1] + 0.5) * step
    L[:, 2] = z
    err = np.linalg.norm(L - P, axis=1)
    return vis[seam_idx], err[seam_idx], L[seam_idx], step

def main(root, n_garments=20, resolutions=(128, 256, 512)):
    dirs = sorted(glob.glob(os.path.join(root, "rand_*")))[:n_garments]
    all_m, rows = [], {r: [] for r in resolutions}
    for gd in dirs:
        out = load(gd)
        if out is None:
            continue
        W, lab = out
        seam_idx, sets = seam_sets(lab)
        if len(seam_idx) < 10:
            continue
        P = W[seam_idx]
        m = margins(P, sets)
        all_m.append(m)
        ext = W.max(0) - W.min(0)
        axes = tuple(np.argsort(-ext))          # (largest, mid, smallest) -> img x,y / depth
        axes = (axes[1], axes[0], axes[2])      # x=mid, y=largest(up), depth=smallest
        for r in resolutions:
            vis, err, L, step = roundtrip(W, seam_idx, r, axes)
            # collision: lifted point within step/2 of a DISJOINT-seam lifted point
            D = np.linalg.norm(L[:, None, :] - L[None, :, :], axis=2)
            n = len(L)
            same = np.zeros((n, n), bool)
            for i in range(n):
                si = sets[i]
                same[i] = [not si.isdisjoint(sets[j]) for j in range(n)]
            D[same] = np.inf
            coll = D.min(1) < step
            rows[r].append((vis.mean(), np.percentile(err, [50, 90, 99]), coll.mean(), step))
    M = np.concatenate(all_m)
    print(f"garments used: {len(all_m)}   seam vertices: {len(M)}")
    print(f"\n=== Level-1 MARGIN m(v)  [same units as mesh] ===")
    for p in (50, 25, 10, 1):
        print(f"  p{p:<3d} = {np.percentile(M, 100-p if False else p):.4f}")
    print(f"  min  = {M.min():.4f}   mean = {M.mean():.4f}")
    print(f"\n=== ORACLE ROUND-TRIP (front+back orthographic depth) ===")
    print(f"{'res':>5} {'visible':>9} {'err p50':>9} {'err p90':>9} {'err p99':>9} {'collide':>9} {'pixel':>9}")
    for r in resolutions:
        a = rows[r]
        vis = np.mean([x[0] for x in a])
        e = np.mean([x[1] for x in a], axis=0)
        c = np.mean([x[2] for x in a])
        s = np.mean([x[3] for x in a])
        print(f"{r:>5} {vis:>9.3f} {e[0]:>9.4f} {e[1]:>9.4f} {e[2]:>9.4f} {c:>9.3f} {s:>9.4f}")
    print("\nNOTE visibility is OPTIMISTIC (vertex-level z-buffer, no triangle occlusion).")

if __name__ == "__main__":
    main(os.path.expanduser("~/mnt/data"),
         n_garments=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
