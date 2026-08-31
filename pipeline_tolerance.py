"""
CORRECT pipeline. The representation under test must be traversed.

  3D panel boundary points (2 copies per seam vertex, one per incident panel)
    -> add 3D prediction error
    -> PROJECT into the front depth map   (xy -> pixel grid, z-buffer keeps front-most)
    -> LIFT back                          (pixel centre, depth stored at that pixel)
    -> decode correspondence on the lifted points

Failure modes this exposes that a 3D-only test cannot:
  occlusion  - a copy is not the front-most in its pixel, so it reads someone else's depth
  merge      - two points that must NOT be sewn share a pixel and become coincident
  split      - noise moves partners into different pixels
"""
import numpy as np, os, glob, sys, depth_oracle as D
from depth_tolerance import front_visible
AX_Z = 2

def project_lift(pts, res, frame):
    """front depth map (front = max z), then lift every point through it."""
    (x0, x1, y0, y1) = frame
    step = max(x1 - x0, y1 - y0) / res
    ix = np.floor((pts[:, 0] - x0) / step).astype(np.int64)
    iy = np.floor((pts[:, 1] - y0) / step).astype(np.int64)
    ix = np.clip(ix, 0, res); iy = np.clip(iy, 0, res)
    pid = ix * (res + 1) + iy
    z = pts[:, AX_Z]
    buf = {}
    o = np.lexsort((-z, pid)); ps, zs = pid[o], z[o]
    first = np.r_[True, ps[1:] != ps[:-1]]
    buf = dict(zip(ps[first], zs[first]))
    front = np.array([buf[p] for p in pid])
    occluded = front > z + 1e-9
    L = np.stack([x0 + (ix + .5) * step, y0 + (iy + .5) * step, front], 1)
    return L, occluded, pid, step

def run(root, n, res_list=(256, 512, 1024), sigmas=(0.5,1,2,3,5,10), R=6):
    rng = np.random.default_rng(31)
    out = {(r, s): [] for r in res_list for s in sigmas}
    used = 0
    for gd in sorted(glob.glob(os.path.join(root, "rand_*")))[:n]:
        o = D.load(gd)
        if o is None: continue
        W, lab = o; si, sets = D.seam_sets(lab)
        if len(si) < 20: continue
        vis = front_visible(W, si)
        P = W[si][vis]; S = [sets[i] for i in np.where(vis)[0]]
        if len(P) < 20: continue
        used += 1
        ids = sorted({x for fs in S for x in fs})
        other = np.delete(W, si, axis=0)                 # rest of the garment (occluders)
        base = np.repeat(P, 2, axis=0); m2 = len(base)
        Sx = [S[i] for i in np.repeat(np.arange(len(P)), 2)]
        M = np.zeros((m2, len(ids)), bool)
        for k, sd in enumerate(ids): M[:, k] = [sd in Sx[i] for i in range(m2)]
        ok = (M.astype(np.uint8) @ M.astype(np.uint8).T) > 0
        np.fill_diagonal(ok, False); eye = np.eye(m2, dtype=bool)
        big = [k for k in range(len(ids)) if M[:, k].sum() > 1]
        frame = (W[:,0].min(), W[:,0].max(), W[:,1].min(), W[:,1].max())
        for s_mm in sigmas:
            s = s_mm / 10.0
            for r in res_list:
                acc = []
                for _ in range(R):
                    Qs = base + rng.normal(0, s, (m2, 3))
                    Qo = other + rng.normal(0, s, other.shape)
                    allp = np.vstack([Qs, Qo])
                    L, occ, pid, step = project_lift(allp, r, frame)
                    Ls, occs = L[:m2], occ[:m2]
                    Dm = np.linalg.norm(Ls[:, None, :] - Ls[None, :, :], axis=2)
                    Dm[eye] = np.inf
                    nn = Dm.argmin(1); good = ok[np.arange(m2), nn]
                    acc.append((good.mean(),
                                np.mean([good[M[:,k]].all()       for k in big]),
                                np.mean([good[M[:,k]].mean()>0.5  for k in big]),
                                occs.mean(), step*10))
                out[(r, s_mm)].append(np.mean(acc, axis=0))
    return out, used

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out, used = run(os.path.expanduser("~/mnt/data"), n)
    print(f"garments={used}   3D noise -> depth projection -> lift -> decode\n")
    print(f"{'res':>5} {'px mm':>6} {'sigma':>7} {'occluded':>9} {'vert':>7} {'seam-all':>9} {'seam-vote':>10} {'garment~':>9}")
    print("-" * 74)
    for r in (256, 512, 1024):
        for s in (0.5,1,2,3,5,10):
            a = np.mean(out[(r, s)], axis=0)
            print(f"{r:>5} {a[4]:>6.2f} {s:>6.1f}mm {a[3]:>9.3f} {a[0]:>7.4f} {a[1]:>9.4f} {a[2]:>10.4f} {a[2]**30:>9.4f}")
        print()
