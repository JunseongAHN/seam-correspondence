"""
How wrong may the predicted DEPTH be before the STITCHING decode breaks?

Setup: seam partners coincide exactly in the GT drape (verified).  A depth
predictor errs per panel per point, so the two copies of a seam point separate
along z by |d_A - d_B|.  We inject that error and ask whether nearest-neighbour
decode still lands on the CORRECT PARTNER SEAM.

Seam-level, not vertex-level: confusing a partner point with its neighbour one
vertex along the same seam is not a stitching error.  Correct == the nearest
copy shares a stitch id.

Noise models
  iid   : d ~ N(0, s) independently per copy      (optimistic)
  panel : one rigid z offset per panel copy-side  (pessimistic / systematic)
Scope: front-visible seam vertices only (front = max z), res 512 raster.
"""
import numpy as np, os, glob, sys, depth_oracle as D

AX_Z = 2
SIGMAS_CM = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]

def front_visible(W, si, res=512):
    ext = W.max(0) - W.min(0)
    lo = W[:, [0, 1]].min(0); step = (W[:, [0, 1]].max(0) - lo).max() / res
    ij = np.clip(np.floor((W[:, [0, 1]] - lo) / step).astype(np.int64), 0, res)
    pid = ij[:, 0] * (res + 1) + ij[:, 1]
    z = W[:, AX_Z]
    o = np.lexsort((-z, pid)); ps, zs = pid[o], z[o]
    first = np.r_[True, ps[1:] != ps[:-1]]
    fz = dict(zip(ps[first], zs[first]))
    return (np.abs(z - np.array([fz[p] for p in pid])) <= 1e-9)[si]

def run(root, n_garments, rng):
    out = {m: {s: [] for s in SIGMAS_CM} for m in ("iid", "panel")}
    per_garment = {m: {s: [] for s in SIGMAS_CM} for m in ("iid", "panel")}
    for gd in sorted(glob.glob(os.path.join(root, "rand_*")))[:n_garments]:
        o = D.load(gd)
        if o is None: continue
        W, lab = o
        si, sets = D.seam_sets(lab)
        if len(si) < 20: continue
        vis = front_visible(W, si)
        P = W[si][vis]; S = [sets[i] for i in np.where(vis)[0]]
        n = len(P)
        if n < 20: continue
        # two copies per seam point (96% of seam verts have multiplicity 2)
        base = np.repeat(P, 2, axis=0)
        owner = np.repeat(np.arange(n), 2)          # which seam point
        side  = np.tile([0, 1], n)                  # copy A / copy B
        Sx = [S[i] for i in owner]
        # same-seam mask: nearest neighbour is CORRECT if stitch ids intersect
        m2 = len(base)
        ok_mask = np.zeros((m2, m2), bool)
        for i in range(m2):
            si_ = Sx[i]
            ok_mask[i] = [not si_.isdisjoint(Sx[j]) for j in range(m2)]
        np.fill_diagonal(ok_mask, False)
        self_mask = np.eye(m2, dtype=bool)
        for model in ("iid", "panel"):
            for s in SIGMAS_CM:
                if model == "iid":
                    dz = rng.normal(0, s, m2)
                else:                                 # rigid per side
                    dz = np.where(side == 0, rng.normal(0, s), rng.normal(0, s))
                Q = base.copy(); Q[:, AX_Z] += dz
                Dm = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)
                Dm[self_mask] = np.inf
                nn = Dm.argmin(1)
                good = ok_mask[np.arange(m2), nn]
                out[model][s].append(good.mean())
                # seam-level: every copy of a seam decodes correctly
                seam_ok = []
                for sid in sorted({x for fs in S for x in fs}):
                    mem = np.array([sid in Sx[i] for i in range(m2)])
                    if mem.sum() < 2: continue
                    seam_ok.append(good[mem].all())
                if seam_ok:
                    per_garment[model][s].append(np.mean(seam_ok))
    return out, per_garment

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    out, pg = run(os.path.expanduser("~/mnt/data"), n, rng)
    print(f"garments={n}  front-visible seam vertices only, 2 copies each\n")
    print(f"{'sigma_z':>8} | {'iid pair':>9} {'iid seam':>9} | {'panel pair':>11} {'panel seam':>11}")
    print("-" * 60)
    for s in SIGMAS_CM:
        print(f"{s*10:>6.0f}mm | {np.mean(out['iid'][s]):>9.4f} {np.mean(pg['iid'][s]):>9.4f} "
              f"| {np.mean(out['panel'][s]):>11.4f} {np.mean(pg['panel'][s]):>11.4f}")
    print("\npair = nearest copy shares a stitch id; seam = ALL copies of that seam correct")
