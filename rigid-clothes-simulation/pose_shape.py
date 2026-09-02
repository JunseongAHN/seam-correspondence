"""Is the across-run spread shape, or is it pose?

  python pose_shape.py

The spread reported so far aligns the whole garment once and then takes a
per-vertex standard deviation.  Under that measure an arm hung a few degrees
differently in each run shows up as large disagreement at the wrist, even if the
cuff itself comes out identically every time -- the same person photographed six
times and registered on the torso, arms raised differently: the fingertips are
far apart, the hand is the same shape.

So align each GROUP on its own and see how much of the spread survives.  What
disappears between one level and the next is the pose variance of that joint.

  whole garment  ->  sub-assembly  ->  panel

Rotation only.  Allowing scale would hide real stretch and reflection is not
physical, so the determinant is corrected.  The alignment rotation is reported in
degrees as well, because "the shoulder wanders 4 degrees between runs" is easier
to act on than a table of centimetres.

The skirt is handled separately: it is 69% of the vertices and it has lobes
rather than a joint, so it also gets a 1-degree-of-freedom test -- align on
rotation about the body's vertical axis alone.  If the spread collapses under
that, the runs differ by where the lobes sit, not by what they are.
"""

import json
import os

import numpy as np

import analyze
import gcd_io
import run_garment as RG

RUNS = ["seed1", "seed2", "seed3", "seed4", "seed5", "inflated"]
SEEDS = RUNS[:5]

SUB = {
    "arm_L": ("left_sleeve_f", "left_sleeve_b", "sl_left_cuff_f", "sl_left_cuff_b"),
    "arm_R": ("right_sleeve_f", "right_sleeve_b", "sl_right_cuff_f", "sl_right_cuff_b"),
    "hood": ("left_hood", "right_hood"),
    "torso": ("left_ftorso", "right_ftorso", "left_btorso", "right_btorso"),
    "skirt": ("skirt_front", "skirt_back", "wb_front", "wb_back"),
}


def kabsch(A, B):
    """rotation taking centred A onto centred B; reflection excluded"""
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def angle_of(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def aligned_spread(Ps, ref, idx):
    """align every run on idx alone, then the per-vertex std across runs"""
    B = Ps[ref][idx] - Ps[ref][idx].mean(0)
    Q, angs = [], []
    for P in Ps:
        A = P[idx] - P[idx].mean(0)
        R = kabsch(A, B)
        Q.append(A @ R.T)
        angs.append(angle_of(R))
    Q = np.stack(Q)
    std = np.linalg.norm(Q.std(0), axis=1)
    dev = np.linalg.norm(Q - Q.mean(0), axis=2).max(0)
    return std, dev, np.array(angs)


def main():
    d = gcd_io.load(RG.GARMENT)
    pn = np.array(d["panel_names"], dtype=object)[np.maximum(d["panel_of_raw"], 0)]
    cls = analyze.class_of_raw(d)
    load = lambda t: np.load(os.path.join(RG.RESULT, "assembly_sym_body_ease5_%s.npy" % t))
    ALL = [load(t) for t in RUNS]
    FIVE = [load(t) for t in SEEDS]

    def globally(Ps):
        ref = Ps[0]
        return np.stack([analyze.procrustes(P, ref) for P in Ps])

    out = {}
    for setname, Ps in (("6 runs", ALL), ("5 seeds", FIVE)):
        G = globally(Ps)
        gstd = np.linalg.norm(G.std(0), axis=1)

        # sub-assembly level
        sub_std = np.full(len(pn), np.nan)
        sub_ang = {}
        for name, keys in SUB.items():
            idx = np.where(np.array([str(x) in keys for x in pn]))[0]
            if not len(idx):
                continue
            s, _, a = aligned_spread(Ps, 0, idx)
            sub_std[idx] = s
            sub_ang[name] = a

        # panel level
        pan_std = np.full(len(pn), np.nan)
        pan_ang = {}
        for nm in sorted(set(str(x) for x in pn)):
            idx = np.where(pn == nm)[0]
            s, _, a = aligned_spread(Ps, 0, idx)
            pan_std[idx] = s
            pan_ang[nm] = a

        print("=" * 96)
        print("%s -- per-vertex spread (cm) at three alignment levels" % setname)
        print("=" * 96)
        print("  %-14s %10s %14s %12s %10s   %s"
              % ("class", "global", "sub-assembly", "panel", "n", "panel align angle p50/max"))
        rows = {}
        for c in sorted(set(cls)):
            k = cls == c
            if not k.any():
                continue
            nm = [x for x in sorted(set(str(y) for y in pn[k]))]
            ang = np.concatenate([pan_ang[x] for x in nm])
            rows[c] = (float(np.median(gstd[k])), float(np.nanmedian(sub_std[k])),
                       float(np.nanmedian(pan_std[k])), int(k.sum()))
            print("  %-14s %10.3f %14.3f %12.3f %10d   %6.2f / %6.2f"
                  % (c, rows[c][0], rows[c][1], rows[c][2], rows[c][3],
                     np.median(ang), ang.max()))
        print("  %-14s %10.3f %14.3f %12.3f %10d"
              % ("OVERALL", np.median(gstd), np.nanmedian(sub_std),
                 np.nanmedian(pan_std), len(pn)))
        print()
        print("  sub-assembly alignment rotation, degrees (this is the joint play):")
        for name in sorted(sub_ang):
            a = sub_ang[name]
            print("    %-8s p50 %6.2f  max %6.2f    per run: %s"
                  % (name, np.median(a), a.max(), " ".join("%.1f" % x for x in a)))
        print()
        out[setname] = rows

    # ---- the skirt on its own -------------------------------------------
    print("=" * 96)
    print("the skirt: lobes, and a one-degree-of-freedom alignment")
    print("=" * 96)
    sk = np.where(np.array([str(x).startswith("skirt") for x in pn]))[0]
    print("  %-10s %8s %10s %12s" % ("run", "lobes", "hem r p50", "hem r p90"))
    for t, P in zip(RUNS, ALL):
        S = P[sk]
        c = S.mean(0)
        hem = S[S[:, 1] < np.quantile(S[:, 1], 0.05)]
        r = np.hypot(hem[:, 0] - c[0], hem[:, 2] - c[2])
        th = np.arctan2(hem[:, 2] - c[2], hem[:, 0] - c[0])
        o = np.argsort(th)
        rr = r[o] - r[o].mean()
        lobes = int((np.diff(np.sign(rr)) != 0).sum() // 2)
        print("  %-10s %8d %10.2f %12.2f" % (t, lobes, np.median(r), np.quantile(r, .9)))

    # 1-dof: rotation about y only, angle chosen to minimise the residual
    print()
    print("  %-10s %12s %14s %14s" % ("set", "global std", "3-dof Kabsch", "1-dof about y"))
    for setname, Ps in (("6 runs", ALL), ("5 seeds", FIVE)):
        G = globally(Ps)
        g = float(np.median(np.linalg.norm(G.std(0), axis=1)[sk]))
        s3 = float(np.median(aligned_spread(Ps, 0, sk)[0]))
        B = Ps[0][sk] - Ps[0][sk].mean(0)
        Q, best = [], []
        for P in Ps:
            A = P[sk] - P[sk].mean(0)
            bt, bv = 0.0, np.inf
            for t in np.linspace(-np.pi, np.pi, 721):
                c, s = np.cos(t), np.sin(t)
                Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                v = float(((A @ Ry.T - B) ** 2).sum())
                if v < bv:
                    bv, bt = v, t
            c, s = np.cos(bt), np.sin(bt)
            Q.append(A @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T)
            best.append(np.degrees(bt))
        s1 = float(np.median(np.linalg.norm(np.stack(Q).std(0), axis=1)))
        print("  %-10s %12.3f %14.3f %14.3f    best y-angles: %s"
              % (setname, g, s3, s1, " ".join("%.1f" % x for x in best)))

    with open(os.path.join(RG.RESULT, "pose_shape.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
