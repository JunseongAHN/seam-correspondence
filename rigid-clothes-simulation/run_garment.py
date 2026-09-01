"""Isometric assembly of one real garment (spec 3).

  python run_garment.py --fast                  # quick look, short ladder
  python run_garment.py --seed 1                # one production run
  python run_garment.py --seed 1 --inflate 1.5  # the wider-basin probe

Initial state is ALWAYS the specification placement plus a random perturbation.
The drape is never an input.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import assembly as A
import body
import gcd_io
import plyio

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR_X = 0.0          # the specification places the garment about x = 0
RESULT = os.path.join(HERE, "result")
GARMENT = r"C:\Users\PC\Downloads\data\rand_00YONAPXZE"

DATA = os.path.dirname(GARMENT)
MEASURES = os.path.join(GARMENT, os.path.basename(GARMENT) + "_body_measurements.yaml")


def garment_dir(name):
    """accept either a full path or a bare id under the dataset directory"""
    return name if os.path.sep in name else os.path.join(DATA, name)


def measures_of(gdir):
    return os.path.join(gdir, os.path.basename(gdir) + "_body_measurements.yaml")

LADDER = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
FAST_LADDER = [1e-1, 1e-3, 1e-5, 1e-7]


def half_space_sets(d, lr=False, fb=False):
    """(mask, axis, sign) triples: sign=+1 means the panel must keep coord>=0."""
    pr = d["panel_of_raw"]
    nm = np.array(d["panel_names"], dtype=object)[np.maximum(pr, 0)]
    out = []
    if lr:                                   # placement puts left_* at x>0, right_* at x<0
        out.append((np.array([str(x).startswith("left_") or "_left_" in str(x) for x in nm]), 0, +1))
        out.append((np.array([str(x).startswith("right_") or "_right_" in str(x) for x in nm]), 0, -1))
    if fb:
        F = ("ftorso", "skirt_front", "wb_front")
        B = ("btorso", "skirt_back", "wb_back")
        out.append((np.array([any(k in str(x) for k in F) for x in nm]), 2, +1))
        out.append((np.array([any(k in str(x) for k in B) for x in nm]), 2, -1))
    return out


def half_space_penalty(d, gar, lr=False, fb=False, mu_rel=1.0):
    """One-sided quadratic penalty keeping each panel on its own side of a plane.

    Only panels that HAVE a side can be constrained.  skirt_front, skirt_back,
    wb_front and wb_back are single pieces spanning both halves, so they get no
    left/right constraint.  Sleeves, cuffs and the hood form tubes across the
    front/back plane, so they get no front/back constraint -- pinning them would
    flatten the tube rather than stop a collapse.

    Returns (mu, clamp).  mu is carried by EVERY vertex of a constrained panel,
    not just the violating ones, so the system matrix is independent of the
    active set and the single factorisation per lambda_b rung still holds; the
    active set enters only through the right-hand side (see assembly.solve).
    """
    sets = half_space_sets(d, lr, fb)
    if not sets:
        return None, None

    mu = np.zeros(gar["n"])
    for m, _, _ in sets:
        mu[m] = mu_rel * diag_scale(gar)

    def clamp(P):
        for m, ax, sg in sets:
            P[m & (sg * P[:, ax] < 0), ax] = 0.0
        return P
    return mu, clamp


def diag_scale(gar):
    """mean diagonal of the ARAP stiffness -- the natural unit for a penalty."""
    W = np.einsum("t,tuc,tuc->tu", gar["area"], gar["G"], gar["G"])
    dg = np.zeros(gar["n"])
    np.add.at(dg, gar["faces"].ravel(), W.ravel())
    return float(dg.mean())


def build(gdir=GARMENT, rest_override=None):
    d = gcd_io.load(gdir, rest_override=rest_override)
    rest_tri = d["rest"][d["faces"]]
    G, area = A.shape_gradients(rest_tri)
    H, R4 = A.build_hinges(d["faces"], d["wid"], rest_tri, d["panel_of_face"])
    Kb, wb = A.hinge_stencils(R4)
    gar = dict(faces=d["faces"], n=len(d["rest"]), G=G, area=area,
               hinges=H, Kb=Kb, wb=wb, pairs=d["pairs"])
    return d, gar


def initial(d, seed, inflate=1.0, amp=0.01, sym=False):
    """placement, optionally scaled about its centroid, plus isotropic noise"""
    P = d["placed"].copy()
    c = P.mean(0)
    P = c + inflate * (P - c)
    scale = float(np.linalg.norm(np.ptp(P, axis=0)))
    rng = np.random.default_rng(seed)
    if amp == 0.0:
        return P                                   # deterministic run
    if not sym:
        N = rng.standard_normal(P.shape)
    else:
        # The mesh is NOT mirror-symmetric (left_ftorso has 1108 vertices, right
        # 1117), so there is no vertex permutation to mirror through and exact
        # symmetry is unreachable.  What we CAN remove is the asymmetry the
        # perturbation itself injects: build the noise from a smooth random
        # field that is an even function of x about the garment's mid-plane.
        Q = P.copy()
        Q[:, 0] = np.abs(Q[:, 0] - MIRROR_X)        # even in x by construction
        Q = Q / max(scale, 1e-9)
        w = rng.standard_normal((24, 3)) * 6.0
        ph = rng.uniform(0, 2 * np.pi, (24, 3))
        A_ = rng.standard_normal((24, 3))
        N = np.einsum("kd,nkd->nd", np.ones((24, 3)) * 0 + 1.0,
                      np.cos(Q @ w.T[:, :, None].squeeze(-1)[:, :3].T if False else
                             (Q[:, None, :] * w[None, :, :]).sum(-1)[:, :, None] + ph[None])
                      * A_[None])
        N /= max(N.std(), 1e-12)
    return P + (amp * scale / np.sqrt(3)) * N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--amp", type=float, default=0.01)
    ap.add_argument("--inflate", type=float, default=1.0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--lam-stop", type=float, default=None,
                    help="stop the lambda_b ladder here instead of 1e-8")
    ap.add_argument("--lam-start", type=float, default=None,
                    help="begin the ladder above 1e-1, decade by decade.  The "
                         "placement puts the skirt panels flat and 30 cm apart, "
                         "an envelope whose bending energy is 3.2x that of the "
                         "round tube it should become; a stiffer opening rung is "
                         "what has to carry it out of that basin.")
    ap.add_argument("--mu", type=float, default=1.0,
                    help="half-space penalty weight, relative to the mean ARAP diagonal")
    ap.add_argument("--half-lr", action="store_true",
                    help="left_* panels stay at x>=0, right_* at x<=0 (centre panels free)")
    ap.add_argument("--half-fb", action="store_true",
                    help="front torso/skirt/wb stay at z>=0, back at z<=0 (sleeves/hood free)")
    ap.add_argument("--body", action="store_true",
                    help="keep the garment outside an analytic body proxy built from "
                         "<name>_body_measurements.yaml (NOT from the drape)")
    ap.add_argument("--anchor", type=float, default=0.0,
                    help="weight of a weak pull towards the specification placement, "
                         "relative to the mean ARAP diagonal; 0 = off")
    ap.add_argument("--sym", action="store_true",
                    help="mirror-symmetric perturbation, so the whole problem stays symmetric")
    ap.add_argument("--per-lambda", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--garment", default=None, help="garment id or directory")
    ap.add_argument("--outdir", default=None, help="where to write, default result/")
    ap.add_argument("--rest", default=None,
                    help="npy of replacement flat pattern coordinates; the "
                         "placement is rebuilt from them (perturb_pattern.py)")
    a = ap.parse_args()

    ladder = FAST_LADDER if a.fast else LADDER
    if a.lam_start and a.lam_start > ladder[0]:
        pre, w = [], a.lam_start
        while w > ladder[0]:
            pre.append(w)
            w /= 10.0
        ladder = pre + ladder
    if a.lam_stop:
        ladder = [x for x in ladder if x >= a.lam_stop]
    per_lambda = a.per_lambda if a.per_lambda else (120 if a.fast else 400)
    tag = a.tag or (("fast_" if a.fast else "") + ("sym_" if a.sym else "")
                    + ("lr_" if a.half_lr else "") + ("fb_" if a.half_fb else "")
                    + ("body_" if a.body else "")
                    + (("lam%g_" % a.lam_start) if a.lam_start else "")
                    + (("anch%g_" % a.anchor) if a.anchor else "") +
                    ("inflated" if a.inflate != 1.0 else "seed%d" % a.seed))

    log = lambda s: (sys.stdout.write(s + "\n"), sys.stdout.flush())
    t0 = time.time()
    gdir = garment_dir(a.garment) if a.garment else GARMENT
    outdir = a.outdir or RESULT
    ro = np.load(a.rest) if a.rest else None
    d, gar = build(gdir, rest_override=ro)
    log("built: %d verts, %d faces, %d hinges, %d seam pairs  (%.1fs)"
        % (gar["n"], len(gar["faces"]), len(gar["hinges"]), len(gar["pairs"]), time.time() - t0))

    P0 = initial(d, a.seed, a.inflate, amp=a.amp, sym=a.sym)
    t0 = time.time()
    pk = None
    if a.body:
        # the obstacle is fixed in the absolute placement frame, so the solve must
        # not re-centre the garment each iteration
        pk = body.pack(body.primitives(
            body.load_measurements(measures_of(gdir)), d["placed"],
            np.array(d["panel_names"], dtype=object)[np.maximum(d["panel_of_raw"], 0)],
            d["rest"]))
        clamp = body.projector(pk)
        mu = np.full(gar["n"], a.mu * diag_scale(gar))
        recenter = False
    else:
        mu, clamp = half_space_penalty(d, gar, lr=a.half_lr, fb=a.half_fb, mu_rel=a.mu)
        recenter = True
    nu = anchor = None
    if a.anchor:
        nu = np.full(gar["n"], a.anchor * diag_scale(gar))
        anchor = d["placed"]
        recenter = False
    P, asm, hist, viol, nfac = A.solve_annealed(
        gar, P0, ladder, per_lambda=per_lambda, max_iter=a.max_iter, log=log,
        mu=mu, clamp=clamp, nu=nu, anchor=anchor, recenter=recenter)
    secs = time.time() - t0

    e = hist[-1]
    gap = np.linalg.norm(P[d["pairs"][:, 0]] - P[d["pairs"][:, 1]], axis=1)
    F = A.deformation_gradients(P, gar["faces"], gar["G"])
    _, s = A.best_rotations(F)
    hs = {}
    if pk is not None:
        dep = body.penetration(P.copy(), pk)[0]
        hs["body"] = dict(n=int(gar["n"]), frac=float((dep > 0).mean()),
                          max_cm=float(dep.max()))
    for m, ax, sg in half_space_sets(d, lr=a.half_lr, fb=a.half_fb):
        v = np.maximum(-sg * P[m, ax], 0.0)
        hs["ax%d_sg%+d" % (ax, sg)] = dict(n=int(m.sum()), frac=float((v > 0).mean()),
                                           max_cm=float(v.max()) if len(v) else 0.0)
    meta = dict(tag=tag, seed=a.seed, inflate=a.inflate, ladder=ladder, mu_rel=a.mu,
                rest_override=a.rest, amp=a.amp,
                anchor=a.anchor, body=bool(a.body), half_space=hs,
                placed_dev_p50=float(np.median(np.linalg.norm(P - d["placed"], axis=1))),
                per_lambda=per_lambda, iterations=len(hist), factorizations=nfac,
                seconds=secs, mono_violations=len(viol),
                E_arap=e["E_arap"], E_bend=e["E_bend"], E_stitch=e["E_stitch"],
                max_sigma_dev=float(np.abs(s - 1).max()),
                p50_sigma_dev=float(np.median(np.abs(s - 1).max(1))),
                seam_gap_max=float(gap.max()), seam_gap_p50=float(np.median(gap)),
                history=hist[::5])
    os.makedirs(outdir, exist_ok=True)
    plyio.write_ply(os.path.join(outdir, "assembly_%s.ply" % tag), P, d["faces"], d["panel_of_raw"])
    np.save(os.path.join(outdir, "assembly_%s.npy" % tag), P)
    with open(os.path.join(outdir, "assembly_%s.json" % tag), "w") as f:
        json.dump(meta, f)
    log("%s: %d iters, %d fac, %.1f min | max|s-1| %.3e  gap max %.3e cm  mono viol %d"
        % (tag, len(hist), nfac, secs / 60, meta["max_sigma_dev"], meta["seam_gap_max"], len(viol)))


if __name__ == "__main__":
    main()
