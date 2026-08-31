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
import gcd_io
import plyio

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result")
GARMENT = r"C:\Users\PC\Downloads\data\rand_00YONAPXZE"

LADDER = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
FAST_LADDER = [1e-1, 1e-3, 1e-5, 1e-7]


def build(garment_dir=GARMENT):
    d = gcd_io.load(garment_dir)
    rest_tri = d["rest"][d["faces"]]
    G, area = A.shape_gradients(rest_tri)
    H, R4 = A.build_hinges(d["faces"], d["wid"], rest_tri, d["panel_of_face"])
    Kb, wb = A.hinge_stencils(R4)
    gar = dict(faces=d["faces"], n=len(d["rest"]), G=G, area=area,
               hinges=H, Kb=Kb, wb=wb, pairs=d["pairs"])
    return d, gar


def initial(d, seed, inflate=1.0, amp=0.01):
    """placement, optionally scaled about its centroid, plus isotropic noise"""
    P = d["placed"].copy()
    c = P.mean(0)
    P = c + inflate * (P - c)
    scale = float(np.linalg.norm(np.ptp(P, axis=0)))
    rng = np.random.default_rng(seed)
    return P + (amp * scale / np.sqrt(3)) * rng.standard_normal(P.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--inflate", type=float, default=1.0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--per-lambda", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    ladder = FAST_LADDER if a.fast else LADDER
    per_lambda = a.per_lambda if a.per_lambda else (120 if a.fast else 400)
    tag = a.tag or (("fast_" if a.fast else "") +
                    ("inflated" if a.inflate != 1.0 else "seed%d" % a.seed))

    log = lambda s: (sys.stdout.write(s + "\n"), sys.stdout.flush())
    t0 = time.time()
    d, gar = build()
    log("built: %d verts, %d faces, %d hinges, %d seam pairs  (%.1fs)"
        % (gar["n"], len(gar["faces"]), len(gar["hinges"]), len(gar["pairs"]), time.time() - t0))

    P0 = initial(d, a.seed, a.inflate)
    t0 = time.time()
    P, asm, hist, viol, nfac = A.solve_annealed(
        gar, P0, ladder, per_lambda=per_lambda, max_iter=a.max_iter, log=log)
    secs = time.time() - t0

    e = hist[-1]
    gap = np.linalg.norm(P[d["pairs"][:, 0]] - P[d["pairs"][:, 1]], axis=1)
    F = A.deformation_gradients(P, gar["faces"], gar["G"])
    _, s = A.best_rotations(F)
    meta = dict(tag=tag, seed=a.seed, inflate=a.inflate, ladder=ladder,
                per_lambda=per_lambda, iterations=len(hist), factorizations=nfac,
                seconds=secs, mono_violations=len(viol),
                E_arap=e["E_arap"], E_bend=e["E_bend"], E_stitch=e["E_stitch"],
                max_sigma_dev=float(np.abs(s - 1).max()),
                p50_sigma_dev=float(np.median(np.abs(s - 1).max(1))),
                seam_gap_max=float(gap.max()), seam_gap_p50=float(np.median(gap)),
                history=hist[::5])
    os.makedirs(RESULT, exist_ok=True)
    plyio.write_ply(os.path.join(RESULT, "assembly_%s.ply" % tag), P, d["faces"], d["panel_of_raw"])
    np.save(os.path.join(RESULT, "assembly_%s.npy" % tag), P)
    with open(os.path.join(RESULT, "assembly_%s.json" % tag), "w") as f:
        json.dump(meta, f)
    log("%s: %d iters, %d fac, %.1f min | max|s-1| %.3e  gap max %.3e cm  mono viol %d"
        % (tag, len(hist), nfac, secs / 60, meta["max_sigma_dev"], meta["seam_gap_max"], len(viol)))


if __name__ == "__main__":
    main()
