"""Case 1: rectangle W = 2*pi, H = 4  ->  cylinder of radius exactly 1.

  python run_case1.py                 # primary + the lambda_b = 0 counterexample
  python run_case1.py --sweep         # the lambda_b sweep of section 4.2
  python run_case1.py --seam-hinges   # diagnostic: penalise bending across the seam too

The initial state is ALWAYS the flat panel plus a random z perturbation.  Nothing
is ever pre-rolled and nothing analytic is injected: the roll-up has to be found
by the solver.
"""

import argparse
import json
import os
import time

import numpy as np

import arap
import mesh as meshmod

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result")
W, H = 2.0 * np.pi, 4.0
LADDER = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


def initial(m, amp=0.01, seed=0):
    """Flat, plus a z perturbation of amplitude 0.01*W (spec 2.3).

    The perturbation is not optional: at the exactly flat state every optimal
    R_t has a zero third row, so the ARAP right-hand side has no z component and
    z stays 0 forever.  Flat is an exact fixed point of the iteration."""
    rng = np.random.default_rng(seed)
    return np.hstack([m.rest, amp * W * rng.standard_normal((m.n, 1))])


# The lambda_b ladder, the w_s schedule and per_lambda are IDENTICAL for both
# cases -- deliberately.  Anything tuned per case would be a free parameter per
# case, which is exactly what section 8 forbids.  Only the mesh differs.
def ladder_for(lam_b):
    """lambda_b continuation, stiff -> target.  A single small lambda_b started
    from flat does NOT find the cylinder (the sheet folds flat, cross-section
    winding number 0); starting stiff picks the one-tube mode, and softening
    afterwards removes the lambda_b radius bias."""
    if lam_b == 0.0:
        return [0.0]
    return [L for L in LADDER if L > lam_b] + [lam_b]


def run(tag, lam_b, seam_hinges=False, nw=64, nh=32, per_lambda=400, save=True, seed=0):
    m = meshmod.make_rectangle(W, H, nw, nh)
    P0 = initial(m, seed=seed)
    ladder = ladder_for(lam_b)
    t0 = time.time()
    P, asm, hist, viol, nfac = arap.solve_annealed(
        m, P0, ladder, seam_hinges=seam_hinges, w0=1e-2, w1=1e4, factor=2.0,
        iters_per_stage=10, per_lambda=per_lambda)
    secs = time.time() - t0

    ring = [(nh // 2) * (nw + 1) + a for a in range(nw)]
    meta = dict(case=1, tag=tag, lam_b=lam_b, lam_ladder=ladder, seam_hinges=bool(seam_hinges),
                init="planar + 0.01*W random z (seed %d)" % seed,
                mesh=dict(W=W, H=H, nw=nw, nh=nh), char_len=W, seconds=secs,
                factorizations=nfac, per_lambda=per_lambda,
                poly_radius=float((np.pi / nw) / np.sin(np.pi / nw)),
                winding=arap.winding_number(P, ring),
                mono_violations=len(viol), history=hist)
    if save:
        meshmod.write_ply(os.path.join(HERE, tag + ".ply"), P, m.faces)
        with open(os.path.join(HERE, tag + ".json"), "w") as f:
            json.dump(meta, f)
        os.makedirs(RESULT, exist_ok=True)
        meshmod.write_ply(os.path.join(RESULT, tag + "_input.ply"), P0, m.faces)
        meshmod.write_ply(os.path.join(RESULT, tag + "_output.ply"), P, m.faces)
    return m, asm, P, meta


def quick(m, asm, P):
    X = P - P.mean(0)
    ax = np.linalg.eigh(X.T @ X)[1][:, -1]
    d = np.linalg.norm(X - np.outer(X @ ax, ax), axis=1)
    F = arap.deformation_gradients(P, m.faces, asm.G)
    _, s = arap.best_rotations(F)
    g = np.linalg.norm(P[m.pairs[:, 0]] - P[m.pairs[:, 1]], axis=1).max()
    return d.mean(), d.std(), float(np.abs(s - 1).max()), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam-b", type=float, default=1e-8)
    ap.add_argument("--nw", type=int, default=64)
    ap.add_argument("--nh", type=int, default=32)
    ap.add_argument("--per-lambda", type=int, default=400)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--seam-hinges", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    if a.sweep:
        poly = (np.pi / a.nw) / np.sin(np.pi / a.nw)
        print("section 4.2  lambda_b sweep, case 1  (nw=%d, seam hinge %s, planar init)"
              % (a.nw, "INCLUDED" if a.seam_hinges else "excluded"))
        print("  exact-isometry polygon radius for this mesh: %.6f" % poly)
        print("  the prediction under test is r = poly / (1 + 3*lambda_b), i.e. the")
        print("  minimiser DOES move with lambda_b because ARAP is a soft penalty.")
        print("  %-9s %8s %10s %10s %10s %11s %11s"
              % ("lambda_b", "winding", "radius", "std", "max|s-1|", "E_bend", "poly/(1+3L)"))
        for lam in (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
            m, asm, P, meta = run("_sweep", lam, a.seam_hinges, a.nw, a.nh,
                                  a.per_lambda, save=False)
            r, sd, ms, _ = quick(m, asm, P)
            print("  %-9g %8.4f %10.6f %10.3e %10.3e %11.5g %11.6f"
                  % (lam, meta["winding"], r, sd, ms, meta["history"][-1]["E_bend"],
                     poly / (1 + 3 * lam)))
        return

    suffix = ("_seamhinge" if a.seam_hinges else "") + a.suffix
    for tag, lam in (("simulationresult1" + suffix, a.lam_b),
                     ("simulationresult1_nobend" + suffix, 0.0)):
        m, asm, P, meta = run(tag, lam, a.seam_hinges, a.nw, a.nh, a.per_lambda)
        r, sd, ms, g = quick(m, asm, P)
        print("%-36s lam_b=%-8g it=%4d %5.1fs winding %6.3f  radius %.6f +- %.6f  max|s-1| %.2e  gap %.1e"
              % (tag + ".ply", lam, len(meta["history"]), meta["seconds"], meta["winding"],
                 r, sd, ms, g))


if __name__ == "__main__":
    main()
