"""Case 2: 120-degree sector of radius L = 3  ->  cone with sin(beta) = 1/3.

  python run_case2.py                 # primary + the lambda_b = 0 counterexample
  python run_case2.py --sweep         # lambda_b sweep (the apex is the interesting part)
  python run_case2.py --seam-hinges   # diagnostic: penalise bending across the seam too

The initial state is ALWAYS the flat sector plus a random z perturbation.  Nothing
is ever pre-rolled and nothing analytic is injected.
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
L, ANGLE = 3.0, 2.0 * np.pi / 3.0
LADDER = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


def initial(m, amp=0.01, seed=0):
    """Flat, plus a z perturbation of amplitude 0.01*L (spec 3.3)."""
    rng = np.random.default_rng(seed)
    return np.hstack([m.rest, amp * L * rng.standard_normal((m.n, 1))])


# The lambda_b ladder, the w_s schedule and per_lambda are IDENTICAL for both
# cases -- deliberately.  Anything tuned per case would be a free parameter per
# case, which is exactly what section 8 forbids.  Only the mesh differs.
def ladder_for(lam_b):
    if lam_b == 0.0:
        return [0.0]
    return [x for x in LADDER if x > lam_b] + [lam_b]


def run(tag, lam_b, seam_hinges=False, nr=40, ntheta=48, per_lambda=400, save=True, seed=0):
    m = meshmod.make_sector(L, ANGLE, nr, ntheta)
    P0 = initial(m, seed=seed)
    ladder = ladder_for(lam_b)
    t0 = time.time()
    P, asm, hist, viol, nfac = arap.solve_annealed(
        m, P0, ladder, seam_hinges=seam_hinges, w0=1e-2, w1=1e4, factor=2.0,
        iters_per_stage=10, per_lambda=per_lambda)
    secs = time.time() - t0

    ring = [1 + (nr // 2 - 1) * (ntheta + 1) + k for k in range(ntheta)]
    meta = dict(case=2, tag=tag, lam_b=lam_b, lam_ladder=ladder, seam_hinges=bool(seam_hinges),
                init="planar + 0.01*L random z (seed %d)" % seed,
                mesh=dict(L=L, angle=ANGLE, nr=nr, ntheta=ntheta), char_len=L, seconds=secs,
                factorizations=nfac, per_lambda=per_lambda,
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
    import verify
    f = verify.fit_cone(P, m.faces, asm.A)
    F = arap.deformation_gradients(P, m.faces, asm.G)
    _, s = arap.best_rotations(F)
    dev = np.abs(s - 1).max(1)
    rc = np.linalg.norm(m.rest[m.faces].mean(1), axis=1)
    far = dev[rc > 0.2 * L].max()                       # away from the apex
    g = np.linalg.norm(P[m.pairs[:, 0]] - P[m.pairs[:, 1]], axis=1).max()
    return f["sin_beta"], f["sin_beta_line"], float(dev.max()), float(far), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam-b", type=float, default=1e-8)
    ap.add_argument("--nr", type=int, default=40)
    ap.add_argument("--ntheta", type=int, default=48)
    ap.add_argument("--per-lambda", type=int, default=400)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--seam-hinges", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    if a.sweep:
        print("lambda_b sweep, case 2  (nr=%d ntheta=%d, seam hinge %s, planar init)"
              % (a.nr, a.ntheta, "INCLUDED" if a.seam_hinges else "excluded"))
        print("  at the apex the bending weight 3/A_e blows up while the ARAP weight A_t")
        print("  vanishes, so max|s-1| there is set almost entirely by lambda_b.")
        print("  %-9s %8s %11s %12s %11s %11s %10s"
              % ("lambda_b", "winding", "sin b (N)", "sin b (line)", "max|s-1|", "off-apex", "E_bend"))
        for lam in (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2):
            m, asm, P, meta = run("_sweep", lam, a.seam_hinges, a.nr, a.ntheta,
                                  a.per_lambda, save=False)
            sb, sl, ms, far, _ = quick(m, asm, P)
            print("  %-9g %8.4f %11.6f %12.6f %11.3e %11.3e %10.5g"
                  % (lam, meta["winding"], sb, sl, ms, far, meta["history"][-1]["E_bend"]))
        print("  analytic sin(beta) = 0.333333")
        return

    suffix = ("_seamhinge" if a.seam_hinges else "") + a.suffix
    for tag, lam in (("simulationresult2" + suffix, a.lam_b),
                     ("simulationresult2_nobend" + suffix, 0.0)):
        m, asm, P, meta = run(tag, lam, a.seam_hinges, a.nr, a.ntheta, a.per_lambda)
        sb, sl, ms, far, g = quick(m, asm, P)
        print("%-36s lam_b=%-8g it=%4d %5.1fs winding %6.3f  sin(b) %.6f / %.6f  max|s-1| %.2e (%.2e off-apex)  gap %.1e"
              % (tag + ".ply", lam, len(meta["history"]), meta["seconds"], meta["winding"],
                 sb, sl, ms, far, g))


if __name__ == "__main__":
    main()
