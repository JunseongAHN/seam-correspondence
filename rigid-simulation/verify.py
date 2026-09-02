"""Verification for the two analytically-known assembly cases (spec section 5).

Everything here is measured from the resulting point cloud alone -- no estimator
is seeded with the analytic answer.
"""

import json
import os
import sys

import numpy as np

import arap
import mesh as meshmod

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# 5.1 / 5.5  principal stretches and the ARAP identity
# --------------------------------------------------------------------------

def stretch_report(m, P):
    G, A = arap.shape_gradients(m.rest, m.faces)
    F = arap.deformation_gradients(P, m.faces, G)
    R, s = arap.best_rotations(F)
    lhs = np.einsum("tdc,tdc->t", F - R, F - R)
    rhs = ((s - 1.0) ** 2).sum(1)
    ident = float(np.abs(lhs - rhs).max())
    assert ident < 1e-9, "identity broken: max err %.3e" % ident
    s1, s2 = s[:, 0], s[:, 1]
    dev = np.abs(s - 1.0).max(axis=1)
    return dict(
        s1=[float(s1.min()), float(np.median(s1)), float(s1.max())],
        s2=[float(s2.min()), float(np.median(s2)), float(s2.max())],
        max_dev=float(dev.max()), p50_dev=float(np.median(dev)),
        p999_dev=float(np.quantile(dev, 0.999)),
        worst_face=int(dev.argmax()), identity_err=ident, dev=dev,
    )


# --------------------------------------------------------------------------
# 5.3  seam gap
# --------------------------------------------------------------------------

def seam_report(m, P, char_len):
    d = np.linalg.norm(P[m.pairs[:, 0]] - P[m.pairs[:, 1]], axis=1)
    return dict(max=float(d.max()), p50=float(np.median(d)),
                tol=1e-4 * char_len, ok=bool(d.max() < 1e-4 * char_len))


# --------------------------------------------------------------------------
# 5.4  case 1: cylinder
# --------------------------------------------------------------------------

def face_normals(P, faces):
    a = P[faces[:, 1]] - P[faces[:, 0]]
    b = P[faces[:, 2]] - P[faces[:, 0]]
    n = np.cross(a, b)
    return n / np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-300)


def cylinder_report(m, P):
    """Axis by PCA (as the spec asks) and, as a cross-check, by the null
    direction of the face-normal covariance.  The second does not depend on the
    height/radius aspect ratio; PCA does, and silently returns the wrong
    eigenvector once H^2/12 < r^2/2."""
    X = P - P.mean(0)
    evals, evecs = np.linalg.eigh(X.T @ X / len(X))
    ax_pca = evecs[:, -1]

    N = face_normals(P, m.faces)
    _, nv = np.linalg.eigh(N.T @ N)
    ax_nrm = nv[:, 0]
    if ax_nrm @ ax_pca < 0:
        ax_nrm = -ax_nrm

    out = {"axis_angle_deg": float(np.degrees(np.arccos(np.clip(abs(ax_pca @ ax_nrm), -1, 1)))),
           "pca_var_along": float(evals[-1]), "pca_var_across": float(evals[0] + evals[1])}
    for tag, ax in (("pca", ax_pca), ("normal", ax_nrm)):
        t = X @ ax
        d = np.linalg.norm(X - np.outer(t, ax), axis=1)
        out["radius_" + tag] = [float(d.mean()), float(d.std()), float(d.min()), float(d.max())]
        out["height_" + tag] = float(t.max() - t.min())
    return out


def cross_section_turning(m, P, meta):
    """Turning angle of the mid-height cross-section polygon.  A circular
    cylinder turns by 360/nw at every vertex; a crease shows up as one large
    angle at the seam (vertex 0)."""
    nw, nh = meta["mesh"]["nw"], meta["mesh"]["nh"]
    idx = [(nh // 2) * (nw + 1) + a for a in range(nw)]     # drop the duplicate seam copy
    Q = P[idx]
    e1 = np.roll(Q, -1, 0) - Q
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e0 = Q - np.roll(Q, 1, 0)
    e0 /= np.linalg.norm(e0, axis=1)[:, None]
    t = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", e0, e1), -1, 1)))
    return dict(seam=float(t[0]), median=float(np.median(t)), max=float(t.max()),
                uniform=360.0 / nw, argmax=int(t.argmax()))


# --------------------------------------------------------------------------
# 5.4  case 2: cone
# --------------------------------------------------------------------------

def fit_cone(P, faces, areas=None):
    """Cone (apex, unit axis, half-angle beta) from the surface alone.

    Two independent closed-form estimators, no analytic seeding:

    (a) Every unit normal of a cone of half-angle beta satisfies  N . axis =
        -sin(beta)  EXACTLY, whatever the apex is.  So the axis is the direction
        of least variance of the normals and sin(beta) is |mean(N . axis)|.
        Scale free, apex free, and it degrades gracefully to a cylinder
        (sin beta -> 0).

    (b) With the axis known, the cross-sectional radius is affine in the axial
        coordinate: rho(t) = tan(beta) * (t - t_apex).  A straight-line fit of
        rho against t therefore gives tan(beta) and the apex independently.

    An apex/axis/beta fit that minimises the algebraic residual
    (p-a).n - |p-a| cos(beta) is NOT usable here: pushing the apex to infinity
    drives every such residual to zero, so the estimator collapses to
    sin(beta) = 0 with a tiny residual.  Both estimators below are free of that.
    """
    N = face_normals(P, faces)
    w = np.ones(len(N)) if areas is None else np.abs(areas)
    w = w / w.sum()
    Nm = (w[:, None] * N).sum(0)
    C = (w[:, None] * N).T @ N - np.outer(Nm, Nm)
    _, ev = np.linalg.eigh(C)
    ax = ev[:, 0]                                       # least-variance direction
    proj = N @ ax
    if proj.mean() > 0:                                 # point the axis apex->base
        ax, proj = -ax, -proj
    sin_beta_n = float(-(w * proj).sum())

    t = P @ ax
    Pp = P - np.outer(t, ax)
    ctr = Pp.mean(0)
    rho = np.linalg.norm(Pp - ctr, axis=1)
    A = np.stack([t, np.ones_like(t)], 1)
    slope, inter = np.linalg.lstsq(A, rho, rcond=None)[0]
    tan_beta_l = float(abs(slope))
    sin_beta_l = tan_beta_l / np.sqrt(1.0 + tan_beta_l ** 2)
    t_apex = float(-inter / slope) if abs(slope) > 1e-12 else float(t.min())
    apex = ctr + t_apex * ax

    resid = rho - (slope * t + inter)
    return dict(apex=apex, axis=ax, sin_beta=sin_beta_n, sin_beta_line=float(sin_beta_l),
                t_apex=t_apex, rho_rms_residual=float(np.sqrt(np.mean(resid ** 2))),
                normal_proj_std=float(np.sqrt(max(0.0, (w * (proj + sin_beta_n) ** 2).sum()))))


def cone_report(m, P):
    G, A = arap.shape_gradients(m.rest, m.faces)
    fit = fit_cone(P, m.faces, A)
    a, n = fit["apex"], fit["axis"]
    v = P - a
    slant = np.linalg.norm(v, axis=1)
    rim = slant > 0.999 * slant.max()                   # the sector's outer arc
    t = v[rim] @ n
    d = np.linalg.norm(v[rim] - np.outer(t, n), axis=1)
    sb = fit["sin_beta"]
    return dict(sin_beta=sb, sin_beta_line=fit["sin_beta_line"],
                beta_deg=float(np.degrees(np.arcsin(min(1.0, sb)))),
                base_radius=float(d.mean()), base_radius_std=float(d.std()),
                height=float(t.mean()), slant_max=float(slant.max()),
                normal_proj_std=fit["normal_proj_std"],
                rms_residual=fit["rho_rms_residual"])


# --------------------------------------------------------------------------
# 5.6  monotone decrease
# --------------------------------------------------------------------------

def monotone_report(hist):
    """Both steps are exact minimizations at fixed weights, so E must not
    increase *within* a continuation stage.  It jumps up at a stage boundary
    because w_s just grew; that is the schedule, not a bug."""
    worst, n_bad = 0.0, 0
    for a, b in zip(hist[:-1], hist[1:]):
        if a["stage"] != b["stage"] or b["E"] <= a["E"]:
            continue
        rel = (b["E"] - a["E"]) / max(abs(a["E"]), 1e-30)
        worst = max(worst, rel)
        if rel > 1e-9:
            n_bad += 1
    return dict(violations=n_bad, worst_rel_increase=worst)


def energy_curve(hist, k=8):
    n = len(hist)
    picks = sorted(set([0] + [int(round(i * (n - 1) / (k - 1))) for i in range(k)]))
    return [(hist[i]["it"], hist[i]["w_s"], hist[i]["E_arap"], hist[i]["E_bend"],
             hist[i]["E_stitch"]) for i in picks]


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def load(tag):
    P, F = meshmod.read_ply(os.path.join(HERE, tag + ".ply"))
    with open(os.path.join(HERE, tag + ".json")) as f:
        meta = json.load(f)
    return P, F, meta


def build_mesh(meta):
    c = meta["mesh"]
    if meta["case"] == 1:
        return meshmod.make_rectangle(c["W"], c["H"], c["nw"], c["nh"])
    return meshmod.make_sector(c["L"], c["angle"], c["nr"], c["ntheta"])


def report_one(tag, curve=True):
    P, F, meta = load(tag)
    m = build_mesh(meta)
    st = stretch_report(m, P)
    sm = seam_report(m, P, meta["char_len"])
    mo = monotone_report(meta["history"])
    e = meta["history"][-1]
    ok = lambda c: "PASS" if c else "FAIL"

    print("=" * 82)
    print("%s   case %d | lambda_b=%g | seam hinge %s | init=%s"
          % (tag, meta["case"], meta["lam_b"],
             "INCLUDED" if meta["seam_hinges"] else "excluded (spec 1.5)", meta["init"]))
    print("=" * 82)
    print("  iterations %d   factorizations %d   wall %.1fs   w_s %g -> %g"
          % (len(meta["history"]), meta["factorizations"], meta["seconds"],
             meta["history"][0]["w_s"], meta["history"][-1]["w_s"]))
    print("  lambda_b ladder %s" % " -> ".join("%g" % x for x in meta.get("lam_ladder", [meta["lam_b"]])))
    print("  winding number of the mid cross-section %.4f   (1 = one tube; 0 = folded flat)"
          % abs(meta.get("winding", float("nan"))))
    print("  5.1 stretch   sigma1 min/p50/max %.6f / %.6f / %.6f" % tuple(st["s1"]))
    print("                sigma2 min/p50/max %.6f / %.6f / %.6f" % tuple(st["s2"]))
    print("                max|sigma-1| %.3e   p50 %.3e   p99.9 %.3e     %s (< 1e-3)"
          % (st["max_dev"], st["p50_dev"], st["p999_dev"], ok(st["max_dev"] < 1e-3)))
    print("  5.5 identity  max| ||F-R||^2 - (s1-1)^2-(s2-1)^2 | = %.3e   PASS" % st["identity_err"])
    print("  5.2 energy    E_ARAP %.6e   E_bend %.6e   E_stitch %.6e"
          % (e["E_arap"], e["E_bend"], e["E_stitch"]))
    if curve:
        print("                iter      w_s        E_ARAP        E_bend      E_stitch")
        for it, w, ea, eb, es in energy_curve(meta["history"]):
            print("                %4d %8.3g  %12.5e %12.5e %12.5e" % (it, w, ea, eb, es))
    print("  5.3 seam gap  max %.3e   p50 %.3e   tol %.1e                 %s"
          % (sm["max"], sm["p50"], sm["tol"], ok(sm["ok"])))
    print("  5.6 monotone  in-stage violations %d (worst rel. rise %.2e)      %s"
          % (mo["violations"], mo["worst_rel_increase"], ok(mo["violations"] == 0)))

    if meta["case"] == 1:
        cy = cylinder_report(m, P)
        rp, rn = cy["radius_pca"], cy["radius_normal"]
        poly = meta["poly_radius"]
        tw = cross_section_turning(m, P, meta)
        print("  5.4 cylinder  PCA axis vs normal-covariance axis differ by %.4f deg" % cy["axis_angle_deg"])
        print("                radius (PCA)    mean %.6f std %.6f min %.6f max %.6f" % tuple(rp))
        print("                radius (normal) mean %.6f std %.6f min %.6f max %.6f" % tuple(rn))
        print("                height %.6f (analytic 4)" % cy["height_pca"])
        print("                |mean radius - 1| = %.3e                        %s (< 1e-3)"
              % (abs(rp[0] - 1.0), ok(abs(rp[0] - 1.0) < 1e-3)))
        print("                radius std        = %.3e                        %s (< 1e-3)"
              % (rp[1], ok(rp[1] < 1e-3)))
        print("                cross-section turning: seam %.3f deg, median %.3f deg,"
              % (tw["seam"], tw["median"]))
        print("                  max %.3f deg at vertex %d  (a circle turns %.3f at every vertex)"
              % (tw["max"], tw["argmax"], tw["uniform"]))
        print("                [discretization] an exact isometry puts the vertices on a regular")
        print("                  %d-gon of edge W/%d, circumradius %.6f -- so 1.000000 is NOT"
              % (meta["mesh"]["nw"], meta["mesh"]["nw"], poly))
        print("                  attainable. residual vs the polygon radius: %+.3e" % (rp[0] - poly))
        # RMS radial error against the analytic cylinder: captures BOTH the bias
        # and the spread.  Using the std alone would flatter a uniformly collapsed
        # tube (radius 0.19 everywhere has a small std but is completely wrong).
        X = P - P.mean(0)
        axp = np.linalg.eigh(X.T @ X)[1][:, -1]
        dd = np.linalg.norm(X - np.outer(X @ axp, axp), axis=1)
        shape_err = float(np.sqrt(np.mean((dd - 1.0) ** 2)))
        shape_name = "rms|r-1|"
        print("                rms radial error vs the analytic r=1 cylinder: %.3e" % shape_err)
    else:
        co = cone_report(m, P)
        print("  5.4 cone      sin(beta) %.6f  (analytic 0.333333)  err %.3e   %s (< 1e-3)"
              % (co["sin_beta"], abs(co["sin_beta"] - 1 / 3), ok(abs(co["sin_beta"] - 1 / 3) < 1e-3)))
        print("                beta %.4f deg (analytic 19.4712)" % co["beta_deg"])
        print("                base radius %.6f +- %.6f (analytic 1)" % (co["base_radius"], co["base_radius_std"]))
        print("                height      %.6f            (analytic 2.828427)" % co["height"])
        print("                cross-check sin(beta) from the rho-vs-t line fit: %.6f (err %.3e)"
              % (co["sin_beta_line"], abs(co["sin_beta_line"] - 1 / 3)))
        print("                normal-projection std %.3e   rho-fit rms residual %.3e"
              % (co["normal_proj_std"], co["rms_residual"]))
        shape_err, shape_name = abs(co["sin_beta"] - 1 / 3), "|sin b - 1/3|"

    return dict(tag=tag, case=meta["case"], lam_b=meta["lam_b"], seam=meta["seam_hinges"],
                E_arap=e["E_arap"], E_bend=e["E_bend"], E_stitch=e["E_stitch"],
                max_dev=st["max_dev"], shape_err=shape_err, shape_name=shape_name,
                gap=sm["max"])


DEFAULT = ["simulationresult1", "simulationresult1_nobend",
           "simulationresult2", "simulationresult2_nobend"]


def main():
    tags = sys.argv[1:] or DEFAULT
    rows = []
    for t in tags:
        if os.path.exists(os.path.join(HERE, t + ".ply")):
            rows.append(report_one(t))
            print()
        else:
            print("(missing %s.ply -- run the case script first)" % t)
    if not rows:
        return
    print("=" * 82)
    print("SECTION 4.1  --  does isometry alone pick the shape?")
    print("=" * 82)
    print("  %-38s %10s %10s %11s %11s" % ("run", "E_ARAP", "max|s-1|", "E_bend", "shape err"))
    for r in rows:
        print("  %-38s %10.3e %10.3e %11.4e %11.3e"
              % (r["tag"], r["E_arap"], r["max_dev"], r["E_bend"], r["shape_err"]))
    print()
    print("  Read the columns pairwise (lambda_b = 0 vs lambda_b > 0 for the same case):")
    print("  E_ARAP and max|sigma-1| are comparable -- both states are (near-)isometric,")
    print("  so ISOMETRY ALONE DOES NOT PICK THE SHAPE.  E_bend and the shape error are")
    print("  what separate them: BENDING PICKS THE SHAPE.")


if __name__ == "__main__":
    main()
