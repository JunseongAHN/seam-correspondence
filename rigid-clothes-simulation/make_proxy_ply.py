"""Tessellate the analytic body proxy and write it next to the panels, so the
proxy can be eyeballed before it is used as a constraint.

  python make_proxy_ply.py        ->  result/proxy/*.ply
"""

import os

import numpy as np

import body
import gcd_io
import plyio
import run_garment as RG

OUT = os.path.join(RG.HERE, "result", "proxy")
NSEG = 48


def cylinder_mesh(p0, p1, r):
    """A plain circular cylinder with flat caps -- exactly the shape the
    constraint enforces, so what is drawn is what is enforced."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = p1 - p0
    n = n / np.linalg.norm(n)
    u = np.array([1.0, 0.0, 0.0])
    if abs(n @ u) > 0.9:
        u = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(n, u)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    th = np.linspace(0.0, 2.0 * np.pi, NSEG, endpoint=False)
    ring = r * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    V = np.vstack([p0 + ring, p1 + ring, p0[None], p1[None]])
    c0, c1 = 2 * NSEG, 2 * NSEG + 1

    F = []
    for i in range(NSEG):
        j = (i + 1) % NSEG
        F.append([i, j, NSEG + j])
        F.append([i, NSEG + j, NSEG + i])
        F.append([c0, j, i])                      # bottom cap
        F.append([c1, NSEG + i, NSEG + j])        # top cap
    return V, np.array(F, np.int64)


def sphere_mesh(c, r, nu=32, nv=16):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, np.pi, nv)
    V = np.stack([(r * np.sin(v)[:, None] * np.cos(u)[None]).ravel(),
                  (r * np.cos(v)[:, None] * np.ones_like(u)[None]).ravel(),
                  (r * np.sin(v)[:, None] * np.sin(u)[None]).ravel()], 1) + np.asarray(c)
    F = []
    for a in range(nv - 1):
        for i in range(nu):
            j = (i + 1) % nu
            F.append([a * nu + i, a * nu + j, (a + 1) * nu + j])
            F.append([a * nu + i, (a + 1) * nu + j, (a + 1) * nu + i])
    return V, np.array(F, np.int64)


def build_proxy(prim):
    C, S = prim
    V, F = [], []
    off = 0
    for c in C:
        v, f = cylinder_mesh(*c)
        V.append(v); F.append(f + off); off += len(v)
    for c in S:
        v, f = sphere_mesh(*c)
        V.append(v); F.append(f + off); off += len(v)
    return np.concatenate(V), np.concatenate(F)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = gcd_io.load(RG.GARMENT)
    pn = np.array(d["panel_names"], dtype=object)[np.maximum(d["panel_of_raw"], 0)]
    m = body.load_measurements(RG.MEASURES)
    prim = body.primitives(m, d["placed"], pn)
    C, S = prim
    Vb, Fb = build_proxy(prim)
    plyio.write_ply(os.path.join(OUT, "body_proxy.ply"), Vb, Fb, np.full(len(Vb), 7))

    k = body.aspect_ratio(m)
    print("aspect a/b = %.3f  ->  cylinder radius = the ellipse semi-minor axis" % k)
    print("%d cylinders + %d sphere(s):" % (len(C), len(S)))
    for p0, p1, r in C:
        print("  x %+6.1f -> %+6.1f  y %6.1f -> %6.1f  z %+5.1f -> %+5.1f   r %5.2f  (d %5.1f cm)"
              % (p0[0], p1[0], p0[1], p1[1], p0[2], p1[2], r, 2 * r))

    for c, r in S:
        print("  SPHERE centre %s  r %5.2f  (d %5.1f cm)" % (np.round(c, 1), r, 2 * r))
    pk = body.pack(prim)
    for nm in ("placed", "drape"):
        X = d[nm]
        dep = body.penetration(X.copy(), pk)[0]
        print("%-7s inside the proxy: %5d / %d (%5.2f%%)  max depth %.2f cm"
              % (nm, (dep > 0).sum(), len(X), 100 * (dep > 0).mean(), max(dep.max(), 0.0)))
        V = np.vstack([X, Vb])
        F = np.vstack([d["faces"], Fb + len(X)])
        pid = np.concatenate([d["panel_of_raw"], np.full(len(Vb), 7)])
        plyio.write_ply(os.path.join(OUT, "%s_with_body.ply" % nm), V, F, pid)

    # front and back alone against the body, which is what the constraint is for
    for side, keys in (("front", ("ftorso", "skirt_front", "wb_front")),
                       ("back", ("btorso", "skirt_back", "wb_back"))):
        keep = np.array([any(q in str(x) for q in keys) for x in pn])
        idx = np.where(keep)[0]
        ren = -np.ones(len(pn), np.int64)
        ren[idx] = np.arange(len(idx))
        fk = d["faces"][keep[d["faces"]].all(1)]
        V = np.vstack([d["placed"][idx], Vb])
        F = np.vstack([ren[fk], Fb + len(idx)])
        pid = np.concatenate([d["panel_of_raw"][idx], np.full(len(Vb), 7)])
        plyio.write_ply(os.path.join(OUT, "placed_%s_with_body.ply" % side), V, F, pid)
        print("wrote placed_%s_with_body.ply  (%d verts)" % (side, len(idx)))


if __name__ == "__main__":
    main()
