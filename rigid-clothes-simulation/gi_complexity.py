"""How much capacity does a geometry image of this shape need?

  python gi_complexity.py [garment_id]

A geometry image is a field of 3D positions over the flat panel.  Our assembly
and the GarmentCode drape are two such fields over the SAME domain and the SAME
mesh, so their complexity can be compared directly, with nothing else changing.

Three measures, cheapest first.

1. Characteristic wavelength.  The Rayleigh quotient of the field against the
   cotangent Laplacian of the FLAT panel, f'Lf / f'Mf, is a mean squared spatial
   frequency in 1/cm^2; 2*pi over its root is the wavelength of the features the
   field carries.  Long means smooth means a small receptive field suffices.

2. Modal reconstruction.  Take the first k eigenvectors of L phi = lambda M phi
   on the flat panel -- the domain's own Fourier basis -- project the field onto
   them, and measure the RMS error in centimetres.  This is capacity in the most
   literal sense: how many numbers per panel are needed to say where the surface
   is, to a stated tolerance.

3. Metric distortion.  |sigma - 1| of the map from the flat panel to 3D.  Near
   isometry means a kernel of fixed size in the image covers a fixed physical
   size on the surface, so a convolution means the same thing everywhere.
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import assembly as A
import gcd_io
import run_garment as RG

PANELS = ("skirt_front", "left_ftorso", "left_sleeve_f", "left_hood", "wb_front")
KS = (10, 30, 100, 300)


def cotan_laplacian(V2, faces, n):
    """L (positive semi-definite) and lumped mass M, on the FLAT panel."""
    T = V2[faces]
    e0, e1, e2 = T[:, 2] - T[:, 1], T[:, 0] - T[:, 2], T[:, 1] - T[:, 0]
    area = 0.5 * np.abs(np.cross(e2, -e1))
    cot = np.stack([-(e1 * e2).sum(1), -(e2 * e0).sum(1), -(e0 * e1).sum(1)], 1) / \
        (4.0 * np.maximum(area, 1e-12))[:, None]
    I, J, W = [], [], []
    for k, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
        i, j, w = faces[:, a], faces[:, b], cot[:, k]
        I += [i, j, i, j]
        J += [j, i, i, j]
        W += [-w, -w, w, w]
    L = sp.coo_matrix((np.concatenate(W), (np.concatenate(I), np.concatenate(J))),
                      shape=(n, n)).tocsc()
    m = np.zeros(n)
    np.add.at(m, faces.ravel(), np.repeat(area / 3.0, 3))
    return L, m


def main(gid=None):
    gdir = RG.garment_dir(gid) if gid else RG.GARMENT
    d = gcd_io.load(gdir)
    R, F = d["rest"], d["faces"]
    pn = np.array(d["panel_names"], dtype=object)[np.maximum(d["panel_of_raw"], 0)]
    fields = [("ARAP assembly", np.load(os.path.join(RG.RESULT,
                                                     "assembly_sym_body_ease5_seed1.npy"))),
              ("drape", d["drape"])]

    print("=" * 78)
    print("1. characteristic wavelength of the geometry image (cm; longer = smoother)")
    print("=" * 78)
    print("  %-16s %14s %14s   %s" % ("panel", "ARAP assembly", "drape", "ratio"))
    for nm in PANELS:
        k = pn == nm
        if not k.any():
            continue
        idx = np.where(k)[0]
        ren = -np.ones(len(pn), np.int64)
        ren[idx] = np.arange(len(idx))
        fk = ren[F[k[F].all(1)]]
        L, m = cotan_laplacian(R[idx], fk, len(idx))
        out = []
        for _, P in fields:
            f = P[idx] - P[idx].mean(0)
            num = float(np.einsum("nd,nd->", f, L @ f))
            den = float((m[:, None] * f * f).sum())
            out.append(2.0 * np.pi / np.sqrt(max(num / max(den, 1e-30), 1e-30)))
        print("  %-16s %14.1f %14.1f   %5.2fx" % (nm, out[0], out[1], out[0] / out[1]))

    print()
    print("=" * 78)
    print("2. modal reconstruction: RMS error in cm using the first k eigenmodes")
    print("=" * 78)
    for nm in PANELS:
        k = pn == nm
        if not k.any():
            continue
        idx = np.where(k)[0]
        ren = -np.ones(len(pn), np.int64)
        ren[idx] = np.arange(len(idx))
        fk = ren[F[k[F].all(1)]]
        L, m = cotan_laplacian(R[idx], fk, len(idx))
        M = sp.diags(np.maximum(m, 1e-12))
        kmax = min(max(KS) + 5, len(idx) - 2)
        try:
            w, phi = spl.eigsh(L.tocsc() + 1e-9 * M, k=kmax, M=M, sigma=-1e-6, which="LM")
        except Exception as e:
            print("  %-16s eigensolve failed: %s" % (nm, e))
            continue
        o = np.argsort(w)
        phi = phi[:, o]
        print("  %-16s (%d vertices)" % (nm, len(idx)))
        print("      %-16s %s" % ("field", "  ".join("k=%-6d" % q for q in KS)))
        for lbl, P in fields:
            f = P[idx] - P[idx].mean(0)
            c = phi.T @ (M @ f)
            row = []
            for q in KS:
                if q > phi.shape[1]:
                    row.append("     -")
                    continue
                err = f - phi[:, :q] @ c[:q]
                row.append("%8.3f" % np.sqrt((err ** 2).sum(1).mean()))
            print("      %-16s %s" % (lbl, "  ".join(row)))

    print()
    print("=" * 78)
    print("3. how uniformly the image grid lands on the surface")
    print("=" * 78)
    print("  A geometry image resamples the surface on the flat panel's grid, so what")
    print("  matters is the singular values of that map.  sigma1*sigma2 is how much")
    print("  surface one pixel covers: above 1 the surface is under-sampled there,")
    print("  below 1 the resolution is spent for nothing.  sigma1/sigma2 is how far")
    print("  from square a pixel lands, which is what makes a square kernel mean")
    print("  different things in different places.")
    print()
    G, area = A.shape_gradients(R[F])
    w = area / area.sum()

    def q(x, p):                                    # area-weighted quantile
        o = np.argsort(x)
        return float(x[o][np.searchsorted(np.cumsum(w[o]), p)])

    print("  %-16s %28s   %22s" % ("", "area scale sigma1*sigma2", "anisotropy s1/s2"))
    print("  %-16s %7s %7s %7s %7s   %7s %7s" %
          ("field", "p01", "p50", "p99", "max", "p50", "p99"))
    stats = {}
    for lbl, P in fields:
        Fg = A.deformation_gradients(P, F, G)
        _, sv = A.best_rotations(Fg)
        s1 = np.maximum(sv[:, 0], sv[:, 1])
        s2 = np.minimum(sv[:, 0], sv[:, 1])
        ar = s1 * s2
        an = s1 / np.maximum(s2, 1e-9)
        stats[lbl] = (ar, an, s1, s2, sv)
        print("  %-16s %7.3f %7.3f %7.3f %7.3f   %7.3f %7.3f"
              % (lbl, q(ar, .01), q(ar, .5), q(ar, .99), ar.max(), q(an, .5), q(an, .99)))

    print()
    print("  what that costs in pixels: an isometric parametrisation needs 1.00x.")
    print("  To hold the WORST 1%% of the surface at a chosen 3D sampling density,")
    print("  the image has to be finer everywhere by the p99 area scale:")
    for lbl in stats:
        ar = stats[lbl][0]
        print("     %-16s %5.2fx  (and %4.1f%% of the area is under-sampled by >10%%)"
              % (lbl, q(ar, .99), 100 * w[ar > 1.1].sum()))

    print()
    print("  and how much a fixed square kernel varies in physical size:")
    for lbl in stats:
        _, _, s1, s2, _ = stats[lbl]
        print("     %-16s largest / smallest local scale over the surface = %5.2fx"
              % (lbl, q(s1, .99) / max(q(s2, .01), 1e-9)))

    print()
    print("  distance from isometry, |sigma - 1|:")
    print("  %-16s %10s %10s %10s" % ("field", "p50", "p90", "p99"))
    for lbl in stats:
        dv = np.abs(stats[lbl][4] - 1).max(1)
        print("  %-16s %10.4f %10.4f %10.4f"
              % (lbl, q(dv, .5), q(dv, .9), q(dv, .99)))


if __name__ == "__main__":
    main(*sys.argv[1:2])
