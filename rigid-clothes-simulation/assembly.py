"""Isometric assembly solver for a real garment.

Same energy as the toy cases, generalised in two ways the garment forces:

  * rest coordinates are per TRIANGLE, not per vertex.  A seam vertex is stored
    once per incident panel and each panel gives it a different flat position,
    so there is no single rest position per vertex -- but every triangle lies in
    exactly one panel and always has one.

  * hinge stencils are built by UNFOLDING the two incident triangles into a
    common plane from their own rest side lengths.  Two triangles sharing an
    edge can always be laid flat, seam or not, so the rest dihedral is always
    defined and always flat.  ALL interior edges are penalised, no exceptions.
    (Excluding seam hinges is what made the toy cylinder collapse to a teardrop.)

  E = E_ARAP + lambda_b * E_bend + w_s * E_stitch
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


# ---------------------------------------------------------------------------
# ARAP with per-triangle rest
# ---------------------------------------------------------------------------

def shape_gradients(rest_tri):
    """rest_tri (M,3,2) -> G (M,3,2) with F_t = sum_v p_v g_v^T, and areas (M,).

    The area returned is |signed area|.  In this dataset every BACK panel is
    parametrised mirrored relative to its 3D winding (100% of back-panel faces
    have negative signed rest area, 0% of front-panel faces), so signed areas
    would put negative weights on half the mesh and make the stiffness matrix
    indefinite.  Using the absolute value is not a patch: mirroring the rest
    sends F -> F M for a reflection M, which leaves the singular values -- and
    hence ||F-R||^2 over the Stiefel manifold -- unchanged.
    """
    d1 = rest_tri[:, 1] - rest_tri[:, 0]
    d2 = rest_tri[:, 2] - rest_tri[:, 0]
    E = np.stack([d1, d2], axis=2)
    det = E[:, 0, 0] * E[:, 1, 1] - E[:, 0, 1] * E[:, 1, 0]
    if np.any(np.abs(det) < 1e-14):
        raise ValueError("degenerate rest triangle: min |det| = %.3e" % np.abs(det).min())
    Einv = np.linalg.inv(E)
    gj, gk = Einv[:, 0, :], Einv[:, 1, :]
    return np.stack([-(gj + gk), gj, gk], axis=1), 0.5 * np.abs(det)


def deformation_gradients(P, faces, G):
    return np.einsum("tvd,tvc->tdc", P[faces], G)


def best_rotations(F):
    """argmin over the Stiefel manifold V_2(R^3) of ||F-R||^2 is U V^T.
    No determinant fix: V_2(R^3) is connected, unlike SO(3) inside O(3).

    U V^T = F (F^T F)^{-1/2}, and F^T F is 2x2, so no SVD is needed: for a 2x2
    SPD matrix C, sqrt(C) = (C + sqrt(det C) I) / sqrt(tr C + 2 sqrt(det C)),
    which inverts in closed form.  Same answer as np.linalg.svd to ~1e-15 and
    about twenty times quicker over 57k triangles -- this runs once per solver
    iteration.  Singular values come from the eigenvalues of C.
    """
    C = np.einsum("tdu,tdv->tuv", F, F)
    tr = C[:, 0, 0] + C[:, 1, 1]
    det = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0]
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
    sig = np.sqrt(np.maximum(np.stack([tr + disc, tr - disc], 1) * 0.5, 0.0))

    sd = np.sqrt(np.maximum(det, 0.0))                    # = sig[:,0]*sig[:,1]
    den = np.sqrt(np.maximum(tr + 2.0 * sd, 1e-300))
    # S = sqrt(C) = (C + sd I)/den ; S^{-1} = (adj S)/det S, det S = sd
    S = C.copy()
    S[:, 0, 0] += sd
    S[:, 1, 1] += sd
    S /= den[:, None, None]
    inv = np.empty_like(S)
    dS = np.maximum(S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0], 1e-300)
    inv[:, 0, 0] = S[:, 1, 1] / dS
    inv[:, 1, 1] = S[:, 0, 0] / dS
    inv[:, 0, 1] = -S[:, 0, 1] / dS
    inv[:, 1, 0] = -S[:, 1, 0] / dS
    return np.einsum("tdu,tuv->tdv", F, inv), sig


def arap_energy(F, R, A):
    d = F - R
    return float(np.sum(A * np.einsum("tdc,tdc->t", d, d)))


# ---------------------------------------------------------------------------
# hinges: unfold every interior edge of the WELDED topology
# ---------------------------------------------------------------------------

def build_hinges(faces, wid, rest_tri, panel_of_face=None):
    """-> hinges (H,4) RAW vertex ids (v0,v1 shared edge; v2,v3 opposite),
          rest4  (H,4,2) the two triangles unfolded into one plane.

    Topology comes from the welded mesh so that seams -- where the raw mesh is
    torn into separate panels -- still produce hinges.

    When both triangles are in the SAME panel their rest frames already agree, so
    their own rest coordinates are used directly.  Unfolding is only needed across
    a seam.  This matters: a panel can weld two of its own distinct vertices onto
    one point (a dart or notch), and there the welded-id matching that orients the
    unfold is ambiguous -- 102 of 84,596 same-panel hinges in this garment.
    """
    inc = {}
    for t, f in enumerate(faces):
        w = wid[f]
        for c in range(3):
            a, b = w[c], w[(c + 1) % 3]
            k = (a, b) if a < b else (b, a)
            inc.setdefault(k, []).append((t, c))

    H, R4 = [], []
    for (wa, wb), lst in inc.items():
        if len(lst) != 2:
            continue                                   # boundary or non-manifold
        (tA, cA), (tB, cB) = lst
        fA, fB = faces[tA], faces[tB]
        a0, a1, a2 = fA[cA], fA[(cA + 1) % 3], fA[(cA + 2) % 3]
        b0, b1, b3 = fB[cB], fB[(cB + 1) % 3], fB[(cB + 2) % 3]
        # orient B's shared edge the same way as A's
        if wid[b0] != wid[a0]:
            b0, b1 = b1, b0
        xa = rest_tri[tA]
        xb = rest_tri[tB]
        ia = [cA, (cA + 1) % 3, (cA + 2) % 3]
        ib = [cB, (cB + 1) % 3, (cB + 2) % 3]
        if wid[fB[ib[0]]] != wid[a0]:
            ib[0], ib[1] = ib[1], ib[0]
        A0, A1, A2 = xa[ia[0]], xa[ia[1]], xa[ia[2]]
        B0, B1, B3 = xb[ib[0]], xb[ib[1]], xb[ib[2]]
        LA = np.linalg.norm(A1 - A0)
        LB = np.linalg.norm(B1 - B0)
        L = 0.5 * (LA + LB)                            # seams differ slightly; average
        p, q = np.linalg.norm(A2 - A0), np.linalg.norm(A2 - A1)
        pp, qq = np.linalg.norm(B3 - B0), np.linalg.norm(B3 - B1)
        X2 = (L * L + p * p - q * q) / (2 * L)
        X3 = (L * L + pp * pp - qq * qq) / (2 * L)
        Y2 = np.sqrt(max(p * p - X2 * X2, 1e-24))
        Y3 = np.sqrt(max(pp * pp - X3 * X3, 1e-24))
        H.append((a0, a1, a2, b3))
        if panel_of_face is not None and panel_of_face[tA] == panel_of_face[tB]:
            R4.append([A0, A1, A2, B3])            # shared rest frame; no unfold needed
        else:
            R4.append([[0.0, 0.0], [L, 0.0], [X2, Y2], [X3, -Y3]])
    return np.array(H, np.int64).reshape(-1, 4), np.array(R4, float).reshape(-1, 4, 2)


def _cot(u, v):
    dot = np.einsum("ij,ij->i", u, v)
    crs = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    return dot / np.maximum(crs, 1e-300)


def hinge_stencils(rest4):
    """Bergou quadratic bending: K is the affine dependency of the four unfolded
    rest points (sum K = 0, sum K x = 0), scaled by the cotangent formula."""
    if len(rest4) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    x0, x1, x2, x3 = rest4[:, 0], rest4[:, 1], rest4[:, 2], rest4[:, 3]
    e = x1 - x0
    c01 = _cot(e, x2 - x0)
    c02 = _cot(e, x3 - x0)
    c03 = _cot(-e, x2 - x1)
    c04 = _cot(-e, x3 - x1)
    K = np.stack([c03 + c04, c01 + c02, -c01 - c03, -c02 - c04], axis=1)
    tri = lambda a, b, c: 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                                       - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    A = tri(x0, x1, x2) + tri(x0, x1, x3)
    return K, 3.0 / np.maximum(A, 1e-300)


def bending_energy(P, hinges, K, w):
    if len(hinges) == 0:
        return 0.0
    Kp = np.einsum("ha,had->hd", K, P[hinges])
    return float(np.sum(w * np.einsum("hd,hd->h", Kp, Kp)))


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

class Assembly:
    """L(w_s) = L0 + w_s D^T D,  L0 = K_cot + lambda_b H_bend + eps I.

    D^T D has rank <= n_pairs (967 here), so one Woodbury update carries the
    whole w_s continuation on a single factorization of L0.  lambda_b sits
    inside L0, so each rung of the lambda_b ladder costs one factorization.
    """

    def __init__(self, gar, lam_b, eps=1e-8, woodbury=True, mu=None, nu=None, anchor=None):
        self.g, self.lam_b = gar, lam_b
        self.mu, self.nu, self.anchor = mu, nu, anchor
        n = gar["n"]
        self.G, self.A = gar["G"], gar["area"]
        self.hinges, self.Kb, self.wb = gar["hinges"], gar["Kb"], gar["wb"]

        W = np.einsum("t,tuc,tvc->tuv", self.A, self.G, self.G)
        r0 = np.repeat(gar["faces"], 3, axis=1).ravel()
        c0 = np.tile(gar["faces"], (1, 3)).ravel()
        v0 = W.ravel()
        vb = self.wb[:, None, None] * self.Kb[:, :, None] * self.Kb[:, None, :]
        r1 = np.repeat(self.hinges, 4, axis=1).ravel()
        c1 = np.tile(self.hinges, (1, 4)).ravel()
        self.eps = eps * float(np.abs(v0).sum() / n)
        L0 = sp.coo_matrix((np.concatenate([v0, lam_b * vb.ravel()]),
                            (np.concatenate([r0, r1]), np.concatenate([c0, c1]))),
                           shape=(n, n)).tocsc()
        L0 = L0 + self.eps * sp.identity(n, format="csc")
        dg = None
        if mu is not None:
            # one-sided obstacle penalty.  mu is CONSTANT (every constrained
            # vertex carries it, violating or not), so the matrix does not change
            # with the active set and the single factorisation stands.
            dg = np.asarray(mu, float)
        if nu is not None:
            dg = np.asarray(nu, float) if dg is None else dg + np.asarray(nu, float)
        if dg is not None:
            L0 = L0 + sp.diags(dg, format="csc")
        self.lu = spl.splu(L0)
        self.L0 = L0

        p = gar["pairs"]
        self.pairs = p
        self.woodbury = woodbury and len(p) > 0
        if self.woodbury:
            D = sp.coo_matrix(
                (np.concatenate([np.ones(len(p)), -np.ones(len(p))]),
                 (np.concatenate([np.arange(len(p)), np.arange(len(p))]),
                  np.concatenate([p[:, 0], p[:, 1]]))), shape=(len(p), n)).tocsr()
            self.D = D
            self.Y = self.lu.solve(np.asarray(D.T.todense()))     # (n, n_pairs)
            self.C = D @ self.Y
            # C = D L0^-1 D^T is symmetric PSD, and the w_s continuation only
            # shifts its spectrum: (C + I/w_s)^-1 = V (Lam + I/w_s)^-1 V^T.  One
            # eigendecomposition per factorisation therefore carries every rung of
            # the w_s ladder, instead of one dense inverse per w_s value (~25 of
            # them, and each cost more than this eigh).
            self._lam, self._V = np.linalg.eigh(self.C)

    def solve_global(self, b, w_s):
        z = self.lu.solve(b)
        if not self.woodbury or w_s == 0.0:
            return z
        t = self._V.T @ (self.D @ z)
        t /= (self._lam + 1.0 / w_s)[:, None]
        return z - self.Y @ (self._V @ t)

    def energies(self, P, R=None):
        F = deformation_gradients(P, self.g["faces"], self.G)
        if R is None:
            R, _ = best_rotations(F)
        e_a = arap_energy(F, R, self.A)
        e_b = bending_energy(P, self.hinges, self.Kb, self.wb)
        d = P[self.pairs[:, 0]] - P[self.pairs[:, 1]]
        return e_a, e_b, float(np.sum(d * d))


def arap_rhs(R, faces, G, A, n):
    contrib = np.einsum("t,tdc,tvc->tvd", A, R, G).reshape(-1, 3)
    idx = faces.ravel()
    # np.add.at is an unbuffered ufunc loop and costs ~10x what bincount does
    return np.stack([np.bincount(idx, contrib[:, d], minlength=n) for d in range(3)], 1)


def geometric_schedule(w0, w1, factor, iters, tail):
    s, w = [], float(w0)
    while w < w1:
        s.append((w, iters))
        w *= factor
    s.append((float(w1), tail))
    return s


def solve(asm, P0, schedule, max_iter, tol=1e-10, log=None, clamp=None, recenter=True):
    g = asm.g
    P = P0.copy()
    hist, viol, it = [], [], 0
    for stage, (w_s, n_it) in enumerate(schedule):
        prev = None
        for _ in range(n_it):
            if it >= max_iter:
                break
            F = deformation_gradients(P, g["faces"], asm.G)
            R, _ = best_rotations(F)
            # energy at the current P with its own optimal rotations -- the same
            # F and R the global step is about to use, so no second SVD
            e_a = arap_energy(F, R, asm.A)
            e_b = bending_energy(P, asm.hinges, asm.Kb, asm.wb)
            dd = P[asm.pairs[:, 0]] - P[asm.pairs[:, 1]]
            e_s = float(np.sum(dd * dd))
            b = arap_rhs(R, g["faces"], asm.G, asm.A, g["n"])
            e_c = e_n = 0.0
            if asm.nu is not None:
                # two-sided anchor to the specification placement (an input, not
                # the drape): picks the point of the isometric continuum nearest
                # the pose the pattern was designed around.
                e_n = float(np.sum(asm.nu[:, None] * (P - asm.anchor) ** 2))
                b = b + asm.nu[:, None] * asm.anchor
            if asm.mu is not None:
                # Z = the feasible point nearest P.  Adding mu*|x - Z|^2 to the
                # energy makes the local step exact for a fixed Z and the global
                # step a linear solve with the SAME matrix; on free coordinates
                # Z = P so the term is purely proximal and vanishes at the fixed
                # point, on violating ones it is the half-space penalty itself.
                Z = clamp(P.copy())
                e_c = float(np.sum(asm.mu[:, None] * (P - Z) ** 2))
                b = b + asm.mu[:, None] * Z
            tot = e_a + asm.lam_b * e_b + w_s * e_s + e_c + e_n
            P = asm.solve_global(b, w_s)
            if recenter:
                P -= P.mean(0)
            hist.append(dict(it=it, stage=stage, w_s=w_s, E=tot, E_arap=e_a,
                             E_bend=e_b, E_stitch=e_s, E_half=e_c, E_anchor=e_n))
            it += 1
            if prev is not None:
                if tot > prev * (1 + 1e-9) + 1e-14:
                    viol.append((it, prev, tot))
                if abs(prev - tot) <= tol * max(abs(prev), 1e-30) and stage == len(schedule) - 1:
                    prev = tot
                    break
            prev = tot
        if log:
            log("    stage %2d  w_s=%9.3g  it=%5d  E=%.8g  E_arap=%.4g E_bend=%.4g "
                "E_stitch=%.3g E_obst=%.3g E_anch=%.3g"
                % (stage, w_s, it, hist[-1]["E"], hist[-1]["E_arap"],
                   hist[-1]["E_bend"], hist[-1]["E_stitch"], hist[-1]["E_half"],
                   hist[-1]["E_anchor"]))
        if it >= max_iter:
            break
    return P, hist, viol


def solve_annealed(gar, P0, ladder, w0=1e-2, w1=1e4, factor=2.0, iters_per_stage=10,
                   per_lambda=400, max_iter=20000, log=None, mu=None, clamp=None,
                   nu=None, anchor=None, recenter=True):
    """lambda_b continuation, stiff -> target.  From a flat/placed start a single
    small lambda_b folds the sheet flat instead of finding the shape; starting
    stiff picks the long-wavelength mode and softening afterwards removes the
    lambda_b bias."""
    P, hist, viol, off, used = P0.copy(), [], [], 0, 0
    asm = None
    for i, lam in enumerate(ladder):
        if log:
            log("  lambda_b = %g  (rung %d/%d)" % (lam, i + 1, len(ladder)))
        asm = Assembly(gar, lam, mu=mu, nu=nu, anchor=anchor)
        sched = geometric_schedule(w0 if i == 0 else w1, w1, factor, iters_per_stage, per_lambda)
        budget = min(per_lambda + iters_per_stage * len(sched), max_iter - used)
        if budget <= 0:
            break
        P, h, v = solve(asm, P, sched, max_iter=budget, log=log, clamp=clamp,
                        recenter=recenter)
        for rec in h:
            rec["stage"] += off
            rec["lam_b"] = lam
            rec["it"] += used
        off += len(sched)
        used += len(h)
        hist += h
        viol += v
    return P, asm, hist, viol, len(ladder)
