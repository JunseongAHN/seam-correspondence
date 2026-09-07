"""Per-triangle ARAP: deformation gradients, the SVD local step, and the
constant part of the global linear system.

The rotations are per *triangle*, never per vertex one-ring.  A stitched
vertex's one-ring does not lie flat (that is exactly the angle deficit we are
solving for), so it has no rest position in R^2; a triangle always does.
"""

import numpy as np


def shape_gradients(rest, faces):
    """Constant per-triangle vectors g_i, g_j, g_k in R^2 with

        F_t = sum_{v in t} p_v g_v^T = P_t E_t^{-1}

    Returns G (M,3,2) ordered as the face's own (i, j, k), and areas A (M,).
    """
    x = rest[faces]                                   # (M,3,2)
    d1 = x[:, 1] - x[:, 0]
    d2 = x[:, 2] - x[:, 0]
    E = np.stack([d1, d2], axis=2)                    # (M,2,2), columns
    det = E[:, 0, 0] * E[:, 1, 1] - E[:, 0, 1] * E[:, 1, 0]
    if np.any(np.abs(det) < 1e-14):
        raise ValueError("degenerate rest triangle: |det E| = %.3e" % np.abs(det).min())
    Einv = np.linalg.inv(E)                           # (M,2,2)
    gj, gk = Einv[:, 0, :], Einv[:, 1, :]             # rows of E^{-1}
    gi = -(gj + gk)
    G = np.stack([gi, gj, gk], axis=1)                # (M,3,2)
    return G, 0.5 * det


def deformation_gradients(P, faces, G):
    """F_t in R^{3x2} for every triangle, shape (M,3,2)."""
    return np.einsum("tvd,tvc->tdc", P[faces], G)


def best_rotations(F):
    """Local step.  argmin over {R in R^{3x2} : R^T R = I_2} of ||F-R||_F^2.

    With the thin SVD F = U S V^T the maximiser of tr(R^T F) on the Stiefel
    manifold V_2(R^3) is R = U V^T.  V_2(R^3) is *connected* -- unlike SO(3)
    inside O(3) -- so the determinant/reflection fix used by 3x3 ARAP does not
    apply and must not be applied here.

    Returns R (M,3,2) and the singular values s (M,2) = the principal stretches.
    """
    U, s, Vt = np.linalg.svd(F, full_matrices=False)   # U (M,3,2), s (M,2), Vt (M,2,2)
    return U @ Vt, s


def arap_energy(F, R, A):
    d = F - R
    return float(np.sum(A * np.einsum("tdc,tdc->t", d, d)))


def stiffness_triplets(faces, G, A):
    """COO triplets of K[u,v] = sum_t A_t (g_u . g_v) -- the cotangent
    Laplacian (stiffness matrix) of the flat rest mesh.  Constant."""
    W = np.einsum("t,tuc,tvc->tuv", A, G, G)          # (M,3,3)
    rows = np.repeat(faces, 3, axis=1).ravel()
    cols = np.tile(faces, (1, 3)).ravel()
    return rows.astype(np.int64), cols.astype(np.int64), W.ravel()


def arap_rhs(R, faces, G, A, n):
    """b[v] = sum_t A_t R_t g_v, shape (N,3)."""
    contrib = np.einsum("t,tdc,tvc->tvd", A, R, G)    # (M,3,3): (tri, local vert, xyz)
    b = np.zeros((n, 3))
    np.add.at(b, faces.ravel(), contrib.reshape(-1, 3))
    return b


# ---------------------------------------------------------------------------
# global step
# ---------------------------------------------------------------------------

import bending as _bending

try:
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spl
    HAVE_SCIPY = True
except ImportError:                                    # numpy-only fallback
    HAVE_SCIPY = False


class Assembly:
    """Everything that does not change during the solve.

    The global matrix is  L(w_s) = L0 + w_s * D^T D  with

        L0 = K_cot + lambda_b * H_bend + eps * I     (constant)
        D  = the (n_pairs x N) seam difference operator.

    w_s moves during the continuation but D^T D has rank <= n_pairs (33 and 40
    for the two cases), so a Woodbury update absorbs the ENTIRE w_s schedule
    against one factorization of L0 -- zero re-factorizations per iteration,
    which is what spec 1.8 demands and what spec 1.9 would otherwise break.

    lambda_b does sit inside L0, so `solve_annealed` builds one Assembly (hence
    one factorization) per lambda_b rung: 4 for case 1, 7 for case 2, over the
    whole run rather than per iteration.
    """

    def __init__(self, m, lam_b, eps=1e-8, seam_hinges=False):
        self.m, self.lam_b = m, lam_b
        self.G, self.A = shape_gradients(m.rest, m.faces)
        h = m.hinges()
        Kb, wb = _bending.hinge_stencils(m.rest, h)
        if seam_hinges and len(m.seam_hinges):
            Ks, ws = _bending.hinge_stencils(m.rest, m.seam_hinges, rest4=m.seam_hinge_rest)
            h = np.vstack([h, m.seam_hinges])
            Kb, wb = np.vstack([Kb, Ks]), np.concatenate([wb, ws])
        self.hinges, self.Kb, self.wb = h, Kb, wb

        n = m.n
        r0, c0, v0 = stiffness_triplets(m.faces, self.G, self.A)
        r1, c1, v1 = _bending.bending_triplets(self.hinges, self.Kb, self.wb)
        rows = np.concatenate([r0, r1])
        cols = np.concatenate([c0, c1])
        vals = np.concatenate([v0, lam_b * v1])
        self.eps = eps * float(np.abs(v0).sum() / n)

        if HAVE_SCIPY:
            L0 = _sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
            L0 = L0 + self.eps * _sp.identity(n, format="csc")
            self._lu = _spl.splu(L0.tocsc())
            self._solve0 = self._lu.solve
        else:
            L0 = np.zeros((n, n))
            np.add.at(L0, (rows, cols), vals)
            L0[np.diag_indices(n)] += self.eps
            self._inv = np.linalg.inv(L0)              # one factorization, reused
            self._solve0 = lambda B: self._inv @ B

        # seam difference operator and the Woodbury pieces (all built once)
        p = m.pairs
        if len(p):
            D = np.zeros((len(p), n))
            D[np.arange(len(p)), p[:, 0]] = 1.0
            D[np.arange(len(p)), p[:, 1]] = -1.0
            self.D = D
            self.Y = self._solve0(D.T)                 # (N, n_pairs)
            self.C = D @ self.Y                        # (n_pairs, n_pairs)
        else:
            self.D = self.Y = self.C = None

    def solve_global(self, b, w_s):
        """(L0 + w_s D^T D)^{-1} b  by Woodbury -- no re-factorization."""
        z = self._solve0(b)
        if self.D is None or w_s == 0.0:
            return z
        M = self.C + np.eye(len(self.C)) / w_s
        return z - self.Y @ np.linalg.solve(M, self.D @ z)

    # ---- energies -----------------------------------------------------
    def energies(self, P, R=None):
        F = deformation_gradients(P, self.m.faces, self.G)
        if R is None:
            R, _ = best_rotations(F)
        e_a = arap_energy(F, R, self.A)
        e_b = _bending.bending_energy(P, self.hinges, self.Kb, self.wb)
        d = P[self.m.pairs[:, 0]] - P[self.m.pairs[:, 1]] if len(self.m.pairs) else np.zeros((0, 3))
        e_s = float(np.sum(d * d))
        return e_a, e_b, e_s

    def total(self, P, w_s, R=None):
        e_a, e_b, e_s = self.energies(P, R)
        return e_a + self.lam_b * e_b + w_s * e_s


def solve(asm, P0, schedule, max_iter=500, tol=1e-10, verbose=False):
    """Local/global alternation with seam-weight continuation.

    schedule: list of (w_s, n_iter).  Energy is monotone within a stage (both
    steps are exact minimizations at fixed weights); it jumps *up* at a stage
    boundary because w_s just increased, which is not a bug.
    """
    m = asm.m
    P = P0.copy()
    hist = []
    it = 0
    mono_viol = []
    for stage, (w_s, n_it) in enumerate(schedule):
        prev = None
        for _ in range(n_it):
            if it >= max_iter:
                break
            F = deformation_gradients(P, m.faces, asm.G)
            R, s = best_rotations(F)
            b = arap_rhs(R, m.faces, asm.G, asm.A, m.n)
            P = asm.solve_global(b, w_s)
            P -= P.mean(0)          # fix the translation gauge (all three energies are translation invariant)
            e_a, e_b, e_s = asm.energies(P)
            tot = e_a + asm.lam_b * e_b + w_s * e_s
            hist.append(dict(it=it, stage=stage, w_s=w_s, E=tot, E_arap=e_a,
                             E_bend=e_b, E_stitch=e_s))
            it += 1
            if prev is not None:
                if tot > prev * (1 + 1e-9) + 1e-14:
                    mono_viol.append((it, prev, tot))
                if abs(prev - tot) <= tol * max(abs(prev), 1e-30) and stage == len(schedule) - 1:
                    prev = tot
                    break
            prev = tot
        if verbose:
            print("  stage %2d w_s=%9.3g  it=%3d  E=%.9g" % (stage, w_s, it, hist[-1]["E"]))
        if it >= max_iter:
            break
    return P, hist, mono_viol


def geometric_schedule(w0=1e-2, w1=1e4, factor=2.0, iters_per_stage=10, tail=None):
    """w_s from w0 to w1, multiplied by `factor` every `iters_per_stage`
    iterations, then a final stage held at w1 until convergence."""
    sched, w = [], float(w0)
    while w < w1:
        sched.append((w, iters_per_stage))
        w *= factor
    sched.append((float(w1), tail if tail is not None else 300))
    return sched


def solve_annealed(m, P0, lam_stages, seam_hinges=False, w0=1e-2, w1=1e4,
                   factor=2.0, iters_per_stage=10, per_lambda=300, tol=1e-10):
    """Bending-stiffness continuation on top of the seam-weight continuation.

    Starting from the flat sheet, a single fixed small lambda_b does NOT find the
    cylinder: the sheet is so floppy that it folds flat (cross-section winding
    number 0) while the seam closes, and that is a genuine local minimum.  So
    lambda_b is annealed from stiff to the target, exactly the same kind of
    continuation the spec already prescribes for w_s -- a stiff sheet picks the
    long-wavelength (one tube) mode, and softening afterwards removes the
    lambda_b radius bias without changing the mode.

    lambda_b sits inside L0, so each lambda_b value costs one factorization
    (len(lam_stages) in total, NOT one per iteration).  Every w_s in the
    schedule is free within a stage thanks to the Woodbury update.
    """
    P = P0.copy()
    hist, viol, off = [], [], 0
    asm = None
    for i, lam in enumerate(lam_stages):
        asm = Assembly(m, lam_b=lam, seam_hinges=seam_hinges)
        sched = geometric_schedule(w0 if i == 0 else w1, w1, factor, iters_per_stage,
                                   tail=per_lambda)
        budget = per_lambda + iters_per_stage * len(sched)
        P, h, v = solve(asm, P, sched, max_iter=budget, tol=tol)
        for rec in h:                       # keep stage ids globally unique: the energy
            rec["stage"] += off             # functional itself changes when lambda_b does,
            rec["lam_b"] = lam              # so monotonicity only holds within a stage
            rec["it"] += len(hist)
        off += len(sched)
        hist += h
        viol += v
    return P, asm, hist, viol, len(lam_stages)


def winding_number(P, ring_idx):
    """Signed turns of a closed polygon of vertex indices, in its own best-fit
    plane.  1 = the sheet rolled into one tube; 0 = it folded flat; 2 = scrolled
    twice.  This is the diagnostic that tells a correct roll-up from a wrong one."""
    Q = P[list(ring_idx)]
    C = Q - Q.mean(0)
    _, ev = np.linalg.eigh(C.T @ C)
    xy = np.stack([C @ ev[:, 1], C @ ev[:, 2]], 1)
    e = np.roll(xy, -1, 0) - xy
    ang = np.arctan2(e[:, 1], e[:, 0])
    d = np.diff(np.concatenate([ang, ang[:1]]))
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(d.sum() / (2 * np.pi))
