"""Bergou et al. 2006 quadratic bending, exact for a flat rest state.

For a hinge with flat rest positions x0..x3 the 1x4 stencil K is the affine
dependency of those four coplanar points: sum_i K_i = 0 and sum_i K_i x_i = 0.
Four points in the plane have a one-dimensional affine dependency, so K is
determined up to scale; the cotangent formula below fixes that scale.  Because
K annihilates every affine image of the rest state, the energy is exactly zero
on any planar (or rigidly moved) configuration and is quadratic in p.
"""

import numpy as np


def _cot(u, v):
    """cot of the angle between 2D vectors u and v, batched."""
    dot = np.einsum("ij,ij->i", u, v)
    crs = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    return dot / crs


def hinge_stencils(rest, hinges, rest4=None):
    """K (H,4) and weights w (H,) with E_e = w_e * ||K_e . p||^2.

    hinges rows are (v0, v1, v2, v3): v0-v1 is the shared edge, v2 opposite it
    in one triangle, v3 in the other.
    """
    if len(hinges) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    X = rest[hinges] if rest4 is None else np.asarray(rest4, float)
    x0, x1 = X[:, 0], X[:, 1]
    x2, x3 = X[:, 2], X[:, 3]
    e = x1 - x0
    c01 = _cot(e, x2 - x0)          # at x0, triangle (x0,x1,x2)
    c02 = _cot(e, x3 - x0)          # at x0, triangle (x0,x1,x3)
    c03 = _cot(-e, x2 - x1)         # at x1, triangle (x0,x1,x2)
    c04 = _cot(-e, x3 - x1)         # at x1, triangle (x0,x1,x3)
    K = np.stack([c03 + c04, c01 + c02, -c01 - c03, -c02 - c04], axis=1)

    def tri_area(a, b, c):
        u, v = b - a, c - a
        return 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])

    A = tri_area(x0, x1, x2) + tri_area(x0, x1, x3)
    return K, 3.0 / A


def bending_triplets(hinges, K, w):
    """COO triplets of H = sum_e w_e K_e^T K_e (N x N, applied per coordinate)."""
    if len(hinges) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    vals = w[:, None, None] * K[:, :, None] * K[:, None, :]   # (H,4,4)
    rows = np.repeat(hinges, 4, axis=1).ravel()
    cols = np.tile(hinges, (1, 4)).ravel()
    return rows, cols, vals.ravel()


def bending_energy(P, hinges, K, w):
    if len(hinges) == 0:
        return 0.0
    Kp = np.einsum("ha,had->hd", K, P[hinges])
    return float(np.sum(w * np.einsum("hd,hd->h", Kp, Kp)))
