"""Unit tests for the isometric-shell assembly solver.

    "$PY" test_assembly.py                          # everything, ~66 s
    "$PY" test_assembly.py ShapeGradients Hinges    # named cases only
    GCD_TEST_DIR=/nowhere "$PY" test_assembly.py    # only the synthetic ones

Most of the suite runs on tiny synthetic meshes in milliseconds and needs no
dataset.  The end-to-end tests load a real GarmentCodeData sample from
GCD_TEST_DIR (overridable by environment variable) and are skipped when it is
absent.  They use the cheapest ladder that still passes the four gates
README.md lists under "끝났다고 성공이 아니다":

    mono_violations == 0
    seam_gap_max    <  1e-3 cm
    max_sigma_dev   <  1.0
    E_arap / E_bend / max_sigma_dev finite
"""

import os
import pickle
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import assembly as A
import body
import gcd_io
import run_garment


GCD_TEST_DIR = os.environ.get(
    "GCD_TEST_DIR", r"C:\Users\POMCHECKER\gcd_data\test\part0")
TROUSERS = os.path.join(GCD_TEST_DIR, "rand_1328ERLDIC")
SHIRT = os.path.join(GCD_TEST_DIR, "rand_0B1T21D8NX")


def _have(d):
    return os.path.isdir(d) and os.path.exists(
        os.path.join(d, os.path.basename(d) + "_sim.ply"))


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------

def _grid(nx, ny):
    """(nx*ny, 2) lattice points and its consistently wound triangulation."""
    idx = lambda i, j: i * ny + j
    f = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            f.append((idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)))
            f.append((idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)))
    pts = np.array([[i, j] for i in range(nx) for j in range(ny)], float)
    return pts, np.array(f, np.int64)


def two_panel_mesh(nx=3, ny=3):
    """Two flat panels welded along one column of vertices.

    The second panel's rest frame is MIRRORED and translated away, the way the
    dataset parametrises its back panels: every face of panel 1 therefore has
    negative signed rest area, which is exactly the case shape_gradients'
    docstring is about.  The seam column is stored twice -- once per panel --
    so the raw mesh is panel-separated while the welded mesh is not, as in the
    real data.
    """
    ptsL, fL = _grid(nx, ny)
    ptsR, fR = _grid(nx, ny)
    nL = len(ptsL)
    mirror = np.array([[-1.0, 0.0], [0.0, 1.0]])
    rest = np.vstack([ptsL, ptsR @ mirror.T + np.array([20.0, 3.0])])
    faces = np.vstack([fL, fR + nL])

    wid = np.arange(2 * nL)
    for j in range(ny):                        # L column nx-1  ==  R column 0
        wid[nL + j] = (nx - 1) * ny + j
    _, wid = np.unique(wid, return_inverse=True)
    wid = np.asarray(wid, np.int64).ravel()

    pairs = np.array([[(nx - 1) * ny + j, nL + j] for j in range(ny)], np.int64)
    panel_of_face = np.array([0] * len(fL) + [1] * len(fR), np.int64)
    return dict(rest=rest, faces=faces, wid=wid, pairs=pairs, nx=nx, ny=ny,
                panel_of_face=panel_of_face, n=len(rest))


def synthetic_gar(nx=3, ny=3):
    """The `gar` dict Assembly consumes, built from two_panel_mesh."""
    m = two_panel_mesh(nx, ny)
    rest_tri = m["rest"][m["faces"]]
    G, area = A.shape_gradients(rest_tri)
    H, R4 = A.build_hinges(m["faces"], m["wid"], rest_tri, m["panel_of_face"])
    Kb, wb = A.hinge_stencils(R4)
    gar = dict(faces=m["faces"], n=m["n"], G=G, area=area,
               hinges=H, Kb=Kb, wb=wb, pairs=m["pairs"])
    return m, gar


def flat_positions(m):
    """A 3D embedding with the seam closed: undo the mirror+shift of panel 1."""
    nx, ny = m["nx"], m["ny"]
    half = m["n"] // 2
    P = np.zeros((m["n"], 3))
    P[:half, :2] = m["rest"][:half]
    a = m["rest"][half:]
    P[half:, 0] = (nx - 1) + (20.0 - a[:, 0])
    P[half:, 1] = a[:, 1] - 3.0
    return P


def rot3(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


# ---------------------------------------------------------------------------
# shape_gradients
# ---------------------------------------------------------------------------

class ShapeGradients(unittest.TestCase):

    def test_reproduces_a_known_affine_map(self):
        """F_t = sum_v p_v g_v^T must return the affine map's linear part."""
        rng = np.random.default_rng(0)
        m = two_panel_mesh(4, 3)
        G, _ = A.shape_gradients(m["rest"][m["faces"]])
        B = rng.standard_normal((3, 2))                 # rest plane -> R^3
        t = rng.standard_normal(3)
        P = m["rest"] @ B.T + t
        F = A.deformation_gradients(P, m["faces"], G)
        np.testing.assert_allclose(F, np.broadcast_to(B, F.shape), atol=1e-12)

    def test_translation_invariance(self):
        """G rows sum to zero, so F cannot see a translation."""
        m = two_panel_mesh()
        G, _ = A.shape_gradients(m["rest"][m["faces"]])
        np.testing.assert_allclose(G.sum(1), 0.0, atol=1e-12)

    def test_areas_positive_on_mirrored_panels(self):
        """Panel 1's rest is mirrored -- signed areas are negative there and the
        returned areas must still be positive, or half the mesh would carry a
        negative stiffness weight."""
        m = two_panel_mesh(4, 4)
        rest_tri = m["rest"][m["faces"]]
        d1 = rest_tri[:, 1] - rest_tri[:, 0]
        d2 = rest_tri[:, 2] - rest_tri[:, 0]
        signed = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        pof = m["panel_of_face"]
        self.assertTrue((signed[pof == 0] > 0).all())
        self.assertTrue((signed[pof == 1] < 0).all())   # the mirrored panel
        _, area = A.shape_gradients(rest_tri)
        self.assertTrue((area > 0).all())
        np.testing.assert_allclose(area, np.abs(signed), rtol=1e-12)

    def test_mirroring_leaves_arap_energy_unchanged(self):
        """The docstring's argument for abs(): mirroring the rest sends F -> F M,
        which leaves the singular values and ||F-R||^2 alone."""
        rng = np.random.default_rng(3)
        rest = rng.standard_normal((5, 3, 2))
        G, area = A.shape_gradients(rest)
        mirror = np.array([[-1.0, 0.0], [0.0, 1.0]])
        Gm, aream = A.shape_gradients(rest @ mirror.T)
        P = rng.standard_normal((15, 3))
        faces = np.arange(15, dtype=np.int64).reshape(5, 3)
        F = A.deformation_gradients(P, faces, G)
        Fm = A.deformation_gradients(P, faces, Gm)
        R, sig = A.best_rotations(F)
        Rm, sigm = A.best_rotations(Fm)
        np.testing.assert_allclose(area, aream, rtol=1e-12)
        np.testing.assert_allclose(np.sort(sig, 1), np.sort(sigm, 1), rtol=1e-10)
        self.assertAlmostEqual(A.arap_energy(F, R, area),
                               A.arap_energy(Fm, Rm, aream), places=10)

    def test_degenerate_rest_triangle_raises(self):
        bad = np.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        with self.assertRaises(ValueError):
            A.shape_gradients(bad)


# ---------------------------------------------------------------------------
# best_rotations
# ---------------------------------------------------------------------------

class BestRotations(unittest.TestCase):

    def test_pure_rotation_is_recovered_exactly(self):
        rng = np.random.default_rng(1)
        F = np.empty((20, 3, 2))
        for i in range(20):
            F[i] = rot3(rng.standard_normal(3), rng.uniform(-np.pi, np.pi))[:, :2]
        R, sig = A.best_rotations(F)
        np.testing.assert_allclose(R, F, atol=1e-12)
        np.testing.assert_allclose(sig, 1.0, atol=1e-12)

    def test_matches_numpy_svd(self):
        """the closed form claims ~1e-15 against np.linalg.svd"""
        rng = np.random.default_rng(2)
        F = rng.standard_normal((500, 3, 2))
        R, sig = A.best_rotations(F)
        U, s, Vt = np.linalg.svd(F, full_matrices=False)
        np.testing.assert_allclose(R, U @ Vt, atol=1e-12)
        np.testing.assert_allclose(sig, s, rtol=1e-12)

    def test_R_has_orthonormal_columns(self):
        """1e-11, not the docstring's 1e-15: the closed form goes through
        C = F^T F, which squares the condition number.  On 200 random F the
        worst is 1.3e-12, at cond(C) = 4.4e4; np.linalg.svd stays at 1.6e-15.
        See test_closed_form_degrades_on_ill_conditioned_F."""
        rng = np.random.default_rng(4)
        F = rng.standard_normal((200, 3, 2))
        R, _ = A.best_rotations(F)
        RtR = np.einsum("tdu,tdv->tuv", R, R)
        np.testing.assert_allclose(RtR, np.broadcast_to(np.eye(2), RtR.shape),
                                   atol=1e-11)

    def test_closed_form_degrades_on_ill_conditioned_F(self):
        """Pins the measured accuracy of R = F (F^T F)^{-1/2} as a function of
        the singular-value ratio.  It is NOT ~1e-15 in general -- the error
        grows like the square of the ratio, reaching 2.5e-6 at 1e6 -- but the
        singular values themselves stay accurate, and the solver never sees a
        ratio anywhere near that (the max_sigma_dev < 1 gate caps it near 10).
        """
        rng = np.random.default_rng(30)
        measured = {}
        for ratio in (1e1, 1e2, 1e3, 1e4, 1e6):
            U = np.linalg.qr(rng.standard_normal((3, 3)))[0][:, :2]
            V = np.linalg.qr(rng.standard_normal((2, 2)))[0]
            F = ((U * np.array([1.0, 1.0 / ratio])) @ V.T)[None]
            R, sig = A.best_rotations(F)
            Uu, ss, Vt = np.linalg.svd(F, full_matrices=False)
            measured[ratio] = np.abs(R - Uu @ Vt).max()
            np.testing.assert_allclose(sig, ss, atol=1e-10)
        self.assertLess(measured[1e1], 1e-14)
        self.assertLess(measured[1e2], 1e-11)
        self.assertLess(measured[1e3], 1e-9)
        self.assertLess(measured[1e6], 1e-4)
        # monotone degradation, i.e. the error really is conditioning-driven
        self.assertGreater(measured[1e6], measured[1e3])

    def test_no_determinant_fix_reflections_are_allowed(self):
        """V_2(R^3) is connected: a mirrored F must give R = F, not a flip."""
        F = np.array([[[1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]])
        R, sig = A.best_rotations(F)
        np.testing.assert_allclose(R, F, atol=1e-14)
        np.testing.assert_allclose(sig, 1.0, atol=1e-14)

    def test_scale_invariance_of_R(self):
        rng = np.random.default_rng(5)
        F = rng.standard_normal((50, 3, 2))
        R, sig = A.best_rotations(F)
        R2, sig2 = A.best_rotations(3.5 * F)
        np.testing.assert_allclose(R, R2, atol=1e-11)
        np.testing.assert_allclose(3.5 * sig, sig2, rtol=1e-11)

    def test_minimises_the_frobenius_distance(self):
        """R must beat every other point of the Stiefel manifold we can sample."""
        rng = np.random.default_rng(6)
        F = rng.standard_normal((1, 3, 2))
        R, _ = A.best_rotations(F)
        best = np.sum((F - R) ** 2)
        for _ in range(300):
            M = rot3(rng.standard_normal(3), rng.uniform(-np.pi, np.pi))[:, :2]
            self.assertLessEqual(best, np.sum((F[0] - M) ** 2) + 1e-12)


# ---------------------------------------------------------------------------
# hinges
# ---------------------------------------------------------------------------

class Hinges(unittest.TestCase):

    def test_hinge_count_matches_euler(self):
        """Two 3x3 grid panels: 8 interior edges each, plus the 2 seam edges
        that only exist once the mesh is welded."""
        m = two_panel_mesh(3, 3)
        H, R4 = A.build_hinges(m["faces"], m["wid"], m["rest"][m["faces"]],
                               m["panel_of_face"])
        self.assertEqual(len(H), 18)
        self.assertEqual(R4.shape, (18, 4, 2))
        # a hinge's two shared-edge endpoints are distinct welded points
        self.assertTrue((m["wid"][H[:, 0]] != m["wid"][H[:, 1]]).all())

    def test_seam_hinges_exist_and_are_unfolded(self):
        """Across a seam the two triangles get a fresh common frame; inside one
        panel their own rest coordinates are reused."""
        m = two_panel_mesh(3, 3)
        H, R4 = A.build_hinges(m["faces"], m["wid"], m["rest"][m["faces"]],
                               m["panel_of_face"])
        half = m["n"] // 2
        panel_of_raw = np.array([0] * half + [1] * half)
        seam = panel_of_raw[H[:, 2]] != panel_of_raw[H[:, 3]]
        self.assertEqual(int(seam.sum()), 2)
        # the unfolded frame is canonical: x0 at the origin, x1 on +x, x2 above
        # the edge and x3 below it
        s4 = R4[seam]
        np.testing.assert_allclose(s4[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(s4[:, 1, 1], 0.0, atol=1e-12)
        self.assertTrue((s4[:, 1, 0] > 0).all())
        self.assertTrue((s4[:, 2, 1] > 0).all())
        self.assertTrue((s4[:, 3, 1] < 0).all())
        # same-panel hinges keep their own frame, so they are NOT canonicalised
        self.assertGreater(np.abs(R4[~seam, 0]).max(), 1e-6)

    def test_unfold_preserves_the_rest_side_lengths(self):
        """The unfold is rigid per triangle: on this mesh every edge is 1 or
        sqrt(2) in the rest pattern and must stay so after unfolding."""
        m = two_panel_mesh(4, 3)
        H, R4 = A.build_hinges(m["faces"], m["wid"], m["rest"][m["faces"]],
                               m["panel_of_face"])
        allowed = np.array([1.0, np.sqrt(2.0)])
        for (i, j) in ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3)):
            L = np.linalg.norm(R4[:, i] - R4[:, j], axis=1)
            self.assertLess(np.abs(L[:, None] - allowed[None]).min(1).max(), 1e-9)

    def test_stencil_is_affine_invariant(self):
        """The invariant hinge_stencils' docstring states: sum K = 0 and
        sum_a K_a x_a = 0 on the unfolded rest points."""
        m = two_panel_mesh(4, 4)
        H, R4 = A.build_hinges(m["faces"], m["wid"], m["rest"][m["faces"]],
                               m["panel_of_face"])
        K, w = A.hinge_stencils(R4)
        np.testing.assert_allclose(K.sum(1), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.einsum("ha,had->hd", K, R4), 0.0,
                                   atol=1e-11)
        self.assertTrue((w > 0).all())
        self.assertTrue(np.isfinite(w).all())

    def test_stencil_on_random_valid_unfolds(self):
        rng = np.random.default_rng(7)
        n = 500
        x0 = np.zeros((n, 2))
        x1 = np.stack([rng.uniform(0.5, 3.0, n), np.zeros(n)], 1)
        x2 = np.stack([rng.uniform(-1.0, 4.0, n), rng.uniform(0.2, 3.0, n)], 1)
        x3 = np.stack([rng.uniform(-1.0, 4.0, n), -rng.uniform(0.2, 3.0, n)], 1)
        R4 = np.stack([x0, x1, x2, x3], 1)
        K, w = A.hinge_stencils(R4)
        scale = np.abs(K).max(1)
        np.testing.assert_allclose(K.sum(1) / scale, 0.0, atol=1e-10)
        np.testing.assert_allclose(
            np.einsum("ha,had->hd", K, R4) / scale[:, None], 0.0, atol=1e-9)

    def test_empty_hinge_set(self):
        K, w = A.hinge_stencils(np.zeros((0, 4, 2)))
        self.assertEqual(K.shape, (0, 4))
        self.assertEqual(A.bending_energy(np.zeros((3, 3)),
                                          np.zeros((0, 4), np.int64), K, w), 0.0)

    def test_bending_energy_vanishes_on_an_affine_image_of_the_unfold(self):
        """sum K x = 0 in the rest plane, so any affine image of a hinge's own
        unfolded rest carries no bending energy."""
        rng = np.random.default_rng(8)
        m = two_panel_mesh(4, 4)
        H, R4 = A.build_hinges(m["faces"], m["wid"], m["rest"][m["faces"]],
                               m["panel_of_face"])
        K, _ = A.hinge_stencils(R4)
        B = rng.standard_normal((3, 2))
        t = rng.standard_normal(3)
        np.testing.assert_allclose(np.einsum("ha,had->hd", K, R4 @ B.T + t),
                                   0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# energies
# ---------------------------------------------------------------------------

class Energies(unittest.TestCase):

    def setUp(self):
        self.m, self.gar = synthetic_gar(4, 4)
        rng = np.random.default_rng(9)
        self.P = flat_positions(self.m) + 0.15 * rng.standard_normal(
            (self.m["n"], 3))

    def _energies(self, P):
        F = A.deformation_gradients(P, self.gar["faces"], self.gar["G"])
        R, sig = A.best_rotations(F)
        return (A.arap_energy(F, R, self.gar["area"]),
                A.bending_energy(P, self.gar["hinges"], self.gar["Kb"],
                                 self.gar["wb"]), F, R, sig)

    def test_rigid_motion_leaves_both_energies_unchanged(self):
        e_a, e_b, _, _, _ = self._energies(self.P)
        self.assertGreater(e_a, 0.0)
        self.assertGreater(e_b, 0.0)
        Q = rot3([0.3, -0.7, 0.2], 1.1)
        t = np.array([12.0, -4.0, 7.5])
        e_a2, e_b2, _, _, _ = self._energies(self.P @ Q.T + t)
        self.assertAlmostEqual(e_a, e_a2, delta=1e-10 * max(1.0, abs(e_a)))
        self.assertAlmostEqual(e_b, e_b2, delta=1e-10 * max(1.0, abs(e_b)))

    def test_uniform_scale_follows_the_formula(self):
        """E_arap(sP) = sum_t A_t (s^2 ||F||_F^2 - 2 s (sig1+sig2) + 2), since R
        is scale invariant and R^T R = I so <F,R> = sig1 + sig2."""
        _, _, F, R, sig = self._energies(self.P)
        Aw = self.gar["area"]
        nf2 = np.einsum("tdc,tdc->t", F, F)
        for s in (0.5, 1.0, 2.0, 7.3):
            pred = float(np.sum(Aw * (s * s * nf2 - 2.0 * s * sig.sum(1) + 2.0)))
            got, _, _, _, _ = self._energies(s * self.P)
            self.assertAlmostEqual(got, pred, delta=1e-9 * max(1.0, abs(pred)))

    def test_uniform_scale_is_quadratic_in_bending(self):
        _, e_b, _, _, _ = self._energies(self.P)
        for s in (0.5, 2.0, 7.3):
            _, got, _, _, _ = self._energies(s * self.P)
            self.assertAlmostEqual(got, s * s * e_b,
                                   delta=1e-9 * max(1.0, abs(e_b)))

    def test_arap_energy_is_zero_on_an_isometric_embedding(self):
        """Each panel rigidly embedded in R^3 -- no stretch anywhere."""
        m, gar = synthetic_gar(4, 4)
        half = m["n"] // 2
        P = np.zeros((m["n"], 3))
        Q = rot3([0.2, 0.9, -0.3], 0.8)
        P[:half] = np.hstack([m["rest"][:half], np.zeros((half, 1))]) @ Q.T
        P[half:] = np.hstack([m["rest"][half:], np.zeros((half, 1))]) + 5.0
        F = A.deformation_gradients(P, gar["faces"], gar["G"])
        R, sig = A.best_rotations(F)
        self.assertLess(A.arap_energy(F, R, gar["area"]), 1e-20)
        np.testing.assert_allclose(sig, 1.0, atol=1e-12)

    def test_arap_rhs_matches_a_scatter_reference(self):
        m, gar = synthetic_gar(3, 3)
        rng = np.random.default_rng(10)
        R = rng.standard_normal((len(gar["faces"]), 3, 2))
        got = A.arap_rhs(R, gar["faces"], gar["G"], gar["area"], gar["n"])
        ref = np.zeros((gar["n"], 3))
        for t, f in enumerate(gar["faces"]):
            for v in range(3):
                ref[f[v]] += gar["area"][t] * (R[t] @ gar["G"][t, v])
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_assembly_energies_agrees_with_the_free_functions(self):
        m, gar = synthetic_gar(4, 3)
        asm = A.Assembly(gar, lam_b=1e-3)
        P = flat_positions(m) + 0.1
        e_a, e_b, e_s = asm.energies(P)
        F = A.deformation_gradients(P, gar["faces"], gar["G"])
        R, _ = A.best_rotations(F)
        self.assertAlmostEqual(e_a, A.arap_energy(F, R, gar["area"]), places=12)
        self.assertAlmostEqual(
            e_b, A.bending_energy(P, gar["hinges"], gar["Kb"], gar["wb"]),
            places=12)
        d = P[gar["pairs"][:, 0]] - P[gar["pairs"][:, 1]]
        self.assertAlmostEqual(e_s, float(np.sum(d * d)), places=12)


# ---------------------------------------------------------------------------
# the linear algebra inside Assembly
# ---------------------------------------------------------------------------

class SolveGlobal(unittest.TestCase):

    def setUp(self):
        self.m, self.gar = synthetic_gar(4, 4)
        # Every production run carries an obstacle penalty (--body or
        # --half-*), and without one L0's only defence against the ARAP
        # stiffness's translation nullspace is eps ~ 4.5e-8, which leaves
        # cond(L0) ~ 1.8e8.  The regularised matrix is the one where an exact
        # comparison of two solvers is meaningful; the singular one is covered
        # by test_unregularised_L0_is_ill_conditioned below.
        self.mu = np.full(self.gar["n"], 0.37)

    def _asm(self, lam_b=1e-2, **kw):
        kw.setdefault("mu", self.mu)
        return A.Assembly(self.gar, lam_b=lam_b, **kw)

    def test_woodbury_matches_the_dense_augmented_system(self):
        """L(w_s) = L0 + w_s D^T D, solved densely, over the whole w_s ladder
        (geometric_schedule tops out at 1e4)."""
        asm = self._asm()
        L0 = asm.L0.toarray()
        D = asm.D.toarray()
        rng = np.random.default_rng(11)
        b = rng.standard_normal((self.gar["n"], 3))
        for w_s in (1e-2, 1.0, 1e2, 1e3, 1e4):
            got = asm.solve_global(b, w_s)
            ref = np.linalg.solve(L0 + w_s * (D.T @ D), b)
            np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-11)

    def test_woodbury_survives_a_w_s_far_past_the_ladder(self):
        """At w_s = 1e8 the augmented system's condition number is ~5e8 and the
        two solvers part company at 5e-9 -- still a residual-accurate answer,
        which is what the continuation needs."""
        asm = self._asm()
        L0 = asm.L0.toarray()
        D = asm.D.toarray()
        rng = np.random.default_rng(25)
        b = rng.standard_normal((self.gar["n"], 3))
        for w_s in (1e6, 1e8):
            L = L0 + w_s * (D.T @ D)
            x = asm.solve_global(b, w_s)
            res = np.linalg.norm(L @ x - b) / np.linalg.norm(b)
            ref = np.linalg.norm(L @ np.linalg.solve(L, b) - b) / np.linalg.norm(b)
            self.assertLess(res, max(1e-9, 100.0 * ref))

    def test_unregularised_L0_is_ill_conditioned(self):
        """Recorded rather than asserted away: with no mu/nu the ARAP stiffness
        is singular on translations and eps alone damps it, so L0's condition
        number is ~1e8 and the Woodbury and dense solutions agree only to a few
        parts in 1e9.  Both are residual-accurate; the disagreement lives in
        the near-null direction."""
        asm = A.Assembly(self.gar, lam_b=1e-2)
        L0 = asm.L0.toarray()
        D = asm.D.toarray()
        self.assertGreater(np.linalg.cond(L0), 1e6)
        rng = np.random.default_rng(26)
        b = rng.standard_normal((self.gar["n"], 3))
        for w_s in (1.0, 1e4):
            L = L0 + w_s * (D.T @ D)
            x = asm.solve_global(b, w_s)
            ref = np.linalg.solve(L, b)
            self.assertLess(np.linalg.norm(L @ x - b),
                            100.0 * np.linalg.norm(L @ ref - b) + 1e-12)
            self.assertLess(np.abs(x - ref).max() / np.abs(ref).max(), 1e-5)

    def test_w_s_zero_is_the_plain_solve(self):
        asm = self._asm()
        rng = np.random.default_rng(12)
        b = rng.standard_normal((self.gar["n"], 3))
        np.testing.assert_allclose(asm.solve_global(b, 0.0),
                                   np.linalg.solve(asm.L0.toarray(), b),
                                   rtol=1e-9, atol=1e-11)

    def test_woodbury_flag_off_drops_the_stitch_term(self):
        """woodbury=False is not an alternative solver for the same system: it
        returns the L0 solve, i.e. w_s is ignored.  Pinned down here so the
        flag's meaning is explicit."""
        rng = np.random.default_rng(13)
        b = rng.standard_normal((self.gar["n"], 3))
        on = self._asm(woodbury=True)
        off = self._asm(woodbury=False)
        L0 = off.L0.toarray()
        for w_s in (1.0, 1e4):
            np.testing.assert_allclose(off.solve_global(b, w_s),
                                       np.linalg.solve(L0, b),
                                       rtol=1e-9, atol=1e-11)
            self.assertGreater(
                np.abs(on.solve_global(b, w_s) - off.solve_global(b, w_s)).max(),
                1e-6)

    def test_L0_is_symmetric_and_positive_definite(self):
        for asm in (A.Assembly(self.gar, lam_b=1e-2), self._asm()):
            L0 = asm.L0.toarray()
            np.testing.assert_allclose(L0, L0.T, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(L0).min(), 0.0)

    def test_C_is_symmetric_psd_and_its_eigenvalues_are_cached(self):
        asm = self._asm()
        C = np.asarray(asm.C)
        np.testing.assert_allclose(C, C.T, atol=1e-10)
        ev = np.linalg.eigvalsh(C)
        self.assertGreaterEqual(ev.min(), -1e-12)
        np.testing.assert_allclose(np.sort(asm._lam), np.sort(ev), atol=1e-10)

    def test_penalty_diagonals_enter_L0(self):
        mu = np.linspace(1.0, 2.0, self.gar["n"])
        nu = np.full(self.gar["n"], 0.25)
        base = A.Assembly(self.gar, lam_b=1e-2).L0.toarray()
        both = A.Assembly(self.gar, lam_b=1e-2, mu=mu, nu=nu).L0.toarray()
        np.testing.assert_allclose(both - base, np.diag(mu + nu), atol=1e-12)

    def test_lambda_b_enters_L0_linearly(self):
        L0 = A.Assembly(self.gar, lam_b=0.0).L0.toarray()
        L1 = A.Assembly(self.gar, lam_b=1.0).L0.toarray()
        L2 = A.Assembly(self.gar, lam_b=2.0).L0.toarray()
        np.testing.assert_allclose(L2 - L0, 2.0 * (L1 - L0), atol=1e-10)

    def test_no_pairs_disables_woodbury(self):
        gar = dict(self.gar)
        gar["pairs"] = np.zeros((0, 2), np.int64)
        asm = A.Assembly(gar, lam_b=1e-2, mu=self.mu)
        self.assertFalse(asm.woodbury)
        rng = np.random.default_rng(14)
        b = rng.standard_normal((gar["n"], 3))
        np.testing.assert_allclose(asm.solve_global(b, 1e4),
                                   np.linalg.solve(asm.L0.toarray(), b),
                                   rtol=1e-9, atol=1e-11)


# ---------------------------------------------------------------------------
# schedule and the iteration
# ---------------------------------------------------------------------------

class Schedule(unittest.TestCase):

    def test_geometric_schedule_shape(self):
        s = A.geometric_schedule(1e-2, 1e4, 2.0, 10, 400)
        ws = [w for w, _ in s]
        self.assertAlmostEqual(ws[0], 1e-2)
        self.assertAlmostEqual(ws[-1], 1e4)
        self.assertEqual(s[-1][1], 400)
        self.assertTrue(all(n == 10 for _, n in s[:-1]))
        self.assertTrue(all(w < 1e4 for w in ws[:-1]))
        for a, b in zip(ws, ws[1:-1]):
            self.assertAlmostEqual(b / a, 2.0)

    def test_geometric_schedule_degenerate_when_w0_ge_w1(self):
        self.assertEqual(A.geometric_schedule(1e4, 1e4, 2.0, 10, 400),
                         [(1e4, 400)])


class SolveLoop(unittest.TestCase):

    def test_descent_is_monotone_and_the_seam_closes(self):
        m, gar = synthetic_gar(4, 4)
        rng = np.random.default_rng(15)
        P0 = flat_positions(m) + 0.3 * rng.standard_normal((m["n"], 3))
        asm = A.Assembly(gar, lam_b=1e-3)
        sched = A.geometric_schedule(1e-2, 1e4, 2.0, 10, 200)
        P, hist, viol = A.solve(asm, P0, sched, max_iter=5000)
        self.assertEqual(viol, [], "monotonicity violations: %r" % (viol[:5],))
        self.assertTrue(np.isfinite(P).all())
        gap = np.linalg.norm(P[gar["pairs"][:, 0]] - P[gar["pairs"][:, 1]],
                             axis=1)
        self.assertLess(gap.max(), 1e-3)
        F = A.deformation_gradients(P, gar["faces"], gar["G"])
        _, sig = A.best_rotations(F)
        self.assertLess(np.abs(sig - 1.0).max(), 1.0)

    def test_energy_history_is_consistent_with_the_returned_shape(self):
        m, gar = synthetic_gar(4, 4)
        asm = A.Assembly(gar, lam_b=1e-3)
        P, hist, _ = A.solve(asm, flat_positions(m) + 0.05, [(1e3, 60)],
                             max_iter=60, recenter=False)
        e_a, e_b, e_s = asm.energies(P)
        # the last record is the energy BEFORE the final global step, so it can
        # only be larger than the energy at the returned P
        self.assertLessEqual(e_a + 1e-3 * e_b + 1e3 * e_s,
                             hist[-1]["E"] + 1e-12)

    def test_solve_is_deterministic(self):
        m, gar = synthetic_gar(4, 4)
        rng = np.random.default_rng(16)
        P0 = flat_positions(m) + 0.2 * rng.standard_normal((m["n"], 3))
        asm = A.Assembly(gar, lam_b=1e-3)
        sched = A.geometric_schedule(1e-2, 1e3, 4.0, 5, 50)
        P1, _, _ = A.solve(asm, P0, sched, max_iter=1000)
        P2, _, _ = A.solve(asm, P0, sched, max_iter=1000)
        np.testing.assert_array_equal(P1, P2)

    def test_solve_does_not_mutate_its_input(self):
        m, gar = synthetic_gar(3, 3)
        asm = A.Assembly(gar, lam_b=1e-3)
        P0 = flat_positions(m)
        keep = P0.copy()
        A.solve(asm, P0, [(1e2, 5)], max_iter=5)
        np.testing.assert_array_equal(P0, keep)

    def test_max_iter_is_respected(self):
        m, gar = synthetic_gar(3, 3)
        asm = A.Assembly(gar, lam_b=1e-3)
        sched = A.geometric_schedule(1e-2, 1e4, 2.0, 10, 400)
        _, hist, _ = A.solve(asm, flat_positions(m), sched, max_iter=17)
        self.assertEqual(len(hist), 17)

    def test_recenter_puts_the_centroid_at_the_origin(self):
        """The obstacle penalty is what makes recenter=False observable: with
        no mu the right-hand side is translation-free (G rows sum to zero), so
        the solve lands at a zero-mean shape whether it re-centres or not."""
        m, gar = synthetic_gar(3, 3)
        mu = np.full(gar["n"], 5.0 * run_garment.diag_scale(gar))
        asm = A.Assembly(gar, lam_b=1e-3, mu=mu)
        keep = lambda P: P                      # obstacle that never binds
        P0 = flat_positions(m) + 100.0
        P, _, _ = A.solve(asm, P0, [(1e2, 5)], max_iter=5, clamp=keep,
                          recenter=True)
        np.testing.assert_allclose(P.mean(0), 0.0, atol=1e-9)
        Q, _, _ = A.solve(asm, P0, [(1e2, 5)], max_iter=5, clamp=keep,
                          recenter=False)
        self.assertGreater(np.abs(Q.mean(0)).max(), 1.0)

    def test_obstacle_penalty_pushes_the_shape_out_of_a_half_space(self):
        """asm.mu + clamp is the one-sided obstacle: a shape started below a
        floor at z = 0 must come back up, and descent must stay monotone (the
        hard clamp HANDOFF.md rejected gave 1507 violations)."""
        m, gar = synthetic_gar(4, 4)
        mu = np.full(gar["n"], 5.0 * run_garment.diag_scale(gar))
        asm = A.Assembly(gar, lam_b=1e-3, mu=mu)

        def clamp(P):
            P[:, 2] = np.maximum(P[:, 2], 0.0)
            return P

        P0 = flat_positions(m)
        P0[:, 2] -= 1.0
        P, hist, viol = A.solve(asm, P0, [(1e3, 200)], max_iter=200,
                                clamp=clamp, recenter=False)
        self.assertGreater(P[:, 2].min(), P0[:, 2].min())
        self.assertGreater(P[:, 2].min(), -1e-3)
        self.assertTrue(all(h["E_half"] >= 0.0 for h in hist))
        # This configuration relaxes all the way to E ~ 1e-13, below the
        # absolute floor solve() uses (prev*(1+1e-9) + 1e-14), so the last few
        # dozen iterations register as "violations" that are pure round-off.
        # Assert descent up to round-off rather than the raw counter.
        emax = max(h["E"] for h in hist)
        for it, prev, tot in viol:
            self.assertLess(tot - prev, 1e-12 * emax)
        self.assertTrue(all(v[1] < 1e-9 for v in viol))

    def test_solve_annealed_reports_one_factorisation_per_rung(self):
        m, gar = synthetic_gar(3, 3)
        rng = np.random.default_rng(17)
        P0 = flat_positions(m) + 0.2 * rng.standard_normal((m["n"], 3))
        ladder = [1e-1, 1e-2, 1e-3]
        P, asm, hist, viol, nfac = A.solve_annealed(
            gar, P0, ladder, per_lambda=40, max_iter=2000)
        self.assertEqual(nfac, len(ladder))
        self.assertEqual(asm.lam_b, ladder[-1])
        self.assertEqual(hist[0]["lam_b"], ladder[0])
        self.assertEqual(hist[-1]["lam_b"], ladder[-1])
        self.assertEqual([h["it"] for h in hist], list(range(len(hist))))
        self.assertEqual(viol, [])


# ---------------------------------------------------------------------------
# run_garment / gcd_io helpers that need no garment
# ---------------------------------------------------------------------------

class Initial(unittest.TestCase):

    def test_amp_zero_returns_the_placement_untouched(self):
        """Not bit-identical: initial() does c + inflate*(P-c) before it looks
        at amp, so even inflate=1 round-trips through the centroid and moves
        21% of the coordinates by one ulp.  Deterministic, but not an identity.
        """
        rng = np.random.default_rng(18)
        placed = rng.standard_normal((40, 3)) * 10.0
        P = run_garment.initial(dict(placed=placed), seed=1, amp=0.0)
        np.testing.assert_allclose(P, placed, rtol=1e-14, atol=1e-14)

    def test_inflate_scales_about_the_centroid(self):
        rng = np.random.default_rng(19)
        placed = rng.standard_normal((40, 3)) * 10.0
        P = run_garment.initial(dict(placed=placed), seed=1, inflate=2.0, amp=0.0)
        c = placed.mean(0)
        np.testing.assert_allclose(P, c + 2.0 * (placed - c), atol=1e-12)

    def test_sym_perturbation_is_even_in_x(self):
        """--sym builds the noise from a field that depends only on |x|, so two
        vertices mirrored about x=0 receive the same displacement."""
        rng = np.random.default_rng(20)
        half = rng.standard_normal((30, 3)) * 10.0
        placed = np.vstack([half, half * np.array([-1.0, 1.0, 1.0])])
        dP = run_garment.initial(dict(placed=placed), seed=3, amp=0.05,
                                 sym=True) - placed
        np.testing.assert_allclose(dP[:30], dP[30:], atol=1e-12)
        self.assertGreater(np.abs(dP).max(), 0.0)

    def test_asymmetric_perturbation_is_not_even_in_x(self):
        rng = np.random.default_rng(21)
        half = rng.standard_normal((30, 3)) * 10.0
        placed = np.vstack([half, half * np.array([-1.0, 1.0, 1.0])])
        dP = run_garment.initial(dict(placed=placed), seed=3, amp=0.05,
                                 sym=False) - placed
        self.assertGreater(np.abs(dP[:30] - dP[30:]).max(), 1e-3)

    def test_initial_is_reproducible_for_a_seed(self):
        rng = np.random.default_rng(22)
        d = dict(placed=rng.standard_normal((40, 3)) * 10.0)
        a = run_garment.initial(d, seed=7, amp=0.05, sym=True)
        b = run_garment.initial(d, seed=7, amp=0.05, sym=True)
        c = run_garment.initial(d, seed=8, amp=0.05, sym=True)
        np.testing.assert_array_equal(a, b)
        self.assertGreater(np.abs(a - c).max(), 1e-6)

    def test_diag_scale_is_the_mean_arap_diagonal(self):
        """with lam_b = 0 and eps = 0, L0 IS the ARAP stiffness."""
        m, gar = synthetic_gar(4, 3)
        asm = A.Assembly(gar, lam_b=0.0, eps=0.0)
        self.assertAlmostEqual(run_garment.diag_scale(gar),
                               float(asm.L0.diagonal().mean()), places=10)


class EulerXYZ(unittest.TestCase):

    def test_is_a_proper_rotation(self):
        rng = np.random.default_rng(23)
        for _ in range(20):
            R = gcd_io.euler_xyz(rng.uniform(-180, 180, 3))
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_single_axis_cases(self):
        np.testing.assert_allclose(gcd_io.euler_xyz([90, 0, 0]) @ [0, 1, 0],
                                   [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(gcd_io.euler_xyz([0, 90, 0]) @ [0, 0, 1],
                                   [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(gcd_io.euler_xyz([0, 0, 90]) @ [1, 0, 0],
                                   [0, 1, 0], atol=1e-12)

    def test_composition_order_is_Rz_Ry_Rx(self):
        """i.e. EXTRINSIC x-y-z, equivalently INTRINSIC z-y-x.  The docstring
        calls it "intrinsic x-y-z", which would be Rx @ Ry @ Rz -- a different
        matrix.  Only the naming is off; this pins the actual convention."""
        a = [20.0, -35.0, 50.0]
        R = gcd_io.euler_xyz(a)
        Rx = gcd_io.euler_xyz([a[0], 0, 0])
        Ry = gcd_io.euler_xyz([0, a[1], 0])
        Rz = gcd_io.euler_xyz([0, 0, a[2]])
        np.testing.assert_allclose(R, Rz @ Ry @ Rx, atol=1e-12)
        self.assertGreater(np.abs(R - Rx @ Ry @ Rz).max(), 1e-3)


class ConnectedComponents(unittest.TestCase):

    def test_raw_mesh_splits_into_one_component_per_panel(self):
        m = two_panel_mesh(4, 4)
        comp, ncomp = gcd_io.connected_components(m["faces"], m["n"])
        self.assertEqual(ncomp, 2)
        half = m["n"] // 2
        self.assertEqual(len(set(comp[:half].tolist())), 1)
        self.assertEqual(len(set(comp[half:].tolist())), 1)
        self.assertNotEqual(comp[0], comp[half])


# ---------------------------------------------------------------------------
# gcd_io on a real garment
# ---------------------------------------------------------------------------

@unittest.skipUnless(_have(TROUSERS), "dataset absent: %s" % TROUSERS)
class GarmentIO(unittest.TestCase):

    d = None

    @classmethod
    def setUpClass(cls):
        cls.d = gcd_io.load(TROUSERS)

    def test_welding_reduces_the_vertex_count(self):
        d = self.d
        self.assertLess(d["n_welded"], len(d["rest"]))
        self.assertEqual(int(d["wid"].max()) + 1, d["n_welded"])
        self.assertEqual(len(d["labels"]), d["n_welded"])

    def test_seam_pairs_are_coincident_in_the_drape(self):
        """The correspondence IS byte-identical xyz -- not a tolerance."""
        d = self.d
        p = d["pairs"]
        self.assertGreater(len(p), 0)
        np.testing.assert_array_equal(d["drape"][p[:, 0]], d["drape"][p[:, 1]])
        np.testing.assert_array_equal(d["wid"][p[:, 0]], d["wid"][p[:, 1]])

    def test_seam_pair_count_matches_the_welding_multiplicity(self):
        d = self.d
        self.assertEqual(len(d["pairs"]), len(d["rest"]) - d["n_welded"])

    def test_seam_pairs_are_far_apart_in_the_rest_pattern(self):
        """A seam vertex has a different flat position per incident panel; if
        the rest coordinates agreed there would be nothing to stitch."""
        d = self.d
        p = d["pairs"]
        sep = np.linalg.norm(d["rest"][p[:, 0]] - d["rest"][p[:, 1]], axis=1)
        self.assertGreater(float(np.median(sep)), 1.0)

    def test_panels_partition_the_mesh(self):
        d = self.d
        pr = d["panel_of_raw"]
        self.assertEqual(len(pr), len(d["rest"]))
        self.assertTrue((pr >= 0).all())
        self.assertTrue((pr < len(d["panel_names"])).all())
        counts = np.bincount(pr, minlength=len(d["panel_names"]))
        self.assertEqual(int(counts.sum()), len(d["rest"]))
        self.assertTrue((counts > 0).all())
        self.assertEqual(len(set(d["panel_names"])), len(d["panel_names"]))

    def test_every_face_lies_in_exactly_one_panel(self):
        d = self.d
        self.assertTrue(
            (d["panel_of_raw"][d["faces"]] == d["panel_of_face"][:, None]).all())

    def test_panel_labels_agree_with_the_segmentation_file(self):
        """Every welded vertex is either a stitch marker or carries the label of
        a panel the specification names."""
        d = self.d
        names = set(d["panel_names"])
        for l in set(str(x) for x in d["labels"]):
            self.assertTrue(l in names or l.startswith("stitch"), l)

    def test_uv_calibration_reproduces_orig_lens(self):
        """README: the anisotropic fit lands inside 1-2%, and a single isotropic
        scale does not."""
        d = self.d
        r = d["uv_report"]
        self.assertLess(r["err_median"], 0.02)
        self.assertLess(r["err_p90"], 0.02)
        self.assertLess(r["err_median"], r["isotropic_err_median"])
        self.assertGreater(r["n_edges"], 100)
        self.assertGreater(r["Kx"], 0.0)
        self.assertGreater(r["Ky"], 0.0)

        # recompute independently, straight off the returned rest coordinates
        with open(os.path.join(
                TROUSERS,
                os.path.basename(TROUSERS) + "_orig_lens.pickle"), "rb") as fh:
            ol = pickle.load(fh)
        ol = {(int(min(a, b)), int(max(a, b))): float(v)
              for (a, b), v in ol.items()}
        wid, rest = d["wid"], d["rest"]
        err, seen = [], set()
        for f in d["faces"]:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                k = (min(wid[a], wid[b]), max(wid[a], wid[b]))
                e = (min(a, b), max(a, b))
                if k in ol and e not in seen:
                    seen.add(e)
                    err.append(abs(np.linalg.norm(rest[a] - rest[b]) / ol[k] - 1.0))
        err = np.array(err)
        self.assertEqual(len(err), r["n_edges"])
        self.assertLess(float(np.median(err)), 0.01)
        self.assertLess(float(np.quantile(err, 0.9)), 0.02)

    def test_anisotropy_is_real(self):
        d = self.d
        self.assertGreater(abs(d["Kx"] / d["Ky"] - 1.0), 0.01)

    def test_placement_is_not_the_drape(self):
        d = self.d
        self.assertEqual(d["placed"].shape, d["drape"].shape)
        self.assertGreater(np.abs(d["placed"] - d["drape"]).max(), 1.0)

    def test_placement_is_isometric_to_the_rest_pattern(self):
        """Each panel is placed by a rigid transform, so every rest edge length
        survives it exactly."""
        d = self.d
        f = d["faces"]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            lr = np.linalg.norm(d["rest"][f[:, a]] - d["rest"][f[:, b]], axis=1)
            lp = np.linalg.norm(d["placed"][f[:, a]] - d["placed"][f[:, b]],
                                axis=1)
            np.testing.assert_allclose(lp, lr, rtol=1e-9, atol=1e-9)

    def test_rest_override_is_honoured_and_validated(self):
        d = self.d
        ro = d["rest"].copy()
        ro[:, 0] += 3.0
        d2 = gcd_io.load(TROUSERS, rest_override=ro)
        np.testing.assert_allclose(d2["rest"], ro, atol=1e-12)
        with self.assertRaises(ValueError):
            gcd_io.load(TROUSERS, rest_override=np.zeros((3, 2)))

    def test_hinges_on_the_real_mesh_keep_the_affine_invariant(self):
        """The invariant the unfold is supposed to guarantee, on tens of
        thousands of real hinges rather than a synthetic four-point patch."""
        d = self.d
        rest_tri = d["rest"][d["faces"]]
        H, R4 = A.build_hinges(d["faces"], d["wid"], rest_tri,
                               d["panel_of_face"])
        K, w = A.hinge_stencils(R4)
        self.assertGreater(len(H), 1000)
        scale = np.abs(K).max(1)
        np.testing.assert_allclose(K.sum(1) / scale, 0.0, atol=1e-10)
        np.testing.assert_allclose(
            np.einsum("ha,had->hd", K, R4) / scale[:, None], 0.0, atol=1e-7)
        self.assertTrue(np.isfinite(w).all())
        self.assertTrue((w > 0).all())

    def test_real_rest_areas_are_positive_although_back_panels_are_mirrored(self):
        d = self.d
        rest_tri = d["rest"][d["faces"]]
        _, area = A.shape_gradients(rest_tri)
        self.assertTrue((area > 0).all())
        d1 = rest_tri[:, 1] - rest_tri[:, 0]
        d2 = rest_tri[:, 2] - rest_tri[:, 0]
        signed = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        # the dataset really does mirror one side of the pattern
        self.assertGreater(float((signed < 0).mean()), 0.2)
        self.assertGreater(float((signed > 0).mean()), 0.2)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

CHEAP_LADDER = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]     # --lam-start 1e-3 --lam-stop 1e-5
CHEAP_PER_LAMBDA = 30


def solve_cheap(gdir, per_lambda=CHEAP_PER_LAMBDA, ladder=None):
    """run_garment.py --amp 0 --body --sym --mu 0.02, in process."""
    ladder = CHEAP_LADDER if ladder is None else ladder
    d, gar = run_garment.build(gdir)
    P0 = run_garment.initial(d, seed=1, amp=0.0, sym=True)
    pk = body.pack(body.primitives(
        body.load_measurements(run_garment.measures_of(gdir)), d["placed"],
        np.array(d["panel_names"], dtype=object)[np.maximum(d["panel_of_raw"], 0)],
        d["rest"]))
    mu = np.full(gar["n"], 0.02 * run_garment.diag_scale(gar))
    P, asm, hist, viol, nfac = A.solve_annealed(
        gar, P0, ladder, per_lambda=per_lambda, max_iter=20000,
        mu=mu, clamp=body.projector(pk), recenter=False)
    gap = np.linalg.norm(P[d["pairs"][:, 0]] - P[d["pairs"][:, 1]], axis=1)
    F = A.deformation_gradients(P, gar["faces"], gar["G"])
    _, s = A.best_rotations(F)
    e = hist[-1]
    meta = dict(mono_violations=len(viol), seam_gap_max=float(gap.max()),
                max_sigma_dev=float(np.abs(s - 1).max()),
                E_arap=e["E_arap"], E_bend=e["E_bend"], E_stitch=e["E_stitch"],
                iterations=len(hist), factorizations=nfac)
    return d, gar, P, meta


class GateMixin(object):
    """The four gates README.md lists under 끝났다고 성공이 아니다."""

    def check_gates(self, meta):
        self.assertEqual(meta["mono_violations"], 0)
        self.assertLess(meta["seam_gap_max"], 1e-3)
        self.assertLess(meta["max_sigma_dev"], 1.0)
        for k in ("E_arap", "E_bend", "max_sigma_dev"):
            self.assertTrue(np.isfinite(meta[k]), "%s = %r" % (k, meta[k]))


@unittest.skipUnless(_have(TROUSERS), "dataset absent: %s" % TROUSERS)
class EndToEndTrousers(unittest.TestCase, GateMixin):
    """rand_1328ERLDIC -- 6 panels, ~10.2k verts, the easy one."""

    res = None

    @classmethod
    def setUpClass(cls):
        cls.res = solve_cheap(TROUSERS)

    def test_gates(self):
        self.check_gates(self.res[3])

    def test_one_factorisation_per_rung(self):
        self.assertEqual(self.res[3]["factorizations"], len(CHEAP_LADDER))

    def test_shape_is_finite_and_three_dimensional(self):
        d, gar, P, meta = self.res
        self.assertTrue(np.isfinite(P).all())
        self.assertEqual(P.shape, d["placed"].shape)
        self.assertGreater(float(np.ptp(P, axis=0).min()), 1.0)

    def test_the_solve_is_neither_the_placement_nor_the_drape(self):
        d, gar, P, meta = self.res
        self.assertGreater(np.abs(P - d["placed"]).max(), 1.0)
        self.assertGreater(np.abs(P - d["drape"]).max(), 1.0)

    def test_max_sigma_dev_is_in_the_documented_band(self):
        """the task's note for this garment: max|s-1| ~ 0.12"""
        self.assertLess(self.res[3]["max_sigma_dev"], 0.3)

    def test_bit_identical_on_a_second_run(self):
        """amp 0 makes the whole pipeline deterministic; a two-rung ladder is
        enough to see it, and much cheaper than repeating the full solve."""
        a = solve_cheap(TROUSERS, per_lambda=5, ladder=[1e-1, 1e-2])
        b = solve_cheap(TROUSERS, per_lambda=5, ladder=[1e-1, 1e-2])
        np.testing.assert_array_equal(a[2], b[2])
        self.assertEqual(a[3], b[3])


@unittest.skipUnless(_have(SHIRT), "dataset absent: %s" % SHIRT)
class EndToEndShirt(unittest.TestCase, GateMixin):
    """rand_0B1T21D8NX -- 14 panels with sleeves, ~8.4k verts, the hard one."""

    res = None

    @classmethod
    def setUpClass(cls):
        cls.res = solve_cheap(SHIRT)

    def test_gates(self):
        """NOTE the margin: at this cheap setting max_sigma_dev comes out
        0.905 against a gate of 1.0.  The gate passes, but a small change to
        the ladder or the obstacle can push this garment over -- it is the
        run to watch, not a comfortable pass."""
        self.check_gates(self.res[3])
        self.assertGreater(self.res[3]["max_sigma_dev"], 0.5)

    def test_the_sleeved_garment_really_is_the_harder_case(self):
        """more panels, more seam pairs and far more stretch than the trousers"""
        d = self.res[0]
        self.assertGreaterEqual(len(d["panel_names"]), 10)
        self.assertGreater(len(d["pairs"]), 500)

    def test_shape_is_finite(self):
        self.assertTrue(np.isfinite(self.res[2]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
