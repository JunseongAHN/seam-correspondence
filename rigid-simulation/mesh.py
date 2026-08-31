"""Flat rest meshes, seam pairings, hinge extraction, ASCII PLY I/O.

Every mesh here is a *flat* panel: rest positions live in R^2.  Seams are
*pairs of distinct vertices* that a soft penalty pulls together -- the
vertices are never merged, so the seam line carries no hinge.
"""

import numpy as np


class Mesh:
    """A flat panel plus its seam pairing.

    rest   (N,2) float64 -- flat material coordinates
    faces  (M,3) int32   -- triangles, consistently counter-clockwise in the plane
    pairs  (K,2) int32   -- stitched vertex pairs
    """

    def __init__(self, rest, faces, pairs, name="", seam_hinges=None, seam_hinge_rest=None):
        self.rest = np.ascontiguousarray(rest, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int32)
        self.pairs = np.ascontiguousarray(pairs, dtype=np.int32).reshape(-1, 2)
        self.name = name
        # Seam-crossing hinges.  The spec excludes these from the bending penalty on
        # the grounds that their rest dihedral is undefined -- true for a general
        # seam, but NOT for these two cases, where the flat rest continues across the
        # seam isometrically (a translation for the rectangle, a rotation for the
        # sector).  So we can build them, and `seam_hinge_rest` carries the unrolled
        # rest positions the stencil must be computed from.  Off by default.
        self.seam_hinges = (np.zeros((0, 4), np.int32) if seam_hinges is None
                            else np.asarray(seam_hinges, np.int32).reshape(-1, 4))
        self.seam_hinge_rest = (np.zeros((0, 4, 2)) if seam_hinge_rest is None
                                else np.asarray(seam_hinge_rest, float).reshape(-1, 4, 2))

    @property
    def n(self):
        return self.rest.shape[0]

    def areas(self):
        a = self.rest[self.faces[:, 1]] - self.rest[self.faces[:, 0]]
        b = self.rest[self.faces[:, 2]] - self.rest[self.faces[:, 0]]
        return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    def hinges(self):
        """Interior edges as (v0, v1, v2, v3): v0-v1 is the shared edge, v2 and
        v3 are the opposite vertices of the two incident triangles.

        Only edges shared by exactly two triangles appear.  A seam is made of
        *duplicated* boundary vertices, so no edge ever spans it -- seam-crossing
        hinges are excluded structurally, not by a filter.
        """
        seen = {}
        for t, (i, j, k) in enumerate(self.faces):
            for a, b, opp in ((i, j, k), (j, k, i), (k, i, j)):
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    seen[key].append(opp)
                else:
                    seen[key] = [opp]
        out = []
        for (a, b), opps in seen.items():
            if len(opps) == 2:
                out.append((a, b, opps[0], opps[1]))
        return np.array(sorted(out), dtype=np.int32).reshape(-1, 4)


def make_rectangle(W=2.0 * np.pi, H=4.0, nw=64, nh=32):
    """Rectangle [0,W] x [0,H] on a uniform (nw+1) x (nh+1) grid.

    Seam: the x=0 edge to the x=W edge, matched by y index.
    """
    ix, iy = np.meshgrid(np.arange(nw + 1), np.arange(nh + 1), indexing="ij")
    rest = np.stack([ix.ravel(order="F") * (W / nw), iy.ravel(order="F") * (H / nh)], axis=1)

    def vid(a, b):
        return b * (nw + 1) + a

    faces = []
    for b in range(nh):
        for a in range(nw):
            v00, v10, v11, v01 = vid(a, b), vid(a + 1, b), vid(a + 1, b + 1), vid(a, b + 1)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    pairs = [(vid(0, b), vid(nw, b)) for b in range(nh + 1)]

    # seam hinges: the x=0 copy is the canonical shared edge; the far triangle's
    # rest position is translated by -W, which is exactly the seam's rest isometry
    sh, shr = [], []
    dx = W / nw
    for b in range(nh):
        sh.append((vid(0, b), vid(0, b + 1), vid(1, b + 1), vid(nw - 1, b)))
        y0, y1 = rest[vid(0, b), 1], rest[vid(0, b + 1), 1]
        shr.append([[0.0, y0], [0.0, y1], [dx, y1], [-dx, y0]])
    return Mesh(rest, faces, pairs, name="rectangle", seam_hinges=sh, seam_hinge_rest=shr)


def make_sector(L=3.0, angle=2.0 * np.pi / 3.0, nr=40, ntheta=48):
    """Circular sector of radius L and central angle `angle`, polar grid.

    r=0 collapses to a single apex vertex (index 0); the first ring is a
    triangle fan, so no triangle is degenerate.

    Seam: the theta=0 straight edge to the theta=angle straight edge, matched by
    r.  The apex is already shared and is *not* given a duplicate pair.
    """
    th = np.linspace(0.0, angle, ntheta + 1)
    rr = np.linspace(0.0, L, nr + 1)[1:]          # rings 1..nr, apex handled separately
    ring = np.stack([np.outer(rr, np.cos(th)).ravel(), np.outer(rr, np.sin(th)).ravel()], axis=1)
    rest = np.vstack([[0.0, 0.0], ring])

    def vid(j, k):                                 # j in 1..nr, k in 0..ntheta
        return 1 + (j - 1) * (ntheta + 1) + k

    faces = []
    for k in range(ntheta):                        # apex fan
        faces.append((0, vid(1, k), vid(1, k + 1)))
    for j in range(1, nr):                         # quad rings, split like the rectangle
        for k in range(ntheta):
            v00, v10, v11, v01 = vid(j, k), vid(j + 1, k), vid(j + 1, k + 1), vid(j, k + 1)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    pairs = [(vid(j, 0), vid(j, ntheta)) for j in range(1, nr + 1)]

    # seam hinges: theta=0 copy is canonical; the far triangle's rest position is
    # rotated by -angle, which is exactly the seam's rest isometry
    c, sn = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -sn], [sn, c]])
    sh, shr = [], []
    for j in range(nr):
        if j == 0:
            quad = (0, vid(1, 0), vid(1, 1), vid(1, ntheta - 1))
        else:
            quad = (vid(j, 0), vid(j + 1, 0), vid(j + 1, 1), vid(j, ntheta - 1))
        sh.append(quad)
        x = rest[list(quad)].copy()
        x[3] = R @ x[3]
        shr.append(x)
    return Mesh(rest, faces, pairs, name="sector", seam_hinges=sh, seam_hinge_rest=shr)


def write_ply(path, P, faces):
    """ASCII PLY: vertices as `x y z`, faces as `3 i j k`."""
    P = np.asarray(P, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % len(P))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("element face %d\n" % len(faces))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for p in P:
            f.write("%.17g %.17g %.17g\n" % (p[0], p[1], p[2]))
        for t in faces:
            f.write("3 %d %d %d\n" % (t[0], t[1], t[2]))


def read_ply(path):
    with open(path) as f:
        lines = f.read().split("\n")
    nv = nf = 0
    h = 0
    for i, ln in enumerate(lines):
        if ln.startswith("element vertex"):
            nv = int(ln.split()[-1])
        elif ln.startswith("element face"):
            nf = int(ln.split()[-1])
        elif ln.strip() == "end_header":
            h = i + 1
            break
    P = np.array([[float(x) for x in lines[h + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[h + nv + i].split()[1:4]] for i in range(nf)], dtype=np.int32)
    return P, F
