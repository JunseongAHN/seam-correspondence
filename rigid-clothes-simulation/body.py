"""Analytic body proxy: plain circular cylinders plus one sphere for the head.

Built from <name>_body_measurements.yaml -- an INPUT to GarmentCode's pattern
generation, not an output of the drape -- and from the specification placement,
which is likewise an input.  Nothing here is fitted to the drape.

Three decisions worth stating.

1. Every primitive is a PLAIN cylinder (constant radius, flat caps) or a SPHERE,
   so the closest point on it is closed form.  No taper, no anisotropic scaling,
   no iteration.  The sphere is the cheaper of the two: one norm, no axial
   projection.

2. Each primitive is INSCRIBED in the part it stands for, so the proxy is a
   conservative under-estimate of the body and can never push the garment
   further out than a real body would.  The torso cross-section is an ellipse:
   `hip_back_width` = 54.82 cannot be a half-width, since no ellipse of
   circumference 103.48 reaches a semi-axis of 27.41 (even the degenerate b -> 0
   ellipse has perimeter 4a = 109.6), so the `*_back_width` entries are arcs over
   the back half, ~C/2.  The one genuine chord in the file is `shoulder_w`
   = 36.46; with the chest circumference its back arc implies (2 * back_width
   = 95.36), Ramanujan's perimeter gives b = 11.8 and an aspect ratio
   a/b = 1.548.  That single ratio carries down the torso and each level's (a, b)
   then follows from its own measured circumference.  The cylinder radius is the
   SEMI-MINOR axis b, the half-depth -- the circle inscribed in the true
   cross-section.  Taking C/(2*pi) instead would make the torso 32.9 cm deep at
   the hips where the body is 25.6, and would wedge the front and back apart.

3. The ARM axis is read off the placement rather than from `arm_pose_angle`.
   Taking that angle from the vertical puts the arm axis 13 cm above the cuff:
   the sleeve sits at x 12.6..36.2, y 112.6..138.6, but the angle sends the axis
   through y = 133 at the cuff's x = 30.7 and on out to x = 56.7, well past the
   sleeve.  Rather than guess the convention, the arm is taken as the principal
   axis of the placed sleeve panels -- a sleeve is a tube around the arm, so its
   principal axis IS the arm -- truncated to the span the sleeve covers, since an
   obstacle is only needed where the garment is.  Its radius is the measured
   armscye-to-wrist taper evaluated at the far end, so it stays inscribed.

Frame: x left-right, y up from the floor, +z front -- the absolute frame the
specification placement already uses.
"""

import numpy as np
import yaml


def load_measurements(path):
    return yaml.safe_load(open(path))["body"]


def ellipse_perimeter(a, b):
    """Ramanujan's second approximation; good to 1e-5 at these aspect ratios."""
    return np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b)))


def aspect_ratio(m):
    """a/b of the torso, from the only chord width in the file (shoulder_w) and
    the chest circumference its back arc implies."""
    a = m["shoulder_w"] / 2.0
    C = 2.0 * m["back_width"]
    lo, hi = 1e-3, a
    for _ in range(60):                          # perimeter is monotone in b
        mid = 0.5 * (lo + hi)
        if ellipse_perimeter(a, mid) < C:
            lo = mid
        else:
            hi = mid
    return 2.0 * a / (lo + hi)


def half_depth(C, k):
    """Semi-minor axis of the ellipse of circumference C with a/b = k.  Ramanujan's
    perimeter is homogeneous of degree 1, so one evaluation fixes the scale."""
    return C / ellipse_perimeter(k, 1.0)


def ring_radius(rest, panel_names, keys):
    """Radius of the tube the listed panels can form, from the FLAT pattern.

    In this pattern the panel's x extent is the "around the body" direction, and
    that reading validates against the measurements it should reproduce: the four
    torso panels sum to 99.90 cm against a measured bust of 99.84, and the two
    waistband panels to 84.39 against a measured waist of 84.33 -- 0.07%.
    """
    tot = 0.0
    for nm in sorted(set(str(x) for x in panel_names)):
        if any(k in nm for k in keys):
            tot += float(np.ptp(rest[panel_names == nm][:, 0]))
    return tot / (2.0 * np.pi)


def arm_axis(placed, panel_names, m, rest):
    """[(p0, p1, r)] for the two arms, from the placed sleeve panels.

    The two sides are averaged and mirrored so the proxy is exactly symmetric
    even though the two sleeve meshes are not.
    """
    r_armscye = m["armscye_depth"] / 2.0
    r_wrist = m["wrist"] / (2.0 * np.pi)
    P0s, P1s = [], []
    for side, sg in (("left", 1.0), ("right", -1.0)):
        is_side = np.array([side in str(x) for x in panel_names])
        cuff = is_side & np.array(["cuff" in str(x) for x in panel_names])
        slv = is_side & np.array(["sleeve" in str(x) for x in panel_names])
        if not slv.any():
            return []                                 # sleeveless garment
        if not cuff.any():
            # no cuff: the sleeve's own lowest ring is the far end
            Y = placed[slv][:, 1]
            cuff = slv.copy()
            cuff[slv] = Y <= np.quantile(Y, 0.15)
        far = placed[cuff].mean(0)                    # centre of the cuff ring
        Y = placed[slv][:, 1]
        top = placed[slv][Y >= np.quantile(Y, 0.85)]
        near = top.mean(0)                            # centre of the armscye end
        for p in (near, far):
            p[0] *= sg                                # mirror the right side onto the left
        P0s.append(near)
        P1s.append(far)
    p0 = np.mean(P0s, 0)
    p1 = np.mean(P1s, 0)

    f = min(np.linalg.norm(p1 - p0) / m["arm_length"], 1.0)
    r = r_armscye + f * (r_wrist - r_armscye)         # taper at the far end
    # ...but the obstacle must also fit inside what the GARMENT can wrap, which
    # here is the binding constraint: the cuff ring is only 19.9 cm around
    # (r = 3.17) against an arm of r = 5.15 there.  This dress has no ease at all
    # -- its torso ring is 99.90 cm against a measured bust of 99.84 -- so an
    # obstacle even slightly too large has no isometric solution and the solver
    # answers by tearing: with r = 5.15 the cuff came out at p50 |sigma-1| = 1.49.
    # cap by what one arm's tightest ring can wrap, when there is a cuff to read
    cap = ring_radius(rest, panel_names, ("left_cuff",))
    if cap > 1e-6:
        r = min(r, cap)
    out = []
    for sg in (1.0, -1.0):
        M = np.array([sg, 1.0, 1.0])
        out.append((p0 * M, p1 * M, r))
    return out


def primitives(m, placed, panel_names, rest):
    """(cylinders, spheres) = ([(p0, p1, r)], [(centre, r)]), centimetres."""
    tau = 2.0 * np.pi
    k = aspect_ratio(m)
    y_waist = m["_waist_level"]
    y_hip = y_waist - m["hips_line"]
    y_bust = y_waist + m["vert_bust_line"]
    y_shldr = m["height"] - m["head_l"]              # the head sits on the shoulders

    b_hip = half_depth(m["hips"], k)
    b_waist = half_depth(m["waist"], k)
    b_bust = half_depth(m["bust"], k)
    b_shldr = half_depth(2.0 * m["back_width"], k)
    a_hip = k * b_hip

    r_thigh = m["leg_circ"] / tau
    r_ankle = m["wrist"] / tau                       # ankle ~ wrist
    r_head = m["head_l"] / 3.0                       # head length ~ 3 head radii

    def col(x, y0, y1, r):                           # a vertical cylinder
        return (np.array([x, y0, 0.0]), np.array([x, y1, 0.0]), r)

    C = [col(0.0, y_hip, y_waist, min(b_hip, b_waist)),
         col(0.0, y_waist, y_bust, min(b_waist, b_bust)),
         col(0.0, y_bust, y_shldr, min(b_bust, b_shldr))]

    # legs: two columns inscribed in the hip ellipse, each split thigh / calf
    dx = max(a_hip - r_thigh, 0.0)
    y_knee = 0.5 * y_hip
    r_knee = 0.5 * (r_thigh + r_ankle)
    for s in (-1.0, 1.0):
        C.append(col(s * dx, y_knee, y_hip, r_knee))
        C.append(col(s * dx, 0.0, y_knee, r_ankle))

    C += list(arm_axis(placed, panel_names, m, rest))

    S = [(np.array([0.0, m["height"] - r_head, 0.0]), r_head)]
    return C, S


def pack(prim):
    C, S = prim
    P0 = np.array([c[0] for c in C])
    D = np.array([c[1] for c in C]) - P0
    L = np.linalg.norm(D, axis=1)
    Sc = np.array([s[0] for s in S]) if S else np.zeros((0, 3))
    Sr = np.array([s[1] for s in S]) if S else np.zeros(0)
    N = D / L[:, None]
    return (P0, N, L, np.array([c[2] for c in C]), Sc, Sr,
            np.einsum("kd,kd->k", P0, N), np.einsum("kd,kd->k", P0, P0),
            np.einsum("kd,kd->k", Sc, Sc))


def penetration(X, packed):
    """(depth, exit point) per vertex, for the primitive it is deepest inside.

    depth <= 0 means the vertex is outside every primitive.  For a point inside a
    finite cylinder the nearest boundary point is exactly one of three -- radially
    out to the wall, or straight out through either cap -- and for a sphere it is
    radially out from the centre.  Both are closed form, so the projection is
    exact per primitive.

    The test itself needs only scalars: |X - P0|^2 comes from three matrix
    products and the radial distance from |W|^2 - s^2, so no (n, k, 3) array is
    ever formed.  Only the handful of vertices that are actually inside get their
    exit point built.  That matters -- this runs once per solver iteration.
    """
    P0, N, L, R, Sc, Sr, P0N, P0P0, ScSc = packed
    X2 = np.einsum("nd,nd->n", X, X)[:, None]

    s = X @ N.T - P0N[None]                                 # (n, k) axial coordinate
    w2 = X2 - 2.0 * (X @ P0.T) + P0P0[None]                 # |X - P0|^2
    rho = np.sqrt(np.maximum(w2 - s * s, 0.0))

    d_wall = R[None] - rho                                  # all three > 0 <=> inside
    depth = np.minimum(d_wall, np.minimum(s, L[None] - s))

    k = np.argmax(depth, axis=1)
    i = np.arange(len(X))
    dep = depth[i, k]

    out = X
    hit = dep > 0.0
    if hit.any():
        out = X.copy()
        j = np.where(hit)[0]
        kj = k[j]
        wall, c0 = d_wall[j, kj], s[j, kj]
        c1 = L[kj] - c0
        Nj = N[kj]
        base = P0[kj] + c0[:, None] * Nj                    # foot on the axis
        v = X[j] - base
        rj = rho[j, kj]
        u = np.where(rj[:, None] > 1e-9, v / np.maximum(rj[:, None], 1e-9),
                     np.array([1.0, 0.0, 0.0]))             # on the axis: send to +x
        pick = np.argmin(np.stack([wall, c0, c1], 1), axis=1)[:, None]
        out[j] = np.where(pick == 0, base + R[kj][:, None] * u,
                          np.where(pick == 1, X[j] - c0[:, None] * Nj,
                                   X[j] + c1[:, None] * Nj))

    if len(Sr):
        ds = np.sqrt(np.maximum(X2 - 2.0 * (X @ Sc.T) + ScSc[None], 0.0))
        sdep = Sr[None] - ds
        q = np.argmax(sdep, axis=1)
        sd = sdep[i, q]
        # BOTH conditions: the sphere must be deeper AND actually contain the
        # vertex.  Without "sd > 0" every vertex merely nearer the sphere than to
        # any cylinder -- both distances negative -- gets teleported onto it.
        take = (sd > dep) & (sd > 0.0)
        if take.any():
            if out is X:
                out = X.copy()
            j = np.where(take)[0]
            v = X[j] - Sc[q[j]]
            n = np.linalg.norm(v, axis=1, keepdims=True)
            v = np.where(n > 1e-9, v / np.maximum(n, 1e-9), np.array([0.0, 1.0, 0.0]))
            out[j] = Sc[q[j]] + Sr[q[j]][:, None] * v
            dep = np.where(take, sd, dep)
    return dep, out


def projector(packed, sweeps=2):
    """clamp(P): move every vertex inside the body out onto its surface.

    Leaving one primitive can land a vertex inside a neighbour, so the union is
    swept a few times; each sweep is exact for the primitive it acts on.
    """
    def clamp(P):
        for _ in range(sweeps):
            dep, out = penetration(P, packed)
            if not (dep > 0.0).any():
                break
            P = out
        return P
    return clamp
