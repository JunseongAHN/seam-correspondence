"""Edge polyline reconstruction and the sagitta shape descriptor.

The tagged-union curvature encoding (k_t + k1..k10, where k_t switches the meaning of
the ten parameter slots) is fine when k_t comes from a specification file and is exact.
It is not usable when the input is a DXF polyline, where k_t has to be *estimated*: a
circular arc and a weakly-curved quadratic Bezier fit sampled points about equally well,
so k_t flips, and eleven feature dimensions change meaning at once.

The sagitta profile removes the type decision entirely.  Both a specification curve and a
DXF polyline are reduced to the same thing -- signed perpendicular deviation from the
chord, at K uniform arclength positions, divided by the chord length:

    v_k = ((p(s_k) - p0) . n) / L ,   s_k = k/(K+1) of the arc,  k = 1..K

with n the left normal of the chord in the panel's ACW traversal frame.  The values are
dimensionless, always mean the same thing, and vary continuously with the shape, so a
refitting error is a small perturbation rather than a categorical error.  Two edges that
get stitched together have equal length and mirror-image curvature, which in this
encoding is a sign flip plus a reversal -- directly visible to an inner product.
"""
from __future__ import annotations
import math

import numpy as np

from .config import KT_STRAIGHT, KT_CIRCLE, KT_QUADRATIC, KT_CUBIC, KT_BSPLINE

SAMPLES_PER_EDGE = 32   # curve -> polyline resolution before resampling


def arc_polyline(p0, p1, r, large, sweep, n=SAMPLES_PER_EDGE):
    """SVG-style circular arc, same convention as viz/visualize.py:arc_points.
    GCD circle params are [radius, large_arc, right]."""
    chord = p1 - p0
    d = float(np.linalg.norm(chord))
    if d < 1e-9:
        return np.stack([p0, p1])
    r = max(float(r), d / 2 + 1e-9)
    mid = (p0 + p1) / 2
    h = math.sqrt(max(r * r - d * d / 4, 0.0))
    perp = np.array([-chord[1], chord[0]]) / d
    center = mid + (h if bool(large) != bool(sweep) else -h) * perp
    a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
    a1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
    if sweep:
        while a1 < a0:
            a1 += 2 * math.pi
    else:
        while a1 > a0:
            a1 -= 2 * math.pi
    t = np.linspace(a0, a1, n)
    return np.stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)], axis=1)


def edge_polyline(edge, n=SAMPLES_PER_EDGE):
    """Parsed Edge -> polyline in the same panel-local frame as edge.start/edge.end.

    Bezier control points in edge.kparams are already absolute and already translated
    with the panel bbox, so they need no further conversion.  B-splines fall back to the
    chord (the parameter layout is not pinned down; they do not occur in GCD.v2 specs
    seen so far -- callers that care should count edge.kt == KT_BSPLINE)."""
    p0 = np.asarray(edge.start, float)
    p1 = np.asarray(edge.end, float)
    kp = edge.kparams
    if edge.kt == KT_QUADRATIC and len(kp) >= 2:
        q = np.array(kp[:2], float)
        t = np.linspace(0, 1, n)[:, None]
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * q + t ** 2 * p1
    if edge.kt == KT_CUBIC and len(kp) >= 4:
        q1 = np.array(kp[0:2], float)
        q2 = np.array(kp[2:4], float)
        t = np.linspace(0, 1, n)[:, None]
        return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * q1
                + 3 * (1 - t) * t ** 2 * q2 + t ** 3 * p1)
    if edge.kt == KT_CIRCLE and len(kp) >= 1:
        large = kp[1] if len(kp) > 1 else 0
        sweep = kp[2] if len(kp) > 2 else 0
        return arc_polyline(p0, p1, kp[0], large, sweep, n)
    return np.stack([p0, p1])


def sagitta_profile(pts, K):
    """Polyline -> (K,) signed perpendicular deviation from the chord, divided by the
    chord length, sampled at uniform arclength fractions k/(K+1).

    Endpoints are excluded: their deviation is identically zero and carries nothing."""
    pts = np.asarray(pts, float)
    out = np.zeros(K, dtype=np.float64)
    if len(pts) < 3:
        return out
    p0, p1 = pts[0], pts[-1]
    chord = p1 - p0
    L = float(np.hypot(chord[0], chord[1]))
    if L < 1e-12:
        return out
    nrm = np.array([-chord[1], chord[0]]) / L      # left normal, ACW frame
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return out
    s /= s[-1]
    targets = np.arange(1, K + 1) / (K + 1.0)
    qx = np.interp(targets, s, pts[:, 0])
    qy = np.interp(targets, s, pts[:, 1])
    return ((np.stack([qx, qy], 1) - p0) @ nrm) / L
