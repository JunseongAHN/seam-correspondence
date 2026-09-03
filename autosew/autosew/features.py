"""Pattern -> node features (M,24), cycle-graph neighbor indices (M,2), GT tensors.

Feature layout (indices, see supp Table 1 + preprocessing §3.1):
   0: x0/100    1: y0/100     start vertex (panel-local, bbox-LL at origin)
   2: x1/100    3: y1/100     end vertex
   4: chord length /100
   5: ox        6: oy         unit vector start->end
   7: curvature type k_t (raw 0..5, or /5 if cfg.curvature_type_norm)
   8..17: curvature params k1..k10 (abs frame: /100; rel frame: raw), zero-padded
  18: sin(alpha_l)  19: cos(alpha_l)   interior angle at start vertex
  20: sin(alpha_r)  21: cos(alpha_r)   interior angle at end vertex
  22: per-panel edge count N_e, min-max -> [0,1]
  23: panel ID u (see cfg.panel_id_mode)

With cfg.curvature_encoding == "sagitta" (the DXF/FBX industrial track) dims 7..17 are
replaced by K = cfg.sagitta_samples signed sagitta values (see curves.py) and the six
trailing features shift to follow them; K = 11 leaves the layout 24-dim and the trailing
six at 18..23, so the two encodings differ in dims 7..17 and nowhere else.

With cfg.arc_features two more are appended (24, 25 in that layout): the ARC length /100
and arc/chord.  Dim 4 is the chord, but what a seam matches is the arc -- the length of
fabric you sew along -- and on GCD the arc agrees between the two sides of a stitch far
more often than the chord, on exactly the seam types the model is worst at.
"""
from __future__ import annotations
import math
import random

import numpy as np

from .config import AutoSewConfig, KT_QUADRATIC, KT_CUBIC
from .curves import edge_polyline, sagitta_profile
from .gcd_parser import Pattern

F_DIM = 24
N_CURV = 11   # dims 7..17: the curvature block, whatever encodes it
N_ARC = 2     # appended when cfg.arc_features: arclength/100, arclength/chord


def feature_dim(cfg: AutoSewConfig) -> int:
    """Node feature width for this config; set cfg.in_dim to match."""
    k = cfg.sagitta_samples if cfg.curvature_encoding == "sagitta" else N_CURV
    return F_DIM - N_CURV + k + (N_ARC if cfg.arc_features else 0)


def _interior_angles(panel):
    """Interior angle at each traversal vertex of the ACW cycle, via chord directions.
    Returns list a[i] = interior angle at start vertex of edge i (angle between edge i-1 and edge i)."""
    n = len(panel.edges)
    dirs = []
    for e in panel.edges:
        dx, dy = e.end[0] - e.start[0], e.end[1] - e.start[1]
        L = math.hypot(dx, dy) or 1.0
        dirs.append((dx / L, dy / L))
    ang = []
    for i in range(n):
        a = dirs[i - 1]  # incoming
        b = dirs[i]      # outgoing
        turn = math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
        interior = math.pi - turn  # ACW polygon: convex vertex -> turn>0 -> interior<pi
        ang.append(interior)
    return ang


def pattern_to_tensors(p: Pattern, cfg: AutoSewConfig, rng: random.Random | None = None):
    """Returns dict of numpy arrays:
       x        (M,24) float32
       nbr      (M,2)  int64   cycle neighbors (prev, next) within panel
       gt_pairs (K,2)  int64   unordered GT stitch pairs (i<j)
       stitched (M,)   bool    node participates in >=1 stitch
    """
    keys = p.edge_key_list()
    M = len(keys)
    nk = cfg.sagitta_samples if cfg.curvature_encoding == "sagitta" else N_CURV
    tail = 7 + nk                     # first index after the curvature block
    arc0 = tail + 6                   # first index after panel_u, where the arc pair goes
    x = np.zeros((M, feature_dim(cfg)), dtype=np.float32)
    nbr = np.zeros((M, 2), dtype=np.int64)

    n_panels = len(p.panels)
    if cfg.panel_id_mode == "random_norm":
        assert rng is not None
        ids = list(range(n_panels))
        rng.shuffle(ids)
    else:
        ids = list(range(n_panels))

    lo, hi = cfg.edge_count_minmax
    node = 0
    for pi, panel in enumerate(p.panels):
        n = len(panel.edges)
        base = node
        angles = _interior_angles(panel)
        for j, e in enumerate(panel.edges):
            s, t = e.start, e.end
            dx, dy = t[0] - s[0], t[1] - s[1]
            L = math.hypot(dx, dy)
            ox, oy = (dx / L, dy / L) if L > 0 else (0.0, 0.0)
            x[node, 0] = s[0] / cfg.scale_div
            x[node, 1] = s[1] / cfg.scale_div
            x[node, 2] = t[0] / cfg.scale_div
            x[node, 3] = t[1] / cfg.scale_div
            x[node, 4] = L / cfg.scale_div
            x[node, 5] = ox
            x[node, 6] = oy
            poly = edge_polyline(e) if (cfg.curvature_encoding == "sagitta"
                                        or cfg.arc_features) else None
            if cfg.curvature_encoding == "sagitta":
                x[node, 7:tail] = sagitta_profile(poly, nk)
            else:
                x[node, 7] = e.kt / 5.0 if cfg.curvature_type_norm else float(e.kt)
                kp = e.kparams if cfg.curvature_frame == "abs" else e.kparams_rel
                for q, val in enumerate(kp[:10]):
                    if cfg.curvature_frame == "abs":
                        # circle params = [radius, flag, flag]: scale the radius only
                        from .config import KT_CIRCLE
                        if e.kt == KT_CIRCLE:
                            v = val / cfg.scale_div if q == 0 else val
                        else:
                            v = val / cfg.scale_div
                    else:
                        v = val
                    x[node, 8 + q] = v
            al = angles[j]
            ar = angles[(j + 1) % n]
            x[node, tail + 0] = math.sin(al)
            x[node, tail + 1] = math.cos(al)
            x[node, tail + 2] = math.sin(ar)
            x[node, tail + 3] = math.cos(ar)
            x[node, tail + 4] = min(max((n - lo) / max(hi - lo, 1e-6), 0.0), 1.0)
            if cfg.panel_id_mode == "index_raw":
                x[node, tail + 5] = float(ids[pi])
            else:
                x[node, tail + 5] = ids[pi] / cfg.max_panels_norm
            if cfg.arc_features:
                # what a seam matches is the arc, not the chord: the length of fabric you
                # sew along.  The ratio is >= 1 and says how much the edge bows out.
                arc = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
                x[node, arc0 + 0] = arc / cfg.scale_div
                x[node, arc0 + 1] = arc / L if L > 1e-9 else 1.0
            # cycle neighbors (panel with 1 edge -> self)
            nbr[node, 0] = base + (j - 1) % n
            nbr[node, 1] = base + (j + 1) % n
            node += 1

    pairs = sorted(p.gt_pairs())
    gt = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    stitched = np.zeros(M, dtype=bool)
    for a, b in pairs:
        stitched[a] = True
        stitched[b] = True

    return {"x": x, "nbr": nbr, "gt_pairs": gt, "stitched": stitched, "name": p.name}
