"""DXF (ASTM-ish, as CLO exports) -> the same edge features the model wants.

The boundary arrives as a polyline, so the curve TYPE and its control points are gone.
With cfg.curvature_encoding == "tagged" they have to be refitted, and that is where this
path loses accuracy: a circular arc and a weakly-curved quadratic Bezier fit the samples
about equally well, k_t flips, and eleven feature dimensions change meaning at once.
With "sagitta" no type is ever decided -- the descriptor is read straight off the
polyline, which is what the DXF actually gives us.  Everything else (coordinates, chord
length, direction, interior angles, edge count, panel id) is recoverable exactly.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import ezdxf
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "autosew"))
from autosew.config import AutoSewConfig, KT_STRAIGHT, KT_CIRCLE, KT_QUADRATIC, KT_CUBIC
from autosew.curves import sagitta_profile
from autosew.features import N_CURV, feature_dim

MM = 10.0


def read_panels(path):
    """-> [(name, boundary points in cm, edge start indices)] in block order."""
    doc = ezdxf.readfile(path)
    out = []
    for b in doc.blocks:
        if b.name.startswith("*"):
            continue
        poly = next((e for e in b if e.dxftype() == "POLYLINE" and e.dxf.layer == "8"), None)
        if poly is None:
            poly = next((e for e in b if e.dxftype() == "POLYLINE" and e.dxf.layer == "1"), None)
        if poly is None:
            continue
        pts = np.array([list(v.dxf.location)[:2] for v in poly.vertices]) / MM
        turn = np.array([list(p.dxf.location)[:2] for p in b
                         if p.dxftype() == "POINT" and p.dxf.layer == "2"]).reshape(-1, 2) / MM
        turn = np.unique(np.round(turn, 6), axis=0)
        idx = sorted({int(np.argmin(np.linalg.norm(pts - t, axis=1))) for t in turn})
        out.append((b.name, pts, idx))
    return out


def fit_edge(seg):
    """seg: points from one edge start to the next (inclusive). -> (kt, params_abs)."""
    p0, p1 = seg[0], seg[-1]
    chord = p1 - p0
    L = np.hypot(*chord)
    if len(seg) <= 2 or L < 1e-12:
        return KT_STRAIGHT, []
    # perpendicular offsets tell us how curved this is
    n = np.array([-chord[1], chord[0]]) / L
    t = ((seg - p0) @ chord) / (L * L)
    off = (seg - p0) @ n
    if np.abs(off).max() < 1e-6 * max(L, 1.0):
        return KT_STRAIGHT, []
    # quadratic Bezier: one control point, solved in least squares over t
    B = np.stack([(1 - t) ** 2, 2 * (1 - t) * t, t ** 2], 1)
    q, *_ = np.linalg.lstsq(B[:, 1:2], seg - B[:, 0:1] * p0 - B[:, 2:3] * p1, rcond=None)
    res_q = np.abs(B @ np.stack([p0, q[0], p1]) - seg).max()
    # cubic Bezier: two control points
    B3 = np.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], 1)
    c, *_ = np.linalg.lstsq(B3[:, 1:3], seg - B3[:, 0:1] * p0 - B3[:, 3:4] * p1, rcond=None)
    res_c = np.abs(B3 @ np.stack([p0, c[0], c[1], p1]) - seg).max()
    # circular arc: fit a centre equidistant from every sample
    A = np.hstack([2 * seg, np.ones((len(seg), 1))])
    sol, *_ = np.linalg.lstsq(A, (seg ** 2).sum(1), rcond=None)
    ctr = sol[:2]; r = np.sqrt(max(sol[2] + ctr @ ctr, 0.0))
    res_a = np.abs(np.linalg.norm(seg - ctr, axis=1) - r).max()
    tol = 1e-4 * max(L, 1.0)
    if res_a <= tol and res_a <= res_q:
        cross = chord[0] * (seg[len(seg)//2] - p0)[1] - chord[1] * (seg[len(seg)//2] - p0)[0]
        ang = 2 * np.arcsin(min(L / (2 * max(r, L / 2)), 1.0))
        return KT_CIRCLE, [r, 1.0 if ang > np.pi else 0.0, 1.0 if cross > 0 else 0.0]
    if res_q <= tol:
        return KT_QUADRATIC, [q[0][0], q[0][1]]
    return KT_CUBIC, [c[0][0], c[0][1], c[1][0], c[1][1]]


def build(path, cfg):
    panels = read_panels(path)
    rows, keys = [], []
    lo, hi = cfg.edge_count_minmax
    sag = cfg.curvature_encoding == "sagitta"
    nk = cfg.sagitta_samples if sag else N_CURV
    tail = 7 + nk
    for pi, (name, pts, idx) in enumerate(panels):
        n = len(idx)
        segs, starts, ends, kts, kps = [], [], [], [], []
        for k in range(n):
            a, b = idx[k], idx[(k + 1) % n]
            seg = pts[a:b + 1] if b > a else np.vstack([pts[a:], pts[:b + 1]])
            segs.append(seg); starts.append(seg[0]); ends.append(seg[-1])
            kt, kp = (KT_STRAIGHT, []) if sag else fit_edge(seg)
            kts.append(kt); kps.append(kp)
        starts = np.array(starts); ends = np.array(ends)
        # ACW canonicalisation, on the chord polygon like the parser does
        area = np.sum(starts[:, 0] * np.roll(starts[:, 1], -1)
                      - np.roll(starts[:, 0], -1) * starts[:, 1])
        if area < 0:
            order = list(range(n))[::-1]
            starts, ends = ends[order], starts[order]
            kts = [kts[i] for i in order]; kps = [kps[i] for i in order]
            segs = [segs[i][::-1] for i in order]
        # panel-local: bbox lower-left to the origin
        mn = np.vstack([starts, ends]).min(0)
        starts -= mn; ends -= mn
        kps = [[v - (mn[0] if j % 2 == 0 else mn[1]) for j, v in enumerate(p)]
               if kt in (KT_QUADRATIC, KT_CUBIC) else p for kt, p in zip(kts, kps)]
        d = ends - starts
        Ls = np.hypot(d[:, 0], d[:, 1])
        dirs = d / np.maximum(Ls, 1e-12)[:, None]
        ang = []
        for i in range(n):
            aa, bb = dirs[i - 1], dirs[i]
            ang.append(np.pi - np.arctan2(aa[0]*bb[1]-aa[1]*bb[0], aa@bb))
        for j in range(n):
            x = np.zeros(feature_dim(cfg), np.float32)
            x[0:2] = starts[j] / cfg.scale_div; x[2:4] = ends[j] / cfg.scale_div
            x[4] = Ls[j] / cfg.scale_div; x[5:7] = dirs[j]
            if sag:
                # straight off the polyline; no curve type is ever decided
                x[7:tail] = sagitta_profile(segs[j], nk)
            else:
                x[7] = kts[j]
                for qi, v in enumerate(kps[j][:10]):
                    x[8+qi] = v / cfg.scale_div if not (kts[j] == KT_CIRCLE and qi) else v
                if kts[j] == KT_CIRCLE and kps[j]:
                    x[8] = kps[j][0] / cfg.scale_div
            aL, aR = ang[j], ang[(j+1) % n]
            x[tail+0], x[tail+1] = np.sin(aL), np.cos(aL)
            x[tail+2], x[tail+3] = np.sin(aR), np.cos(aR)
            x[tail+4] = min(max((n - lo) / max(hi - lo, 1e-6), 0.0), 1.0)
            x[tail+5] = pi / cfg.max_panels_norm
            rows.append(x); keys.append((name, j))
    return np.array(rows), keys


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dxf", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    x, keys = build(a.dxf, AutoSewConfig())
    np.save(a.out, x)
    print(f"{a.dxf}: M={len(keys)}  x.shape={x.shape}")
