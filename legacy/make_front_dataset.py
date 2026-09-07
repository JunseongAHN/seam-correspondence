#!/usr/bin/env python
"""Garment-level front-view dataset:  panel mask (res^2)  ->  height map h (res^2).

One sample = one garment.  Orthographic front view (front = max z), proper
triangle rasterisation with a z-buffer -- NOT vertex splat, which under-counts
occlusion (see 2026-08-31-depth-pi2-findings.md sec.9).

Per garment it writes an .npz:
    panel_id  int16   [res,res]  which panel owns the pixel, -1 = background
    mask      uint8   [res,res]  1 where the garment covers the pixel
    h         float32 [res,res]  front-most surface z in cm, NaN outside
    h_norm    float32 [res,res]  h mapped to [-1,1] by the stored z range, 0 outside
plus panel names, the frame, and coverage stats.

Image convention: row 0 = top (max y), col 0 = left (min x).

    python make_front_dataset.py --root <default_body/data> --n 10 --res 1024
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise-inject"))
from ninject.io_gcd import load, panel_membership


def face_panels(F, labels):
    """-> (panel_names, face_panel[int16])  face owns the panel all 3 verts share."""
    panels, memb, _ = panel_membership(F, labels)
    fp = np.full(len(F), -1, np.int16)
    for pk in range(len(panels)):
        own = np.array([pk in memb[v] for v in range(len(labels))])
        sel = own[F].all(1) & (fp < 0)
        fp[sel] = pk
    return panels, fp


def rasterize(V, F, fp, res, x0, y0, span):
    """Orthographic +z front view with a z-buffer. -> (z[res,res], pid[res,res]) in [x,y]."""
    step = span / res
    px = (V[:, 0] - x0) / step
    py = (V[:, 1] - y0) / step
    z = V[:, 2]
    zbuf = np.full((res, res), -np.inf)
    pid = np.full((res, res), -1, np.int16)
    fx0, fx1, fx2 = px[F[:, 0]], px[F[:, 1]], px[F[:, 2]]
    fy0, fy1, fy2 = py[F[:, 0]], py[F[:, 1]], py[F[:, 2]]
    area = (fx1 - fx0) * (fy2 - fy0) - (fx2 - fx0) * (fy1 - fy0)
    bx0 = np.clip(np.floor(np.minimum(np.minimum(fx0, fx1), fx2)).astype(int), 0, res - 1)
    bx1 = np.clip(np.ceil(np.maximum(np.maximum(fx0, fx1), fx2)).astype(int), 0, res - 1)
    by0 = np.clip(np.floor(np.minimum(np.minimum(fy0, fy1), fy2)).astype(int), 0, res - 1)
    by1 = np.clip(np.ceil(np.maximum(np.maximum(fy0, fy1), fy2)).astype(int), 0, res - 1)
    for t in range(len(F)):
        a = area[t]
        if abs(a) < 1e-12:
            continue
        i0, i1, j0, j1 = bx0[t], bx1[t], by0[t], by1[t]
        X = (np.arange(i0, i1 + 1) + .5)[:, None]
        Y = (np.arange(j0, j1 + 1) + .5)[None, :]
        w0 = ((fx1[t] - X) * (fy2[t] - Y) - (fx2[t] - X) * (fy1[t] - Y)) / a
        w1 = ((fx2[t] - X) * (fy0[t] - Y) - (fx0[t] - X) * (fy2[t] - Y)) / a
        w2 = 1.0 - w0 - w1
        ins = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not ins.any():
            continue
        v = F[t]
        zz = w0 * z[v[0]] + w1 * z[v[1]] + w2 * z[v[2]]
        sub = zbuf[i0:i1 + 1, j0:j1 + 1]
        upd = ins & (zz > sub)
        sub[upd] = zz[upd]
        pid[i0:i1 + 1, j0:j1 + 1][upd] = fp[t]
    return zbuf, pid


def layer_count(V, F, res, x0, y0, span):
    """How many distinct panels project onto each covered pixel (overlap census)."""
    step = span / res
    ix = np.clip(((V[:, 0] - x0) / step).astype(int), 0, res - 1)
    iy = np.clip(((V[:, 1] - y0) / step).astype(int), 0, res - 1)
    return ix, iy


def one(root, g, res, margin):
    gd = os.path.join(root, g)
    W, F, labels = load(gd, g)
    panels, fp = face_panels(F, labels)
    keep = fp >= 0
    F, fp = F[keep], fp[keep]

    cx, cy = (W[:, 0].min() + W[:, 0].max()) / 2, (W[:, 1].min() + W[:, 1].max()) / 2
    span = max(np.ptp(W[:, 0]), np.ptp(W[:, 1])) * (1 + margin)
    x0, y0 = cx - span / 2, cy - span / 2

    zbuf, pid = rasterize(W, F, fp, res, x0, y0, span)
    # to image convention: [row=y from top, col=x]
    pid = pid.T[::-1, :].copy()
    zb = zbuf.T[::-1, :].copy()
    mask = (pid >= 0).astype(np.uint8)
    h = np.where(mask.astype(bool), zb, np.nan).astype(np.float32)

    zmin, zmax = float(np.nanmin(h)), float(np.nanmax(h))
    hn = np.zeros_like(h)
    hn[mask.astype(bool)] = 2 * (h[mask.astype(bool)] - zmin) / (zmax - zmin) - 1

    vis = sorted(set(np.unique(pid).tolist()) - {-1})
    stats = dict(garment=g, res=res, panels_total=len(panels), panels_visible=len(vis),
                 panels_hidden=[panels[k] for k in range(len(panels)) if k not in vis],
                 coverage=float(mask.mean()),
                 z_min_cm=zmin, z_max_cm=zmax, z_span_cm=zmax - zmin,
                 px_mm=span / res * 10, span_cm=float(span),
                 frame=dict(x0=float(x0), y0=float(y0), span=float(span)),
                 panel_px={panels[k]: int((pid == k).sum()) for k in vis})
    return dict(panel_id=pid, mask=mask, h=h, h_norm=hn.astype(np.float32),
                panels=np.array(panels, dtype=object)), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data_front1024")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--margin", type=float, default=0.04)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--garments", default=None)
    a = ap.parse_args()

    names = ([x.strip() for x in a.garments.split(",")] if a.garments else
             sorted(d for d in os.listdir(a.root)
                    if os.path.isdir(os.path.join(a.root, d)))[:a.n])
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
    os.makedirs(od, exist_ok=True)
    index = []
    for i, g in enumerate(names):
        t = time.time()
        try:
            arrs, st = one(a.root, g, a.res, a.margin)
        except Exception as e:
            print(f"[{i+1}/{len(names)}] {g} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        np.savez_compressed(os.path.join(od, g + ".npz"), **arrs)
        st["seconds"] = round(time.time() - t, 1)
        index.append(st)
        print(f"[{i+1}/{len(names)}] {g}  cover {st['coverage']*100:5.1f}%  "
              f"panels {st['panels_visible']}/{st['panels_total']}  "
              f"z {st['z_min_cm']:+6.1f}..{st['z_max_cm']:+6.1f}cm  "
              f"{st['px_mm']:.2f}mm/px  {st['seconds']}s", flush=True)
    json.dump(index, open(os.path.join(od, "index.json"), "w"), indent=1)
    print(f"\nwrote {len(index)} samples -> {a.out}/")


if __name__ == "__main__":
    main()
