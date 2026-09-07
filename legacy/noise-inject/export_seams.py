#!/usr/bin/env python
"""Export seam-decode correctness for the browser sample viewer.

For each garment, writes the SAME panel surfaces as export_vtkjs.py, plus a
seam-copy point cloud (one noise draw per sigma x smoothing cell) carrying,
per copy: its actual post-noise 3D position, whether its 1-NN decode was
correct (`good`, same test as ninject/pipeline.py:decode/score), and the
index of the copy it matched to (so the viewer can draw a line to a wrong
match).

    python export_seams.py --root <data> --garments rand_00YONAPXZE,rand_...

Writes data/samples/<garment>/{panels.json,panels.bin,seams.json,seams.bin}
and data/samples/index.json (garment list + accuracy pulled from measure.json
if present, so the sidebar doesn't have to recompute anything).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from view_vtk import build_panels, DEF_ROOT
import measure as M
from ninject.pipeline import project_lift, decode

SIGMAS = [1, 2, 3, 5, 10, 20]
ITERS = [0, 5, 15, 30, 60]


def export_panels(panels, center, out_dir, garment):
    """Same layout export_vtkjs.py writes -- kept identical so nothing else
    that reads panels.json/panels.bin needs to change."""
    chunks, off, meta = [], 0, []

    def put(arr):
        nonlocal off
        b = np.ascontiguousarray(arr).tobytes()
        chunks.append(b)
        o = off
        off += len(b)
        return o

    for P in panels:
        pos = put((P.V - center).astype(np.float32))
        fac = put(P.F.astype(np.uint32))
        loops = [{"off": put(np.asarray(L, np.uint32)), "n": len(L)} for L in P.loops]
        meta.append({"name": P.name, "front": bool(P.front), "nv": len(P.V),
                     "nf": len(P.F), "pos": pos, "faces": fac, "loops": loops})

    blob = b"".join(chunks)
    open(os.path.join(out_dir, "panels.bin"), "wb").write(blob)
    lo, hi = (np.vstack([P.V for P in panels]) - center).min(0), \
             (np.vstack([P.V for P in panels]) - center).max(0)
    json.dump({"garment": garment, "panels": meta,
               "bbox": [lo.tolist(), hi.tolist()], "bytes": len(blob)},
              open(os.path.join(out_dir, "panels.json"), "w", encoding="utf-8"))


def export_seams(root, garment, center, res, seed, w, sigmas, iters, out_dir, panels):
    """Per cell: the displaced panel surfaces, the noised copies, the lifted
    copies the decoder actually compared, and the decode result.

    `panels` (from view_vtk.build_panels) and `P` (from measure.prepare) are the
    same panels in the same order with the same vertex order, so one displacement
    array concatenated in panel order indexes both.
    """
    P, copies, share, member, frame, info = M.prepare(root, garment)
    n = len(copies)
    if [len(p["V"]) for p in P] != [len(q.V) for q in panels]:
        raise ValueError(f"{garment}: panel layout differs between builders")

    chunks, off, cells = [], 0, []

    def put(arr):
        """Append and pad to 4 bytes -- the browser's typed-array views on this
        blob need 4-byte-aligned offsets, and `good` is uint8."""
        nonlocal off
        b = np.ascontiguousarray(arr).tobytes()
        pad = -len(b) % 4
        chunks.append(b + b"\0" * pad)
        o = off
        off += len(b) + pad
        return o

    # --- per garment: GT copy positions, the true partner, the owning panel.
    # copies are laid out as sibling pairs (2k, 2k+1) sharing one welded vertex,
    # so the true correspondence is i^1 -- exported rather than assumed.
    gt = np.array([P[pi]["V"][li] for (_, pi, li) in copies])
    pair = np.empty(n, np.uint32)
    vs = [c[0] for c in copies]
    for i in range(0, n, 2):
        if vs[i] != vs[i + 1]:
            raise ValueError(f"{garment}: copies are not sibling-paired at {i}")
        pair[i], pair[i + 1] = i + 1, i
    gt_o = put((gt - center).astype(np.float32))
    pair_o = put(pair)
    panel_o = put(np.array([pi for (_, pi, _) in copies], np.uint32))

    for s in sigmas:
        for it in iters:
            rng = np.random.default_rng(seed)
            sc = s / 10.0
            disp = []
            for p in P:
                d = M.N.white(rng, len(p["V"]), sc)
                d = M.N.smooth_on_graph(d, p["nbr"][0], p["nbr"][1], w, it)
                disp.append(d)
            pts = np.array([P[pi]["V"][li] + disp[pi][li] for (_, pi, li) in copies])
            allp = np.vstack([pts] + [p["V"] + d for p, d in zip(P, disp)])
            lifted, _, _ = project_lift(allp, res, frame)
            good, nn, mutual = decode(lifted[:n], share)

            # panel displacement, int16-quantised (error ~ scale/32767, microns)
            flat = np.vstack(disp)
            scale = float(np.abs(flat).max()) / 32767.0 or 1.0
            cells.append({
                "sigma": s, "iters": it,
                "pos": put((pts - center).astype(np.float32)),
                "lift": put((lifted[:n] - center).astype(np.float32)),
                "good": put(good.astype(np.uint8)),
                "nn": put(nn.astype(np.uint32)),
                "disp": put(np.round(flat / scale).astype(np.int16)),
                "dscale": scale,
                "vertex": float(good.mean()), "mutual": mutual,
            })

    blob = b"".join(chunks)
    open(os.path.join(out_dir, "seams.bin"), "wb").write(blob)
    json.dump({"garment": garment, "n_copies": n, "n_verts": int(sum(len(p["V"]) for p in P)),
               "sigmas": sigmas, "iters": iters, "seed": seed, "w": w, "res": res,
               "gt": gt_o, "pair": pair_o, "panel": panel_o,
               "cells": cells, "info": info},
              open(os.path.join(out_dir, "seams.json"), "w", encoding="utf-8"))
    return info, len(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEF_ROOT)
    ap.add_argument("--garments", required=True, help="comma-separated list")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--sigmas", default=None)
    ap.add_argument("--iters", default=None)
    a = ap.parse_args()

    sigmas = ([float(x) if '.' in x else int(x) for x in a.sigmas.split(',')]
              if a.sigmas else SIGMAS)
    iters = [int(x) for x in a.iters.split(',')] if a.iters else ITERS
    garments = [g.strip() for g in a.garments.split(',') if g.strip()]

    here = os.path.dirname(os.path.abspath(__file__))
    root_out = os.path.join(here, "data", "samples")
    os.makedirs(root_out, exist_ok=True)

    measure_fp = os.path.join(here, "data", "measure.json")
    per_garment_scores = {}
    if os.path.exists(measure_fp):
        per_garment_scores = json.load(open(measure_fp)).get("per_garment", {})

    index = []
    for g in garments:
        out_dir = os.path.join(root_out, g)
        os.makedirs(out_dir, exist_ok=True)
        panels, center, W = build_panels(a.root, g)
        export_panels(panels, center, out_dir, g)
        info, nbytes = export_seams(a.root, g, center, a.res, a.seed, a.w,
                                    sigmas, iters, out_dir, panels)
        index.append({"garment": g, "panels": len(panels),
                      "n_seam_vertices": info["n_seam_vertices"], "n_seams": info["n_seams"],
                      "scores": per_garment_scores.get(g, {})})
        print(f"{g}: panels {len(panels)}  seam-v {info['n_seam_vertices']}  "
              f"seams {info['n_seams']}  seams.bin {nbytes/1e6:.1f} MB", flush=True)

    json.dump({"root": a.root, "sigmas": sigmas, "iters": iters, "garments": index},
              open(os.path.join(root_out, "index.json"), "w", encoding="utf-8"))
    print(f"wrote data/samples/index.json  ({len(garments)} garments)")


if __name__ == "__main__":
    main()
