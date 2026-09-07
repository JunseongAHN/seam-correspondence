#!/usr/bin/env python
"""Merge the per-garment tmp_<garment>.json outputs (from parallel measure.py runs)
into a single data/measure.json with the same schema measure.py itself writes.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
KEYS = ("vertex", "mutual", "seam_all", "seam_vote", "garment", "sep_mm")

garments = sys.argv[1:]
per, infos, failed = {}, {}, {}
root = sigmas = iters = res = draws = seed = w = None
wall = 0.0

for g in garments:
    fp = os.path.join(DATA, f"tmp_{g}.json")
    d = json.load(open(fp))
    m = d["meta"]
    root = m["root"]; res = m["res"]; draws = m["draws"]; seed = m["seed"]; w = m["w"]
    sigmas = m["sigmas"]; iters = m["iters"]
    wall += m["wall_s"]
    failed.update(m.get("failed", {}))
    infos.update(m.get("per_garment_info", {}))
    per.update(d.get("per_garment", {}))

cells = [f"{s}|{it}" for s in sigmas for it in iters]
mean = {c: {k: float(np.mean([per[g][c][k] for g in per])) for k in KEYS} for c in cells}
sd = {c: {k: float(np.std([per[g][c][k] for g in per], ddof=1)) if len(per) > 1 else 0.0
          for k in KEYS} for c in cells}
for c in cells:
    mean[c]["n_garments"] = len(per)

out = dict(meta=dict(mode="multi", root=root, garments=sorted(per), n_garments=len(per),
                      failed=failed, sigmas=sigmas, iters=iters, res=res,
                      draws=draws, seed=seed, w=w, wall_s=round(wall, 1),
                      per_garment_info=infos,
                      smoothing="mesh-graph neighbour average (same as view_vtk.py)"),
           rows=mean, sd=sd, per_garment=per)
json.dump(out, open(os.path.join(DATA, "measure.json"), "w"), indent=1)
print(f"wrote data/measure.json  ({len(per)} garments, {len(failed)} failed)")

HDR = (f"\n{'sigma':>7} {'smooth':>7} {'sep mm':>7} {'vertex':>7} {'mutual':>7} "
       f"{'s.all':>7} {'s.vote':>7} {'garment':>8}")
print(HDR); print("-" * 66)
for s in sigmas:
    for it in iters:
        r = mean[f"{s}|{it}"]
        e = sd[f"{s}|{it}"]
        print(f"{s:>5}mm {it:>5}it {r['sep_mm']:>7.1f} {r['vertex']:>7.4f} "
              f"{r['mutual']:>7.3f} {r['seam_all']:>7.4f} {r['seam_vote']:>7.4f} "
              f"{r['garment']:>8.2f}   +-{e['seam_vote']:.4f} vote  +-{e['garment']:.2f} garment")
    print()
if failed:
    print("failed:", failed)
