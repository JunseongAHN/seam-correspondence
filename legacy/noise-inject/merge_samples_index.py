#!/usr/bin/env python
"""Rebuild data/samples/index.json from each garment's own seams.json.

Needed because export_seams.py writes index.json itself, and running it once
per garment in parallel (for speed) means the last process to finish clobbers
everyone else's entry.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_OUT = os.path.join(HERE, "data", "samples")
MEASURE_FP = os.path.join(HERE, "data", "measure.json")

garments = sys.argv[1:]
per_garment_scores = {}
if os.path.exists(MEASURE_FP):
    per_garment_scores = json.load(open(MEASURE_FP)).get("per_garment", {})

sigmas = iters = root = None
index = []
for g in garments:
    d = json.load(open(os.path.join(ROOT_OUT, g, "seams.json")))
    sigmas, iters = d["sigmas"], d["iters"]
    info = d["info"]
    index.append({"garment": g, "panels": info["panels"],
                  "n_seam_vertices": info["n_seam_vertices"], "n_seams": info["n_seams"],
                  "scores": per_garment_scores.get(g, {})})

json.dump({"sigmas": sigmas, "iters": iters, "garments": index},
          open(os.path.join(ROOT_OUT, "index.json"), "w", encoding="utf-8"))
print(f"wrote data/samples/index.json  ({len(garments)} garments)")
