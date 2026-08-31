#!/usr/bin/env python
"""Export the panel surfaces for the vtk.js viewer.

  python export_vtkjs.py
  python export_vtkjs.py --garment rand_023FMIGQK0

Writes data/panels.json (manifest) and data/panels.bin (geometry).  Same panels
and same full resolution as view_vtk.py -- no decimation.  Coordinates are
centred on the garment so the browser does not have to.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from view_vtk import build_panels, DEF_ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=DEF_ROOT)
ap.add_argument("--garment", default="rand_00YONAPXZE")
a = ap.parse_args()

panels, center, W = build_panels(a.root, a.garment)
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "data")
os.makedirs(out, exist_ok=True)

chunks, off, meta = [], 0, []


def put(arr):
    """Append a 4-byte-wide array to the blob and return its byte offset."""
    global off
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
with open(os.path.join(out, "panels.bin"), "wb") as f:
    f.write(blob)

lo, hi = (W - center).min(0), (W - center).max(0)
with open(os.path.join(out, "panels.json"), "w", encoding="utf-8") as f:
    json.dump({"garment": a.garment, "panels": meta,
               "bbox": [lo.tolist(), hi.tolist()], "bytes": len(blob)}, f)

print(f"{a.garment}: panels {len(panels)}  verts {sum(p['nv'] for p in meta)}  "
      f"faces {sum(p['nf'] for p in meta)}  bin {len(blob)/1e6:.2f} MB")
