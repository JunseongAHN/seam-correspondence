"""How unusual are this garment's sleeve seams, measured against GarmentCodeData?

Every model gets the sleeve completely wrong.  The two edges of a sleeve seam here differ
in length by up to 3.4x.  GCD's sleeve/torso seams are the reference: if this garment sits
far out in that distribution, the model is being asked for something it has barely seen.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "autosew"))
sys.path.insert(0, str(R / "dxfcheck"))
from autosew.curves import edge_polyline
from autosew.dataset import find_spec_files
from autosew.gcd_parser import parse_specification
from dxf_to_features import read_panels

GCD_TEST = r"C:\Users\POMCHECKER\gcd_data\test"
PARTS = [("sleeve", r"sleeve|cuff"), ("skirt", r"skirt|gore|godet"),
         ("pants", r"pant|trouser|leg"), ("collar", r"collar|hood|lapel"),
         ("torso", r"torso|bodice|front|back|wb|waistband")]
# which CLO pieces are the sleeve
SLEEVE = {"6_M", "11_M", "8_M", "10_M"}


def part_of(n):
    n = n.lower()
    for p, pat in PARTS:
        if re.search(pat, n):
            return p
    return "other"


# ---------- the CLO garment
E = {}
for name, pts, idx in read_panels(str(R / "clo_example" / "panel_seperated.dxf")):
    n = len(idx)
    for k in range(n):
        a, b = idx[k], idx[(k + 1) % n]
        seg = pts[a:b + 1] if b > a else np.vstack([pts[a:], pts[:b + 1]])
        E[(name, k)] = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())

gt = json.loads((R / "clo_example" / "panel_seperated_gt.json").read_text())
clo = {"sleeve to body": [], "sleeve to sleeve": [], "everything else": []}
for st in gt["stitches"]:
    (p1, e1), (p2, e2) = st
    a, b = E[(p1, e1)], E[(p2, e2)]
    r = max(a, b) / min(a, b)
    s1, s2 = p1 in SLEEVE, p2 in SLEEVE
    kind = ("sleeve to sleeve" if s1 and s2 else
            "sleeve to body" if s1 or s2 else "everything else")
    clo[kind].append((r, f"{p1}#{e1} ~ {p2}#{e2}", a, b))

print("the CLO garment's seams, by length ratio")
for kind in ("sleeve to body", "sleeve to sleeve", "everything else"):
    v = sorted(clo[kind], reverse=True)
    print(f"\n  {kind}  ({len(v)} seams)")
    for r, lbl, a, b in v:
        print(f"    {lbl:<34}{a:>8.2f}{b:>8.2f}   ratio {r:>6.3f}")
    print(f"    median ratio {np.median([x[0] for x in v]):.3f}")

# ---------- GarmentCodeData, for comparison
print("\n\nGarmentCodeData: length ratio of the two edges of a stitch")
gcd = defaultdict(list)
for f in find_spec_files(GCD_TEST, limit=400):
    try:
        p = parse_specification(f)
    except Exception:
        continue
    arc, pan = {}, {}
    for panel in p.panels:
        for e in panel.edges:
            k = (panel.name, e.idx_in_panel)
            arc[k] = float(np.linalg.norm(np.diff(edge_polyline(e), axis=0), axis=1).sum())
            pan[k] = panel.name
    for sides in p.stitches:
        ss = [s for s in sides if s in arc]
        if len(ss) != 2 or min(arc[ss[0]], arc[ss[1]]) < 1e-6:
            continue
        a, b = ss
        r = max(arc[a], arc[b]) / min(arc[a], arc[b])
        ka, kb = part_of(pan[a]), part_of(pan[b])
        if pan[a] == pan[b]:
            gcd["within a panel"].append(r)
        elif "sleeve" in (ka, kb) and "torso" in (ka, kb):
            gcd["sleeve to torso"].append(r)
        elif "skirt" in (ka, kb) and "torso" in (ka, kb):
            gcd["skirt to torso"].append(r)
        elif ka == "sleeve" and kb == "sleeve":
            gcd["sleeve to sleeve"].append(r)
        elif ka == kb:
            gcd["same part"].append(r)
        else:
            gcd["everything else"].append(r)

print(f"  {'seam kind':<20}{'n':>7}{'median':>9}{'p75':>8}{'p90':>8}{'p99':>8}{'max':>8}")
for k, v in sorted(gcd.items(), key=lambda kv: -len(kv[1])):
    v = np.array(v)
    print(f"  {k:<20}{len(v):>7}{np.median(v):>9.3f}{np.percentile(v,75):>8.3f}"
          f"{np.percentile(v,90):>8.3f}{np.percentile(v,99):>8.3f}{v.max():>8.3f}")

ref = np.array(gcd["sleeve to torso"])
print(f"\nwhere this garment's sleeve-to-body seams sit in GCD's sleeve-to-torso "
      f"distribution (n={len(ref)}):")
for r, lbl, a, b in sorted(clo["sleeve to body"], reverse=True):
    pct = (ref < r).mean() * 100
    print(f"  {lbl:<34} ratio {r:>6.3f}   above {pct:5.1f}% of GCD sleeve seams"
          f"   ({(ref >= r).sum()} of {len(ref)} are this extreme or more)")

print(f"\nGCD sleeve-to-torso seams within 10% in length: "
      f"{(ref < 1.1).mean()*100:.1f}%")
print(f"this garment's sleeve-to-body seams within 10%: "
      f"{np.mean([r < 1.1 for r, *_ in clo['sleeve to body']])*100:.1f}%")

# ---------- the waist, the other seam every model misses
WAIST = [(("3_M", 4), ("Pattern_634078_M", 1)),
         (("Pattern_669883_M", 3), ("9_M", 0))]
sk = np.array(gcd["skirt to torso"])
print(f"\n\nthe waist seams, against GCD's skirt-to-torso distribution (n={len(sk)}):")
print(f"  GCD skirt-to-torso: median {np.median(sk):.3f}  p75 {np.percentile(sk,75):.3f}"
      f"  p90 {np.percentile(sk,90):.3f}  p99 {np.percentile(sk,99):.3f}"
      f"  max {sk.max():.3f}   within 10%: {(sk<1.1).mean()*100:.1f}%")
for a, b in WAIST:
    x, y = E[a], E[b]
    r = max(x, y) / min(x, y)
    pct = (sk < r).mean() * 100
    print(f"  {a[0]+'#'+str(a[1])+' ~ '+b[0]+'#'+str(b[1]):<40}"
          f"{x:>8.2f}{y:>8.2f}   ratio {r:>6.3f}   above {pct:5.1f}% of GCD waist seams"
          f"   ({(sk >= r).sum()} of {len(sk)} this extreme or more)")
