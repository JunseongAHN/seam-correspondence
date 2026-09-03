"""Is any panel's traversal direction backwards?

Pieces are canonicalised to anticlockwise in the DXF layout.  But a mirrored piece -- a
left sleeve against a right one -- is laid out face-up like everything else and gets
flipped over when it is sewn, so its physical traversal is the opposite of its layout
one.  If that is not accounted for, every sagitta value on that panel has the wrong sign,
and a pair of edges that really do interlock will not look like it.

Test: search all 2^10 assignments of "this panel is flipped" for the one that makes the
ground-truth stitches agree in shape best, and see whether it beats leaving them alone.
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "autosew"))
sys.path.insert(0, str(R / "dxfcheck"))
from autosew.curves import sagitta_profile
from dxf_to_features import read_panels

K = 11
E, PANELS = {}, []
for name, pts, idx in read_panels(str(R / "clo_example" / "panel_seperated.dxf")):
    PANELS.append(name)
    n = len(idx)
    for k in range(n):
        a, b = idx[k], idx[(k + 1) % n]
        seg = pts[a:b + 1] if b > a else np.vstack([pts[a:], pts[:b + 1]])
        E[(name, k)] = dict(
            arc=float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()),
            sag=sagitta_profile(seg, K))

gt = json.loads((R / "clo_example" / "panel_seperated_gt.json").read_text())
STITCH = [((a[0], a[1]), (b[0], b[1])) for a, b in gt["stitches"]]


def prof(key, flipped):
    """Sagitta of this edge under the given traversal direction.  Reversing the traversal
    negates the signed offset and reverses the sampling order."""
    s = E[key]["sag"]
    return -s[::-1] if flipped else s


def residual(flip):
    """Total shape mismatch of the ground-truth stitches under this flip assignment.

    Two edges sewn together are traversed in opposite senses when the pieces meet, so a
    genuinely interlocking pair satisfies s2 == -reverse(s1) -- which, once both are
    expressed in their own physical traversal, is just s2 == s1 read the other way."""
    tot, per = 0.0, []
    for a, b in STITCH:
        s1 = prof(a, flip[a[0]])
        s2 = prof(b, flip[b[0]])
        amp = max(np.abs(s1).max(), np.abs(s2).max(), 1e-9)
        r = min(np.abs(s2 - c).max() for c in (s1, -s1, s1[::-1], -s1[::-1])) / amp
        tot += r
        per.append(r)
    return tot, per


base_flip = {p: False for p in PANELS}
base, base_per = residual(base_flip)
print(f"{len(PANELS)} panels, {len(STITCH)} ground-truth stitches")
print(f"as parsed (no panel flipped): total shape mismatch {base:.3f}"
      f"   mean {base/len(STITCH):.3f}")

best = (base, base_flip)
scores = []
for bits in product([False, True], repeat=len(PANELS)):
    f = dict(zip(PANELS, bits))
    t, _ = residual(f)
    scores.append(t)
    if t < best[0]:
        best = (t, f)

scores = np.array(scores)
print(f"\nover all {len(scores)} flip assignments:"
      f"  best {scores.min():.3f}   median {np.median(scores):.3f}   worst {scores.max():.3f}")
flipped = [p for p, v in best[1].items() if v]
print(f"best assignment flips: {flipped if flipped else '(nothing)'}"
      f"   total {best[0]:.3f}")
print(f"improvement over as-parsed: {base - best[0]:.3f}"
      f"  ({(base - best[0]) / base * 100:.1f}%)")

if best[0] < base - 1e-9:
    _, per = residual(best[1])
    print(f"\n{'stitch':<38}{'as parsed':>12}{'best flip':>12}{'change':>10}")
    for (a, b), p0, p1 in zip(STITCH, base_per, per):
        lbl = f"{a[0]}#{a[1]} ~ {b[0]}#{b[1]}"
        print(f"{lbl:<38}{p0:>12.3f}{p1:>12.3f}{p1-p0:>10.3f}")
else:
    print("\nno flip assignment beats the parse as it stands -- panel orientation is not "
          "what is making the shapes disagree")

# a flip is only meaningful if the panel has curved edges at all
print(f"\n{'panel':<20}{'edges':>7}{'max |sagitta|':>15}   flippable?")
for p in PANELS:
    ks = [k for k in E if k[0] == p]
    amp = max(np.abs(E[k]["sag"]).max() for k in ks)
    print(f"{p:<20}{len(ks):>7}{amp:>15.4f}   "
          f"{'yes' if amp > 0.01 else 'no - too straight to tell'}")
