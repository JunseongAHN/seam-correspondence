"""Synthetic toy patterns (GCD-specification-shaped dicts) for CPU tests.

Families:
  bodice : front + back rectangles (IDENTICAL shapes -> panel-ID hard case),
           stitches L-L, R-R, TOP-TOP; hems unstitched. One seam pair curved.
  skirt  : two different-size rectangles, side seams only.
  multi  : wide torso + two half back panels; torso TOP stitched to BOTH back tops
           (edge with GT degree 2 -> multi-edge case, 2-to-1).
One panel per pattern is emitted in CW order on purpose (exercises ACW canonicalization).
"""
import random


def _rect(w, h):
    # ACW: (0,0)->(w,0)->(w,h)->(0,h); edges: 0 bottom, 1 right, 2 top, 3 left
    return {
        "vertices": [[0, 0], [w, 0], [w, h], [0, h]],
        "edges": [
            {"endpoints": [0, 1]},
            {"endpoints": [1, 2]},
            {"endpoints": [2, 3]},
            {"endpoints": [3, 0]},
        ],
    }


def _rect_cw(w, h):
    # same rectangle, CW traversal (edge i here = edge (3-i)+... independent indexing)
    return {
        "vertices": [[0, 0], [0, h], [w, h], [w, 0]],
        "edges": [
            {"endpoints": [0, 1]},   # left, upward
            {"endpoints": [1, 2]},   # top
            {"endpoints": [2, 3]},   # right, downward
            {"endpoints": [3, 0]},   # bottom
        ],
    }


def make_pattern(kind, rng: random.Random):
    j = lambda a, b: rng.uniform(a, b)
    if kind == "bodice":
        w, h = j(30, 60), j(40, 70)
        front = _rect(w, h)
        back = _rect_cw(w, h)  # identical shape, CW on purpose
        front["edges"][1]["curvature"] = [0.5, 0.12]   # right edge curved (legacy quadratic)
        back["edges"][2]["curvature"] = [0.5, -0.12]   # its counterpart (CW right edge)
        panels = {"front": front, "back": back}
        # front: 0 bottom,1 right,2 top,3 left ; back(CW): 0 left,1 top,2 right,3 bottom
        stitches = [
            [{"panel": "front", "edge": 1}, {"panel": "back", "edge": 2}],
            [{"panel": "front", "edge": 3}, {"panel": "back", "edge": 0}],
            [{"panel": "front", "edge": 2}, {"panel": "back", "edge": 1}],
        ]
    elif kind == "skirt":
        w1, h1 = j(50, 80), j(50, 90)
        w2 = w1 * j(0.95, 1.05)
        panels = {"sfront": _rect(w1, h1), "sback": _rect(w2, h1)}
        stitches = [
            [{"panel": "sfront", "edge": 1}, {"panel": "sback", "edge": 3}],
            [{"panel": "sfront", "edge": 3}, {"panel": "sback", "edge": 1}],
        ]
    elif kind == "multi":
        w, h = j(40, 70), j(30, 60)
        torso = _rect(w, h)
        bl = _rect(w * 0.5, h)
        br = _rect(w * 0.5, h)
        panels = {"torso": torso, "back_l": bl, "back_r": br}
        stitches = [
            [{"panel": "torso", "edge": 2}, {"panel": "back_l", "edge": 2}],  # torso.top deg=2
            [{"panel": "torso", "edge": 2}, {"panel": "back_r", "edge": 2}],
            [{"panel": "torso", "edge": 1}, {"panel": "back_r", "edge": 1}],
            [{"panel": "torso", "edge": 3}, {"panel": "back_l", "edge": 3}],
        ]
    else:
        raise ValueError(kind)
    return {"pattern": {"panels": panels, "stitches": stitches, "units_in_meter": 100}}


def make_set(n, seed=0, kinds=("bodice", "skirt", "multi")):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        out.append((f"{kind}_{i:04d}", make_pattern(kind, rng)))
    return out
