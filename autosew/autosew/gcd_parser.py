"""Parser: GarmentCode(Data) *_specification.json -> Pattern.

Format assumptions (verify against real data with scripts/validate_data.py BEFORE training):
  spec["pattern"]["panels"] : {panel_name: {"vertices": [[x,y],...],
                                            "edges": [{"endpoints":[a,b], "curvature": ...}, ...]}}
  spec["pattern"]["stitches"]: [[{"panel":p,"edge":e},{"panel":q,"edge":f}], ...]
    (a stitch may join >2 sides in M-E data; each listed side pairs with every other side)
  spec["pattern"]["units_in_meter"]: 100 for cm (GCD default). We convert everything to cm.

Curvature encodings handled:
  - absent / None                          -> straight
  - [x, y] (legacy)                        -> quadratic Bezier, edge-relative control point
  - {"type": "quadratic", "params": [[x,y]]}
  - {"type": "cubic",     "params": [[x1,y1],[x2,y2]]}
  - {"type": <contains "circ" or "arc">, "params": [...]} -> circular arc, params kept raw
  - anything else -> KT_UNKNOWN (counted; investigate if frequent)

Edge-relative -> absolute control point (GarmentCode convention):
  abs = start + cx * (end - start) + cy * perp(end - start),  perp(v) = (-v_y, v_x)
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .config import KT_STRAIGHT, KT_CIRCLE, KT_QUADRATIC, KT_CUBIC, KT_BSPLINE, KT_UNKNOWN


@dataclass
class Edge:
    panel: str
    idx_in_panel: int          # ORIGINAL json index (stitch GT refers to this)
    start: tuple               # panel-local cm, after ACW canonicalization
    end: tuple
    kt: int
    kparams: list              # up to 10 floats; abs-frame control points already converted
    kparams_rel: list          # relative-frame params (for curvature_frame="rel")


@dataclass
class Panel:
    name: str
    order_idx: int             # position in the panels dict (file order)
    edges: list                # [Edge] in ACW traversal order
    n_edges_raw: int
    was_reversed: bool = False


@dataclass
class Pattern:
    name: str
    panels: list               # [Panel]
    stitches: list             # list[set[(panel_name, edge_idx)]] - each stitch = set of >=2 sides
    meta: dict = field(default_factory=dict)

    def edge_key_list(self):
        """Global node order: panels in file order, edges in ACW traversal order.
        Returns list of (panel_name, original_edge_idx)."""
        keys = []
        for p in self.panels:
            for e in p.edges:
                keys.append((p.name, e.idx_in_panel))
        return keys

    def gt_pairs(self):
        """Ground-truth unordered node-index pairs (multi-edge stitches expand pairwise)."""
        key2node = {k: i for i, k in enumerate(self.edge_key_list())}
        pairs = set()
        for sides in self.stitches:
            nodes = sorted(key2node[s] for s in sides if s in key2node)
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    pairs.add((nodes[a], nodes[b]))
        return pairs


def _perp(v):
    return (-v[1], v[0])


def _rel_to_abs(start, end, c):
    ex, ey = end[0] - start[0], end[1] - start[1]
    px, py = _perp((ex, ey))
    return (start[0] + c[0] * ex + c[1] * px, start[1] + c[0] * ey + c[1] * py)


def _parse_curvature(curv, start, end, rel_coords=True):
    """-> (kt, abs_params, rel_params); *_params padded/truncated to 10 by caller.
    rel_coords: GCD properties.curvature_coords == "relative" (v2 default)."""
    def conv(c):
        if rel_coords:
            a = _rel_to_abs(start, end, c)
            return a, (c[0], c[1])
        # already absolute: rel version left empty-equivalent (mirror of abs)
        return (c[0], c[1]), (c[0], c[1])

    if curv is None:
        return KT_STRAIGHT, [], []
    if isinstance(curv, list) and len(curv) == 2 and all(isinstance(t, (int, float)) for t in curv):
        a, r = conv(curv)
        return KT_QUADRATIC, [a[0], a[1]], [r[0], r[1]]
    if isinstance(curv, dict):
        ctype = str(curv.get("type", "")).lower()
        params = curv.get("params", [])
        if "quadratic" in ctype:
            a, r = conv(params[0])
            return KT_QUADRATIC, [a[0], a[1]], [r[0], r[1]]
        if "cubic" in ctype:
            out_a, out_r = [], []
            for c in params[:2]:
                a, r = conv(c)
                out_a += [a[0], a[1]]
                out_r += [r[0], r[1]]
            return KT_CUBIC, out_a, out_r
        if "circ" in ctype or "arc" in ctype:
            flat = []
            for p in params:
                if isinstance(p, (list, tuple)):
                    flat += [float(t) for t in p]
                elif isinstance(p, bool):
                    flat.append(1.0 if p else 0.0)
                else:
                    flat.append(float(p))
            return KT_CIRCLE, flat[:10], flat[:10]
        if "spline" in ctype or "bspline" in ctype or ctype == "b-spline":
            flat = []
            for p in params:
                if isinstance(p, (list, tuple)):
                    flat += [float(t) for t in p]
                else:
                    flat.append(float(p))
            return KT_BSPLINE, flat[:10], flat[:10]
        return KT_UNKNOWN, [], []
    return KT_UNKNOWN, [], []


def _reverse_rel_params(kt, rel):
    """Curvature transform when an edge is traversed backwards (rel frame: x->1-x, y->-y,
    control-point order reversed for cubic)."""
    if kt == KT_QUADRATIC and len(rel) == 2:
        return [1.0 - rel[0], -rel[1]]
    if kt == KT_CUBIC and len(rel) == 4:
        return [1.0 - rel[2], -rel[3], 1.0 - rel[0], -rel[1]]
    # circle/bspline: keep raw, direction flags unknown -> leave as-is (validate on real data)
    return list(rel)


def parse_specification(path_or_dict, name=None) -> Pattern:
    if isinstance(path_or_dict, (str, Path)):
        p = Path(path_or_dict)
        spec = json.loads(p.read_text())
        name = name or p.stem
    else:
        spec = path_or_dict
        name = name or "pattern"
    pat = spec["pattern"] if "pattern" in spec else spec
    props = spec.get("properties", {}) if isinstance(spec, dict) else {}

    unit = float(pat.get("units_in_meter") or props.get("units_in_meter") or 100.0)
    to_cm = 100.0 / unit  # cm per stored unit
    rel_coords = str(props.get("curvature_coords", "relative")).lower() != "absolute"

    # deterministic panel ordering: GCD panel_order when present, else dict order
    order = pat.get("panel_order")
    if order and set(order) == set(pat["panels"].keys()):
        panel_names = list(order)
    else:
        panel_names = list(pat["panels"].keys())

    panels = []
    for order_idx, pname in enumerate(panel_names):
        pdata = pat["panels"][pname]
        verts = [(v[0] * to_cm, v[1] * to_cm) for v in pdata["vertices"]]
        raw_edges = pdata["edges"]
        n_raw = len(raw_edges)

        # signed area of the chord polygon in edge-traversal order
        pts = [verts[e["endpoints"][0]] for e in raw_edges]
        area = 0.0
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            area += x0 * y1 - x1 * y0
        reversed_ = area < 0

        seq = list(range(n_raw))
        if reversed_:
            seq = seq[::-1]

        edges = []
        for j in seq:
            e = raw_edges[j]
            a, b = e["endpoints"][0], e["endpoints"][1]
            if reversed_:
                a, b = b, a
            start, end = verts[a], verts[b]
            curv = e.get("curvature", None)
            kt, kabs, krel = _parse_curvature(curv, verts[e["endpoints"][0]], verts[e["endpoints"][1]],
                                              rel_coords=rel_coords)
            if reversed_:
                if rel_coords:
                    krel = _reverse_rel_params(kt, krel)
                    # recompute abs from transformed rel for Bezier types
                    if kt == KT_QUADRATIC and len(krel) == 2:
                        aa = _rel_to_abs(start, end, krel)
                        kabs = [aa[0], aa[1]]
                    elif kt == KT_CUBIC and len(krel) == 4:
                        aa1 = _rel_to_abs(start, end, krel[0:2])
                        aa2 = _rel_to_abs(start, end, krel[2:4])
                        kabs = [aa1[0], aa1[1], aa2[0], aa2[1]]
                else:
                    # absolute control points: positions unchanged, cubic order swaps
                    if kt == KT_CUBIC and len(kabs) == 4:
                        kabs = kabs[2:4] + kabs[0:2]
                        krel = list(kabs)
                if kt == KT_CIRCLE and len(kabs) > 2:
                    # circle params are [radius, large_arc, right]; traversing the arc
                    # backwards flips its sweep direction, while the radius and the
                    # large-arc flag are orientation-independent
                    kabs = list(kabs)
                    kabs[2] = 1.0 - float(kabs[2])
                    krel = list(kabs)
            edges.append(Edge(panel=pname, idx_in_panel=j, start=start, end=end,
                              kt=kt, kparams=kabs, kparams_rel=krel))

        # translate: panel bbox lower-left corner -> origin (chord bbox over vertices)
        xs = [e.start[0] for e in edges] + [e.end[0] for e in edges]
        ys = [e.start[1] for e in edges] + [e.end[1] for e in edges]
        mx, my = min(xs), min(ys)
        for e in edges:
            e.start = (e.start[0] - mx, e.start[1] - my)
            e.end = (e.end[0] - mx, e.end[1] - my)
            if e.kt in (KT_QUADRATIC, KT_CUBIC):
                e.kparams = [t - (mx if i % 2 == 0 else my) for i, t in enumerate(e.kparams)]

        panels.append(Panel(name=pname, order_idx=order_idx, edges=edges,
                            n_edges_raw=n_raw, was_reversed=reversed_))

    stitches = []
    for st in pat.get("stitches", []):
        sides = set()
        for side in st:
            # GCD stitches may carry a trailing orientation tag string, e.g. "right_wrong"
            # (fabric-face flag on collar attachments etc.) -- not an edge side; skip.
            if isinstance(side, dict) and "panel" in side and "edge" in side:
                sides.add((side["panel"], int(side["edge"])))
        if len(sides) >= 2:
            stitches.append(sides)

    return Pattern(name=name, panels=panels, stitches=stitches,
                   meta={"units_in_meter": unit, "n_stitches_raw": len(pat.get("stitches", []))})
