"""specification.json -> ASTM-style DXF, matching the layer layout CLO exports.

Panel outline goes on layer 8 as a closed POLYLINE (densely sampled, so curves
survive as points), edge boundaries go on layer 2 as POINTs ("turn points"), and
the interior curve samples go on layer 3.  That is the structure observed in
panel.dxf, so a reader written against this file also reads CLO's output.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "autosew"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "autosew" / "viz"))
from visualize import load_raw, sample_edge

MM = 10.0            # spec is cm; CLO exports mm


def panel_polyline(verts, edges, n=24):
    """Closed boundary plus, for each edge, the index of its first point."""
    pts, starts = [], []
    for e in edges:
        p = sample_edge(verts[e["endpoints"][0]], verts[e["endpoints"][1]],
                        e.get("curvature"), n)
        starts.append(len(pts))
        pts.extend(p[:-1])                      # drop the duplicated join
    return np.asarray(pts) * MM, starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    panels, _ = load_raw(a.spec)
    L = []
    w = lambda c, v: L.extend((str(c), str(v)))
    w(0, "SECTION"); w(2, "BLOCKS")
    gap = 0.0
    for name, pd in panels.items():
        pts, starts = panel_polyline(pd["verts"], pd["edges"])
        pts = pts + np.array([gap, 0.0])
        gap += (pts[:, 0].max() - pts[:, 0].min()) + 300.0
        w(0, "BLOCK"); w(8, "1"); w(2, name); w(70, 0)
        w(10, 0.0); w(20, 0.0); w(30, 0.0); w(3, name); w(1, "")
        w(0, "POLYLINE"); w(8, "8"); w(66, 1); w(70, 1)
        w(10, 0.0); w(20, 0.0); w(30, 0.0)
        for p in pts:
            w(0, "VERTEX"); w(8, "8"); w(10, f"{p[0]:.6f}"); w(20, f"{p[1]:.6f}"); w(30, 0.0)
        w(0, "SEQEND"); w(8, "8")
        for i, s in enumerate(starts):
            w(0, "POINT"); w(8, "2")
            w(10, f"{pts[s][0]:.6f}"); w(20, f"{pts[s][1]:.6f}"); w(30, 0.0)
        for i in range(len(pts)):
            if i in starts: continue
            w(0, "POINT"); w(8, "3")
            w(10, f"{pts[i][0]:.6f}"); w(20, f"{pts[i][1]:.6f}"); w(30, 0.0)
        w(0, "TEXT"); w(8, "15"); w(10, 0.0); w(20, 0.0); w(30, 0.0); w(40, 5.0); w(1, name)
        w(0, "ENDBLK"); w(8, "1")
    w(0, "ENDSEC")
    w(0, "SECTION"); w(2, "ENTITIES")
    for name in panels:
        w(0, "INSERT"); w(8, "1"); w(2, name); w(10, 0.0); w(20, 0.0); w(30, 0.0)
    w(0, "ENDSEC"); w(0, "EOF")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: {len(panels)} panels, "
          f"{sum(len(p['edges']) for p in panels.values())} edges")


if __name__ == "__main__":
    main()
