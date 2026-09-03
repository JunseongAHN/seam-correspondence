"""Draw every ground-truth stitch on the pattern, at true scale, numbered.

Each stitch gets a number; both of its edges are drawn in the same colour with that
number at their midpoints, so a pairing that joins the wrong two edges is visible rather
than having to be read out of a table.

    & $PY dxfcheck/draw_gt.py --out report/gt-check.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "dxfcheck"))
from dxf_to_features import panel_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", default=str(R / "clo_example" / "panel_seperated.dxf"))
    ap.add_argument("--gt", default=str(R / "clo_example" / "panel_seperated_gt.json"))
    ap.add_argument("--out", default=str(R / "report" / "gt-check.png"))
    a = ap.parse_args()

    E = panel_edges(a.dxf)
    P = {}
    for (name, k), seg in E.items():
        P.setdefault(name, []).append(seg[:-1])
    P = {n: np.vstack(v) for n, v in P.items()}

    gt = json.loads(Path(a.gt).read_text())
    stitches = [((s[0][0], s[0][1]), (s[1][0], s[1][1])) for s in gt["stitches"]]

    fig, ax = plt.subplots(figsize=(15, 12))
    for name, pts in P.items():
        ax.plot(*np.vstack([pts, pts[:1]]).T, color="#d8d8d8", lw=0.9, zorder=1)
        c = pts.mean(0)
        ax.text(c[0], c[1], name, fontsize=8, ha="center", color="#999", zorder=2)

    cmap = plt.get_cmap("tab20")
    rows = []
    for n, (ka, kb) in enumerate(stitches, 1):
        col = cmap((n - 1) % 20)
        arcs = []
        for key in (ka, kb):
            seg = E[key]
            arc = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
            arcs.append(arc)
            ax.plot(*seg.T, color=col, lw=3.4, zorder=4, solid_capstyle="round")
            m = seg[len(seg) // 2]
            ax.text(m[0], m[1], str(n), fontsize=8.5, color="white", ha="center",
                    va="center", zorder=6, weight="bold",
                    bbox=dict(fc=col, ec="none", boxstyle="circle,pad=0.22"))
        m1 = E[ka][len(E[ka]) // 2]
        m2 = E[kb][len(E[kb]) // 2]
        ax.plot([m1[0], m2[0]], [m1[1], m2[1]], color=col, lw=0.7, ls=":", zorder=3,
                alpha=.7)
        rows.append((n, ka, kb, arcs[0], arcs[1], max(arcs) / min(arcs)))

    x0 = min(p[:, 0].min() for p in P.values())
    y0 = min(p[:, 1].min() for p in P.values())
    ax.plot([x0, x0 + 10], [y0 - 7, y0 - 7], color="k", lw=3)
    ax.text(x0 + 5, y0 - 11, "10 cm", ha="center", fontsize=9)
    ax.set_aspect("equal")
    ax.set_title(f"{len(stitches)} ground-truth stitches, drawn at true scale\n"
                 "both edges of a stitch share a colour and a number",
                 fontsize=12)
    ax.set_xlabel("cm"); ax.set_ylabel("cm")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=115, bbox_inches="tight")
    print(f"wrote {a.out}\n")

    print(f"{'#':>3}  {'edge A':<24}{'cm':>8}   {'edge B':<24}{'cm':>8}{'ratio':>8}   note")
    for n, ka, kb, a1, a2, r in rows:
        note = "CHECK" if r > 1.41 else ("wide" if r > 1.15 else "")
        print(f"{n:>3}  {ka[0]+'#'+str(ka[1]):<24}{a1:>8.2f}   "
              f"{kb[0]+'#'+str(kb[1]):<24}{a2:>8.2f}{r:>8.3f}   {note}")

    print(f"\n{'panel':<20}{'edges':>7}   edge lengths (index: cm)")
    for name in P:
        ks = sorted(k for k in E if k[0] == name)
        s = "  ".join(f"{k[1]}:{np.linalg.norm(np.diff(E[k], axis=0), axis=1).sum():.1f}"
                      for k in ks)
        used = {k for st in stitches for k in st if k[0] == name}
        free = [k[1] for k in ks if k not in used]
        print(f"{name:<20}{len(ks):>7}   {s}"
              + (f"      free: {free}" if free else "      all sewn"))


if __name__ == "__main__":
    main()
