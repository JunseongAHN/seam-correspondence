"""Figures for REPORT.md.  Each one is a measurement, not an illustration.

    & $PY dxfcheck/make_report_figures.py --out report/
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "autosew"))
sys.path.insert(0, str(R / "dxfcheck"))
from autosew.config import AutoSewConfig, KT_STRAIGHT
from autosew.curves import edge_polyline, sagitta_profile
from autosew.dataset import find_spec_files
from autosew.gcd_parser import parse_specification
import dxf_to_features as D

K = 11
CLO = str(R / "clo_example" / "panel_seperated.dxf")
GCD_TEST = r"C:\Users\POMCHECKER\gcd_data\test"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, RED, GREY, GREEN = "#2b6cb0", "#c53030", "#8a8a8a", "#2f855a"


def fig_turn_points(out):
    """Which turn points CLO exports, and which of them are not corners at all."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    for ax, name in zip(axes, ("Pattern_634078_M", "3_M")):
        pts = idx_all = None
        for nm, p, i in D.read_panels(CLO):          # filtered
            if nm == name:
                pts, idx_keep = p, i
        # unfiltered set, for the comparison
        import ezdxf
        doc = ezdxf.readfile(CLO)
        blk = doc.blocks.get(name)
        poly = None
        for layer in D.BOUNDARY_LAYERS if hasattr(D, "BOUNDARY_LAYERS") else ("8", "1"):
            poly = next((e for e in blk if e.dxftype() == "POLYLINE"
                         and e.dxf.layer == layer), None)
            if poly is not None:
                break
        P = np.array([list(v.dxf.location)[:2] for v in poly.vertices]) / D.MM
        turn = np.array([list(e.dxf.location)[:2] for e in blk
                         if e.dxftype() == "POINT" and e.dxf.layer == "2"]).reshape(-1, 2) / D.MM
        idx_all = sorted({int(np.argmin(np.linalg.norm(P - t, axis=1)))
                          for t in np.unique(np.round(turn, 6), axis=0)})
        dropped = [i for i in idx_all if i not in idx_keep]

        ax.plot(*np.vstack([P, P[:1]]).T, color=GREY, lw=1.0)
        ax.plot(P[idx_keep, 0], P[idx_keep, 1], "o", ms=6, color=BLUE,
                label=f"real corner ({len(idx_keep)})")
        if dropped:
            ax.plot(P[dropped, 0], P[dropped, 1], "X", ms=11, color=RED,
                    label=f"smooth, dropped ({len(dropped)})")
        cx = (P[:, 0].min() + P[:, 0].max()) / 2
        ax.axvline(cx, color=RED, ls=":", lw=1, alpha=.6)
        ax.set_title(f"{name}: {len(idx_all)} turn points \u2192 {len(idx_keep)} edges")
        ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=7.5)
    fig.suptitle("CLO exports a turn point on the mirror axis where the outline is smooth.\n"
                 "It splits one seam across two edges, which whole-edge matching cannot express.",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / "01-turn-points.png", bbox_inches="tight")
    plt.close(fig)


def fig_panel_u(out):
    """What destroying the panel-id feature costs each model."""
    models = ["r1\n(tagged, index)", "s12_tagged", "s12_sagitta", "rand_sagitta\n(random ids)"]
    base = [0.9086, 0.8346, 0.8210, 0.7552]
    zero = [0.6110, 0.5919, 0.6273, 0.7577]
    x = np.arange(len(models)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - w/2, base, w, label="panel ids intact", color=BLUE)
    ax.bar(x + w/2, zero, w, label="panel ids zeroed (what a DXF gives)", color=RED)
    for i, (b, z) in enumerate(zip(base, zero)):
        ax.annotate(f"{z-b:+.3f}", (i + w/2, z + .012), ha="center", fontsize=8,
                    color=GREEN if z >= b else RED, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("test TF1"); ax.set_ylim(0, 1.0); ax.legend(fontsize=8)
    ax.set_title("A quarter of the reported accuracy came from panel ORDER,\n"
                 "which a DXF does not have. Training with shuffled ids removes the dependence.",
                 fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / "02-panel-order.png", bbox_inches="tight")
    plt.close(fig)


def fig_chord_vs_arc(out, limit=250):
    """Do the two sides of a stitch agree better on chord or on arc?"""
    PARTS = [("sleeve", r"sleeve|cuff"), ("skirt", r"skirt|gore|godet"),
             ("pants", r"pant|trouser|leg"), ("collar", r"collar|hood|lapel"),
             ("torso", r"torso|bodice|front|back|wb|waistband")]

    def part(n):
        n = n.lower()
        for p, pt in PARTS:
            if re.search(pt, n):
                return p
        return "other"

    cfg = AutoSewConfig()
    rat = defaultdict(list)
    for f in find_spec_files(GCD_TEST, limit=limit):
        try:
            p = parse_specification(f)
        except Exception:
            continue
        ch, arc, pan = {}, {}, {}
        for panel in p.panels:
            for e in panel.edges:
                k = (panel.name, e.idx_in_panel)
                poly = edge_polyline(e)
                ch[k] = float(np.hypot(e.end[0] - e.start[0], e.end[1] - e.start[1]))
                arc[k] = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
                pan[k] = panel.name
        for sides in p.stitches:
            ss = [s for s in sides if s in ch]
            if len(ss) != 2:
                continue
            a, b = ss
            if pan[a] == pan[b]:
                continue
            ka, kb = part(pan[a]), part(pan[b])
            kind = "same part" if ka == kb else f"{'/'.join(sorted((ka, kb)))}"
            if min(ch[a], ch[b]) > 1e-6:
                rat[kind].append((max(ch[a], ch[b]) / min(ch[a], ch[b]),
                                  max(arc[a], arc[b]) / min(arc[a], arc[b])))
    keep = [k for k in rat if len(rat[k]) >= 100]
    keep.sort(key=lambda k: -np.median([r for r, _ in rat[k]]))
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = np.arange(len(keep)); w = 0.36
    c = [(np.array(rat[k])[:, 0] < 1.1).mean() * 100 for k in keep]
    a = [(np.array(rat[k])[:, 1] < 1.1).mean() * 100 for k in keep]
    ax.bar(x - w/2, c, w, label="chord (dim 4, the paper's feature)", color=GREY)
    ax.bar(x + w/2, a, w, label="arc length (added)", color=BLUE)
    for i, (cc, aa) in enumerate(zip(c, a)):
        if aa - cc > 0.5:
            ax.annotate(f"+{aa-cc:.0f}pt", (i + w/2, aa + 1.5), ha="center",
                        fontsize=8, color=GREEN, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(keep, fontsize=8)
    ax.set_ylabel("% of stitches whose two edges agree within 10%")
    ax.set_ylim(0, 108); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("A seam matches the ARC -- the fabric you sew along -- not the chord.\n"
                 "The gain lands exactly on the seam types the model is worst at.",
                 fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / "03-chord-vs-arc.png", bbox_inches="tight")
    plt.close(fig)


def fig_clo_profiles(out):
    """The seams the model finds are mirror pairs; the ones it misses are not.

    Edges come from panel_edges(), which applies the same anticlockwise canonicalisation
    build() does.  Reading read_panels() directly would name a different edge on every
    panel that got reversed -- which is exactly the mistake that produced an earlier,
    wrong version of this figure."""
    E = {k: (float(np.hypot(*(s[-1] - s[0]))), sagitta_profile(s, K), s)
         for k, s in D.panel_edges(CLO).items()}

    cases = [("side seam  3_M#5 \u2194 9_M#1\nPREDICTED", ("3_M", 5), ("9_M", 1)),
             ("waist  3_M#4 \u2194 Pattern_634078_M#1\nMISSED", ("3_M", 4),
              ("Pattern_634078_M", 1)),
             ("armhole  3_M#2 \u2194 11_M#1\nMISSED", ("3_M", 2), ("11_M", 1))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    t = np.arange(1, K + 1) / (K + 1)
    for ax, (title, k1, k2) in zip(axes, cases):
        L1, s1, _ = E[k1]; L2, s2, _ = E[k2]
        a1 = float(np.linalg.norm(np.diff(E[k1][2], axis=0), axis=1).sum())
        a2 = float(np.linalg.norm(np.diff(E[k2][2], axis=0), axis=1).sum())
        ax.plot(t, s1, "o-", color=BLUE, ms=4, label=f"{k1[0]}#{k1[1]}  {a1:.1f} cm")
        ax.plot(t, s2, "s-", color=RED, ms=4, label=f"{k2[0]}#{k2[1]}  {a2:.1f} cm")
        ax.axhline(0, color="k", lw=.6)
        amp = max(np.abs(s1).max(), np.abs(s2).max(), 1e-9)
        res = min(np.abs(s2 - c).max() for c in (s1, -s1, s1[::-1], -s1[::-1]))
        ax.set_title(f"{title}\nshape mismatch {res/amp:.2f}", fontsize=9)
        ax.set_xlabel("position along the edge"); ax.legend(fontsize=7.5)
    axes[0].set_ylabel("sagitta  (deviation / chord)")
    fig.suptitle("The seams the model finds have interlocking shapes; the ones it misses do not.",
                 fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / "04-clo-seam-shapes.png", bbox_inches="tight")
    plt.close(fig)


def fig_encoding_roundtrip(out):
    """Sagitta converges with sampling density; the tagged encoding never does."""
    npts = [3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128]
    sag = [1.196e-1, 7.604e-2, 4.020e-2, 2.706e-2, 1.204e-2, 3.889e-3,
           3.841e-3, 9.133e-4, 7.075e-4, 2.196e-4, 4.314e-5]
    tag = [3.291, 1.174, 1.174, 1.174, 1.174, 1.174, 1.174, 1.101, 1.085, 1.000, 1.000]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.loglog(npts, tag, "s-", color=RED, label="tagged  (k_t + 10 typed slots)")
    ax.loglog(npts, sag, "o-", color=BLUE, label="sagitta profile")
    ax.set_xlabel("points sampled along the edge")
    ax.set_ylabel("max error vs the exact curve")
    ax.legend(fontsize=8)
    ax.set_title("More samples cannot repair a categorical flip.\n"
                 "The tagged encoding sits at ~1.0 no matter how much data it gets.",
                 fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / "05-encoding-density.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "report"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_turn_points, fig_panel_u, fig_chord_vs_arc, fig_clo_profiles,
               fig_encoding_roundtrip):
        fn(out)
        print(f"  {fn.__name__}")
    for p in sorted(out.glob("*.png")):
        print(f"{p.name:<28}{p.stat().st_size/1024:8.0f} KB")


if __name__ == "__main__":
    main()
