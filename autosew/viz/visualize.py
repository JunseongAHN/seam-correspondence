"""AutoSew input/GT/prediction visualizer.

Draws the 2D pattern (panels laid out by their translation, front/back split),
colors each stitched edge PAIR with a shared color, and connects the pair with a line.
Unstitched edges are gray. Reads the RAW specification.json (independent of the
training parser), so what you see is what is in the file.

GT only:
  python viz/visualize.py --spec <..._specification.json> --out viz/out/g.png
Prediction overlay (after training; needs torch + a checkpoint):
  python viz/visualize.py --spec <...> --ckpt runs/full/best.pt --out viz/out/g_pred.png
Batch sample:
  python viz/visualize.py --data_dir <PART_DIR> --n 6 --out_dir viz/out
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for --ckpt mode


# ---------------- geometry ----------------

def rel_to_abs(p0, p1, c):
    e = p1 - p0
    perp = np.array([-e[1], e[0]])
    return p0 + c[0] * e + c[1] * perp


def arc_points(p0, p1, r, large, sweep, n=32):
    """SVG-style circular arc from p0 to p1. GCD circle params: [radius, large_arc, right]."""
    chord = p1 - p0
    d = float(np.linalg.norm(chord))
    if d < 1e-9:
        return np.stack([p0, p1])
    r = max(float(r), d / 2 + 1e-9)
    mid = (p0 + p1) / 2
    h = math.sqrt(max(r * r - d * d / 4, 0.0))
    perp = np.array([-chord[1], chord[0]]) / d
    center = mid + (h if bool(large) != bool(sweep) else -h) * perp
    a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
    a1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
    if sweep:  # increasing angle (CCW in y-up coords)
        while a1 < a0:
            a1 += 2 * math.pi
    else:
        while a1 > a0:
            a1 -= 2 * math.pi
    t = np.linspace(a0, a1, n)
    return np.stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)], axis=1)


def sample_edge(p0, p1, curv, n=24):
    """polyline points for one edge (raw JSON curvature, relative coords)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    if curv is None:
        return np.stack([p0, p1])
    if isinstance(curv, list):  # legacy quadratic, relative ctrl
        q = rel_to_abs(p0, p1, curv)
        t = np.linspace(0, 1, n)[:, None]
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * q + t ** 2 * p1
    if isinstance(curv, dict):
        typ = str(curv.get("type", "")).lower()
        par = curv.get("params", [])
        if "quadratic" in typ:
            q = rel_to_abs(p0, p1, par[0])
            t = np.linspace(0, 1, n)[:, None]
            return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * q + t ** 2 * p1
        if "cubic" in typ:
            q1 = rel_to_abs(p0, p1, par[0]); q2 = rel_to_abs(p0, p1, par[1])
            t = np.linspace(0, 1, n)[:, None]
            return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * q1
                    + 3 * (1 - t) * t ** 2 * q2 + t ** 3 * p1)
        if "circ" in typ or "arc" in typ:
            r = par[0]; large = par[1] if len(par) > 1 else 0; sweep = par[2] if len(par) > 2 else 0
            return arc_points(p0, p1, r, large, sweep)
    return np.stack([p0, p1])


# ---------------- layout ----------------

def load_raw(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    pat = spec["pattern"] if "pattern" in spec else spec
    panels = {}
    order = pat.get("panel_order") or list(pat["panels"].keys())
    for name in order:
        pd = pat["panels"][name]
        panels[name] = {
            "verts": np.asarray(pd["vertices"], float),
            "edges": pd["edges"],
            "tr": np.asarray(pd.get("translation", [0, 0, 0]), float),
            "rot": np.asarray(pd.get("rotation", [0, 0, 0]), float),
        }
    stitches = []
    for st in pat.get("stitches", []):
        sides = [(s["panel"], int(s["edge"])) for s in st
                 if isinstance(s, dict) and "panel" in s and "edge" in s]
        if len(sides) >= 2:
            stitches.append(sides)
    return panels, stitches


def place_panels(panels, gap=25.0):
    """2D placement: rotate by rz, translate (tx,ty); back panels (tz<0) shifted right."""
    placed = {}
    for name, p in panels.items():
        rz = math.radians(p["rot"][2])
        R = np.array([[math.cos(rz), -math.sin(rz)], [math.sin(rz), math.cos(rz)]])
        v = p["verts"] @ R.T + p["tr"][:2]
        placed[name] = {"verts": v, "edges": p["edges"], "back": p["tr"][2] < 0}
    front = [n for n, p in placed.items() if not p["back"]]
    back = [n for n, p in placed.items() if p["back"]]

    def bbox(names):
        pts = np.concatenate([placed[n]["verts"] for n in names]) if names else np.zeros((1, 2))
        return pts.min(0), pts.max(0)

    if front and back:
        fmin, fmax = bbox(front)
        bmin, _ = bbox(back)
        dx = fmax[0] + gap - bmin[0]
        for n in back:
            placed[n]["verts"] = placed[n]["verts"] + np.array([dx, 0.0])
    return placed


# ---------------- drawing ----------------

def stitch_colors(n, seed=0):
    if n == 0:
        return []
    phi = 0.61803398875
    hs = [(0.05 + i * phi) % 1.0 for i in range(n)]
    return [plt.cm.hsv(h) for h in hs]


def edge_polyline(placed_panel, eidx, n=24):
    e = placed_panel["edges"][eidx]
    v = placed_panel["verts"]
    p0, p1 = v[e["endpoints"][0]], v[e["endpoints"][1]]
    return sample_edge(p0, p1, e.get("curvature"), n)


def draw_pattern(ax, placed, stitches, pred_pairs=None, title=""):
    """GT mode: pred_pairs=None. Pred mode: pred_pairs = set of ((p,e),(q,f)) sorted tuples."""
    stitched_edges = {}
    for si, sides in enumerate(stitches):
        for pe in sides:
            stitched_edges.setdefault(pe, []).append(si)
    # panel outlines + unstitched edges
    for name, pp in placed.items():
        for i in range(len(pp["edges"])):
            if (name, i) not in stitched_edges:
                pts = edge_polyline(pp, i)
                ax.plot(pts[:, 0], pts[:, 1], color="0.55", lw=1.0, zorder=1)
        c = pp["verts"].mean(0)
        ax.text(c[0], c[1], name, fontsize=5, ha="center", va="center",
                color="0.35", zorder=5)
    # stitched edges, colored per stitch, connected by a line
    colors = stitch_colors(len(stitches))
    for si, sides in enumerate(stitches):
        col = colors[si]
        mids = []
        for (pn, ei) in sides:
            pts = edge_polyline(placed[pn], ei)
            ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2.4, zorder=3,
                    solid_capstyle="round")
            mids.append(pts[len(pts) // 2])
        for a in range(len(mids)):
            for b in range(a + 1, len(mids)):
                ax.plot([mids[a][0], mids[b][0]], [mids[a][1], mids[b][1]],
                        color=col, lw=0.8, alpha=0.55, zorder=2)
    # prediction overlay
    if pred_pairs is not None:
        gt_pairs = set()
        for sides in stitches:
            for a in range(len(sides)):
                for b in range(a + 1, len(sides)):
                    gt_pairs.add(tuple(sorted((sides[a], sides[b]))))
        def mid(pe):
            pts = edge_polyline(placed[pe[0]], pe[1])
            return pts[len(pts) // 2]
        for pr in pred_pairs & gt_pairs:
            m1, m2 = mid(pr[0]), mid(pr[1])
            ax.plot([m1[0], m2[0]], [m1[1], m2[1]], color="green", lw=1.6, alpha=0.9, zorder=4)
        for pr in pred_pairs - gt_pairs:
            m1, m2 = mid(pr[0]), mid(pr[1])
            ax.plot([m1[0], m2[0]], [m1[1], m2[1]], color="red", lw=1.8, zorder=6)
        for pr in gt_pairs - pred_pairs:
            m1, m2 = mid(pr[0]), mid(pr[1])
            ax.plot([m1[0], m2[0]], [m1[1], m2[1]], color="black", lw=1.4,
                    ls=(0, (4, 3)), zorder=6)
        n_ok = len(pred_pairs & gt_pairs); n_fp = len(pred_pairs - gt_pairs); n_fn = len(gt_pairs - pred_pairs)
        title += f"   pred: correct {n_ok} / FP {n_fp} / FN {n_fn}"
        ax.plot([], [], color="green", lw=1.6, label="correct")
        ax.plot([], [], color="red", lw=1.8, label="FP")
        ax.plot([], [], color="black", lw=1.4, ls=(0, (4, 3)), label="FN")
        ax.legend(loc="upper right", fontsize=6)
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")


def predict_pairs(spec_path, ckpt_path):
    import torch
    from autosew.config import AutoSewConfig
    from autosew.gcd_parser import parse_specification
    from autosew.features import pattern_to_tensors
    from autosew.dataset import collate
    from autosew.model import AutoSewGNN
    from autosew.sinkhorn import log_assignment
    from autosew.metrics import hard_assign_single, _slice_logP

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = AutoSewConfig(**{k: (tuple(v) if k == "edge_count_minmax" else v)
                           for k, v in ck["cfg"].items()})
    model = AutoSewGNN(cfg)
    model.load_state_dict(ck["model"]); model.eval()
    pat = parse_specification(spec_path)
    s = pattern_to_tensors(pat, cfg)
    batch = collate([s])
    with torch.no_grad():
        f = model(batch["x"], batch["nbr"], batch["mask"])
        logP = log_assignment(model.scores(f), model.dustbin_z, batch["mask"], cfg)
    Mb = int(batch["mask"][0].sum())
    pred = hard_assign_single(_slice_logP(logP[0], Mb), Mb, cfg.tau_multi, cfg.hard_mode)
    keys = pat.edge_key_list()
    return {tuple(sorted((keys[i], keys[j]))) for (i, j) in pred}


def render(spec_path, out_path, ckpt=None):
    panels, stitches = load_raw(spec_path)
    placed = place_panels(panels)
    pred = predict_pairs(spec_path, ckpt) if ckpt else None
    pts = np.concatenate([p["verts"] for p in placed.values()])
    w, h = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
    fw = min(16, max(7, w / 30)); fh = max(4, fw * (h / max(w, 1e-6)) * 1.1 + 0.6)
    fig, ax = plt.subplots(figsize=(fw, min(fh, 12)))
    name = Path(spec_path).stem.replace("_specification", "")
    draw_pattern(ax, placed, stitches, pred,
                 title=f"{name}  ({len(panels)} panels, {len(stitches)} stitches)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print("saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="viz/out")
    args = ap.parse_args()
    if args.spec:
        out = args.out or str(Path(args.out_dir) /
                              (Path(args.spec).stem.replace("_specification", "")
                               + ("_pred" if args.ckpt else "_gt") + ".png"))
        render(args.spec, out, args.ckpt)
    elif args.data_dir:
        files = sorted(Path(args.data_dir).rglob("*specification.json"))
        random.Random(args.seed).shuffle(files)
        for f in files[:args.n]:
            out = str(Path(args.out_dir) / (f.stem.replace("_specification", "")
                                            + ("_pred" if args.ckpt else "_gt") + ".png"))
            try:
                render(f, out, args.ckpt)
            except Exception as e:
                print("[fail]", f, type(e).__name__, e)
    else:
        ap.error("need --spec or --data_dir")


if __name__ == "__main__":
    main()
