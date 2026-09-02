"""AutoSew 3D check: panel outlines placed in 3D (spec translation+rotation) + stitch lines.

- Panel outlines drawn at their 3D placement; stitched edge pairs share a color and are
  connected by a line (same color scheme as the 2D tool).
- --mesh underlays the draped `_sim.ply` next to the spec (gray points; seam vertices
  colored per stitch when `_sim_segmentation.txt` exists -- cross-check 2D colors vs 3D seams).
- --ckpt overlays predictions: green = correct, red = FP, black dashed = FN.

Euler convention for panel rotation (audited 2026-08-31): xyz, degrees, sign +1, no flip.

GT placement:      python viz/visualize3d.py --spec <...>_specification.json --show
with draped mesh:  python viz/visualize3d.py --spec <...> --mesh --show
prediction:        python viz/visualize3d.py --spec <...> --ckpt runs/full/best.pt --mesh --show
PNG (3 views):     drop --show, add --out viz/out/name_3d.png
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visualize import load_raw, sample_edge, stitch_colors  # noqa: E402


# ---------------- placement ----------------

def euler_xyz_deg(rot):
    rx, ry, rz = [math.radians(a) for a in rot]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # extrinsic x->y->z (data is near single-axis; order forgiving)


def place3d(panels):
    placed = {}
    for name, p in panels.items():
        R = euler_xyz_deg(p["rot"])
        placed[name] = {"R": R, "t": p["tr"], "verts": p["verts"], "edges": p["edges"]}
    return placed


def edge_polyline3d(pp, eidx, n=24):
    e = pp["edges"][eidx]
    v = pp["verts"]
    pts2 = sample_edge(v[e["endpoints"][0]], v[e["endpoints"][1]], e.get("curvature"), n)
    pts3 = np.concatenate([pts2, np.zeros((len(pts2), 1))], axis=1)
    return pts3 @ pp["R"].T + pp["t"]


# ---------------- ply ----------------

_PLY_T = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
          "uchar": "u1", "uint8": "u1", "char": "i1", "short": "i2", "ushort": "u2",
          "int": "i4", "int32": "i4", "uint": "u4"}


def read_ply_vertices(path):
    """Minimal reader: vertex positions from ascii or binary_little_endian PLY."""
    with open(path, "rb") as f:
        header, fmt, nvert, props = [], None, 0, []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                cur = line.split()[1]
                if cur == "vertex":
                    nvert = int(line.split()[2])
            elif line.startswith("property") and nvert and not props and False:
                pass
            if line == "end_header":
                break
        # re-scan header for vertex properties (those between 'element vertex' and next element)
        in_v = False
        for line in header:
            if line.startswith("element"):
                in_v = line.split()[1] == "vertex"
            elif line.startswith("property") and in_v:
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError("list property in vertex element unsupported")
                props.append((parts[2], _PLY_T[parts[1]]))
        if fmt == "ascii":
            rows = np.loadtxt(f, max_rows=nvert, dtype=float)
            names = [n for n, _ in props]
            xyz = rows[:, [names.index("x"), names.index("y"), names.index("z")]]
        elif fmt == "binary_little_endian":
            dt = np.dtype([(n, "<" + t) for n, t in props])
            arr = np.frombuffer(f.read(dt.itemsize * nvert), dtype=dt, count=nvert)
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(float)
        else:
            raise ValueError(f"unsupported ply format {fmt}")
    if np.ptp(xyz) < 10:   # meters -> cm heuristic (spec is cm)
        xyz = xyz * 100.0
    return xyz


def sibling(spec_path, suffix):
    p = Path(spec_path)
    cand = p.parent / (p.stem.replace("_specification", "") + suffix)
    return cand if cand.exists() else None


# ---------------- drawing ----------------

def to_mpl(pts):
    """GCD frame (x, y=up, z=front) -> matplotlib (X=x, Y=-z, Z=y) so the garment stands up."""
    out = pts[:, [0, 2, 1]].copy()
    out[:, 1] *= -1.0
    return out


def draw3d(ax, placed, stitches, pred_pairs=None, mesh=None, seg=None, title=""):
    stitched_edges = {}
    for si, sides in enumerate(stitches):
        for pe in sides:
            stitched_edges.setdefault(pe, []).append(si)
    colors = stitch_colors(len(stitches))

    if mesh is not None:
        mesh = to_mpl(mesh)
        idx = np.random.RandomState(0).choice(len(mesh), min(len(mesh), 8000), replace=False)
        if seg is not None:
            cols = []
            for i in idx:
                lab = seg[i]
                if lab.startswith("stitch_"):
                    k = int(lab.split("_")[1])
                    cols.append(colors[k % len(colors)] if colors else (0.5, 0.5, 0.5, 1))
                else:
                    cols.append((0.75, 0.75, 0.75, 0.25))
            ax.scatter(mesh[idx, 0], mesh[idx, 1], mesh[idx, 2], c=cols, s=1.2, linewidths=0)
        else:
            ax.scatter(mesh[idx, 0], mesh[idx, 1], mesh[idx, 2], color="0.75", s=0.8,
                       alpha=0.25, linewidths=0)

    for name, pp in placed.items():
        for i in range(len(pp["edges"])):
            pts = to_mpl(edge_polyline3d(pp, i))
            sids = stitched_edges.get((name, i))
            if sids:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[sids[0]], lw=2.0)
            else:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="0.45", lw=0.8)

    def mid(pe):
        pts = to_mpl(edge_polyline3d(placed[pe[0]], pe[1]))
        return pts[len(pts) // 2]

    for si, sides in enumerate(stitches):
        mids = [mid(pe) for pe in sides]
        for a in range(len(mids)):
            for b in range(a + 1, len(mids)):
                seg3 = np.stack([mids[a], mids[b]])
                ax.plot(seg3[:, 0], seg3[:, 1], seg3[:, 2], color=colors[si], lw=0.7, alpha=0.5)

    if pred_pairs is not None:
        gt_pairs = set()
        for sides in stitches:
            for a in range(len(sides)):
                for b in range(a + 1, len(sides)):
                    gt_pairs.add(tuple(sorted((sides[a], sides[b]))))
        for style, pairs in (({"color": "green", "lw": 1.6}, pred_pairs & gt_pairs),
                             ({"color": "red", "lw": 2.0}, pred_pairs - gt_pairs),
                             ({"color": "black", "lw": 1.4, "ls": (0, (4, 3))}, gt_pairs - pred_pairs)):
            for pr in pairs:
                m1, m2 = mid(pr[0]), mid(pr[1])
                ax.plot([m1[0], m2[0]], [m1[1], m2[1]], [m1[2], m2[2]], **style)
        title += f"  pred ok {len(pred_pairs & gt_pairs)} / FP {len(pred_pairs - gt_pairs)} / FN {len(gt_pairs - pred_pairs)}"

    pts_all = np.concatenate([to_mpl(edge_polyline3d(pp, i, n=2))
                              for pp in placed.values() for i in range(len(pp["edges"]))])
    if mesh is not None:
        pts_all = np.concatenate([pts_all, mesh])
    c = (pts_all.min(0) + pts_all.max(0)) / 2
    r = float((pts_all.max(0) - pts_all.min(0)).max()) / 2 * 1.05
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--mesh", action="store_true", help="underlay <name>_sim.ply if present")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true", help="interactive window (drag to rotate)")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels, stitches = load_raw(args.spec)
    placed = place3d(panels)
    pred = None
    if args.ckpt:
        from visualize import predict_pairs
        pred = predict_pairs(args.spec, args.ckpt)

    mesh = seg = None
    if args.mesh:
        mp = sibling(args.spec, "_sim.ply")
        if mp:
            mesh = read_ply_vertices(mp)
            sp = sibling(args.spec, "_sim_segmentation.txt")
            if sp:
                seg = [l.strip() for l in open(sp)]
                if len(seg) != len(mesh):
                    seg = None
        else:
            print("[warn] no _sim.ply next to spec; drawing placement only")

    name = Path(args.spec).stem.replace("_specification", "")
    title = f"{name}  ({len(panels)} panels, {len(stitches)} stitches)"
    if args.show:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")
        draw3d(ax, placed, stitches, pred, mesh, seg, title)
        plt.show()
    else:
        fig = plt.figure(figsize=(16, 6))
        for k, (elev, azim) in enumerate([(12, -60), (5, -90), (5, 0)]):
            ax = fig.add_subplot(1, 3, k + 1, projection="3d")
            draw3d(ax, placed, stitches, pred, mesh, seg, title if k == 0 else "")
            ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()
        out = args.out or str(Path("viz/out") / (name + "_3d.png"))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, facecolor="white")
        print("saved", out)


if __name__ == "__main__":
    main()
