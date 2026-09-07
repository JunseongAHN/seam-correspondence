#!/usr/bin/env python
"""Build the viewer's data files.

  python export.py --root <GarmentCodeData dir> --garment rand_XXXX

Writes data/contours.json (geometry for the 3-D view) and data/sweep.json
(the sigma x error-shape table).  The page fetches both, so re-running this and
refreshing the browser is the whole edit loop.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ninject.io_gcd import (load, panel_loops, panel_membership, stitch_sets,
                           front_visible, AX_Z)
from ninject.mesh import panel_surface
from ninject import noise as N
from ninject.pipeline import project_lift, decode, score, share_matrix

DEF_ROOT = r"C:\Users\PC\Downloads\data"
SIGMAS = [1, 2, 3, 5, 10, 20]
SHAPES = [("white", None, "백색"), ("ema", 0.1, "EMA 평활")]


def build_geometry(W, F, labels, panels, loops, out_dir, cell=1.6,
                   n_cloud=2500, seed=2, log=print):
    """Per panel: decimated surface (verts + faces) and its boundary loops as indices.

    Boundary vertices survive decimation untouched, so the outline is exactly the
    measured one while the interior face count drops by ~10x.
    """
    _, memb, _ = panel_membership(F, labels)
    by_panel = {}
    for pk, L in loops:
        by_panel.setdefault(pk, []).append(L)

    items, tot_f, tot_v = [], 0, 0
    for pk in sorted(by_panel):
        own = np.array([pk in memb[v] for v in range(len(labels))])
        sel = own[F].all(1)
        if sel.sum() < 4:
            continue
        Ls = sorted(by_panel[pk], key=len, reverse=True)[:3]
        V2, F2, loops2 = panel_surface(W, F[sel], Ls, cell)
        zs = V2[:, AX_Z]
        items.append(dict(
            name=panels[pk], zmed=round(float(np.median(zs)), 2),
            front=bool(np.median(zs) >= 0),
            verts=[[round(float(c), 2) for c in v] for v in V2],
            faces=[[int(a), int(b), int(c)] for a, b, c in F2],
            loops=loops2))
        tot_f += len(F2); tot_v += len(V2)
        log(f"  {panels[pk]:<18} verts {len(V2):>5}  faces {len(F2):>5}  "
            f"loops {len(loops2)}  z {np.median(zs):>7.2f}")

    rs = np.random.default_rng(seed)
    pick = rs.choice(len(W), min(n_cloud, len(W)), replace=False)
    lo, hi = W.min(0), W.max(0)
    doc = dict(panels=items,
               cloud=[[round(float(c), 1) for c in W[i]] for i in pick],
               center=[round(float(c), 2) for c in (lo + hi) / 2],
               bbox=dict(lo=[round(float(c), 2) for c in lo],
                         hi=[round(float(c), 2) for c in hi]))
    json.dump(doc, open(os.path.join(out_dir, "contours.json"), "w"), separators=(",", ":"))
    log(f"  total: verts {tot_v}  faces {tot_f}")
    return tot_v, tot_f


def build_sweep(W, labels, loops, out_dir, res=1024, draws=6, seed=101, log=print):
    sets = stitch_sets(labels)
    occ = {}
    for li, (_, L) in enumerate(loops):
        for pos, v in enumerate(L):
            occ.setdefault(v, []).append((li, pos))
    seam_v = [v for v in sets if len(occ.get(v, [])) >= 2]
    vis = front_visible(W, np.array(seam_v))
    seam_v = [v for v, ok in zip(seam_v, vis) if ok]
    copies = [(v, li, pos) for v in seam_v for (li, pos) in occ[v][:2]]
    per_copy = [sets[c[0]] for c in copies]
    share, member, ids = share_matrix(per_copy)

    loopv = [np.array(L) for _, L in loops]
    inloop = np.zeros(len(labels), bool)
    for L in loopv:
        inloop[L] = True
    interior = np.where(~inloop)[0]
    frame = (W[:, 0].min(), W[:, 0].max(), W[:, 1].min(), W[:, 1].max())

    def one(sig_mm, alpha, rng):
        s = sig_mm / 10.0
        disp = [N.displace(rng, len(L), s, alpha) for L in loopv]
        pts = np.array([W[v] + disp[li][pos] for (v, li, pos) in copies])
        occl = [W[L] + disp[li] for li, L in enumerate(loopv)]
        occl.append(W[interior] + N.white(rng, len(interior), s))
        lifted, _, px = project_lift(np.vstack([pts] + occl), res, frame)
        good, _, mutual = decode(lifted[:len(pts)], share)
        sc = score(good, member)
        sc.update(mutual=mutual, sep_mm=N.mean_pair_separation_mm(disp), px_mm=px)
        return sc

    rows = {}
    keys = ("vertex", "mutual", "seam_all", "seam_vote", "garment", "sep_mm")
    log(f"{'σ':>5} {'shape':>6} {'sep mm':>7} {'vertex':>7} {'mutual':>7} "
        f"{'s.all':>7} {'s.vote':>7} {'garment':>7}")
    log("-" * 60)
    for s in SIGMAS:
        for tag, alpha, _label in SHAPES:
            rng = np.random.default_rng(seed)
            acc = [one(s, alpha, rng) for _ in range(draws)]
            r = {k: float(np.mean([a[k] for a in acc])) for k in keys}
            rows[f"{s}|{tag}"] = r
            log(f"{s:>4}mm {tag:>6} {r['sep_mm']:>7.1f} {r['vertex']:>7.4f} "
                f"{r['mutual']:>7.3f} {r['seam_all']:>7.4f} {r['seam_vote']:>7.4f} "
                f"{r['garment']:>7.2f}")
        log("")
    meta = dict(sigmas=SIGMAS, shapes=[[t, l] for t, _, l in SHAPES],
                n_seam_vertices=len(seam_v), n_copies=len(copies),
                n_seams=int(sum(member[:, k].sum() > 1 for k in range(member.shape[1]))),
                res=res, draws=draws, px_mm=round(acc[-1]["px_mm"], 2))
    json.dump(dict(meta=meta, rows=rows),
              open(os.path.join(out_dir, "sweep.json"), "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEF_ROOT, help="GarmentCodeData directory")
    ap.add_argument("--garment", default="rand_00YONAPXZE")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--cell", type=float, default=1.6,
                    help="interior decimation cell size in cm (bigger = coarser)")
    ap.add_argument("--skip-sweep", action="store_true")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "data"); os.makedirs(out, exist_ok=True)
    gd = os.path.join(a.root, a.garment)
    print(f"garment: {gd}")
    W, F, labels = load(gd, a.garment)
    panels, loops = panel_loops(W, F, labels)
    nv, nf = build_geometry(W, F, labels, panels, loops, out, cell=a.cell)
    print(f"panels {len(panels)}  loops {len(loops)}  surface verts {nv}  faces {nf}")
    meta = dict(garment=a.garment, panels=len(panels), loops=len(loops),
                surface_verts=nv, surface_faces=nf, cell_cm=a.cell)
    if not a.skip_sweep:
        print()
        meta.update(build_sweep(W, labels, loops, out, res=a.res, draws=a.draws))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("\nwrote data/contours.json, data/sweep.json, data/meta.json")


if __name__ == "__main__":
    main()
