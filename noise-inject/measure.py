#!/usr/bin/env python
"""How wrong is the seam decode under the error you see in view_vtk.py?

Same noise generator as the viewer: white N(0, s^2 I) per panel vertex, then
`iters` rounds of neighbour averaging over that panel's own mesh graph, then the
magnitude scaled back to the white draw.  So every row at a given sigma has the
SAME error size and differs only in how smooth the error is.

    python measure.py
    python measure.py --garment rand_023FMIGQK0 --draws 8
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ninject.io_gcd import load, panel_loops, panel_membership, stitch_sets, front_visible
from ninject import noise as N
from ninject.pipeline import project_lift, decode, score, share_matrix

DEF_ROOT = r"C:\Users\PC\Downloads\data"
SIGMAS = [1, 2, 3, 5, 10, 20]
ITERS = [0, 5, 15, 30, 60]


def build(root, garment):
    gd = os.path.join(root, garment)
    W, F, labels = load(gd, garment)
    panels, loops = panel_loops(W, F, labels)
    _, memb, _ = panel_membership(F, labels)
    by = {}
    for pk, L in loops:
        by.setdefault(pk, []).append(L)
    out = []
    for pk in sorted(by):
        own = np.array([pk in memb[v] for v in range(len(labels))])
        sel = own[F].all(1)
        if sel.sum() < 4:
            continue
        gid = np.unique(F[sel])
        idx = {v: i for i, v in enumerate(gid)}
        Fl = np.array([[idx[a], idx[b], idx[c]] for a, b, c in F[sel]], np.int64)
        ni, np_ = N.build_adjacency(len(gid), Fl)
        out.append(dict(name=panels[pk], gid=gid, V=W[gid], nbr=(ni, np_)))
    return W, labels, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEF_ROOT)
    ap.add_argument("--garment", default="rand_00YONAPXZE")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--w", type=float, default=0.5, help="smoothing weight per iteration")
    ap.add_argument("--sigmas", default=None,
                    help="comma-separated subset, e.g. 1,2,3 (default: all)")
    ap.add_argument("--merge", action="store_true",
                    help="merge into an existing data/measure.json instead of replacing")
    a = ap.parse_args()

    W, labels, P = build(a.root, a.garment)
    print(f"{a.garment}: panels {len(P)}  vertices {sum(len(p['V']) for p in P)}")

    # a seam vertex owns one copy per panel that contains it
    sets = stitch_sets(labels)
    occ = {}
    for pi, p in enumerate(P):
        for li, v in enumerate(p["gid"]):
            if v in sets:
                occ.setdefault(v, []).append((pi, li))
    seam_v = [v for v in sets if len(occ.get(v, [])) >= 2]
    vis = front_visible(W, np.array(seam_v))
    seam_v = [v for v, ok in zip(seam_v, vis) if ok]
    copies = [(v, pi, li) for v in seam_v for (pi, li) in occ[v][:2]]
    share, member, _ = share_matrix([sets[c[0]] for c in copies])
    n_seams = int(sum(member[:, k].sum() > 1 for k in range(member.shape[1])))
    print(f"front-visible seam vertices {len(seam_v)}  copies {len(copies)}  seams {n_seams}")

    frame = (W[:, 0].min(), W[:, 0].max(), W[:, 1].min(), W[:, 1].max())

    def one(sig_mm, iters, rng):
        s = sig_mm / 10.0
        disp, tot, cnt = [], 0.0, 0
        for p in P:
            d = N.white(rng, len(p["V"]), s)
            d = N.smooth_on_graph(d, p["nbr"][0], p["nbr"][1], a.w, iters)
            disp.append(d)
            tot += np.linalg.norm(d, axis=1).sum(); cnt += len(d)
        pts = np.array([P[pi]["V"][li] + disp[pi][li] for (_, pi, li) in copies])
        allp = np.vstack([pts] + [p["V"] + d for p, d in zip(P, disp)])
        lifted, _, px = project_lift(allp, a.res, frame)
        good, _, mutual = decode(lifted[:len(pts)], share)
        sc = score(good, member)
        sc.update(mutual=mutual, sep_mm=(tot / cnt) * np.sqrt(2) * 10, px_mm=px)
        return sc

    sigmas = ([float(x) if '.' in x else int(x) for x in a.sigmas.split(',')]
              if a.sigmas else SIGMAS)
    rows = {}
    keys = ("vertex", "mutual", "seam_all", "seam_vote", "garment", "sep_mm")
    print(f"\n{'σ':>5} {'smooth':>7} {'sep mm':>7} {'vertex':>7} {'mutual':>7} "
          f"{'s.all':>7} {'s.vote':>7} {'garment':>7}")
    print("-" * 64)
    for s in sigmas:
        for it in ITERS:
            rng = np.random.default_rng(a.seed)
            acc = [one(s, it, rng) for _ in range(a.draws)]
            r = {k: float(np.mean([x[k] for x in acc])) for k in keys}
            rows[f"{s}|{it}"] = r
            print(f"{s:>4}mm {it:>6}it {r['sep_mm']:>7.1f} {r['vertex']:>7.4f} "
                  f"{r['mutual']:>7.3f} {r['seam_all']:>7.4f} {r['seam_vote']:>7.4f} "
                  f"{r['garment']:>7.2f}")
        print()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)
    fp = os.path.join(out, "measure.json")
    done = sigmas
    if a.merge and os.path.exists(fp):
        prev = json.load(open(fp))
        prev["rows"].update(rows); rows = prev["rows"]
        done = sorted({int(k.split("|")[0]) for k in rows})
    json.dump(dict(meta=dict(garment=a.garment, sigmas=done, iters=ITERS,
                             n_seam_vertices=len(seam_v), n_copies=len(copies),
                             n_seams=n_seams, res=a.res, draws=a.draws, w=a.w,
                             smoothing="mesh-graph neighbour average (same as view_vtk.py)"),
                   rows=rows),
              open(fp, "w"), indent=1)
    print("wrote data/measure.json")


if __name__ == "__main__":
    main()
