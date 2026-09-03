"""Does the model only find seams whose two edges are mirror images of each other?

With the edge ordering finally right, the successes on the CLO garment are all between
mirror-pair panels -- front against back bodice, front against back skirt -- and every
seam joining two differently shaped pieces is missed.  Measure the shape relation for
each ground-truth stitch and see whether it separates the hits from the misses.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "autosew"))
sys.path.insert(0, str(R / "dxfcheck"))
from autosew.config import AutoSewConfig
from autosew.curves import sagitta_profile
from autosew.dataset import collate
from autosew.metrics import hard_assign_single, _slice_logP
from autosew.model import AutoSewGNN
from autosew.sinkhorn import log_assignment
from dxf_to_features import build, panel_edges

K = 11
CKPTS = [("r1", "autosew/runs/r1/best.pt"),
         ("rand_sagitta", "autosew/runs/rand_sagitta/best.pt"),
         ("r2", "autosew/runs/r2/best_ep13_tf1_8030.pt"),
         ("r3", "autosew/runs/r3/best.pt")]


def predict(path, cfg, m):
    x, keys = build(path, cfg)
    nbr = np.zeros((len(keys), 2), np.int64)
    s = 0
    while s < len(keys):
        e = s
        while e < len(keys) and keys[e][0] == keys[s][0]:
            e += 1
        n = e - s
        for j in range(n):
            nbr[s + j] = [s + (j - 1) % n, s + (j + 1) % n]
        s = e
    b = collate([{"x": x.astype(np.float32), "nbr": nbr,
                  "gt_pairs": np.zeros((0, 2), np.int64),
                  "stitched": np.zeros(len(keys), bool), "name": "clo"}])
    with torch.no_grad():
        lp = log_assignment(m.scores(m(b["x"], b["nbr"], b["mask"])), m.dustbin_z, b["mask"], cfg)
    M = int(b["mask"][0].sum())
    idx = {f"{p}#{e}": i for i, (p, e) in enumerate(keys)}
    return {(min(i, j), max(i, j))
            for i, j in hard_assign_single(_slice_logP(lp[0], M), M,
                                           cfg.tau_multi, cfg.hard_mode)}, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", default=str(R / "clo_example" / "panel_seperated.dxf"))
    ap.add_argument("--gt", default=str(R / "clo_example" / "panel_seperated_gt.json"))
    a = ap.parse_args()

    E = panel_edges(a.dxf)
    sag = {k: sagitta_profile(s, K) for k, s in E.items()}
    arc = {k: float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) for k, s in E.items()}
    doc = json.loads(Path(a.gt).read_text())
    gt = [((s[0][0], s[0][1]), (s[1][0], s[1][1])) for s in doc["stitches"]]

    hits = {}
    for label, rel in CKPTS:
        p = R / rel
        if not p.exists():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = AutoSewConfig(**{k: (tuple(v) if k == "edge_count_minmax" else v)
                               for k, v in ck["cfg"].items()})
        m = AutoSewGNN(cfg); m.load_state_dict(ck["model"]); m.eval()
        pred, idx = predict(a.dxf, cfg, m)
        hits[label] = {st for st in gt
                       if (min(idx[f"{st[0][0]}#{st[0][1]}"], idx[f"{st[1][0]}#{st[1][1]}"]),
                           max(idx[f"{st[0][0]}#{st[0][1]}"], idx[f"{st[1][0]}#{st[1][1]}"]))
                       in pred}

    names = list(hits)
    print(f"{'stitch':<38}{'ratio':>7}{'shape':>8}{'rel':>7}   " + "".join(f"{n:>14}" for n in names))
    rows = []
    for st in gt:
        s1, s2 = sag[st[0]], sag[st[1]]
        amp = max(np.abs(s1).max(), np.abs(s2).max(), 1e-9)
        cands = {"+s": s1, "-s": -s1, "+rev": s1[::-1], "-rev": -s1[::-1]}
        w, v = min(cands.items(), key=lambda kv: np.abs(s2 - kv[1]).max())
        res = float(np.abs(s2 - v).max()) / amp
        la, lb = arc[st[0]], arc[st[1]]
        n_hit = sum(st in hits[n] for n in names)
        rows.append((res, amp, n_hit))
        print(f"{st[0][0]+'#'+str(st[0][1])+' ~ '+st[1][0]+'#'+str(st[1][1]):<38}"
              f"{max(la,lb)/min(la,lb):>7.3f}{res:>8.3f}{w:>7}   "
              + "".join(f"{'YES' if st in hits[n] else '.':>14}" for n in names))

    print(f"\n{'shape mismatch':<26}{'stitches':>10}{'found by':>10}{'of':>5}{'rate':>8}")
    for lo, hi, lbl in ((0.0, 0.05, "interlocking (<0.05)"),
                        (0.05, 0.5, "loosely related"),
                        (0.5, 9.9, "unrelated (>0.5)")):
        sel = [(r, a_, h) for r, a_, h in rows if lo <= r < hi]
        if not sel:
            continue
        tot = len(sel) * len(names)
        got = sum(h for _, _, h in sel)
        print(f"{lbl:<26}{len(sel):>10}{got:>10}{tot:>5}{got/max(tot,1):>8.2f}")

    curved = [(r, a_, h) for r, a_, h in rows if a_ > 0.01]
    print(f"\n(pairs where at least one edge is actually curved: {len(curved)} of {len(rows)};"
          f" a straight-to-straight pair has no shape to compare)")


if __name__ == "__main__":
    main()
