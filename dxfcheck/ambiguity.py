"""When several edges are the same length, can the model still tell them apart?

Every model misses seams whose two edges match to 1.000, and its false positives join the
right panels through a *different* edge of the same length.  So the failure is not that
the lengths disagree -- it is that they agree with too many candidates at once.

Count, for each ground-truth stitch, how many edges the partner panel offers at
indistinguishable length, and see whether the model's success tracks that.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "autosew"))
sys.path.insert(0, str(R / "dxfcheck"))
from autosew.config import AutoSewConfig
from autosew.dataset import collate
from autosew.metrics import hard_assign_single, _slice_logP
from autosew.model import AutoSewGNN
from autosew.sinkhorn import log_assignment
from dxf_to_features import build, panel_edges

TOL = 0.03      # two edges within 3% are indistinguishable by length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", default=str(R / "clo_example" / "panel_seperated.dxf"))
    ap.add_argument("--gt", default=str(R / "clo_example" / "panel_seperated_gt.json"))
    ap.add_argument("--ckpt", default="autosew/runs/rand_sagitta/best.pt")
    a = ap.parse_args()

    E = panel_edges(a.dxf)
    arc = {k: float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) for k, s in E.items()}
    doc = json.loads(Path(a.gt).read_text())
    gt = [((s[0][0], s[0][1]), (s[1][0], s[1][1])) for s in doc["stitches"]]

    ck = torch.load(R / a.ckpt, map_location="cpu", weights_only=False)
    cfg = AutoSewConfig(**{k: (tuple(v) if k == "edge_count_minmax" else v)
                           for k, v in ck["cfg"].items()})
    m = AutoSewGNN(cfg); m.load_state_dict(ck["model"]); m.eval()
    x, keys = build(a.dxf, cfg)
    idx = {f"{p}#{e}": i for i, (p, e) in enumerate(keys)}
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
    pred = {(min(i, j), max(i, j))
            for i, j in hard_assign_single(_slice_logP(lp[0], M), M, cfg.tau_multi, cfg.hard_mode)}

    print(f"checkpoint {a.ckpt}   {len(gt)} ground-truth stitches\n")
    print(f"{'stitch':<38}{'cm':>7}{'cm':>7}{'ratio':>7}"
          f"{'rivals':>8}   found?")
    rows = []
    for ka, kb in gt:
        la, lb = arc[ka], arc[kb]
        # how many OTHER edges in the whole garment are within TOL of this length?
        rivals = sum(1 for k, L in arc.items()
                     if k not in (ka, kb) and abs(L - la) / max(L, la) < TOL)
        na, nb = idx[f"{ka[0]}#{ka[1]}"], idx[f"{kb[0]}#{kb[1]}"]
        hit = (min(na, nb), max(na, nb)) in pred
        rows.append((rivals, hit))
        print(f"{ka[0]+'#'+str(ka[1])+' ~ '+kb[0]+'#'+str(kb[1]):<38}"
              f"{la:>7.2f}{lb:>7.2f}{max(la,lb)/min(la,lb):>7.3f}{rivals:>8}"
              f"   {'YES' if hit else 'no'}")

    print(f"\n{'edges of that length elsewhere':<34}{'stitches':>10}{'found':>8}{'rate':>8}")
    bins = [(0, 0, "unique length"), (1, 2, "1-2 rivals"), (3, 99, "3 or more rivals")]
    for lo, hi, lbl in bins:
        sel = [h for r, h in rows if lo <= r <= hi]
        if sel:
            print(f"{lbl:<34}{len(sel):>10}{sum(sel):>8}{sum(sel)/len(sel):>8.2f}")

    # the same question for the whole garment: how ambiguous is it?
    print(f"\nacross all {len(arc)} edges, how many share a length (within {TOL:.0%}):")
    hist = defaultdict(int)
    for k, L in arc.items():
        n = sum(1 for k2, L2 in arc.items()
                if k2 != k and abs(L2 - L) / max(L, L2) < TOL)
        hist[n] += 1
    for n in sorted(hist):
        print(f"  {n} others of the same length: {hist[n]} edges")


if __name__ == "__main__":
    main()
