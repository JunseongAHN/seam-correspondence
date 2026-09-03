"""Score every checkpoint on the real CLO garment, against hand-drawn ground truth.

Until now the CLO prediction could only be judged by eye, or by which pairs of panels it
joined.  With ground truth drawn by hand in the web demo it can be scored edge by edge,
the same way the GCD test set is -- so the numbers quoted for GCD and for a real
industrial export finally mean the same thing.

    & $PY dxfcheck/score_clo.py
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
from autosew.dataset import collate
from autosew.metrics import hard_assign_single, _slice_logP
from autosew.model import AutoSewGNN
from autosew.sinkhorn import log_assignment
from dxf_to_features import build

CKPTS = [
    ("r1  tagged / index ids  24k", "autosew/runs/r1/best.pt"),
    ("s12_sagitta  index ids   2k", "autosew/runs/s12_sagitta/best.pt"),
    ("rand_sagitta random ids  2k", "autosew/runs/rand_sagitta/best.pt"),
    ("r2  sagitta / random  87.7k", "autosew/runs/r2/best_ep13_tf1_8030.pt"),
    ("r3  + arc features    87.7k", "autosew/runs/r3/best.pt"),
]


def cycle_neighbours(keys):
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
    return nbr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", default=str(R / "clo_example" / "panel_seperated.dxf"))
    ap.add_argument("--gt", default=str(R / "clo_example" / "panel_seperated_gt.json"))
    a = ap.parse_args()

    doc = json.loads(Path(a.gt).read_text())
    print(f"ground truth: {len(doc['stitches'])} stitches, drawn on a {doc['edges']}-edge parse")

    print(f"\n{'model':<30}{'M':>4}{'pred':>6}{'ok':>5}{'FP':>5}{'FN':>5}"
          f"{'TP':>8}{'TR':>8}{'TF1':>8}{'GSP':>6}")
    for label, rel in CKPTS:
        path = R / rel
        if not path.exists():
            print(f"{label:<30}   (no checkpoint yet)")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = AutoSewConfig(**{k: (tuple(v) if k == "edge_count_minmax" else v)
                               for k, v in ck["cfg"].items()})
        m = AutoSewGNN(cfg)
        m.load_state_dict(ck["model"])
        m.eval()

        x, keys = build(a.dxf, cfg)
        if x.shape[0] != doc["edges"]:
            print(f"{label:<30}   parse gives {x.shape[0]} edges, the ground truth was "
                  f"drawn on {doc['edges']} -- indices would not line up, skipping")
            continue
        idx = {f"{p}#{e}": i for i, (p, e) in enumerate(keys)}
        gt = set()
        bad = 0
        for st in doc["stitches"]:
            ns = [idx.get(f"{p}#{e}") for p, e in st]
            if any(n is None for n in ns):
                bad += 1
                continue
            gt.add((min(ns), max(ns)))
        if bad:
            print(f"  ! {bad} ground-truth stitch(es) name edges this parse does not have")

        nbr = cycle_neighbours(keys)
        b = collate([{"x": x.astype(np.float32), "nbr": nbr,
                      "gt_pairs": np.zeros((0, 2), np.int64),
                      "stitched": np.zeros(len(keys), bool), "name": "clo"}])
        with torch.no_grad():
            lp = log_assignment(m.scores(m(b["x"], b["nbr"], b["mask"])),
                                m.dustbin_z, b["mask"], cfg)
        M = int(b["mask"][0].sum())
        pred = {(min(i, j), max(i, j))
                for i, j in hard_assign_single(_slice_logP(lp[0], M), M,
                                               cfg.tau_multi, cfg.hard_mode)}
        ok = len(pred & gt)
        tp = ok / max(len(pred), 1)
        tr = ok / max(len(gt), 1)
        f1 = 2 * tp * tr / max(tp + tr, 1e-9)
        gsp = 1.0 if pred == gt else 0.0
        print(f"{label:<30}{M:>4}{len(pred):>6}{ok:>5}{len(pred)-ok:>5}{len(gt)-ok:>5}"
              f"{tp:>8.3f}{tr:>8.3f}{f1:>8.3f}{gsp:>6.0f}")

        # Separate "which panels go together" from "which edge of them".  A prediction
        # that joins the right two panels through the wrong pair of edges is a different
        # failure from one that joins panels that are not sewn at all.
        def panels(s):
            return {tuple(sorted((keys[i][0], keys[j][0]))) for i, j in s}
        gp, pp = panels(gt), panels(pred)
        pok = len(gp & pp)
        right_panels_wrong_edge = sum(
            1 for i, j in (pred - gt)
            if tuple(sorted((keys[i][0], keys[j][0]))) in gp)
        print(f"      panel pairs: {pok}/{len(gp)} found, {len(pp)-pok} spurious"
              f"   |   of {len(pred)-ok} false stitches, {right_panels_wrong_edge}"
              f" join the RIGHT panels through the wrong edges")

        miss = sorted(gt - pred)
        if miss:
            print("      missed: " + ", ".join(
                f"{keys[i][0]}#{keys[i][1]}~{keys[j][0]}#{keys[j][1]}" for i, j in miss[:20]))
        extra = sorted(pred - gt)
        if extra:
            print("      false : " + ", ".join(
                f"{keys[i][0]}#{keys[i][1]}~{keys[j][0]}#{keys[j][1]}" for i, j in extra[:20]))


if __name__ == "__main__":
    main()
