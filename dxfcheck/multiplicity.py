"""How often does the model put one edge in more than one stitch?

hard_mode="union" takes the row-wise argmax above tau_multi from BOTH rows and unions the
unordered pairs, which is how the paper expresses a multi-edge stitch.  So a prediction is
not one-to-one by construction.  The training data has no multi-edge stitches at all
(has_multi_edge_gt is false in every run), so any multiplicity here is the model hedging,
not a learned structure -- and every extra pair is a false positive against a one-to-one
ground truth.
"""
import argparse
import json
import sys
from collections import Counter
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

CKPTS = [("r1", "autosew/runs/r1/best.pt"),
         ("s12_sagitta", "autosew/runs/s12_sagitta/best.pt"),
         ("rand_sagitta", "autosew/runs/rand_sagitta/best.pt"),
         ("r2", "autosew/runs/r2/best_ep13_tf1_8030.pt"),
         ("r3", "autosew/runs/r3/best.pt")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", default=str(R / "clo_example" / "panel_seperated.dxf"))
    ap.add_argument("--gt", default=str(R / "clo_example" / "panel_seperated_gt.json"))
    a = ap.parse_args()
    doc = json.loads(Path(a.gt).read_text())

    print(f"{'model':<16}{'pred':>6}{'edges used':>12}{'in >1 stitch':>14}"
          f"{'within-panel':>14}{'mutual-only':>13}")
    for label, rel in CKPTS:
        p = R / rel
        if not p.exists():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = AutoSewConfig(**{k: (tuple(v) if k == "edge_count_minmax" else v)
                               for k, v in ck["cfg"].items()})
        m = AutoSewGNN(cfg); m.load_state_dict(ck["model"]); m.eval()
        x, keys = build(a.dxf, cfg)
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
            lp = log_assignment(m.scores(m(b["x"], b["nbr"], b["mask"])),
                                m.dustbin_z, b["mask"], cfg)
        M = int(b["mask"][0].sum())
        lpm = _slice_logP(lp[0], M)
        union = {(min(i, j), max(i, j))
                 for i, j in hard_assign_single(lpm, M, cfg.tau_multi, "union")}
        mutual = {(min(i, j), max(i, j))
                  for i, j in hard_assign_single(lpm, M, cfg.tau_multi, "mutual")}
        deg = Counter()
        for i, j in union:
            deg[i] += 1; deg[j] += 1
        multi = sum(1 for v in deg.values() if v > 1)
        same = sum(1 for i, j in union if keys[i][0] == keys[j][0])
        print(f"{label:<16}{len(union):>6}{len(deg):>12}{multi:>14}{same:>14}{len(mutual):>13}")

    # what the union rule costs, scored against the one-to-one ground truth
    print(f"\nscoring both selection rules against the {len(doc['stitches'])}-stitch "
          f"ground truth")
    print(f"{'model':<16}{'rule':<9}{'pred':>6}{'ok':>5}{'FP':>5}{'TP':>8}{'TR':>8}{'TF1':>8}")
    for label, rel in CKPTS:
        p = R / rel
        if not p.exists():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = AutoSewConfig(**{k: (tuple(v) if k == "edge_count_minmax" else v)
                               for k, v in ck["cfg"].items()})
        m = AutoSewGNN(cfg); m.load_state_dict(ck["model"]); m.eval()
        x, keys = build(a.dxf, cfg)
        idx = {f"{q}#{e}": i for i, (q, e) in enumerate(keys)}
        gt = {(min(idx[f"{s[0][0]}#{s[0][1]}"], idx[f"{s[1][0]}#{s[1][1]}"]),
               max(idx[f"{s[0][0]}#{s[0][1]}"], idx[f"{s[1][0]}#{s[1][1]}"]))
              for s in doc["stitches"]}
        nbr = np.zeros((len(keys), 2), np.int64)
        s0 = 0
        while s0 < len(keys):
            e = s0
            while e < len(keys) and keys[e][0] == keys[s0][0]:
                e += 1
            n = e - s0
            for j in range(n):
                nbr[s0 + j] = [s0 + (j - 1) % n, s0 + (j + 1) % n]
            s0 = e
        b = collate([{"x": x.astype(np.float32), "nbr": nbr,
                      "gt_pairs": np.zeros((0, 2), np.int64),
                      "stitched": np.zeros(len(keys), bool), "name": "clo"}])
        with torch.no_grad():
            lp = log_assignment(m.scores(m(b["x"], b["nbr"], b["mask"])),
                                m.dustbin_z, b["mask"], cfg)
        M = int(b["mask"][0].sum())
        lpm = _slice_logP(lp[0], M)
        for rule in ("union", "mutual"):
            pred = {(min(i, j), max(i, j))
                    for i, j in hard_assign_single(lpm, M, cfg.tau_multi, rule)}
            ok = len(pred & gt)
            tp = ok / max(len(pred), 1); tr = ok / max(len(gt), 1)
            f1 = 2 * tp * tr / max(tp + tr, 1e-9)
            print(f"{label:<16}{rule:<9}{len(pred):>6}{ok:>5}{len(pred)-ok:>5}"
                  f"{tp:>8.3f}{tr:>8.3f}{f1:>8.3f}")


if __name__ == "__main__":
    main()
