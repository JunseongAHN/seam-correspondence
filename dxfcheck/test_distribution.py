"""Per-garment F1 across the held-out split, for the model the demo ships.

A single averaged TF1 hides the shape of the thing: whether the model is uniformly
mediocre or mostly perfect with a tail of failures.  The demo's example garments are
picked from this distribution, so it is also what says whether that pick was fair.

    & $PY dxfcheck/test_distribution.py --limit 1500
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "autosew"))
from autosew.config import AutoSewConfig
from autosew.dataset import collate, find_spec_files
from autosew.features import pattern_to_tensors
from autosew.gcd_parser import parse_specification
from autosew.metrics import hard_assign_single, _slice_logP
from autosew.model import AutoSewGNN
from autosew.sinkhorn import log_assignment

EXAMPLES = {"rand_JYO1DHSFGH", "rand_I88DFY2AKV", "rand_E3IN9RH60H",
            "rand_3C7X2I6WQ7", "rand_GE9NBC1HFY", "rand_1328ERLDIC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="autosew/runs/r2/best_ep13_tf1_8030.pt")
    ap.add_argument("--dir", default=r"C:\Users\POMCHECKER\gcd_data\test")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--rule", default="mutual", choices=["mutual", "union"])
    ap.add_argument("--fig", default=str(R / "report" / "06-test-distribution.png"))
    a = ap.parse_args()

    ck = torch.load(R / a.ckpt, map_location="cpu", weights_only=False)
    cfg = AutoSewConfig(**{k: (tuple(v) if k == "edge_count_minmax" else v)
                           for k, v in ck["cfg"].items()})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoSewGNN(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    print(f"{a.ckpt}  epoch {ck.get('epoch','?')}  {cfg.curvature_encoding}/"
          f"{cfg.panel_id_mode}  in_dim {cfg.in_dim}  rule {a.rule}  device {dev}")

    rng = random.Random(0)
    rows, shown = [], {}
    for f in find_spec_files(a.dir, limit=a.limit):
        try:
            pat = parse_specification(f)
            s = pattern_to_tensors(pat, cfg, rng)
        except Exception:
            continue
        if s["x"].shape[0] < 2 or not len(s["gt_pairs"]):
            continue
        b = collate([s])
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            lp = log_assignment(m.scores(m(b["x"], b["nbr"], b["mask"])),
                                m.dustbin_z, b["mask"], cfg)
        M = int(b["mask"][0].sum())
        pred = {(min(i, j), max(i, j))
                for i, j in hard_assign_single(_slice_logP(lp[0].cpu(), M), M,
                                               cfg.tau_multi, a.rule)}
        gt = {(int(x), int(y)) for x, y in s["gt_pairs"]}
        ok = len(pred & gt)
        p = ok / max(len(pred), 1); r = ok / max(len(gt), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        name = Path(f).stem.replace("_specification", "")
        rows.append(f1)
        if name in EXAMPLES:
            shown[name] = f1

    v = np.array(rows)
    print(f"\n{len(v)} garments scored")
    print(f"  mean {v.mean():.4f}   median {np.median(v):.4f}")
    for q in (1, 5, 10, 25, 50, 75, 90):
        print(f"  p{q:<3} {np.percentile(v, q):.4f}")
    print(f"\n  exactly 1.000 : {(v >= 0.9999).sum():5d}  ({(v >= 0.9999).mean()*100:5.1f}%)")
    print(f"  above 0.9     : {(v > 0.9).sum():5d}  ({(v > 0.9).mean()*100:5.1f}%)")
    print(f"  below 0.5     : {(v < 0.5).sum():5d}  ({(v < 0.5).mean()*100:5.1f}%)")
    print(f"  exactly 0.000 : {(v <= 1e-9).sum():5d}  ({(v <= 1e-9).mean()*100:5.1f}%)")

    if shown:
        print(f"\n  where the demo's examples sit:")
        for n, f1 in sorted(shown.items(), key=lambda kv: kv[1]):
            print(f"    {n:<20} F1 {f1:.3f}   above {(v < f1).mean()*100:5.1f}% of the split")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.hist(v, bins=np.linspace(0, 1, 41), color="#2b6cb0", edgecolor="white", linewidth=.5)
    for n, f1 in shown.items():
        ax.axvline(f1, color="#c53030", lw=1.2, ls="--", alpha=.8)
        ax.annotate(n.replace("rand_", ""), (f1, ax.get_ylim()[1] * 0.92), rotation=90,
                    fontsize=6.5, color="#c53030", ha="right", va="top")
    ax.set_xlabel("per-garment F1"); ax.set_ylabel("garments")
    ax.set_title(f"Held-out split, {len(v)} garments: {(v >= 0.9999).mean()*100:.0f}% exact, "
                 f"{(v < 0.5).mean()*100:.0f}% below 0.5\n"
                 "dashed: the garments the demo offers as examples", fontsize=9.5)
    ax.grid(alpha=.25)
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.fig, dpi=130, bbox_inches="tight")
    print(f"\nwrote {a.fig}")


if __name__ == "__main__":
    main()
