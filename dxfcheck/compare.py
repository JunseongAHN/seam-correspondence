"""How much does going through DXF cost?  Features first, then the prediction."""
import sys, argparse
from pathlib import Path
import numpy as np, torch
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "autosew"))
sys.path.insert(0, str(R / "dxfcheck"))
from autosew.config import AutoSewConfig
from autosew.gcd_parser import parse_specification
from autosew.features import pattern_to_tensors
from autosew.dataset import collate
from autosew.model import AutoSewGNN
from autosew.sinkhorn import log_assignment
from autosew.metrics import hard_assign_single, _slice_logP
from dxf_to_features import build

ap = argparse.ArgumentParser(); ap.add_argument("--spec"); ap.add_argument("--dxf")
ap.add_argument("--ckpt", default=str(R / "autosew/runs/full/best.pt")); a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = AutoSewConfig(**{k: (tuple(v) if k == "edge_count_minmax" else v) for k, v in ck["cfg"].items()})
m = AutoSewGNN(cfg); m.load_state_dict(ck["model"]); m.eval()

pat = parse_specification(a.spec)
s = pattern_to_tensors(pat, cfg)
xs, nbr = s["x"], s["nbr"]
xd, keys = build(a.dxf, cfg)

print(f"M: spec {xs.shape[0]}  dxf {xd.shape[0]}")
if xs.shape != xd.shape:
    print("shape mismatch -- cannot compare elementwise"); sys.exit(1)

NAMES = ["x0","y0","x1","y1","len","ox","oy","k_t"] + [f"k{i}" for i in range(1,11)] + \
        ["sinL","cosL","sinR","cosR","n_edges","panel_u"]
d = np.abs(xs - xd)
print(f"\noverall max|diff| {d.max():.3e}   mean {d.mean():.3e}")
print(f"{'feature':10}{'max|diff|':>12}{'mean':>12}   note")
for f in range(24):
    note = ""
    if f == 7:
        agree = int((xs[:,7] == xd[:,7]).sum())
        note = f"curve type recovered {agree}/{len(xs)}"
    print(f"{NAMES[f]:10}{d[:,f].max():12.3e}{d[:,f].mean():12.3e}   {note}")

def predict(x):
    b = collate([{"x": x.astype(np.float32), "nbr": nbr,
                  "gt_pairs": s["gt_pairs"], "stitched": s["stitched"], "name": "g"}])
    with torch.no_grad():
        lp = log_assignment(m.scores(m(b["x"], b["nbr"], b["mask"])), m.dustbin_z, b["mask"], cfg)
    Mb = int(b["mask"][0].sum())
    return {f"{min(i,j)}-{max(i,j)}" for i,j in
            hard_assign_single(_slice_logP(lp[0], Mb), Mb, cfg.tau_multi, cfg.hard_mode)}

gt = {f"{i}-{j}" for i,j in pat.gt_pairs()}
ps, pd_ = predict(xs), predict(xd)
def sc(p):
    ok = len(p & gt); return ok, len(p)-ok, len(gt)-ok, (2*ok/max(len(p)+len(gt),1))
print(f"\n{'source':8}{'ok':>5}{'FP':>5}{'FN':>5}{'F1':>8}")
for nm, p in (("spec", ps), ("dxf", pd_)):
    ok, fp, fn, f1 = sc(p); print(f"{nm:8}{ok:5}{fp:5}{fn:5}{f1:8.4f}")
print(f"predicted pair sets identical: {ps == pd_}")
if ps != pd_:
    print(f"  only from spec: {sorted(ps-pd_)[:8]}")
    print(f"  only from dxf : {sorted(pd_-ps)[:8]}")
