"""Does the browser build the same features the model was trained on?

The web demo re-implements features.py in TypeScript.  A divergence there does not
raise -- it just feeds the model slightly wrong numbers and quietly costs accuracy --
so it has to be checked numerically, on both input paths (a specification and a DXF).

    node scripts/feature_parity.mjs ts.json          # in webdemo/
    & $PY dxfcheck/feature_parity.py --ts ts.json

The panel-id dimension is excluded: the shipped config shuffles panel ids per sample
(panel_id_mode="random_norm"), so the two sides cannot agree on it by construction --
which is the point of that setting, and is checked separately by the perturbation runs.
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
from autosew.features import feature_dim, pattern_to_tensors, N_CURV
from autosew.gcd_parser import parse_specification
from dxf_to_features import build

SPEC = R / "data" / "rand_00YONAPXZE" / "rand_00YONAPXZE_specification.json"
DXF = R / "clo_example" / "panel_seperated.dxf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True, help="JSON written by scripts/feature_parity.mjs")
    ap.add_argument("--ckpt", default="autosew/runs/r3/frozen_ep5_tf1_8040.pt")
    a = ap.parse_args()

    ck = torch.load(R / a.ckpt, map_location="cpu", weights_only=False)
    cfg = AutoSewConfig(**{k: (tuple(v) if k == "edge_count_minmax" else v)
                           for k, v in ck["cfg"].items()})
    D = feature_dim(cfg)
    nk = cfg.sagitta_samples if cfg.curvature_encoding == "sagitta" else N_CURV
    pid = 7 + nk + 5                      # the panel-id dim, deliberately not compared

    ts = json.loads(Path(a.ts).read_text())
    print(f"checkpoint {a.ckpt}  {cfg.curvature_encoding}/{cfg.panel_id_mode}"
          f"  arc={cfg.arc_features}  dim {D}")
    print(f"browser dump  encoding={ts['encoding']}  dim {ts['featureDim']}")
    if ts["featureDim"] != D:
        print(f"  ! WIDTH MISMATCH: browser emits {ts['featureDim']}, the model wants {D}")
        return 1

    bad = 0
    for name, case in ts["cases"].items():
        src = DXF if name == "clo" else SPEC
        if name == "clo":
            X, keys = build(str(src), cfg)
        else:
            pat = parse_specification(str(src))
            import random
            s = pattern_to_tensors(pat, cfg, random.Random(0))
            X, keys = s["x"], [(p, e) for p, e in pat.edge_key_list()]
        T = np.array(case["x"], np.float64).reshape(case["M"], D)
        if X.shape != T.shape:
            print(f"  {name:<6} SHAPE {X.shape} vs {T.shape}")
            bad += 1
            continue
        keep = [i for i in range(D) if i != pid]
        d = np.abs(X[:, keep].astype(np.float64) - T[:, keep])
        worst = int(d.max(0).argmax())
        print(f"  {name:<6} M={case['M']:<4} max|diff| {d.max():.3e}"
              f"  (worst dim {keep[worst]})"
              f"   keys {'match' if [list(k) for k in keys] == [list(k) for k in case['keys']] else 'DIFFER'}")
        if d.max() > 1e-6:
            bad += 1
    print("\nPARITY OK" if not bad else f"\n{bad} case(s) disagree")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
