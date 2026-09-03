"""Export a trained checkpoint to ONNX for the browser demo.

Two things in sinkhorn.py do not survive the export as written, so they are patched at
export time rather than changed in the repo source (the training path is the reference
implementation and stays untouched):

  * `torch.eye(...)` traces to `EyeLike`, for which onnxruntime-web has no kernel.
    An arange comparison produces the identical boolean matrix and exports cleanly.
  * `Cbar[:, :M, :M] = C` and the three dustbin slice writes are in-place assignments
    into a freshly allocated tensor. Traced, they become ScatterND chains; built with
    `torch.cat` instead they are plain concatenations.

Both are verified numerically against the unpatched module before the export runs, so a
patch that silently changed the maths would fail here rather than in the browser.

    & $PY scripts/export_onnx.py --ckpt runs/r2/best.pt --out ../webdemo/public/model/autosew.onnx
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autosew import sinkhorn as S
from autosew.config import AutoSewConfig
from autosew.model import AutoSewGNN


def extend_scores_exportable(C, z, mask, neg_inf=-1e9):
    """Same result as sinkhorn.extend_scores, without EyeLike or in-place slice writes."""
    B, M, _ = C.shape
    valid = mask.unsqueeze(1) & mask.unsqueeze(2)
    idx = torch.arange(M, device=C.device)
    eye = (idx.unsqueeze(0) == idx.unsqueeze(1)).unsqueeze(0)      # no EyeLike
    C = torch.where(valid & ~eye, C, torch.full_like(C, neg_inf))

    zc = z.to(C.dtype)
    binvec = torch.where(mask, zc.expand(B, M),
                         torch.full((B, M), neg_inf, device=C.device, dtype=C.dtype))
    zrow = zc.expand(B, 1)
    top = torch.cat([C, binvec.unsqueeze(2)], dim=2)               # (B, M, M+1)
    bot = torch.cat([binvec, zrow], dim=1).unsqueeze(1)            # (B, 1, M+1)
    return torch.cat([top, bot], dim=1)


class ExportWrapper(torch.nn.Module):
    """x, nbr, mask -> logP, so the browser gets one graph and no Python loop."""

    def __init__(self, model, cfg):
        super().__init__()
        self.m = model
        self.cfg = cfg

    def forward(self, x, nbr, mask):
        C = self.m.scores(self.m(x, nbr, mask))
        Cbar = extend_scores_exportable(C, self.m.dustbin_z, mask, self.cfg.neg_inf)
        log_mu, log2Mb = S.log_marginals(mask, self.cfg.neg_inf)
        u, v = S.sinkhorn_log(Cbar, log_mu, self.cfg.sinkhorn_iters,
                              self.cfg.symmetric_updates)
        return Cbar + u.unsqueeze(2) + v.unsqueeze(1) + log2Mb.view(-1, 1, 1)


def dummy(M, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, M, D, generator=g)
    nbr = torch.stack([(torch.arange(M) - 1) % M, (torch.arange(M) + 1) % M], 1).unsqueeze(0)
    mask = torch.ones(1, M, dtype=torch.bool)
    return x, nbr, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--data_dir", default=r"C:\Users\POMCHECKER\gcd_data\test",
                    help="garments to verify the export against")
    ap.add_argument("--n_check", type=int, default=40)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = AutoSewConfig(**{k: (tuple(v) if k == "edge_count_minmax" else v)
                           for k, v in ck["cfg"].items()})
    model = AutoSewGNN(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint epoch {ck.get('epoch', '?')}   in_dim {cfg.in_dim}   "
          f"encoding {cfg.curvature_encoding}   panel ids {cfg.panel_id_mode}")

    # --- the patch must be a no-op numerically, on several shapes
    with torch.no_grad():
        for M in (8, 37, 116):
            x, nbr, mask = dummy(M, cfg.in_dim, seed=M)
            C = model.scores(model(x, nbr, mask))
            ref = S.extend_scores(C, model.dustbin_z, mask, cfg.neg_inf)
            new = extend_scores_exportable(C, model.dustbin_z, mask, cfg.neg_inf)
            d = (ref - new).abs().max().item()
            assert d == 0.0, f"extend_scores patch changed the result at M={M}: {d}"
        print("extend_scores patch is bit-identical to the reference at M = 8, 37, 116")

    wrapper = ExportWrapper(model, cfg).eval()
    x, nbr, mask = dummy(37, cfg.in_dim)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (x, nbr, mask), str(out),
        input_names=["x", "nbr", "mask"], output_names=["logP"],
        dynamic_axes={"x": {0: "B", 1: "M"}, "nbr": {0: "B", 1: "M"},
                      "mask": {0: "B", 1: "M"}, "logP": {0: "B", 1: "M1", 2: "M1"}},
        opset_version=a.opset, do_constant_folding=True, dynamo=False,
    )
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")

    import onnx
    g = onnx.load(str(out))
    ops = {}
    for n in g.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"nodes {len(g.graph.node)}   EyeLike {ops.get('EyeLike', 0)}   "
          f"ScatterND {ops.get('ScatterND', 0)}")
    assert ops.get("EyeLike", 0) == 0, "EyeLike survived: onnxruntime-web has no kernel"

    # --- ONNX vs PyTorch on real garments.
    #
    # The right criterion is the PREDICTION, not the raw logP.  100 Sinkhorn iterations
    # of logsumexp accumulate float32 differences that grow with M -- on random input
    # that reaches ~1e-2 by M=196 without any decision changing.  What has to match is
    # the set of stitch pairs the demo will draw.
    import onnxruntime as ort
    from autosew.dataset import PatternDataset
    from autosew.metrics import hard_assign_single, _slice_logP

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ds = PatternDataset.from_dir(a.data_dir, cfg, limit=a.n_check, verbose=False)
    print(f"\nchecking {len(ds)} real garments from {a.data_dir}")
    agree = 0
    worst_logp = 0.0
    diffs = []
    for s in ds.samples:
        M = s["x"].shape[0]
        x = torch.from_numpy(s["x"]).unsqueeze(0)
        nbr = torch.from_numpy(s["nbr"]).unsqueeze(0)
        mask = torch.ones(1, M, dtype=torch.bool)
        with torch.no_grad():
            ref = wrapper(x, nbr, mask).numpy()
        got = sess.run(["logP"], {"x": x.numpy(), "nbr": nbr.numpy().astype(np.int64),
                                  "mask": mask.numpy()})[0]
        finite = np.isfinite(ref) & (ref > -1e8)
        if finite.any():
            worst_logp = max(worst_logp, float(np.abs(ref[finite] - got[finite]).max()))
        pr = hard_assign_single(_slice_logP(torch.from_numpy(ref)[0], M), M,
                                cfg.tau_multi, cfg.hard_mode)
        po = hard_assign_single(_slice_logP(torch.from_numpy(got)[0], M), M,
                                cfg.tau_multi, cfg.hard_mode)
        pr = {(min(i, j), max(i, j)) for i, j in pr}
        po = {(min(i, j), max(i, j)) for i, j in po}
        if pr == po:
            agree += 1
        else:
            diffs.append((s["name"], M, len(pr ^ po), len(pr)))
    print(f"  identical prediction sets: {agree}/{len(ds.samples)}")
    print(f"  max|onnx - torch| over unmasked logP entries: {worst_logp:.3e}")
    for nm, M, d, n in diffs[:5]:
        print(f"    differs: {nm}  M={M}  {d} pair(s) of {n}")
    assert agree == len(ds.samples), "ONNX and PyTorch predict different stitches"


if __name__ == "__main__":
    main()
