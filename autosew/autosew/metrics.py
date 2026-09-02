"""Hard assignment (§5.2) + the 7 paper metrics (§8.1).

TP/TR/TF1  : micro precision/recall/F1 over unordered predicted vs GT pairs
MEP/MER/MEF1: same, restricted to pairs incident to a GT multi-edge node
              (node with GT stitch degree >= 2). [GAP] exact subset definition
              is not in the paper; conditioning on GT nodes keeps denominators stable.
GSP        : fraction of patterns fully resolved. strict = pred set == GT set;
             recall-only variant (paper text literal) also reported as gsp_recall_only.
"""
import numpy as np
import torch


def hard_assign_single(logP, Mb, tau, mode="union"):
    """logP: (M+1,M+1) tensor for ONE pattern (rows/cols beyond Mb are padding).
    Returns set of unordered predicted pairs (i<j), indices < Mb.
    Paper §5.2: P' = 1/2 (P + P^T); per row keep argmax always, plus every other
    entry with p >= tau. Argmax = dustbin -> edge predicted unstitched."""
    P = torch.exp(logP[: Mb + 1, : Mb + 1].detach().float())
    # move the (M+1)-th (bin) row/col adjacency: here logP is (M_pad+1)^2; caller slices
    Ps = 0.5 * (P + P.T)
    pred_dir = [set() for _ in range(Mb)]
    for i in range(Mb):
        row = Ps[i].clone()
        row[i] = -1.0                      # self excluded
        j_star = int(torch.argmax(row).item())
        if j_star == Mb:                   # dustbin wins -> unstitched
            continue
        pred_dir[i].add(j_star)
        for j in range(Mb):
            if j != i and j != j_star and Ps[i, j].item() >= tau:
                pred_dir[i].add(j)
    pairs = set()
    if mode == "union":
        for i in range(Mb):
            for j in pred_dir[i]:
                pairs.add((min(i, j), max(i, j)))
    elif mode == "mutual":
        for i in range(Mb):
            for j in pred_dir[i]:
                if i in pred_dir[j]:
                    pairs.add((min(i, j), max(i, j)))
    else:
        raise ValueError(mode)
    return pairs


def _slice_logP(logP_padded, Mb):
    """(M_pad+1, M_pad+1) -> (Mb+1, Mb+1) with the bin moved to index Mb."""
    Mp = logP_padded.shape[-1] - 1
    idx = list(range(Mb)) + [Mp]
    idx = torch.tensor(idx, device=logP_padded.device)
    return logP_padded.index_select(0, idx).index_select(1, idx)


class MetricAccumulator:
    def __init__(self):
        self.tp = 0; self.npred = 0; self.ngt = 0
        self.me_tp = 0; self.me_npred = 0; self.me_ngt = 0
        self.pat_total = 0; self.pat_strict = 0; self.pat_recall_only = 0

    def add_pattern(self, pred_pairs, gt_pairs):
        gt = set(map(tuple, gt_pairs))
        pred = set(pred_pairs)
        inter = pred & gt
        self.tp += len(inter); self.npred += len(pred); self.ngt += len(gt)

        deg = {}
        for a, b in gt:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        me_nodes = {k for k, d in deg.items() if d >= 2}
        gt_me = {p for p in gt if p[0] in me_nodes or p[1] in me_nodes}
        pred_me = {p for p in pred if p[0] in me_nodes or p[1] in me_nodes}
        self.me_tp += len(pred_me & gt_me)
        self.me_npred += len(pred_me); self.me_ngt += len(gt_me)

        self.pat_total += 1
        if pred == gt:
            self.pat_strict += 1
        if gt <= pred:
            self.pat_recall_only += 1

    @staticmethod
    def _prf(tp, npred, ngt):
        p = tp / npred if npred else 0.0
        r = tp / ngt if ngt else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    def result(self):
        tp_, tr_, tf1 = self._prf(self.tp, self.npred, self.ngt)
        mep, mer, mef1 = self._prf(self.me_tp, self.me_npred, self.me_ngt)
        return {
            "TP": tp_, "TR": tr_, "TF1": tf1,
            "MEP": mep, "MER": mer, "MEF1": mef1,
            "GSP": self.pat_strict / self.pat_total if self.pat_total else 0.0,
            "GSP_recall_only": self.pat_recall_only / self.pat_total if self.pat_total else 0.0,
            "n_patterns": self.pat_total,
            "n_gt_pairs": self.ngt, "n_pred_pairs": self.npred,
            "has_multi_edge_gt": self.me_ngt > 0,
        }


@torch.no_grad()
def evaluate_batch(logP, batch, cfg, acc: MetricAccumulator):
    B = logP.shape[0]
    for b in range(B):
        Mb = int(batch["mask"][b].sum())
        lp = _slice_logP(logP[b], Mb)
        pred = hard_assign_single(lp, Mb, cfg.tau_multi, cfg.hard_mode)
        acc.add_pattern(pred, [tuple(p) for p in batch["gt_pairs"][b]])
