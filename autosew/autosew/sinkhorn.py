"""Differentiable OT: extended score matrix (dustbin), log-space Sinkhorn, NLL loss.

Single-set self-matching: C is (M x M) symmetric, diagonal (self-match) masked out.
Extended C-bar is (M+1 x M+1); last row/col = dustbin with one learnable scalar z
(paper eq. 4). Marginals (SuperGlue convention, cfg.marginal_mode="superglue"):
real edge capacity 1, dustbin capacity M, total mass 2M, normalized; after the
iterations we add log(2M) back so each REAL row of exp(logP) sums to ~1 -> entries
are probabilities and tau_multi (§5.2) applies directly.

Padding: padded nodes get score -1e9 everywhere and marginal exp(-1e9)~0; verified
against per-sample unpadded computation in tests/test_all.py.
"""
import torch


def extend_scores(C, z, mask, neg_inf=-1e9):
    """C: (B,M,M) raw inner products; z: scalar tensor; mask: (B,M) bool.
    Returns Cbar: (B,M+1,M+1)."""
    B, M, _ = C.shape
    valid = mask.unsqueeze(1) & mask.unsqueeze(2)              # (B,M,M)
    eye = torch.eye(M, dtype=torch.bool, device=C.device).unsqueeze(0)
    C = torch.where(valid & ~eye, C, torch.full_like(C, neg_inf))
    Cbar = torch.full((B, M + 1, M + 1), neg_inf, dtype=C.dtype, device=C.device)
    Cbar[:, :M, :M] = C
    zc = z.to(C.dtype)
    binvec = torch.where(mask, zc.expand(B, M),
                         torch.full((B, M), neg_inf, device=C.device, dtype=C.dtype))
    Cbar[:, :M, M] = binvec
    Cbar[:, M, :M] = binvec
    Cbar[:, M, M] = zc
    return Cbar


def log_marginals(mask, neg_inf=-1e9):
    """(B,M) bool -> log_mu: (B,M+1). mu_i = 1/(2M_b) for real nodes, 1/2 for the bin."""
    B, M = mask.shape
    Mb = mask.sum(dim=1).clamp(min=1).to(torch.float32)        # (B,)
    log2Mb = torch.log(2.0 * Mb)
    mu = torch.full((B, M + 1), neg_inf, device=mask.device, dtype=torch.float32)
    mu[:, :M] = torch.where(mask, (-log2Mb).unsqueeze(1).expand(B, M),
                            torch.full((B, M), neg_inf, device=mask.device))
    mu[:, M] = torch.log(Mb) - log2Mb                          # = -log 2
    return mu, log2Mb


def sinkhorn_log(Cbar, log_mu, iters=100, symmetric_updates=False):
    """Returns u, v (B,M+1). Marginals identical on both sides (self-matching)."""
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_mu)
    if symmetric_updates:
        for _ in range(iters):
            lse = torch.logsumexp(Cbar + u.unsqueeze(1), dim=2)   # over columns j of row i? see note
            u = 0.5 * (u + log_mu - lse)
        return u, u
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Cbar + v.unsqueeze(1), dim=2)
        v = log_mu - torch.logsumexp(Cbar + u.unsqueeze(2), dim=1)
    return u, v


def log_assignment(C, z, mask, cfg):
    """Full pipeline: scores -> extended -> Sinkhorn -> logP (B,M+1,M+1),
    where exp(logP) rows (real) sum to ~1."""
    Cbar = extend_scores(C, z, mask, cfg.neg_inf)
    log_mu, log2Mb = log_marginals(mask, cfg.neg_inf)
    u, v = sinkhorn_log(Cbar, log_mu, cfg.sinkhorn_iters, cfg.symmetric_updates)
    logP = Cbar + u.unsqueeze(2) + v.unsqueeze(1) + log2Mb.view(-1, 1, 1)
    return logP


def build_supervision(batch, cfg):
    """-> LongTensor (E,3) of (b, row, col) entries whose logP should be maximized.
    GT matched pairs (both directions if cfg.loss_both_directions) plus
    unstitched real node -> dustbin (and reverse)."""
    entries = []
    M = batch["x"].shape[1]
    for b in range(batch["x"].shape[0]):
        gt = batch["gt_pairs"][b]
        stitched = batch["stitched"][b]
        Mb = int(batch["mask"][b].sum())
        for (i, j) in gt:
            entries.append((b, i, j))
            if cfg.loss_both_directions:
                entries.append((b, j, i))
        for i in range(Mb):
            if not stitched[i]:
                entries.append((b, i, M))
                if cfg.loss_both_directions:
                    entries.append((b, M, i))
    if not entries:
        return None
    return torch.tensor(entries, dtype=torch.long)


def nll_loss(logP, sup):
    """sup: (E,3) (b,row,col). Paper eq. 5: Loss = -sum log P̄_ij (we take the mean)."""
    vals = logP[sup[:, 0], sup[:, 1], sup[:, 2]]
    return -vals.mean()
