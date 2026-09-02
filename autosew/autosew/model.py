"""GraphSAGE encoder (pure PyTorch, no PyG).

h^{l+1}_e = sigma( W^l · CONCAT(h^l_e, AGG({h^l_u : u in N(e)})) )   (paper eq. 1)
AGG = mean (default) or max. Neighbors come as an explicit (B,M,2) index tensor
(cycle graph: prev/next edge in the same panel), padded nodes point at themselves.
"""
import torch
import torch.nn as nn


class SageLayer(nn.Module):
    def __init__(self, din, dout, aggregator="mean"):
        super().__init__()
        self.lin = nn.Linear(2 * din, dout)
        self.aggregator = aggregator

    def forward(self, h, nbr, mask):
        # h: (B,M,D)  nbr: (B,M,K) int64  mask: (B,M) bool
        B, M, D = h.shape
        K = nbr.shape[-1]
        idx = nbr.reshape(B, M * K, 1).expand(-1, -1, D)
        neigh = torch.gather(h, 1, idx).reshape(B, M, K, D)
        if self.aggregator == "mean":
            agg = neigh.mean(dim=2)
        elif self.aggregator == "max":
            agg = neigh.max(dim=2).values
        else:
            raise ValueError(self.aggregator)
        out = self.lin(torch.cat([h, agg], dim=-1))
        return out * mask.unsqueeze(-1)


class AutoSewGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.layer_scheme == "last128":
            dims = [cfg.in_dim] + [cfg.hidden_dim] * (cfg.num_layers - 1) + [cfg.out_dim]
            self.proj = None
        elif cfg.layer_scheme == "proj":
            dims = [cfg.in_dim] + [cfg.hidden_dim] * cfg.num_layers
            self.proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)
        else:
            raise ValueError(cfg.layer_scheme)
        self.layers = nn.ModuleList(
            [SageLayer(dims[i], dims[i + 1], cfg.aggregator) for i in range(len(dims) - 1)]
        )
        self.act = nn.ReLU()
        # learnable dustbin score z (paper eq. 4)
        self.dustbin_z = nn.Parameter(torch.tensor(float(cfg.dustbin_init)))

    def forward(self, x, nbr, mask):
        # x: (B,M,24) -> f: (B,M,D)
        h = x
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            h = layer(h, nbr, mask)
            last = (i == n - 1) and self.proj is None
            if not last or self.cfg.final_activation == "relu":
                h = self.act(h)
            h = h * mask.unsqueeze(-1)
        if self.proj is not None:
            h = self.proj(h)
            if self.cfg.final_activation == "relu":
                h = self.act(h)
            h = h * mask.unsqueeze(-1)
        if self.cfg.l2_normalize:
            h = torch.nn.functional.normalize(h, dim=-1)
        return h

    def scores(self, f):
        """C = <f_i, f_j> (B,M,M), eq. 3."""
        C = torch.bmm(f, f.transpose(1, 2))
        if self.cfg.score_scale == "rsqrt_d":
            C = C / (f.shape[-1] ** 0.5)
        return C
