"""Boundary error models.

white     point-wise independent N(0, s^2 I)            — worst case shape
ema       the SAME draw, low-passed along the contour   — an AI-like output:
          wrong, but the contour still follows the GT shape

The EMA is applied to the DISPLACEMENT, never to the coordinates, so the filter
cannot shrink the curve itself (at s = 0 the result is exactly the GT).
`restore_rms` scales the smoothed displacement back to the white magnitude, so
the two models differ in SHAPE only, not in size.
"""
import numpy as np


def white(rng, n, sigma_cm):
    return rng.normal(0.0, sigma_cm, (n, 3)) if sigma_cm > 0 else np.zeros((n, 3))


def ema_closed(X, alpha):
    """Zero-phase circular EMA on an (n,3) array.

    Forward then backward (removes phase lag); two laps each so the loop closes.
    alpha = 1 is a no-op; smaller alpha = smoother.
    """
    n = len(X)
    if n < 4 or alpha >= 1.0:
        return X.copy()
    f = np.empty_like(X); s = X[0].copy()
    for lap in range(2):
        for i in range(n):
            s = alpha * X[i] + (1 - alpha) * s
            if lap:
                f[i] = s
    out = np.empty_like(X); t = f[-1].copy()
    for lap in range(2):
        for i in range(n - 1, -1, -1):
            t = alpha * f[i] + (1 - alpha) * t
            if lap:
                out[i] = t
    return out


def restore_rms(d_smooth, d_white):
    """Scale so the mean per-point displacement magnitude matches the white draw."""
    m1 = np.linalg.norm(d_smooth, axis=1).mean()
    if m1 < 1e-12:
        return d_smooth
    return d_smooth * (np.linalg.norm(d_white, axis=1).mean() / m1)


def displace(rng, n, sigma_cm, alpha=None, keep_rms=True):
    """-> displacement (n,3).  alpha=None -> white."""
    d = white(rng, n, sigma_cm)
    if alpha is None or sigma_cm <= 0:
        return d
    s = ema_closed(d, alpha)
    return restore_rms(s, d) if keep_rms else s


def mean_pair_separation_mm(disp_list):
    """Two independent copies separate by ~sqrt(2) x the single-copy displacement."""
    tot = sum(np.linalg.norm(d, axis=1).sum() for d in disp_list)
    cnt = sum(len(d) for d in disp_list)
    return (tot / cnt) * np.sqrt(2) * 10 if cnt else 0.0


# ---------------------------------------------------------------- mesh graph
def build_adjacency(n_verts, faces):
    """CSR neighbour lists from triangles. -> (nbr_idx, nbr_ptr)"""
    pairs = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    pairs = np.vstack([pairs, pairs[:, ::-1]])
    order = np.argsort(pairs[:, 0], kind="stable")
    p = pairs[order]
    counts = np.bincount(p[:, 0], minlength=n_verts)
    return p[:, 1].astype(np.int64), np.r_[0, np.cumsum(counts)].astype(np.int64)


def smooth_on_graph(d, nbr_idx, nbr_ptr, w=0.5, iters=0, keep_rms=True):
    """Low-pass a displacement field over a mesh graph.

    d <- (1-w) d + w * mean(neighbours), repeated `iters` times.  This is the
    surface analogue of the contour EMA.  With keep_rms the magnitude is scaled
    back to the input, so the result differs in SHAPE only.
    """
    if iters <= 0 or w <= 0:
        return d
    m0 = np.linalg.norm(d, axis=1).mean()
    x = d.copy()
    cnt = np.maximum(np.diff(nbr_ptr), 1)[:, None]
    for _ in range(int(iters)):
        acc = np.add.reduceat(x[nbr_idx], nbr_ptr[:-1], axis=0)
        x = (1 - w) * x + w * (acc / cnt)
    if not keep_rms:
        return x
    m1 = np.linalg.norm(x, axis=1).mean()
    return x * (m0 / m1) if m1 > 1e-12 else x
