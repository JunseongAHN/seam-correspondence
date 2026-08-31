"""noised boundary -> front depth map -> lift -> nearest-neighbour seam decode."""
import numpy as np

AX_X, AX_Y, AX_Z = 0, 1, 2


def project_lift(pts, res, frame):
    """Orthographic front view (front = max z), z-buffer, then lift every point.

    Returns (lifted[n,3], occluded[n], pixel_mm).
    A point that is not the front-most in its pixel reads the occluder's depth —
    that loss is part of what the representation costs.
    """
    x0, x1, y0, y1 = frame
    step = max(x1 - x0, y1 - y0) / res
    ix = np.clip(np.floor((pts[:, AX_X] - x0) / step).astype(np.int64), 0, res)
    iy = np.clip(np.floor((pts[:, AX_Y] - y0) / step).astype(np.int64), 0, res)
    pid = ix * (res + 1) + iy
    z = pts[:, AX_Z]
    o = np.lexsort((-z, pid)); ps, zs = pid[o], z[o]
    first = np.r_[True, ps[1:] != ps[:-1]]
    buf = dict(zip(ps[first], zs[first]))
    front = np.array([buf[p] for p in pid])
    lifted = np.stack([x0 + (ix + .5) * step, y0 + (iy + .5) * step, front], 1)
    return lifted, front > z + 1e-9, step * 10


def decode(lifted, share_mask):
    """1-nearest-neighbour among the lifted copies (self excluded).

    share_mask[i, j] is True when copies i and j belong to a common stitch.
    -> (correct[n], nn_index[n], mutual_rate)
    """
    n = len(lifted)
    Dm = np.linalg.norm(lifted[:, None, :] - lifted[None, :, :], axis=2)
    Dm[np.eye(n, dtype=bool)] = np.inf
    nn = Dm.argmin(1)
    good = share_mask[np.arange(n), nn]
    mutual = float((nn[nn] == np.arange(n)).mean())
    return good, nn, mutual


def score(good, member):
    """member[n_copies, n_seams] boolean membership.

    -> dict(vertex, seam_all, seam_vote, garment)
    """
    big = [k for k in range(member.shape[1]) if member[:, k].sum() > 1]
    votes = [bool(good[member[:, k]].mean() > 0.5) for k in big]
    return dict(vertex=float(good.mean()),
                seam_all=float(np.mean([good[member[:, k]].all() for k in big])),
                seam_vote=float(np.mean(votes)),
                garment=float(all(votes)),
                n_seams=len(big))


def share_matrix(stitch_sets_per_copy):
    """-> (share[n,n] bool, member[n, n_seams] bool, seam_ids)"""
    ids = sorted({s for fs in stitch_sets_per_copy for s in fs},
                 key=lambda s: int(s.split("_")[1]))
    n = len(stitch_sets_per_copy)
    member = np.zeros((n, len(ids)), bool)
    for k, sd in enumerate(ids):
        member[:, k] = [sd in fs for fs in stitch_sets_per_copy]
    share = (member.astype(np.uint8) @ member.astype(np.uint8).T) > 0
    np.fill_diagonal(share, False)
    return share, member, ids
