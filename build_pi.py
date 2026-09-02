"""
pi_1 -> pi_2 dataset builder.

pi_1  (input) : the SORTED 2D panel layout, placed in 3D by the specification's
                per-panel translation/rotation, projected to the front view.
                channels: mask, panel_id, u, v (panel-local 2D), z0 (placed depth)
pi_2  (target): front-view orthographic depth of the DRAPED mesh, same image frame.

Residual target  d = z_drape - z0  is small and smooth; that is the quantity to learn.

Axes (verified): 0=x left-right, 1=y up, 2=z front-back, front = +z, units = cm.
Known simplifications (v1): edge curvature ignored (polygon approximation);
single front view only, so back-facing seams are out of scope by construction.
"""
import numpy as np, json, os, glob, argparse

AX_X, AX_Y, AX_Z = 0, 1, 2

# ---------- geometry helpers ----------
def euler_xyz(r):
    rx, ry, rz = np.deg2rad(r)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

def inside_poly(poly, pts):
    """vectorised crossing test; poly (n,2), pts (m,2) -> (m,) bool"""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]; x1, y1 = poly[(i + 1) % n]
        cond = ((y0 > y) != (y1 > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-30) + x0
        inside ^= cond & (x < xin)
    return inside

def load_ply(gdir, g):
    raw = open(os.path.join(gdir, g + "_sim.ply"), "rb").read()
    end = raw.find(b"end_header\n") + 11
    hdr = raw[:end].decode()
    nv = int([l for l in hdr.split("\n") if l.startswith("element vertex")][0].split()[-1])
    nf = int([l for l in hdr.split("\n") if l.startswith("element face")][0].split()[-1])
    dt = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("s","<f8"),("t","<f8")])
    V = np.frombuffer(raw, dtype=dt, count=nv, offset=end)
    xyz = np.stack([V["x"], V["y"], V["z"]], 1).astype(np.float64)
    F = np.frombuffer(raw, dtype=np.dtype([("n","u1"),("v","<i4",3)]),
                      count=nf, offset=end + nv*dt.itemsize)["v"]
    return xyz, F

# barycentric supersample of every triangle (7 samples/face, vectorised)
BARY = np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5],[1/3,1/3,1/3]])

def front_depth(xyz, F, frame, res):
    x0, x1, y0, y1 = frame
    P = xyz[F]                                   # (nf,3,3)
    S = np.einsum("kb,fbc->fkc", BARY, P).reshape(-1, 3)
    ix = ((S[:, AX_X] - x0) / (x1 - x0) * res).astype(np.int64)
    iy = ((S[:, AX_Y] - y0) / (y1 - y0) * res).astype(np.int64)
    ok = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res)
    ix, iy, z = ix[ok], iy[ok], S[ok, AX_Z]
    buf = np.full(res * res, -np.inf)            # front = MAX z
    np.maximum.at(buf, iy * res + ix, z)
    D = buf.reshape(res, res)
    return D, np.isfinite(D)

def placed_pi1(spec, frame, res):
    x0, x1, y0, y1 = frame
    names = spec["pattern"]["panel_order"]
    pid = np.zeros((res, res), np.int16)
    U = np.zeros((res, res)); V = np.zeros((res, res)); Z0 = np.full((res, res), -np.inf)
    gx = x0 + (np.arange(res) + .5) / res * (x1 - x0)
    gy = y0 + (np.arange(res) + .5) / res * (y1 - y0)
    GX, GY = np.meshgrid(gx, gy)
    for k, nm in enumerate(names, start=1):
        p = spec["pattern"]["panels"][nm]
        poly2 = np.array(p["vertices"], float)
        R = euler_xyz(p["rotation"]); t = np.array(p["translation"], float)
        P3 = (np.c_[poly2, np.zeros(len(poly2))] @ R.T) + t     # panel plane -> world
        # project the polygon to the image plane and test grid points against it
        poly_img = P3[:, [AX_X, AX_Y]]
        pts = np.c_[GX.ravel(), GY.ravel()]
        m = inside_poly(poly_img, pts).reshape(res, res)
        if not m.any():
            continue
        # plane of the panel: interpolate z and panel-local uv by least squares fit
        A = np.c_[poly_img, np.ones(len(poly_img))]
        cz, *_ = np.linalg.lstsq(A, P3[:, AX_Z], rcond=None)
        cu, *_ = np.linalg.lstsq(A, poly2[:, 0], rcond=None)
        cv, *_ = np.linalg.lstsq(A, poly2[:, 1], rcond=None)
        B = np.c_[GX[m], GY[m], np.ones(m.sum())]
        z = B @ cz
        newer = z > Z0[m]                       # keep front-most panel
        idx = np.where(m)
        sel = (idx[0][newer], idx[1][newer])
        Z0[sel] = z[newer]; pid[sel] = k
        U[sel] = (B @ cu)[newer]; V[sel] = (B @ cv)[newer]
    return pid, U, V, Z0, pid > 0

def build(gdir, frame, res):
    g = os.path.basename(gdir)
    spec = json.load(open(os.path.join(gdir, g + "_specification.json")))
    xyz, F = load_ply(gdir, g)
    Dz, Dm = front_depth(xyz, F, frame, res)
    pid, U, V, Z0, Pm = placed_pi1(spec, frame, res)
    both = Dm & Pm
    resid = np.where(both, Dz - np.where(Pm, Z0, 0), 0.0)
    return dict(pid=pid.astype(np.int16), u=U.astype(np.float32), v=V.astype(np.float32),
                z0=np.where(Pm, Z0, 0).astype(np.float32), pi1_mask=Pm,
                depth=np.where(Dm, Dz, 0).astype(np.float32), pi2_mask=Dm,
                resid=resid.astype(np.float32), both=both,
                stitches=json.dumps(spec["pattern"]["stitches"]),
                panel_order=json.dumps(spec["pattern"]["panel_order"]))

def calibrate(root, n=60):
    lo = np.array([np.inf]*3); hi = -lo
    for gd in sorted(glob.glob(os.path.join(root, "rand_*")))[:n]:
        g = os.path.basename(gd)
        xyz, _ = load_ply(gd, g)
        lo = np.minimum(lo, xyz.min(0)); hi = np.maximum(hi, xyz.max(0))
    pad = 0.05 * (hi - lo)
    return (lo[AX_X]-pad[AX_X], hi[AX_X]+pad[AX_X], lo[AX_Y]-pad[AX_Y], hi[AX_Y]+pad[AX_Y]), lo, hi

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/mnt/data"))
    ap.add_argument("--out",  default=os.path.expanduser("~/pi_dataset"))
    ap.add_argument("--res",  type=int, default=256)
    ap.add_argument("--n",    type=int, default=20)
    a = ap.parse_args()
    frame, lo, hi = calibrate(a.root)
    print("world bbox lo", np.round(lo,1), "hi", np.round(hi,1))
    print("image frame  ", tuple(round(f,1) for f in frame), " res", a.res)
    os.makedirs(a.out, exist_ok=True)
    cov1=[]; cov2=[]; rs=[]
    for gd in sorted(glob.glob(os.path.join(a.root, "rand_*")))[:a.n]:
        d = build(gd, frame, a.res)
        np.savez_compressed(os.path.join(a.out, os.path.basename(gd)+".npz"), **d)
        cov1.append(d["pi1_mask"].mean()); cov2.append(d["pi2_mask"].mean())
        r = d["resid"][d["both"]]
        if r.size: rs.append(np.percentile(np.abs(r), [50,90,99]))
    print(f"built {len(cov1)}  pi1 coverage {np.mean(cov1):.3f}  pi2 coverage {np.mean(cov2):.3f}"
          f"  both-overlap {np.mean([np.load(os.path.join(a.out,f))['both'].mean() for f in os.listdir(a.out)]):.3f}")
    r = np.mean(rs, axis=0)
    print(f"residual |z_drape - z0|  p50={r[0]:.2f}cm  p90={r[1]:.2f}cm  p99={r[2]:.2f}cm")
