"""Dump everything the C++ solver needs from gcd_io.load() into a flat binary.

gcd_io.py itself is host-side and is NOT ported: it reads a _sim.ply, a pickle
and a yaml.  This script runs the whole Python front end (gcd_io + run_garment
.build + run_garment.initial + body.primitives/pack) and writes the pieces the
numerical core consumes:

    faces, wid, panel_of_face, rest (flat pattern), P0 (initial placement),
    seam pairs, the penalty diagonal mu, the optional anchor diagonal nu with
    its target, and the packed body primitives (cylinders + spheres).

Everything after that point in run_garment.main is what simcpp reimplements.

  python dump_garment.py --garment <dir> --out <file.bin> [--mu 0.02] [--body]
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rigid-clothes-simulation"))

import body                     # noqa: E402
import gcd_io                   # noqa: E402
import run_garment as RG        # noqa: E402

MAGIC = b"SIMCPP01"


def w_i32(f, a):
    f.write(np.ascontiguousarray(a, np.int32).tobytes())


def w_f64(f, a):
    f.write(np.ascontiguousarray(a, np.float64).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garment", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--body", action="store_true")
    ap.add_argument("--half-lr", action="store_true")
    ap.add_argument("--half-fb", action="store_true")
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--amp", type=float, default=0.0)
    ap.add_argument("--inflate", type=float, default=1.0)
    ap.add_argument("--sym", action="store_true")
    a = ap.parse_args()

    gdir = a.garment
    d, gar = RG.build(gdir)
    n = gar["n"]
    P0 = RG.initial(d, a.seed, a.inflate, amp=a.amp, sym=a.sym)

    cyl = np.zeros((0, 7))
    sph = np.zeros((0, 4))
    if a.body:
        prim = body.primitives(
            body.load_measurements(RG.measures_of(gdir)), d["placed"],
            np.array(d["panel_names"], dtype=object)[np.maximum(d["panel_of_raw"], 0)],
            d["rest"])
        C, S = prim
        cyl = np.array([np.concatenate([c[0], c[1], [c[2]]]) for c in C]).reshape(-1, 7)
        sph = np.array([np.concatenate([s[0], [s[1]]]) for s in S]).reshape(-1, 4)
        mu = np.full(n, a.mu * RG.diag_scale(gar))
        recenter = 0
    else:
        mu, _ = RG.half_space_penalty(d, gar, lr=a.half_lr, fb=a.half_fb, mu_rel=a.mu)
        if mu is None:
            mu = np.zeros(0)
        recenter = 1
    nu = np.zeros(0)
    anchor = np.zeros((0, 3))
    if a.anchor:
        nu = np.full(n, a.anchor * RG.diag_scale(gar))
        anchor = d["placed"]
        recenter = 0

    with open(a.out, "wb") as f:
        f.write(MAGIC)
        # n, n_faces, n_pairs, n_cyl, n_sph, has_mu, has_nu, recenter
        f.write(struct.pack("<8i", n, len(gar["faces"]), len(gar["pairs"]),
                            len(cyl), len(sph), int(len(mu) > 0), int(len(nu) > 0),
                            recenter))
        w_i32(f, gar["faces"])          # (M,3)
        w_i32(f, d["wid"])              # (n,)
        w_i32(f, d["panel_of_face"])    # (M,)
        w_f64(f, d["rest"])             # (n,2)  flat pattern, cm
        w_f64(f, P0)                    # (n,3)
        w_i32(f, gar["pairs"])          # (K,2)
        if len(mu):
            w_f64(f, mu)
        if len(nu):
            w_f64(f, nu)
            w_f64(f, anchor)
        w_f64(f, cyl)                   # (NC,7)  p0 p1 r
        w_f64(f, sph)                   # (NS,4)  c r

    meta = dict(garment=os.path.basename(gdir.rstrip("/\\")), n=int(n),
                n_faces=int(len(gar["faces"])), n_hinges=int(len(gar["hinges"])),
                n_pairs=int(len(gar["pairs"])), n_cyl=int(len(cyl)),
                n_sph=int(len(sph)), mu_rel=a.mu, recenter=recenter,
                diag_scale=float(RG.diag_scale(gar)))
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
