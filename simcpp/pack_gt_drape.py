"""Pack a solved garment into the compact form the web demo renders.

The demo shows the ground-truth drape beside a predicted stitching, purely as a
reference shape -- it is never solved in the browser, so nothing the solver needs
to RUN has to travel.  What is left is the solved surface and the body proxy:

    magic "GTDRAPE1", int32 n, M, NC, NS,
    pos (n,3) f32, faces (M,3) i32, cyl (NC,7) f32, sph (NS,4) f32

Positions are float32: these are centimetres on a ~100 cm garment, so a float32
carries about 1e-5 cm, far below anything visible.  It halves the download.

  python pack_gt_drape.py --dump <g>.bin --npy <g>.npy --out <g>_drape.bin
"""
import argparse
import json
import os
import struct

import numpy as np

MAGIC_IN = b"SIMCPP01"
MAGIC_OUT = b"GTDRAPE1"


def read_dump(path):
    """faces, cyl and sph out of the solver's input; the rest is not needed here."""
    with open(path, "rb") as f:
        assert f.read(8) == MAGIC_IN, path
        n, M, K, NC, NS, has_mu, has_nu, _ = struct.unpack("<8i", f.read(32))
        faces = np.frombuffer(f.read(M * 3 * 4), np.int32).reshape(M, 3)
        f.read(n * 4)                       # wid
        f.read(M * 4)                       # panel_of_face
        f.read(n * 2 * 8)                   # rest
        f.read(n * 3 * 8)                   # P0
        f.read(K * 2 * 4)                   # pairs
        if has_mu:
            f.read(n * 8)
        if has_nu:
            f.read(n * 8 + n * 3 * 8)
        cyl = np.frombuffer(f.read(NC * 7 * 8), np.float64).reshape(NC, 7)
        sph = np.frombuffer(f.read(NS * 4 * 8), np.float64).reshape(NS, 4)
    return faces, cyl, sph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    faces, cyl, sph = read_dump(a.dump)
    pos = np.load(a.npy)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"{a.npy}: expected (n,3), got {pos.shape}")

    with open(a.out, "wb") as f:
        f.write(MAGIC_OUT)
        f.write(struct.pack("<4i", len(pos), len(faces), len(cyl), len(sph)))
        f.write(np.ascontiguousarray(pos, np.float32).tobytes())
        f.write(np.ascontiguousarray(faces, np.int32).tobytes())
        f.write(np.ascontiguousarray(cyl, np.float32).tobytes())
        f.write(np.ascontiguousarray(sph, np.float32).tobytes())

    print(json.dumps(dict(out=os.path.basename(a.out), n=len(pos), faces=len(faces),
                          cyl=len(cyl), sph=len(sph),
                          kb=round(os.path.getsize(a.out) / 1024))))


if __name__ == "__main__":
    main()
