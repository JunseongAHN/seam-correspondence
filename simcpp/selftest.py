"""Diff the C++ port against the Python one PIECE AT A TIME.

The end-to-end result already agrees to ~1e-10 cm, but that is only useful once
you know which stage broke when it doesn't.  This recomputes each intermediate
in Python from the same dump the C++ read, and diffs it against what
`simcpp --selftest <prefix>` wrote:

    shape_gradients          G, area
    build_hinges             hinges
    hinge_stencils           Kb, wb
    deformation_gradients    F at P0
    best_rotations           R, sigma at P0
    body.projector           Z = clamp(P0)
    arap_rhs (+ mu Z)        b at P0
    Assembly.solve_global    one global step at (lam_b = ladder[0], w_s = w0)

  python selftest.py --dump data/trousers.bin --stage out/st_trousers
"""

import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rigid-clothes-simulation"))

import assembly as A     # noqa: E402
import body              # noqa: E402


def read_dump(path):
    f = open(path, "rb")
    assert f.read(8) == b"SIMCPP01"
    n, M, K, NC, NS, has_mu, has_nu, recenter = struct.unpack("<8i", f.read(32))
    rd = lambda dt, cnt: np.frombuffer(f.read(cnt * np.dtype(dt).itemsize), dt)
    d = dict(n=n, recenter=recenter)
    d["faces"] = rd("<i4", M * 3).reshape(M, 3).astype(np.int64)
    d["wid"] = rd("<i4", n).astype(np.int64)
    d["panel_of_face"] = rd("<i4", M).astype(np.int64)
    d["rest"] = rd("<f8", n * 2).reshape(n, 2)
    d["P0"] = rd("<f8", n * 3).reshape(n, 3)
    d["pairs"] = rd("<i4", K * 2).reshape(K, 2).astype(np.int64)
    d["mu"] = rd("<f8", n) if has_mu else None
    if has_nu:
        d["nu"] = rd("<f8", n)
        d["anchor"] = rd("<f8", n * 3).reshape(n, 3)
    else:
        d["nu"] = d["anchor"] = None
    d["cyl"] = rd("<f8", NC * 7).reshape(NC, 7)
    d["sph"] = rd("<f8", NS * 4).reshape(NS, 4)
    return d


def cmp(name, ref, got, rel=True):
    ref = np.asarray(ref)
    got = np.asarray(got)
    if ref.shape != got.shape:
        print("%-24s SHAPE MISMATCH  py %s  cpp %s" % (name, ref.shape, got.shape))
        return False
    if ref.dtype.kind == "i":
        bad = int((ref != got).sum())
        print("%-24s %s  exact-int mismatches: %d" % (name, ref.shape, bad))
        return bad == 0
    err = np.abs(ref - got)
    den = np.maximum(np.abs(ref), 1e-30) if rel else 1.0
    print("%-24s %-14s max abs %.3e   max rel %.3e"
          % (name, str(ref.shape), err.max(), (err / den).max()))
    return err.max() < 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--stage", required=True, help="prefix simcpp --selftest wrote")
    ap.add_argument("--lam-b", type=float, default=1e-1)
    ap.add_argument("--w0", type=float, default=1e-2)
    a = ap.parse_args()
    d = read_dump(a.dump)
    L = lambda s: np.load(a.stage + "_" + s + ".npy")

    ok = True
    rest_tri = d["rest"][d["faces"]]
    G, area = A.shape_gradients(rest_tri)
    ok &= cmp("shape_gradients G", G.reshape(len(G), 6), L("G"))
    ok &= cmp("shape_gradients area", area, L("area"))

    H, R4 = A.build_hinges(d["faces"], d["wid"], rest_tri, d["panel_of_face"])
    Kb, wb = A.hinge_stencils(R4)
    ok &= cmp("build_hinges hinges", H.astype(np.int32), L("hinges"))
    ok &= cmp("hinge_stencils Kb", Kb, L("Kb"))
    ok &= cmp("hinge_stencils wb", wb, L("wb"))

    P0 = d["P0"]
    F = A.deformation_gradients(P0, d["faces"], G)
    R, sig = A.best_rotations(F)
    ok &= cmp("deformation_gradients", F.reshape(len(F), 6), L("F0"))
    ok &= cmp("best_rotations R", R.reshape(len(R), 6), L("R0"))
    ok &= cmp("best_rotations sigma", sig, L("sig0"))

    pk = body.pack((
        [(c[:3], c[3:6], c[6]) for c in d["cyl"]],
        [(s[:3], s[3]) for s in d["sph"]]))
    Z = body.projector(pk)(P0.copy())
    ok &= cmp("body clamp Z", Z, L("Z0"))

    b = A.arap_rhs(R, d["faces"], G, area, d["n"])
    if d["mu"] is not None:
        b = b + d["mu"][:, None] * Z
    ok &= cmp("arap_rhs + mu Z", b, L("b0"))

    gar = dict(faces=d["faces"], n=d["n"], G=G, area=area, hinges=H, Kb=Kb, wb=wb,
               pairs=d["pairs"])
    asm = A.Assembly(gar, a.lam_b, mu=d["mu"], nu=d["nu"], anchor=d["anchor"])
    P1 = asm.solve_global(b, a.w0)
    ok &= cmp("solve_global (one step)", P1, L("P1"))

    print("\nALL STAGES MATCH" if ok else "\nSOME STAGES DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
