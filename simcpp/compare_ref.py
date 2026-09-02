"""Compare a simcpp result against the Python reference, vertex by vertex.

  python compare_ref.py ref/trousers/assembly_ref.npy out/native_trousers.npy \
      --ref-json ref/trousers/assembly_ref.json --cpp-json out/native_trousers.json
"""

import argparse
import json

import numpy as np

GATES = ["mono_violations", "seam_gap_max", "max_sigma_dev", "E_arap", "E_bend",
         "E_stitch", "seam_gap_p50", "p50_sigma_dev"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cpp")
    ap.add_argument("--ref-json")
    ap.add_argument("--cpp-json")
    a = ap.parse_args()

    A = np.load(a.ref)
    B = np.load(a.cpp)
    if A.shape != B.shape:
        raise SystemExit("shape mismatch: ref %s vs cpp %s" % (A.shape, B.shape))

    d = np.linalg.norm(A - B, axis=1)            # per-vertex distance, cm
    scale = float(np.linalg.norm(np.ptp(A, axis=0)))
    print("vertices           %d" % len(A))
    print("bbox diagonal      %.3f cm" % scale)
    print("max |diff|         %.6e cm   (vertex %d)" % (d.max(), int(d.argmax())))
    print("p99 |diff|         %.6e cm" % np.percentile(d, 99))
    print("p50 |diff|         %.6e cm" % np.median(d))
    print("rms |diff|         %.6e cm" % np.sqrt((d ** 2).mean()))
    print("max |diff| / bbox  %.3e" % (d.max() / max(scale, 1e-30)))
    print("max per-coord diff %.6e cm" % np.abs(A - B).max())

    if a.ref_json and a.cpp_json:
        R = json.load(open(a.ref_json))
        C = json.load(open(a.cpp_json))
        print("\n%-18s %22s %22s %12s" % ("gate", "python", "cpp", "rel.diff"))
        for k in GATES:
            if k not in R or k not in C:
                continue
            r, c = float(R[k]), float(C[k])
            rel = abs(r - c) / max(abs(r), 1e-30)
            print("%-18s %22.12g %22.12g %12.3e" % (k, r, c, rel))
        for k in ("iterations", "factorizations"):
            print("%-18s %22d %22d" % (k, R[k], C[k]))
        print("%-18s %22.2f %22.2f" % ("seconds", R["seconds"], C["seconds"]))


if __name__ == "__main__":
    main()
