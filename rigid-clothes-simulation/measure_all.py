"""Spec section 4: measurements (a)-(f) over all finished assembly runs.

  python measure_all.py [tag_prefix]
"""

import glob
import json
import os
import sys
import time

import numpy as np

import analyze
import gcd_io

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result")
GARMENT = r"C:\Users\PC\Downloads\data\rand_00YONAPXZE"

# spec 4(d): drape reference, 300-garment p50, cm
DRAPE_DXY = {"skirt_front": 15.82, "ftorso": 9.11, "wb_front": 11.13,
             "sleeve_f": 6.26, "cuff_f": 3.09, "hood": 12.88}


def load_runs(prefix=""):
    out = {}
    for f in sorted(glob.glob(os.path.join(RESULT, "assembly_%s*.npy" % prefix))):
        tag = os.path.basename(f)[len("assembly_"):-4]
        meta = os.path.join(RESULT, "assembly_%s.json" % tag)
        out[tag] = (np.load(f), json.load(open(meta)) if os.path.exists(meta) else {})
    return out


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    d = gcd_io.load(GARMENT)
    runs = load_runs(prefix)
    if not runs:
        print("no runs matching assembly_%s*.npy" % prefix)
        return
    tags = list(runs)
    cls = analyze.class_of_raw(d)
    out = {"garment": d["name"], "runs": tags, "uv_report": d["uv_report"]}
    print("garment %s | %d runs: %s" % (d["name"], len(tags), ", ".join(tags)))

    # ---------------- (c) internal consistency -------------------------------
    import assembly as A
    ratio, _, ruffle_face = analyze.seam_length_ratio(d)
    G, area = A.shape_gradients(d["rest"][d["faces"]])
    print()
    print("(c) INTERNAL CONSISTENCY")
    print("  seam rest-length ratio: non-gathered p50 %.5f (max %.5f) | designed gathers %d of %d edges, up to %.5f"
          % (np.median(ratio[ratio <= 1.05]), ratio[ratio <= 1.05].max(),
             int((ratio > 1.05).sum()), len(ratio), ratio.max()))
    print("  faces touching a gathered seam: %d (%.2f%%) -- reported separately"
          % (ruffle_face.sum(), 100 * ruffle_face.mean()))
    print("  %-16s %11s %11s %11s %10s %8s" % ("run", "max|s-1| all", "excl.gather", "gap max cm", "mono viol", "iters"))
    char = float(np.linalg.norm(np.ptp(d["placed"], axis=0)))
    cons = {}
    for t in tags:
        m = runs[t][1]
        F = A.deformation_gradients(runs[t][0], d["faces"], G)
        _, sv = A.best_rotations(F)
        dv = np.abs(sv - 1).max(1)
        m["max_sigma_dev_nogather"] = float(dv[~ruffle_face].max())
        m["max_sigma_dev_gather"] = float(dv[ruffle_face].max())
        ok_s = m["max_sigma_dev_nogather"] < 1e-3
        ok_g = m.get("seam_gap_max", np.nan) < 1e-4 * char
        cons[t] = dict(max_sigma_dev=m.get("max_sigma_dev"),
                       max_sigma_dev_nogather=m["max_sigma_dev_nogather"],
                       max_sigma_dev_gather=m["max_sigma_dev_gather"],
                       seam_gap_max=m.get("seam_gap_max"),
                       mono_violations=m.get("mono_violations"), iterations=m.get("iterations"),
                       half_space=m.get("half_space", {}), anchor=m.get("anchor", 0.0),
                       placed_dev_p50=m.get("placed_dev_p50"),
                       pass_sigma=bool(ok_s), pass_gap=bool(ok_g))
        print("  %-16s %11.3e %11.3e %11.3e %10d %8d   sigma %s  gap %s"
              % (t, m.get("max_sigma_dev", np.nan), m["max_sigma_dev_nogather"],
                 m.get("seam_gap_max", np.nan), m.get("mono_violations", -1),
                 m.get("iterations", -1), "PASS" if ok_s else "FAIL", "PASS" if ok_g else "FAIL"))
    print("  seam-gap tolerance = 1e-4 * characteristic length (%.1f cm) = %.3e cm" % (char, 1e-4 * char))
    if any(c["half_space"].get("body") for c in cons.values()):
        print("  body proxy: vertices left inside / max depth cm / median move from the placement")
        for t, c in cons.items():
            b = c["half_space"].get("body", {})
            print("    %-24s %6.3f%%  %6.3f cm   %7.2f cm"
                  % (t, 100 * b.get("frac", 0.0), b.get("max_cm", 0.0),
                     c.get("placed_dev_p50") or float("nan")))
    out["consistency"] = cons

    # ---------------- (a) initial-value sensitivity --------------------------
    print()
    print("(a) INITIAL-VALUE SENSITIVITY  -- the main result")
    ref = runs[tags[0]][0]
    S = np.stack([analyze.procrustes(runs[t][0], ref) for t in tags])   # not `A`: that is the assembly module
    spread = np.linalg.norm(S.std(0), axis=1)
    maxdev = np.linalg.norm(S - S.mean(0), axis=2).max(0)
    print("  per-vertex spread across %d runs, Procrustes aligned (reflection NOT allowed)" % len(tags))
    print("  %-14s %10s %10s %10s" % ("panel class", "std p50", "std p90", "maxdev p50"))
    sens = {}
    for c in sorted(set(cls)):
        m = cls == c
        sens[c] = dict(std_p50=float(np.median(spread[m])), std_p90=float(np.quantile(spread[m], .9)),
                       maxdev_p50=float(np.median(maxdev[m])), n=int(m.sum()))
        print("  %-14s %10.3f %10.3f %10.3f" % (c, sens[c]["std_p50"], sens[c]["std_p90"], sens[c]["maxdev_p50"]))
    print("  overall        %10.3f %10.3f %10.3f"
          % (np.median(spread), np.quantile(spread, .9), np.median(maxdev)))
    out["sensitivity_cm"] = sens

    # ---------------- (b) angle deficit --------------------------------------
    print()
    print("(b) ANGLE DEFICIT of the welded REST mesh")
    K, is_bnd, tot = analyze.angle_deficit(d)
    Kraw = K[d["wid"]]
    inner = ~is_bnd
    print("  interior vertices %d | boundary %d" % (inner.sum(), is_bnd.sum()))
    q = np.quantile(K[inner], [.01, .25, .5, .75, .99])
    print("  interior deficit rad  p1 %+.4f  p25 %+.4f  p50 %+.4f  p75 %+.4f  p99 %+.4f"
          % tuple(q))
    print("  sum over interior = %.4f rad  (Gauss-Bonnet scale)" % K[inner].sum())
    print("  %-14s %10s %10s %10s %10s" % ("panel class", "sum defc", "p50", "p90 |defc|", "n"))
    defc = {}
    for c in sorted(set(cls)):
        m = (cls == c) & inner[d["wid"]]
        if not m.any():
            continue
        v = Kraw[m]
        defc[c] = dict(sum=float(v.sum()), p50=float(np.median(v)),
                       p90_abs=float(np.quantile(np.abs(v), .9)), n=int(m.sum()))
        print("  %-14s %10.3f %10.4f %10.4f %10d" % (c, defc[c]["sum"], defc[c]["p50"],
                                                     defc[c]["p90_abs"], defc[c]["n"]))
    out["angle_deficit"] = defc

    # (a) vs (b) correlation
    print()
    print("  (a) vs (b): correlation of per-vertex |deficit| with per-vertex spread")
    print("  %-14s %8s %10s %10s" % ("panel class", "pearson", "spread p50", "|defc| p50"))
    corr = {}
    for c in sorted(set(cls)):
        m = (cls == c) & inner[d["wid"]]
        if m.sum() < 20:
            continue
        x, y = np.abs(Kraw[m]), spread[m]
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan
        corr[c] = r
        print("  %-14s %8.3f %10.3f %10.4f" % (c, r, np.median(y), np.median(x)))
    m = inner[d["wid"]]
    rall = float(np.corrcoef(np.abs(Kraw[m]), spread[m])[0, 1])
    print("  overall        %8.3f" % rall)
    out["deficit_spread_corr"] = dict(per_class=corr, overall=rall)

    # ---------------- (d) displacement from the placement --------------------
    print()
    print("(d) DISPLACEMENT FROM THE PLACEMENT, Delta_xy (front-view plane, cm)")
    print("  %-14s %12s %12s %10s" % ("panel class", "ARAP p50", "drape p50", "ratio"))
    dxy = {}
    Pa = analyze.procrustes(runs[tags[0]][0], d["placed"])
    dd = np.linalg.norm((Pa - d["placed"])[:, [0, 1]], axis=1)
    for c in sorted(set(cls)):
        m = cls == c
        v = float(np.median(dd[m]))
        ref_v = DRAPE_DXY.get(c)
        dxy[c] = dict(arap_p50=v, drape_p50=ref_v)
        print("  %-14s %12.2f %12s %10s"
              % (c, v, "%.2f" % ref_v if ref_v else "-", "%.2f" % (v / ref_v) if ref_v else "-"))
    out["delta_xy_cm"] = dxy

    # ---------------- (e) distance to the drape ------------------------------
    print()
    print("(e) DISTANCE TO THE ACTUAL DRAPE after Procrustes (cm) -- not a target")
    print("  %-14s %10s %10s" % ("panel class", "p50", "p90"))
    Pd = analyze.procrustes(runs[tags[0]][0], d["drape"])
    dist = np.linalg.norm(Pd - d["drape"], axis=1)
    dr = {}
    for c in sorted(set(cls)):
        m = cls == c
        dr[c] = dict(p50=float(np.median(dist[m])), p90=float(np.quantile(dist[m], .9)))
        print("  %-14s %10.2f %10.2f" % (c, dr[c]["p50"], dr[c]["p90"]))
    print("  overall        %10.2f %10.2f" % (np.median(dist), np.quantile(dist, .9)))
    out["drape_distance_cm"] = dr

    # ---------------- (f) self-intersection ----------------------------------
    print()
    print("(f) SELF-INTERSECTION (counted, never fixed)")
    t0 = time.time()
    n_hit, pairs = analyze.tri_tri_pairs(runs[tags[0]][0], d["faces"])
    print("  intersecting triangle pairs: %d   (%.1fs)" % (n_hit, time.time() - t0))
    combos = {}
    if n_hit:
        pn = np.array(d["panel_names"], dtype=object)[d["panel_of_face"]]
        for a, b in pairs:
            k = tuple(sorted((str(pn[a]), str(pn[b]))))
            combos[k] = combos.get(k, 0) + 1
        for k, v in sorted(combos.items(), key=lambda kv: -kv[1])[:12]:
            print("    %-38s %6d" % (" x ".join(k), v))
    out["self_intersections"] = dict(pairs=int(n_hit),
                                     by_panel={" x ".join(k): v for k, v in combos.items()})

    os.makedirs(RESULT, exist_ok=True)
    with open(os.path.join(RESULT, "measurements.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.save(os.path.join(RESULT, "angle_deficit.npy"), K)
    with open(os.path.join(RESULT, "angle_deficit.json"), "w") as f:
        json.dump(dict(per_class=defc, n_interior=int(inner.sum()),
                       n_boundary=int(is_bnd.sum())), f, indent=1)
    print()
    print("wrote result/measurements.json, result/angle_deficit.json")


if __name__ == "__main__":
    main()
