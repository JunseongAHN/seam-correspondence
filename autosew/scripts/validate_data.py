"""Run FIRST on the real GCD part, before any training.

python scripts/validate_data.py --data_dir /path/to/part --limit 500

Checks the format assumptions in gcd_parser.py against real files and prints a JSON
report: parse failures, unknown curvature encodings, loop-closure violations,
M / stitch statistics, units, edge-count min-max (feed into cfg.edge_count_minmax).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autosew.config import AutoSewConfig
from autosew.dataset import find_spec_files
from autosew.features import pattern_to_tensors
from autosew.gcd_parser import parse_specification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    files = find_spec_files(args.data_dir, args.limit)
    cfg = AutoSewConfig()
    rep = {"n_files_checked": len(files), "parse_fail": 0, "fail_examples": [],
           "units": Counter(), "kt_hist": Counter(), "panels_per_pattern": [],
           "edges_per_panel_min": 1e9, "edges_per_panel_max": 0,
           "M_per_pattern": [], "stitches_per_pattern": [],
           "unstitched_frac": [], "multi_edge_patterns": 0,
           "loop_violations": 0, "reversed_panels": 0, "panel_name_sample": Counter()}

    for f in files:
        try:
            spec = json.loads(Path(f).read_text())
            pat = spec["pattern"] if "pattern" in spec else spec
            # loop-closure check on raw edges
            for pname, pdata in pat["panels"].items():
                E = pdata["edges"]
                for i, e in enumerate(E):
                    if e["endpoints"][1] != E[(i + 1) % len(E)]["endpoints"][0]:
                        rep["loop_violations"] += 1
                        break
            p = parse_specification(spec, name=Path(f).stem)
            s = pattern_to_tensors(p, cfg)
            M = s["x"].shape[0]
            rep["units"][str(p.meta["units_in_meter"])] += 1
            rep["panels_per_pattern"].append(len(p.panels))
            for pan in p.panels:
                n = len(pan.edges)
                rep["edges_per_panel_min"] = min(rep["edges_per_panel_min"], n)
                rep["edges_per_panel_max"] = max(rep["edges_per_panel_max"], n)
                rep["reversed_panels"] += int(pan.was_reversed)
                rep["panel_name_sample"][pan.name.split("_")[0]] += 1
                for e in pan.edges:
                    rep["kt_hist"][e.kt] += 1
            rep["M_per_pattern"].append(M)
            rep["stitches_per_pattern"].append(len(s["gt_pairs"]))
            rep["unstitched_frac"].append(1.0 - s["stitched"].mean())
            deg = Counter()
            for a, b in s["gt_pairs"]:
                deg[int(a)] += 1
                deg[int(b)] += 1
            if deg and max(deg.values()) >= 2:
                rep["multi_edge_patterns"] += 1
        except Exception as e:
            rep["parse_fail"] += 1
            if len(rep["fail_examples"]) < 5:
                rep["fail_examples"].append(f"{f}: {type(e).__name__}: {e}")

    def q(v):
        v = sorted(v)
        return {"min": v[0], "p50": v[len(v) // 2], "mean": round(sum(v) / len(v), 2), "max": v[-1]} if v else {}

    out = {
        "n_files_checked": rep["n_files_checked"],
        "parse_fail": rep["parse_fail"], "fail_examples": rep["fail_examples"],
        "loop_violations": rep["loop_violations"],
        "units_in_meter": dict(rep["units"]),
        "curvature_type_hist (0=straight,1=circle,2=quad,3=cubic,4=bspline,5=UNKNOWN)": dict(rep["kt_hist"]),
        "panels_per_pattern": q(rep["panels_per_pattern"]),
        "edges_per_panel": [rep["edges_per_panel_min"], rep["edges_per_panel_max"]],
        "M_per_pattern": q(rep["M_per_pattern"]),
        "gt_pairs_per_pattern": q(rep["stitches_per_pattern"]),
        "unstitched_frac": q([round(float(x), 3) for x in rep["unstitched_frac"]]),
        "multi_edge_patterns": rep["multi_edge_patterns"],
        "reversed_panels": rep["reversed_panels"],
        "panel_name_prefixes": dict(rep["panel_name_sample"].most_common(15)),
        "ACTION": "if kt=5 (UNKNOWN) is frequent or loop_violations>0, fix gcd_parser before training; "
                  "set cfg.edge_count_minmax from edges_per_panel.",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
