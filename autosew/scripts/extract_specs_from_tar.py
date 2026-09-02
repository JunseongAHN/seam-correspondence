"""Extract ONLY *_specification.json from a GarmentCodeData v2 part tarball.

The libdrive.ethz.ch share does NOT expose per-garment folders over WebDAV -- each
part is a single data.tar.gz. So spec-only fetching means: download the tarball,
stream it through tarfile, keep the spec members, drop everything else.

  python scripts/extract_specs_from_tar.py --tar <part.tar.gz> --part 1 --out ~/gcd_data

Output layout (matches train.py --data_dir expectations):
  <out>/part<N>/<rand_XXX>/<rand_XXX>_specification.json
"""
import argparse
import os
import sys
import tarfile
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_root = Path(os.path.expanduser(args.out)) / f"part{args.part}"
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_spec = n_skip = n_bad = 0
    # "r|gz" = streaming (non-seekable) read: never holds the whole archive in memory
    with tarfile.open(args.tar, "r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith("_specification.json"):
                continue
            name = Path(m.name).name
            garment = name[: -len("_specification.json")]
            dest = out_root / garment / name
            if dest.exists() and dest.stat().st_size > 0:
                n_skip += 1
                continue
            f = tf.extractfile(m)
            if f is None:
                n_bad += 1
                continue
            data = f.read()
            if data.lstrip()[:1] != b"{":
                n_bad += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            n_spec += 1
            if n_spec % 500 == 0:
                print(f"  {n_spec} specs ({time.time()-t0:.0f}s)", flush=True)

    print(f"[part{args.part}] specs={n_spec} skip={n_skip} bad={n_bad} "
          f"in {(time.time()-t0)/60:.1f}min -> {out_root}", flush=True)
    if n_spec + n_skip == 0:
        print("[part] NO specs found -- check the tarball layout", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
