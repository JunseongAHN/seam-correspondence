"""Lay out a GCD part into gcd_data/{train,valid,test} per the OFFICIAL split json.

train/valid: *_specification.json only (that is all AutoSew needs).
test:        every file for the garment (sim.ply, renders, ...) so results can be
             inspected in 3D later.
Garments absent from the official split are dropped.

  # from an extracted directory (e.g. the already-unpacked part0)
  python scripts/organize_split.py --part 0 --src <.../default_body/data> --root ~/gcd_data
  # from a downloaded tarball
  python scripts/organize_split.py --part 1 --tar <part1.tar.gz> --root ~/gcd_data

Layout: <root>/<split>/part<N>/<garment>/<files>
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

SPLIT_DIR = {"training": "train", "validation": "valid", "test": "test"}


def load_split(split_json, part):
    key = f"garments_5000_{part}"
    s = json.loads(Path(split_json).read_text())
    out = {}
    for split_name, entries in s.items():
        for e in entries:
            p = e.split("/")
            if p[0] == key:
                out[p[2]] = SPLIT_DIR[split_name]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default=None, help="split json (default: <root>/GarmentCodeData_v2_official_train_valid_test_data_split.json)")
    ap.add_argument("--src", default=None, help="extracted .../default_body/data dir")
    ap.add_argument("--tar", default=None, help="part data.tar.gz")
    args = ap.parse_args()
    if bool(args.src) == bool(args.tar):
        ap.error("give exactly one of --src / --tar")

    root = Path(os.path.expanduser(args.root))
    split_json = args.split or (root / "GarmentCodeData_v2_official_train_valid_test_data_split.json")
    where = load_split(split_json, args.part)
    print(f"[part{args.part}] official split: {len(where)} garments "
          f"({sum(v=='train' for v in where.values())} train / "
          f"{sum(v=='valid' for v in where.values())} valid / "
          f"{sum(v=='test' for v in where.values())} test)", flush=True)

    n = {"train": 0, "valid": 0, "test": 0}
    n_files = 0
    dropped = set()
    t0 = time.time()

    def want(garment, fname):
        """Return dest path, or None to skip."""
        sp = where.get(garment)
        if sp is None:
            dropped.add(garment)
            return None
        if sp != "test" and not fname.endswith("_specification.json"):
            return None
        return root / sp / f"part{args.part}" / garment / fname

    if args.tar:
        with tarfile.open(args.tar, "r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                fname = Path(m.name).name
                garment = Path(m.name).parent.name
                dest = want(garment, fname)
                if dest is None:
                    continue
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(f.read())
                n_files += 1
                if fname.endswith("_specification.json"):
                    n[where[garment]] += 1
                    if sum(n.values()) % 500 == 0:
                        print(f"  {sum(n.values())} garments ({time.time()-t0:.0f}s)", flush=True)
    else:
        src = Path(args.src)
        for gdir in sorted(p for p in src.iterdir() if p.is_dir()):
            garment = gdir.name
            for f in sorted(gdir.iterdir()):
                if not f.is_file():
                    continue
                dest = want(garment, f.name)
                if dest is None:
                    continue
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dest)
                n_files += 1
            if garment in where:
                n[where[garment]] += 1
                if sum(n.values()) % 500 == 0:
                    print(f"  {sum(n.values())} garments ({time.time()-t0:.0f}s)", flush=True)

    print(f"[part{args.part}] placed train={n['train']} valid={n['valid']} test={n['test']} "
          f"({n_files} files, {len(dropped)} garments not in split) in {(time.time()-t0)/60:.1f}min",
          flush=True)
    if sum(n.values()) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
