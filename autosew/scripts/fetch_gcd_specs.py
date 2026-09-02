"""Download ONLY *_specification.json from GarmentCodeData v2 (libdrive.ethz.ch Nextcloud).

Panels, edges, curvature, stitches are all inside specification.json -- meshes/renders
are not needed for AutoSew training, so this fetches ~150MB per part instead of ~7GB.

Pure stdlib (urllib + xml + threads). Resumable: existing non-empty files are skipped.

  # 1) count garments per part without downloading (sanity check the share layout)
  python3 scripts/fetch_gcd_specs.py --parts 0 1 2 3 --out ~/gcd_data --dry-run

  # 2) actual download (8 parallel connections; be polite, keep <= 12)
  python3 scripts/fetch_gcd_specs.py --parts 0 1 2 3 --out ~/gcd_data

Output layout (matches the local GCD convention, train.py --data_dir ~/gcd_data works):
  <out>/part<N>/<rand_XXX>/<rand_XXX>_specification.json
"""
import argparse
import base64
import concurrent.futures as cf
import os
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_TOKEN = "4UtC8smtLOGwKoZ"
DEFAULT_BASE = "https://libdrive.ethz.ch"
SHARE_ROOT = "GarmentCodeData_v2"


def make_opener(token):
    auth = base64.b64encode(f"{token}:".encode()).decode()
    op = urllib.request.build_opener()
    op.addheaders = [("Authorization", f"Basic {auth}"),
                     ("User-Agent", "autosew-spec-fetch/1.0")]
    return op


def propfind(opener, base, path, depth=1, timeout=60):
    """List a WebDAV collection. path: share-relative, e.g. 'GarmentCodeData_v2/garments_5000_0'.
    Returns list of child hrefs (URL-decoded, share-relative, dirs end with /)."""
    url = f"{base}/public.php/webdav/" + urllib.parse.quote(path.rstrip("/") + "/")
    req = urllib.request.Request(url, method="PROPFIND")
    req.add_header("Depth", str(depth))
    with opener.open(req, timeout=timeout) as r:
        xml_data = r.read()
    ns = {"d": "DAV:"}
    out = []
    root_marker = "/public.php/webdav/"
    for resp in ET.fromstring(xml_data).findall("d:response", ns):
        href = resp.find("d:href", ns).text
        rel = urllib.parse.unquote(href.split(root_marker, 1)[1])
        if rel.rstrip("/") == path.rstrip("/"):
            continue  # the collection itself
        out.append(rel)
    return out


def fetch_file(opener, base, rel_path, dest: Path, retries=3, timeout=120):
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    url = f"{base}/public.php/webdav/" + urllib.parse.quote(rel_path)
    for attempt in range(retries):
        try:
            with opener.open(url, timeout=timeout) as r:
                data = r.read()
            if not data or data.lstrip()[:1] not in (b"{",):
                raise IOError(f"not JSON ({data[:40]!r})")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
            return "ok"
        except Exception as e:
            if attempt == retries - 1:
                return f"fail: {type(e).__name__}: {e}"
            time.sleep(2.0 * (attempt + 1))
    return "fail"


def part_garment_dirs(opener, base, part):
    """Find garment folders for one part. Handles both .../default_body/data/<g>/ and
    .../default_body/<g>/ layouts."""
    proot = f"{SHARE_ROOT}/garments_5000_{part}/default_body"
    children = propfind(opener, base, proot)
    dirs = [c for c in children if c.endswith("/")]
    data_dir = [c for c in dirs if c.rstrip("/").rsplit("/", 1)[-1] == "data"]
    if data_dir:
        dirs = [c for c in propfind(opener, base, data_dir[0].rstrip("/")) if c.endswith("/")]
    return [d.rstrip("/") for d in dirs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", required=True, help="e.g. --parts 0 1 2 3")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--token", type=str, default=DEFAULT_TOKEN)
    ap.add_argument("--base", type=str, default=DEFAULT_BASE)
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads (be polite, <=12)")
    ap.add_argument("--dry-run", action="store_true", help="list garment counts only")
    ap.add_argument("--limit", type=int, default=None, help="max garments per part (debug)")
    args = ap.parse_args()

    opener = make_opener(args.token)
    out_root = Path(os.path.expanduser(args.out))
    grand_ok = grand_skip = grand_fail = 0

    for part in args.parts:
        t0 = time.time()
        try:
            gdirs = part_garment_dirs(opener, args.base, part)
        except Exception as e:
            print(f"[part{part}] LIST FAILED: {type(e).__name__}: {e}", flush=True)
            print("  -> check token/base URL; test with:")
            print(f"     curl -s -u '{args.token}:' -X PROPFIND -H 'Depth: 1' "
                  f"{args.base}/public.php/webdav/{SHARE_ROOT}/ | head")
            sys.exit(2)
        if args.limit:
            gdirs = gdirs[: args.limit]
        print(f"[part{part}] {len(gdirs)} garment folders listed ({time.time()-t0:.1f}s)", flush=True)
        if args.dry_run:
            for g in gdirs[:3]:
                print("   sample:", g)
            continue

        jobs = []
        for g in gdirs:
            name = g.rsplit("/", 1)[-1]
            rel = f"{g}/{name}_specification.json"
            dest = out_root / f"part{part}" / name / f"{name}_specification.json"
            jobs.append((rel, dest))

        ok = skip = fail = 0
        fails = []
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_file, opener, args.base, rel, dest): rel for rel, dest in jobs}
            for i, fut in enumerate(cf.as_completed(futs)):
                res = fut.result()
                if res == "ok":
                    ok += 1
                elif res == "skip":
                    skip += 1
                else:
                    fail += 1
                    if len(fails) < 5:
                        fails.append((futs[fut], res))
                if (i + 1) % 250 == 0:
                    print(f"[part{part}] {i+1}/{len(jobs)}  ok={ok} skip={skip} fail={fail}", flush=True)
        dt = time.time() - t0
        print(f"[part{part}] DONE ok={ok} skip={skip} fail={fail} in {dt/60:.1f}min", flush=True)
        for f_rel, f_msg in fails:
            print(f"   fail example: {f_rel} -> {f_msg}", flush=True)
        grand_ok += ok; grand_skip += skip; grand_fail += fail

    if not args.dry_run:
        total = grand_ok + grand_skip + grand_fail
        print(f"[TOTAL] ok={grand_ok} skip={grand_skip} fail={grand_fail} / {total}", flush=True)
        if total and grand_fail / total > 0.005:
            print("[TOTAL] fail rate > 0.5% — rerun the same command (resume skips done files); "
                  "if it persists, report the fail examples", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
