"""Build Isometric Shells for as many garments as fit in a time budget.

  python run_batch.py [workers] [minutes]        default 8 workers, 60 minutes

Each garment gets `proxy/<id>/` with the assembly, the per-panel-coloured plys
and its own `log.txt`.  A garment already holding `assembly_shell.json` is
skipped, so the batch can be stopped and restarted.

Three things it has to be careful about.

Garments differ -- one has no sleeves, another no waistband -- so a failure is
recorded and the batch carries on rather than stopping.

**Exit code 0 is not the same as a usable result.**  Every torn run earlier in
this project exited 0: the hard clamp collapsed triangles, the sphere bug
teleported 18% of the mesh, the over-wide arm tore the cuff open 1.6x.  So each
finished garment is gated on the three quantities that were wrong in those cases
-- monotonicity violations, seam gap, and max |sigma-1| -- and anything that
trips a gate is marked SUSPECT with the gate named, not silently counted as done.

And BLAS threads have to be capped: eight processes each spawning sixteen OpenMP
threads on sixteen cores ran 6x slower than eight processes with two.

Nothing is ever cleaned up.  Failed and suspect garments keep their directory and
their log; `proxy/_batch_log.jsonl` holds one json line per garment either way.
"""

import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r"C:\Users\PC\Downloads\data"
PROXY = os.path.join(HERE, "proxy")
LOGJ = os.path.join(PROXY, "_batch_log.jsonl")
PY = sys.executable
ARGS = ["--amp", "0", "--body", "--sym", "--mu", "0.02", "--tag", "shell"]

ENV = dict(os.environ)
ENV.update(PYTHONPATH=HERE, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
           OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")


def todo():
    out = []
    for g in sorted(os.listdir(DATA)):
        if not os.path.isdir(os.path.join(DATA, g)):
            continue
        if os.path.exists(os.path.join(PROXY, g, "assembly_shell.json")):
            continue
        out.append(g)
    return out


def _tail(d, n=4):
    try:
        t = open(os.path.join(d, "log.txt"), errors="replace").read().strip().split("\n")
        return " | ".join(t[-n:])[:300]
    except Exception:
        return ""


def gate(j):
    """which of the failure modes seen earlier in this project this run shows"""
    bad = []
    if j.get("mono_violations", 0) > 0:
        bad.append("mono=%d" % j["mono_violations"])
    if not (j.get("seam_gap_max", 1e9) < 1e-3):
        bad.append("gap=%.1e" % j.get("seam_gap_max", float("nan")))
    if not (j.get("max_sigma_dev", 1e9) < 1.0):
        bad.append("sigma=%.2f" % j.get("max_sigma_dev", float("nan")))
    for k in ("E_arap", "E_bend", "max_sigma_dev"):
        v = j.get(k)
        if v is None or v != v or abs(v) == float("inf"):
            bad.append("nonfinite:%s" % k)
    return bad


def one(g, deadline):
    if time.time() > deadline:
        return g, "skipped(deadline)", 0.0, ""
    d = os.path.join(PROXY, g)
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    log = open(os.path.join(d, "log.txt"), "w")
    try:
        r = subprocess.run([PY, os.path.join(HERE, "run_garment.py"),
                            "--garment", g, "--outdir", d] + ARGS,
                           cwd=HERE, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                           timeout=3600)
        log.flush()
        if r.returncode != 0:
            return g, "FAILED(solve rc=%d)" % r.returncode, time.time() - t0, _tail(d)
        jf = os.path.join(d, "assembly_shell.json")
        if not os.path.exists(jf):
            return g, "FAILED(no output)", time.time() - t0, _tail(d)
        j = json.load(open(jf))
        bad = gate(j)
        note = ("max|s-1| %.3f  gap %.1e  mono %d  E_arap %.1f  %d iters"
                % (j["max_sigma_dev"], j["seam_gap_max"], j["mono_violations"],
                   j["E_arap"], j["iterations"]))
        rp = subprocess.run([PY, os.path.join(HERE, "render_patches.py"), g, d],
                            cwd=HERE, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                            timeout=900)
        if rp.returncode != 0:
            bad.append("render rc=%d" % rp.returncode)
        return (g, "SUSPECT(%s)" % ",".join(bad) if bad else "ok",
                time.time() - t0, note)
    except subprocess.TimeoutExpired:
        return g, "FAILED(timeout)", time.time() - t0, _tail(d)
    except Exception as e:
        return g, "FAILED(%s)" % type(e).__name__, time.time() - t0, "%s | %s" % (e, _tail(d))
    finally:
        log.close()


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    deadline = time.time() + minutes * 60.0
    os.makedirs(PROXY, exist_ok=True)
    queue = todo()
    print("%d garments outstanding, %d workers, %.0f minute budget"
          % (len(queue), workers, minutes), flush=True)

    done = ok = bad = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs, it = {}, iter(queue)
        for _ in range(workers):
            g = next(it, None)
            if g:
                futs[ex.submit(one, g, deadline)] = g
        while futs:
            f = next(cf.as_completed(list(futs)))
            futs.pop(f)
            name, status, secs, note = f.result()
            done += 1
            if status == "ok":
                ok += 1
            elif status.startswith("skipped"):
                done -= 1
            else:
                bad += 1
            print("[%5.1f min] %-18s %-28s %5.1f min  %s"
                  % ((time.time() - t0) / 60, name, status, secs / 60, note), flush=True)
            with open(LOGJ, "a") as lf:
                lf.write(json.dumps(dict(garment=name, status=status,
                                         minutes=round(secs / 60, 2), note=note,
                                         at=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
            if time.time() < deadline:
                nxt = next(it, None)
                if nxt:
                    futs[ex.submit(one, nxt, deadline)] = nxt

    print("\n%d finished, %d ok, %d failed or suspect, %.1f min elapsed"
          % (done, ok, bad, (time.time() - t0) / 60), flush=True)
    print("per-garment log in proxy/<id>/log.txt, one json line each in "
          "proxy/_batch_log.jsonl; nothing is deleted", flush=True)


if __name__ == "__main__":
    main()
