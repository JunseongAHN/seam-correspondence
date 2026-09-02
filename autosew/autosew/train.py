"""Training CLI.

CPU smoke test:   python -m autosew.train --synthetic 24 --epochs 60 --batch 8
Real data (GPU):  python -m autosew.train --data_dir /path/to/part --epochs 18 --batch 16 \
                     --out runs/gcd_part --cache runs/gcd_part/cache.pt
"""
import argparse
import json
import time
from pathlib import Path

import torch

from .config import AutoSewConfig
from .dataset import PatternDataset, split_dataset, collate, loader
from .metrics import MetricAccumulator, evaluate_batch
from .model import AutoSewGNN
from .sinkhorn import log_assignment, build_supervision, nll_loss


def run_epoch(model, ds, cfg, device, opt=None, epoch=0):
    train = opt is not None
    model.train(train)
    total, nb = 0.0, 0
    acc = MetricAccumulator()
    for batch in loader(ds, cfg.batch_size, shuffle=train, device=device, seed=cfg.seed + epoch):
        f = model(batch["x"], batch["nbr"], batch["mask"])
        C = model.scores(f)
        logP = log_assignment(C, model.dustbin_z, batch["mask"], cfg)
        sup = build_supervision(batch, cfg)
        if sup is None:
            continue
        loss = nll_loss(logP, sup.to(device))
        if train:
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        else:
            evaluate_batch(logP, batch, cfg, acc)
        total += float(loss.detach())
        nb += 1
    out = {"loss": total / max(nb, 1)}
    if not train:
        out.update(acc.result())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="max spec files to load")
    ap.add_argument("--cache", type=str, default=None, help="preprocessed tensor cache (.pt)")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="runs/dev")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="config overrides, e.g. --set final_activation=none tau_multi=0.4")
    args = ap.parse_args(argv)

    cfg = AutoSewConfig(lr=args.lr, epochs=args.epochs, batch_size=args.batch, seed=args.seed)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        setattr(cfg, k, v)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    # feature-affecting config: a cache built under one of these is INVALID under another
    FEAT_KEYS = ["scale_div", "curvature_frame", "curvature_type_norm",
                 "panel_id_mode", "max_panels_norm", "edge_count_minmax"]
    feat_cfg = {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in cfg.to_dict().items() if k in FEAT_KEYS}

    if args.synthetic:
        ds = PatternDataset.from_synthetic(args.synthetic, cfg, seed=cfg.seed)
    elif args.cache and Path(args.cache).exists():
        obj = torch.load(args.cache, weights_only=False)
        if isinstance(obj, dict) and "samples" in obj:
            mismatch = {k: (obj["feat_cfg"].get(k), feat_cfg[k]) for k in FEAT_KEYS
                        if obj.get("feat_cfg", {}).get(k) != feat_cfg[k]}
            if mismatch:
                raise SystemExit(f"[cache] REFUSED: {args.cache} was built with different "
                                 f"feature config {mismatch} (cached vs current). "
                                 f"Point --cache at a new path to rebuild.")
            ds = PatternDataset(obj["samples"])
        else:  # legacy cache (bare sample list): no config guard possible
            print("[cache] WARNING: legacy cache without feature-config guard; "
                  "only use it with default feature settings", flush=True)
            ds = PatternDataset(obj)
        print(f"[cache] loaded {len(ds)} samples from {args.cache}", flush=True)
    elif args.data_dir:
        ds = PatternDataset.from_dir(args.data_dir, cfg, limit=args.limit)
        if args.cache:
            torch.save({"samples": ds.samples, "feat_cfg": feat_cfg}, args.cache)
            print(f"[cache] saved -> {args.cache}", flush=True)
    else:
        ap.error("need --data_dir or --synthetic")

    print("[stats]", json.dumps(ds.stats()), flush=True)
    tr, va, te = split_dataset(ds, args.val_frac, args.test_frac, seed=cfg.seed)
    print(f"[split] train {len(tr)} / val {len(va)} / test {len(te)}", flush=True)

    model = AutoSewGNN(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_par/1e6:.2f}M params, device={device}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    best_tf1, hist = -1.0, []
    for ep in range(cfg.epochs):
        t0 = time.time()
        trm = run_epoch(model, tr, cfg, device, opt, epoch=ep)
        vam = run_epoch(model, va, cfg, device) if len(va) else {}
        row = {"epoch": ep, "train_loss": round(trm["loss"], 5),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in vam.items()},
               "sec": round(time.time() - t0, 1),
               "z": round(float(model.dustbin_z.detach()), 3)}
        hist.append(row)
        print(json.dumps(row), flush=True)
        (out / "history.jsonl").open("a").write(json.dumps(row) + "\n")
        if vam.get("TF1", -1) > best_tf1:
            best_tf1 = vam.get("TF1", -1)
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "epoch": ep},
                       out / "best.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "epoch": cfg.epochs - 1},
               out / "last.pt")

    if len(te):
        def rnd(d):
            return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
        results = {"last": run_epoch(model, te, cfg, device)}
        print("[test:last]", json.dumps(rnd(results["last"])), flush=True)
        best_path = out / "best.pt"
        if best_path.exists():  # test with the val-selected checkpoint too (viz uses best.pt)
            ck = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            results["best"] = run_epoch(model, te, cfg, device)
            results["best_epoch"] = ck["epoch"]
            print(f"[test:best ep{ck['epoch']}]", json.dumps(rnd(results["best"])), flush=True)
        (out / "test_metrics.json").write_text(json.dumps(results, indent=2))
    return hist


if __name__ == "__main__":
    main()
