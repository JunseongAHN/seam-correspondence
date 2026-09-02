"""Dataset: scan *_specification.json (GCD part layout agnostic), preprocess, pad+collate."""
from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import AutoSewConfig
from .features import pattern_to_tensors
from .gcd_parser import parse_specification
from .synthetic import make_set


def find_spec_files(data_dir, limit=None):
    files = sorted(Path(data_dir).rglob("*specification.json"))
    if limit:
        files = files[:limit]
    return files


class PatternDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples  # list of dicts from pattern_to_tensors

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    @classmethod
    def from_dir(cls, data_dir, cfg: AutoSewConfig, limit=None, verbose=True):
        files = find_spec_files(data_dir, limit)
        samples, failed = [], 0
        rng = random.Random(cfg.seed)
        for f in files:
            try:
                p = parse_specification(f)
                s = pattern_to_tensors(p, cfg, rng)
                if s["x"].shape[0] >= 2 and len(s["gt_pairs"]) > 0:
                    samples.append(s)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                if verbose and failed <= 5:
                    print(f"[parse-fail] {f}: {type(e).__name__}: {e}")
        if verbose:
            print(f"[dataset] parsed {len(samples)} ok, {failed} failed/empty of {len(files)} files")
        return cls(samples)

    @classmethod
    def from_synthetic(cls, n, cfg: AutoSewConfig, seed=0):
        rng = random.Random(seed)
        samples = []
        for name, spec in make_set(n, seed):
            p = parse_specification(spec, name=name)
            samples.append(pattern_to_tensors(p, cfg, rng))
        return cls(samples)

    def stats(self):
        Ms = [s["x"].shape[0] for s in self.samples]
        deg2 = sum(
            1 for s in self.samples
            if len(s["gt_pairs"]) and np.bincount(s["gt_pairs"].ravel()).max() >= 2
        )
        return {
            "n": len(self.samples),
            "M_min": int(min(Ms)), "M_max": int(max(Ms)), "M_mean": float(np.mean(Ms)),
            "patterns_with_multi_edge_gt": deg2,
        }


def split_dataset(ds: PatternDataset, val_frac=0.1, test_frac=0.1, seed=0):
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    n_test = int(len(idx) * test_frac)
    n_val = int(len(idx) * val_frac)
    test = [ds.samples[i] for i in idx[:n_test]]
    val = [ds.samples[i] for i in idx[n_test:n_test + n_val]]
    train = [ds.samples[i] for i in idx[n_test + n_val:]]
    return PatternDataset(train), PatternDataset(val), PatternDataset(test)


def collate(batch, device="cpu"):
    """Pad to M_max. Padded nodes: zero features, self-neighbors, mask False."""
    B = len(batch)
    Mmax = max(s["x"].shape[0] for s in batch)
    x = torch.zeros(B, Mmax, batch[0]["x"].shape[1])
    nbr = torch.zeros(B, Mmax, 2, dtype=torch.long)
    mask = torch.zeros(B, Mmax, dtype=torch.bool)
    gt_pairs, stitched, names = [], [], []
    for b, s in enumerate(batch):
        M = s["x"].shape[0]
        x[b, :M] = torch.from_numpy(s["x"])
        nbr[b, :M] = torch.from_numpy(s["nbr"])
        if M < Mmax:  # padded nodes point at themselves (safe gather)
            nbr[b, M:] = torch.arange(M, Mmax).unsqueeze(1)
        mask[b, :M] = True
        gt_pairs.append([tuple(p) for p in s["gt_pairs"]])
        stitched.append(s["stitched"])
        names.append(s["name"])
    return {
        "x": x.to(device), "nbr": nbr.to(device), "mask": mask.to(device),
        "gt_pairs": gt_pairs, "stitched": stitched, "names": names,
    }


def loader(ds, batch_size, shuffle, device="cpu", seed=0):
    idx = list(range(len(ds)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield collate([ds.samples[j] for j in idx[i:i + batch_size]], device)
