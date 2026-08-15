#!/usr/bin/env python3
"""
Extract pooled ESM-2 embeddings for the catalog (Slurm/GPU).

Loads one model per job (spec: load once, stream many short sequences), sweeps
the candidate layers from config, mean+max pools each, and writes an HDF5 shard.
Designed for array jobs: pass --shard i --nshards K to process a contiguous
slice of the catalog; merge shards afterwards with 06_merge_embeddings.py.

Usage (on the GPU node):
  python 05_embed_esm2.py --model 650M --shard $SLURM_ARRAY_TASK_ID --nshards 20
  python 05_embed_esm2.py --model 35M   --shard 0 --nshards 1      # dev, one job
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from embed_lib import models, store
from embed_lib.pooling import pool

ROOT = Path(__file__).resolve().parent


def load_catalog(cfg):
    df = pd.read_parquet(ROOT / cfg["catalog"])
    return df[["protein_id", "seq"]].reset_index(drop=True)


def batches_by_tokens(items, max_len, budget):
    """Yield lists of (id, seq) length-sorted, capped so sum(len) <= budget."""
    items = sorted(items, key=lambda x: len(x[1]))
    batch, toks = [], 0
    for pid, seq in items:
        L = min(len(seq), max_len)
        if batch and toks + L > budget:
            yield batch
            batch, toks = [], 0
        batch.append((pid, seq[:max_len], len(seq) > max_len, L))
        toks += L
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.REGISTRY))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg = cfg["models"][args.model]
    layers = mcfg["layers"]
    poolings = cfg["poolings"]
    max_len = cfg["max_len"]
    budget = cfg["batch_tokens"]

    cat = load_catalog(cfg)
    shard = cat.iloc[args.shard::args.nshards].reset_index(drop=True)
    print(f"model={args.model} device={args.device} "
          f"shard {args.shard}/{args.nshards}: {len(shard)}/{len(cat)} sequences")

    model, alphabet, bc = models.load(args.model, device=args.device)

    ids, lengths, trunc = [], [], []
    acc = {(L, p): [] for L in layers for p in poolings}
    items = list(zip(shard["protein_id"], shard["seq"]))
    done = 0
    with torch.no_grad():
        for batch in batches_by_tokens(items, max_len, budget):
            labels = [b[0] for b in batch]
            seqs = [b[1] for b in batch]
            was_trunc = [b[2] for b in batch]
            lens = torch.tensor([b[3] for b in batch])
            _, _, toks = bc(list(zip(labels, seqs)))
            toks = toks.to(args.device)
            out = model(toks, repr_layers=layers, return_contacts=False)
            reps = out["representations"]
            for L in layers:
                r = reps[L]
                for p in poolings:
                    acc[(L, p)].append(pool(r, lens, p).float().cpu().numpy())
            ids.extend(labels)
            lengths.extend([b[3] for b in batch])
            trunc.extend(was_trunc)
            done += len(batch)
            if done % 256 < len(batch):
                print(f"  {done}/{len(shard)}")

    arrays = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    out_dir = ROOT / cfg["embed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"esm2_{args.model}_shard{args.shard}of{args.nshards}.h5"
    store.write(out_dir / tag, ids, lengths, trunc, arrays,
                model=mcfg["name"], dim=mcfg["dim"], layers=layers,
                poolings=poolings, max_len=max_len)
    n_trunc = int(np.sum(trunc))
    print(f"wrote {out_dir/tag}: {len(ids)} proteins, {n_trunc} truncated "
          f"(> {max_len} aa)")


if __name__ == "__main__":
    main()
