#!/usr/bin/env python3
"""
Per-RESIDUE ESM-2 embeddings for the residue-level phase (Slurm/GPU).

Unlike 05 (which pools), this stores every residue's vector for the metal subset
+ Study B, so the residue probe (11) and microprotein scan (12) can work at the
level where metal coordination actually lives. fp16, flat residue table.

By default embeds the whole catalog (Study A gives the probe's residues; Study B
the scan). Array-shardable like 05.

Usage (GPU node):
  python 05b_embed_residues.py --model 650M --shard $SLURM_ARRAY_TASK_ID --nshards 20
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from embed_lib import models, store

ROOT = Path(__file__).resolve().parent


def batches_by_tokens(items, max_len, budget):
    items = sorted(items, key=lambda x: len(x[1]))
    batch, toks = [], 0
    for pid, seq in items:
        L = min(len(seq), max_len)
        if batch and toks + L > budget:
            yield batch
            batch, toks = [], 0
        batch.append((pid, seq[:max_len], L))
        toks += L
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.REGISTRY))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="default: L6 + last (early-vs-late emergence contrast)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    nl = models.REGISTRY[args.model][1]
    layers = args.layers or sorted({min(6, nl), nl})     # early + last
    max_len, budget = cfg["max_len"], cfg["batch_tokens"]

    cat = pd.read_parquet(ROOT / cfg["catalog"])[["protein_id", "seq"]]
    shard = cat.iloc[args.shard::args.nshards].reset_index(drop=True)
    print(f"model={args.model} layers={layers} device={args.device} "
          f"shard {args.shard}/{args.nshards}: {len(shard)} proteins")

    model, alphabet, bc = models.load(args.model, device=args.device)

    proteins = list(shard["protein_id"])
    pidx = {p: i for i, p in enumerate(proteins)}
    res_prot, res_pos, res_aa = [], [], []
    acc = {L: [] for L in layers}
    items = list(zip(shard["protein_id"], shard["seq"]))
    done = 0
    with torch.no_grad():
        for batch in batches_by_tokens(items, max_len, budget):
            labels = [b[0] for b in batch]
            seqs = [b[1] for b in batch]
            lens = [b[2] for b in batch]
            _, _, toks = bc(list(zip(labels, seqs)))
            toks = toks.to(args.device)
            out = model(toks, repr_layers=layers, return_contacts=False)
            reps = out["representations"]
            for bi, (pid, seq, L) in enumerate(zip(labels, seqs, lens)):
                # residues occupy token positions 1..L
                for pos in range(L):
                    res_prot.append(pidx[pid]); res_pos.append(pos); res_aa.append(seq[pos])
                for layer in layers:
                    acc[layer].append(reps[layer][bi, 1:L + 1].float().cpu().numpy())
            done += len(batch)
            if done % 256 < len(batch):
                print(f"  {done}/{len(shard)} proteins")

    arrays = {L: np.concatenate(v, axis=0) for L, v in acc.items()}
    out_dir = ROOT / cfg["embed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"esm2_{args.model}_res_shard{args.shard}of{args.nshards}.h5"
    store.write_residues(out_dir / tag, proteins, res_prot, res_pos, res_aa, arrays,
                         model=models.REGISTRY[args.model][0], dim=models.REGISTRY[args.model][2],
                         layers=layers, max_len=max_len)
    print(f"wrote {out_dir/tag}: {len(res_prot)} residues, layers {layers}")


if __name__ == "__main__":
    main()
