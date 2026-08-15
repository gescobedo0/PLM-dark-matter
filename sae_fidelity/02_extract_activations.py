#!/usr/bin/env python
"""Extract ESMC-6B layer-60 residue activations for the subsample (GPU step).

Resumable: already-extracted ids (per the manifest) are skipped, so a Colab
disconnect just means re-running this cell. Point --data at a Google Drive mount
(or set SAE_DATA_DIR) so the cache survives the session.

Usage (Colab, after pip install -r requirements.txt and HF login):
  python 02_extract_activations.py
  python 02_extract_activations.py --limit 300      # smoke: first 300 proteins
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sae_lib.config import load_config
from sae_lib import esmc, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap #proteins (smoke)")
    ap.add_argument("--shard-size", type=int, default=512, help="proteins per shard")
    ap.add_argument("--backend", default="hf", choices=["hf", "esm"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    sub = pd.read_parquet(cfg["paths"]["subsample"])
    if args.limit:
        sub = sub.groupby("group", group_keys=False).head(
            max(1, args.limit // sub["group"].nunique()))

    out_dir = Path(cfg["paths"]["activations_dir"])
    already = store.done_ids(out_dir)
    todo = sub[~sub["id"].isin(already)]
    print(f"{len(sub)} proteins in subsample; {len(already)} already done; "
          f"{len(todo)} to extract")
    if todo.empty:
        return

    lm = esmc.load_esmc(cfg["hf"]["base_model"], cfg["extraction"]["dtype"],
                        backend=args.backend)
    print(f"loaded {cfg['hf']['base_model']} ({lm.backend}); "
          f"hidden_states count = {lm.n_hidden_states}")

    manifest = store.read_manifest(out_dir)
    shard_idx = manifest["n_shards"]
    items = list(zip(todo["id"], todo["seq"]))

    buf_ids, buf_len, buf_acts = [], [], []

    def flush():
        nonlocal shard_idx, buf_ids, buf_len, buf_acts
        if not buf_ids:
            return
        store.save_activation_shard(out_dir, shard_idx, buf_ids, buf_len,
                                    np.concatenate(buf_acts, axis=0))
        manifest["done_ids"].extend(buf_ids)
        shard_idx += 1
        manifest["n_shards"] = shard_idx
        store.write_manifest(out_dir, manifest)
        print(f"  wrote shard {shard_idx - 1} ({len(buf_ids)} proteins)")
        buf_ids, buf_len, buf_acts = [], [], []

    n = 0
    for pid, acts in esmc.extract_layer(
            lm, items, cfg["sae"]["layer_index"],
            cfg["extraction"]["max_batch_tokens"],
            cfg["extraction"]["drop_special_tokens"]):
        buf_ids.append(pid)
        buf_len.append(acts.shape[0])
        buf_acts.append(acts)
        n += 1
        if len(buf_ids) >= args.shard_size:
            flush()
    flush()
    print(f"done: extracted {n} proteins into {out_dir}")


if __name__ == "__main__":
    main()
