#!/usr/bin/env python3
"""
Merge per-shard embedding HDF5 files (from 05_embed_esm2.py array jobs) into one
file per model: embeddings/esm2_<model>.h5, preserving the store schema.

Usage:  python 06_merge_embeddings.py --model 650M
"""
import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from embed_lib import store

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    edir = ROOT / cfg["embed_dir"]

    shards = sorted(glob.glob(str(edir / f"esm2_{args.model}_shard*of*.h5")))
    if not shards:
        raise SystemExit(f"no shards found for model {args.model} in {edir}")
    print(f"merging {len(shards)} shards for {args.model}")

    layers, poolings = store.available(shards[0])
    with h5py.File(shards[0], "r") as h:
        model = h.attrs["model"]; dim = int(h.attrs["dim"]); max_len = int(h.attrs["max_len"])

    ids, lengths, trunc = [], [], []
    arrays = {(L, p): [] for L in layers for p in poolings}
    for s in shards:
        with h5py.File(s, "r") as h:
            ids.extend(x.decode() if isinstance(x, bytes) else x for x in h["protein_ids"][:])
            lengths.extend(h["length"][:].tolist())
            trunc.extend(h["truncated"][:].tolist())
            for L in layers:
                for p in poolings:
                    arrays[(L, p)].append(h[f"L{L}/{p}"][:])
    arrays = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}

    out = edir / f"esm2_{args.model}.h5"
    store.write(out, ids, lengths, trunc, arrays, model=model, dim=dim,
                layers=layers, poolings=poolings, max_len=max_len)
    print(f"wrote {out}: {len(ids)} proteins, layers={layers}, poolings={poolings}")


if __name__ == "__main__":
    main()
