#!/usr/bin/env python3
"""Merge per-residue embedding shards (05b array jobs) -> esm2_<model>_res.h5."""
import argparse
import glob
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
    shards = sorted(glob.glob(str(edir / f"esm2_{args.model}_res_shard*of*.h5")))
    if not shards:
        raise SystemExit(f"no residue shards for {args.model}")
    layers = store.residue_layers(shards[0])
    with h5py.File(shards[0], "r") as h:
        model, dim, max_len = h.attrs["model"], int(h.attrs["dim"]), int(h.attrs["max_len"])

    proteins, res_prot, res_pos, res_aa = [], [], [], []
    arrays = {L: [] for L in layers}
    offset = 0
    for s in shards:
        with h5py.File(s, "r") as h:
            ps = [x.decode() if isinstance(x, bytes) else x for x in h["proteins"][:]]
            proteins.extend(ps)
            res_prot.append(h["res_protein"][:] + offset)
            res_pos.append(h["res_position"][:])
            res_aa.append(h["res_aa"][:])
            for L in layers:
                arrays[L].append(h[f"L{L}"][:])
            offset += len(ps)
    store.write_residues(
        edir / f"esm2_{args.model}_res.h5", proteins,
        np.concatenate(res_prot), np.concatenate(res_pos),
        np.concatenate([a.astype("S1") for a in res_aa]),
        {L: np.concatenate(v) for L, v in arrays.items()},
        model=model, dim=dim, layers=layers, max_len=max_len)
    print(f"merged {len(shards)} shards -> esm2_{args.model}_res.h5 "
          f"({offset} proteins, {sum(len(x) for x in res_pos)} residues)")


if __name__ == "__main__":
    main()
