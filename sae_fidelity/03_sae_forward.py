#!/usr/bin/env python
"""Run the SAE over cached activations: reconstruction (FVU/MSE) + features.

Two passes over the activation shards:
  pass 1  accumulate the baseline activation mean from the BACKGROUND group
          (the SAE's home distribution) — the FVU denominator.
  pass 2  encode/decode every residue, compute per-residue MSE and FVU, max-pool
          features per protein, and accumulate per-group feature usage and an
          active-magnitude reservoir.

Outputs (under paths.sae_features):
  proteins.parquet   id, group, subclass, n_res, protein_fvu, protein_mse, mean_mag
  pooled.npz         ids, pooled max features (P, codebook) float16
  group_stats.npz    per-group usage vectors + magnitude reservoirs
  baseline.npz       baseline mean + which group defined it

Uses torch on CUDA when available; otherwise the (slower, tested) numpy path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sae_lib.config import load_config
from sae_lib import sae, store, pooling


def _encode_decode_metrics(x, w, baseline, backend):
    """Return (idx, val, mse, fvu) for residue block x (n, hidden)."""
    if backend == "torch":
        import torch
        xt = torch.from_numpy(x).to("cuda", torch.float32)
        We = torch.from_numpy(w.W_enc).to("cuda", torch.float32)
        Wd = torch.from_numpy(w.W_dec).to("cuda", torch.float32)
        be = torch.from_numpy(w.b_enc).to("cuda", torch.float32)
        bd = torch.from_numpy(w.b_dec).to("cuda", torch.float32)
        base = torch.from_numpy(baseline).to("cuda", torch.float32)
        pre = torch.relu((xt - bd) @ We + be)
        val, idx = torch.topk(pre, w.topk, dim=1)
        # decode: gather the k active decoder rows per residue, weight by val
        dec = torch.einsum("nk,nkh->nh", val, Wd[idx]) + bd
        mse = ((xt - dec) ** 2).sum(1)
        fvu = mse / ((xt - base) ** 2).sum(1).clamp_min(1e-12)
        return (idx.cpu().numpy(), val.cpu().numpy(),
                mse.cpu().numpy(), fvu.cpu().numpy())
    # numpy fallback
    idx, val = sae.topk_encode(x.astype(np.float64), w)
    dec = sae.decode(idx, val, w)
    mse = np.sum((x - dec) ** 2, axis=1)
    fvu = mse / np.maximum(np.sum((x - baseline) ** 2, axis=1), 1e-12)
    return idx, val, mse, fvu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--reservoir", type=int, default=200_000,
                    help="magnitude samples kept per group")
    args = ap.parse_args()

    cfg = load_config(args.config)
    codebook = cfg["sae"]["codebook"]
    sub = pd.read_parquet(cfg["paths"]["subsample"])
    grp = dict(zip(sub["id"], sub["group"]))
    scl = dict(zip(sub["id"], sub["subclass"]))

    act_dir = Path(cfg["paths"]["activations_dir"])
    shards = store.shard_paths(act_dir)
    if not shards:
        raise SystemExit(f"no activation shards in {act_dir}; run 02 first")

    try:
        import torch  # noqa
        backend = "torch" if torch.cuda.is_available() else "numpy"
    except ImportError:
        backend = "numpy"
    print(f"SAE forward backend: {backend}")

    # --- pass 1: baseline mean from background residues -------------------
    hidden = cfg["sae"]["hidden_dim"]
    bg_sum = np.zeros(hidden, dtype=np.float64)
    bg_n = 0
    all_sum = np.zeros(hidden, dtype=np.float64)
    all_n = 0
    for sp in shards:
        for pid, acts in store.iter_proteins(sp):
            a = acts.astype(np.float64)
            all_sum += a.sum(0); all_n += a.shape[0]
            if grp.get(pid) == "background":
                bg_sum += a.sum(0); bg_n += a.shape[0]
    if bg_n > 0:
        baseline = (bg_sum / bg_n).astype(np.float32); base_src = "background"
    else:
        baseline = (all_sum / max(all_n, 1)).astype(np.float32); base_src = "all"
    print(f"baseline mean from {base_src} ({bg_n or all_n} residues)")

    # --- load SAE ---------------------------------------------------------
    w = sae.load_sae_weights(cfg["hf"]["sae_repo"], cfg["sae"]["topk"],
                             hidden, codebook)
    print("loaded SAE weights:", w.W_enc.shape, w.W_dec.shape)

    # --- pass 2: reconstruction + features --------------------------------
    rng = np.random.default_rng(cfg["seed"])
    rows, pooled_ids, pooled_rows = [], [], []
    usage = {}                     # group -> (codebook,) summed activation
    mag_res = {}                   # group -> list of sampled magnitudes
    for sp in shards:
        for pid, acts in store.iter_proteins(sp):
            g = grp.get(pid, "unknown")
            idx, val, mse, fvu = _encode_decode_metrics(
                acts.astype(np.float32), w, baseline, backend)
            L = acts.shape[0]
            rows.append({"id": pid, "group": g, "subclass": scl.get(pid, ""),
                         "n_res": L, "protein_fvu": float(fvu.mean()),
                         "protein_mse": float(mse.mean()),
                         "mean_mag": float(val.mean())})
            pooled_ids.append(pid)
            pooled_rows.append(
                pooling.max_pool_sparse(idx, val, codebook, np.array([L]))[0]
                .astype(np.float16))
            u = usage.setdefault(g, np.zeros(codebook))
            np.add.at(u, idx.ravel(), val.ravel())
            r = mag_res.setdefault(g, [])
            flat = val.ravel()
            if len(r) < args.reservoir:
                r.extend(flat[:args.reservoir - len(r)].tolist())

    out_dir = Path(cfg["paths"]["sae_features"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / "proteins.parquet", index=False)
    np.savez(out_dir / "pooled.npz",
             ids=np.array(pooled_ids, dtype=object),
             pooled=np.stack(pooled_rows))
    np.savez(out_dir / "group_stats.npz",
             groups=np.array(list(usage), dtype=object),
             usage=np.stack([usage[g] for g in usage]),
             **{f"mag_{g}": np.array(mag_res[g], dtype=np.float32) for g in mag_res})
    np.savez(out_dir / "baseline.npz", baseline=baseline, source=base_src)
    print(f"wrote SAE features for {len(rows)} proteins -> {out_dir}")


if __name__ == "__main__":
    main()
