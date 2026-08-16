#!/usr/bin/env python3
"""
Residue-level metal-coordination probe (Study A').

Positives = coordinating residues (residue_labels.parquet); negatives =
non-coordinating residues, subsampled to `--neg-ratio`x. Leakage-safe:
GroupKFold on cluster30 (a protein's residues never split across folds).

Three questions, each with logistic + MLP, vs a residue-identity baseline:
  main        : all residues                 -> does the embedding find sites at all?
  aa_matched  : only H/C/D/E residues        -> beyond "is it a His/Cys?" (the key control)
  identity    : 20-D one-hot AA (logistic)   -> the baseline both must beat

--emergence sweeps logistic on the aa_matched control across every residue
embedding file / layer present -> where does *site* signal emerge (size & depth)?

Usage:
  python 11_residue_probe.py --model 650M --layer 33
  python 11_residue_probe.py --emergence
Outputs: results/residue/residue_probe.{csv,md}
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from embed_lib import probe, store

ROOT = Path(__file__).resolve().parent
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
CANDIDATE = set("HCDE")


def build_table(h5, layer, catalog, labels):
    r = store.read_residues(h5, layer)
    pid = np.array(r["proteins"])[r["res_protein"]]
    meta = pd.DataFrame({"protein_id": pid, "position": r["position"], "aa": r["aa"]})
    cat = catalog.set_index("protein_id")[["study", "label", "class", "cluster30"]]
    meta = meta.join(cat, on="protein_id")
    coord = set(zip(labels["protein_id"], labels["position"]))
    meta["y"] = [(p, q) in coord for p, q in zip(meta["protein_id"], meta["position"])]
    return meta, r["emb"]


def onehot_aa(aas):
    X = np.zeros((len(aas), 20), np.float32)
    for i, a in enumerate(aas):
        j = AA_IDX.get(a)
        if j is not None:
            X[i, j] = 1.0
    return X


def subsample(meta, mask, neg_ratio, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.where(mask & meta["y"].to_numpy())[0]
    neg = np.where(mask & ~meta["y"].to_numpy())[0]
    if len(neg) > neg_ratio * len(pos):
        neg = rng.choice(neg, int(neg_ratio * len(pos)), replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return idx


def run_block(meta, emb, idx, tag, splits, rows):
    y = meta["y"].to_numpy()[idx].astype(int)
    g = meta["cluster30"].astype(str).to_numpy()[idx]
    X = emb[idx]
    log = probe.cluster_cv(X, y, g, n_splits=splits)
    mlp = probe.mlp_cluster_cv(X, y, g, n_splits=splits)
    base = probe.cluster_cv(onehot_aa(meta["aa"].to_numpy()[idx]), y, g, n_splits=splits)
    for name, r in [("embed-logistic", log), ("embed-mlp", mlp), ("identity-baseline", base)]:
        rows.append({"subset": tag, "probe": name, "n_pos": int(y.sum()),
                     "n_total": len(y), **{k: r[k] for k in
                     ("auroc_mean", "auroc_std", "mcc_mean", "n_groups")}})
        print(f"  {tag:11} {name:18} AUROC {r['auroc_mean']:.3f}+-{r['auroc_std']:.3f} "
              f"MCC {r['mcc_mean']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="650M")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--emergence", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    catalog = pd.read_parquet(ROOT / cfg["catalog"])
    labels = pd.read_parquet(ROOT / "catalog" / "residue_labels.parquet")
    edir = ROOT / cfg["embed_dir"]
    out = ROOT / "results" / "residue"; out.mkdir(parents=True, exist_ok=True)

    rows = []
    h5 = edir / f"esm2_{args.model}_res.h5"
    layer = args.layer or max(store.residue_layers(h5))
    print(f"== residue probe: {args.model} L{layer} (Study A residues) ==")
    meta, emb = build_table(h5, layer, catalog, labels)
    A = (meta["study"] == "A").to_numpy()
    run_block(meta, emb, subsample(meta, A, args.neg_ratio), "main", args.splits, rows)
    aa_mask = A & meta["aa"].isin(CANDIDATE).to_numpy()
    run_block(meta, emb, subsample(meta, aa_mask, args.neg_ratio), "aa_matched", args.splits, rows)

    if args.emergence:
        print("\n== emergence: logistic on aa_matched control, all models/layers ==")
        for f in sorted(glob.glob(str(edir / "esm2_*_res.h5"))):
            f = Path(f); tag = f.stem.replace("esm2_", "").replace("_res", "")
            for L in store.residue_layers(f):
                m, e = build_table(f, L, catalog, labels)
                aam = (m["study"] == "A").to_numpy() & m["aa"].isin(CANDIDATE).to_numpy()
                idx = subsample(m, aam, args.neg_ratio)
                r = probe.cluster_cv(e[idx], m["y"].to_numpy()[idx].astype(int),
                                     m["cluster30"].astype(str).to_numpy()[idx], n_splits=args.splits)
                rows.append({"subset": "aa_matched", "probe": f"emergence:{tag}:L{L}",
                             "n_pos": int(m["y"].to_numpy()[idx].sum()), "n_total": len(idx),
                             "auroc_mean": r["auroc_mean"], "auroc_std": r["auroc_std"],
                             "mcc_mean": r["mcc_mean"], "n_groups": r["n_groups"]})
                print(f"  {tag:5} L{L:<3} aa_matched logistic AUROC {r['auroc_mean']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(out / "residue_probe.csv", index=False)
    md = ["# Residue-level metal-coordination probe", "",
          "GroupKFold on cluster30. `main` = all residues; `aa_matched` = H/C/D/E only "
          "(the 'beyond residue identity' test). Embedding must beat identity-baseline.", "",
          "| subset | probe | AUROC | MCC | n_pos | n_total |", "|---|---|--:|--:|--:|--:|"]
    for _, r in res.iterrows():
        md.append(f"| {r['subset']} | {r['probe']} | {r['auroc_mean']:.3f}±{r['auroc_std']:.3f} "
                  f"| {r['mcc_mean']:.3f} | {r['n_pos']} | {r['n_total']} |")
    (out / "residue_probe.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {out/'residue_probe.csv'} and .md")


if __name__ == "__main__":
    main()
