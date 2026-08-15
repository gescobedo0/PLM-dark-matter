#!/usr/bin/env python3
"""
Study A probe + composition baseline over the size / layer / pooling sweep.

For each available embeddings/esm2_<model>.h5, runs the cluster-aware logistic
probe (GroupKFold on split_group) for every (layer, pooling), and once runs the
20-D amino-acid-composition baseline. Reports AUROC + MCC. Because probe accuracy
is the only dimensionality-fair comparison, this table is what makes the size
sweep (35M vs 150M vs 650M) and the "where does metal signal emerge?" question
interpretable.

Usage:  python 07_probe.py                 # all models found in embeddings/
        python 07_probe.py --models 650M
Outputs: results/probe_results.csv, results/probe_results.md
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from embed_lib import probe, store

ROOT = Path(__file__).resolve().parent


def study_a(cfg):
    df = pd.read_parquet(ROOT / cfg["catalog"])
    a = df[df["study"] == "A"].copy()
    a["label"] = a["label"].astype(int)
    return a.set_index("protein_id")


def align(cat, ids):
    """Return boolean row selection + aligned (y, groups, seqs) for given ids."""
    keep = [i for i in ids if i in cat.index]
    sub = cat.loc[keep]
    return keep, sub["label"].to_numpy(), sub["split_group"].to_numpy(), sub["seq"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pcfg = cfg["probe"]
    cat = study_a(cfg)
    edir = ROOT / cfg["embed_dir"]

    rows = []
    # composition baseline (independent of embeddings)
    ids = list(cat.index)
    keep, y, groups, seqs = align(cat, ids)
    Xc = probe.aa_composition(seqs)
    assert probe.assert_no_group_leakage(groups, Xc, y, pcfg["n_splits"]), "leakage!"
    base = probe.cluster_cv(Xc, y, groups, n_splits=pcfg["n_splits"], C=pcfg["C"],
                            max_iter=pcfg["max_iter"], standardize=pcfg["standardize"])
    rows.append({"model": "baseline", "repr": "aa_composition_20d", "layer": "-",
                 "pooling": "-", **base})
    print(f"baseline (AA composition): AUROC {base['auroc_mean']:.3f} "
          f"MCC {base['mcc_mean']:.3f}  ({base['n_groups']} clusters)")

    files = ([edir / f"esm2_{m}.h5" for m in args.models] if args.models
             else [Path(p) for p in sorted(glob.glob(str(edir / "esm2_*.h5")))
                   if "shard" not in p])
    for f in files:
        if not f.exists():
            print(f"  (skip, not found: {f.name})")
            continue
        layers, poolings = store.available(f)
        tag = f.stem.replace("esm2_", "")
        for L in layers:
            for p in poolings:
                eids, arr = store.read(f, L, p)
                pos = {i: k for k, i in enumerate(eids)}
                keep, y, groups, _ = align(cat, eids)
                idx = [pos[i] for i in keep]
                X = arr[idx]
                r = probe.cluster_cv(X, y, groups, n_splits=pcfg["n_splits"],
                                     C=pcfg["C"], max_iter=pcfg["max_iter"],
                                     standardize=pcfg["standardize"])
                rows.append({"model": tag, "repr": f.stem, "layer": L,
                             "pooling": p, **r})
                print(f"  {tag:5} L{L:<3} {p:4}: AUROC {r['auroc_mean']:.3f}"
                      f"+-{r['auroc_std']:.3f}  MCC {r['mcc_mean']:.3f}")

    res = pd.DataFrame(rows)
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    res.to_csv(outdir / "probe_results.csv", index=False)
    # markdown, best row per model highlighted
    md = ["# Study A probe results", "",
          "AUROC / MCC, cluster-aware GroupKFold on 30% clusters. "
          "Embeddings must beat the composition baseline to justify 1280-D.", "",
          "| model | layer | pooling | AUROC | MCC | folds |",
          "|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        md.append(f"| {r['model']} | {r['layer']} | {r['pooling']} | "
                  f"{r['auroc_mean']:.3f}±{r['auroc_std']:.3f} | "
                  f"{r['mcc_mean']:.3f} | {r['n_folds']} |")
    (outdir / "probe_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {outdir/'probe_results.csv'} and .md")


if __name__ == "__main__":
    main()
