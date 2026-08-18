#!/usr/bin/env python3
"""
Is the embedding's "which metal" signal anything more than coordinating-residue
IDENTITY (composition)? PC2 of the site-atlas separates ions along a soft->hard
axis (Cu/Zn His/Cys ... Mn Asp/Glu), which is just residue composition — so test
whether ESM beats a composition-only baseline at the discriminate task.

For each ion: among COORDINATING residues, ion-X (positive) vs other-metal
(negative), leakage-safe GroupKFold on cluster30, comparing
  embedding   : logistic on the residue's ESM vector
  composition : logistic on the residue's 20-D amino-acid one-hot (identity only)
If embedding ~= composition, "which metal" is residue counting, not representation.

Also reports the per-ion coordinating-residue AA makeup (the soft<->hard chemistry
that drives the composition baseline).

Usage: python 17_composition_baseline.py --model 650M --layer 33
Outputs: results/composition/report.md, composition_vs_embedding.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from embed_lib import probe, store

ROOT = Path(__file__).resolve().parent
IONS = ["Zn", "Cu", "Fe", "Mn", "Co", "Ni"]
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}


def onehot(aas):
    X = np.zeros((len(aas), 20), np.float32)
    for i, a in enumerate(aas):
        j = AA_IDX.get(a)
        if j is not None:
            X[i, j] = 1.0
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="650M")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    catalog = pd.read_parquet(ROOT / cfg["catalog"])
    labels = pd.read_parquet(ROOT / "catalog" / "residue_labels.parquet")
    h5 = ROOT / cfg["embed_dir"] / f"esm2_{args.model}_res.h5"
    layer = args.layer or max(store.residue_layers(h5))
    out = ROOT / "results" / "composition"; out.mkdir(parents=True, exist_ok=True)

    r = store.read_residues(h5, layer)
    pid = np.array(r["proteins"])[r["res_protein"]]
    meta = pd.DataFrame({"protein_id": pid, "position": r["position"], "aa": r["aa"]})
    cat = catalog.set_index("protein_id")
    meta["study"] = meta["protein_id"].map(cat["study"])
    meta["cluster30"] = meta["protein_id"].map(cat["cluster30"])
    emb = r["emb"]
    coord_ions = labels.groupby(["protein_id", "position"])["ion"].agg(set).to_dict()

    # coordinating residues in Study A (the discriminate universe)
    is_coord = np.array([(p, int(q)) in coord_ions for p, q in zip(meta["protein_id"], meta["position"])])
    mask = is_coord & (meta["study"] == "A").to_numpy()
    idx = np.where(mask)[0]
    ion_sets = [coord_ions[(meta["protein_id"].iat[i], int(meta["position"].iat[i]))] for i in idx]
    groups = meta["cluster30"].astype(str).to_numpy()[idx]
    aa = meta["aa"].to_numpy()[idx]
    Xe, Xc = emb[idx], onehot(aa)
    print(f"coordinating Study A residues: {len(idx)}")

    rows, aa_rows = [], []
    for ion in IONS:
        pos = np.array([ion in s for s in ion_sets])
        if pos.sum() < 10 or (~pos).sum() < 10:
            rows.append({"ion": ion, "n_pos": int(pos.sum()), "embedding_auroc": float("nan"),
                         "composition_auroc": float("nan"), "delta": float("nan")})
            continue
        y = pos.astype(int)
        e = probe.cluster_cv(Xe, y, groups, n_splits=args.splits)["auroc_mean"]
        c = probe.cluster_cv(Xc, y, groups, n_splits=args.splits)["auroc_mean"]
        rows.append({"ion": ion, "n_pos": int(pos.sum()), "embedding_auroc": round(e, 3),
                     "composition_auroc": round(c, 3), "delta": round(e - c, 3)})
        # AA makeup of this ion's coordinating residues
        sub_aa = aa[pos]
        frac = {a: round(float(np.mean(sub_aa == a)), 2) for a in "HCDE"}
        frac["other"] = round(float(np.mean(~np.isin(sub_aa, list("HCDE")))), 2)
        aa_rows.append({"ion": ion, **frac})
    res = pd.DataFrame(rows)
    res.to_csv(out / "composition_vs_embedding.csv", index=False)
    aa_df = pd.DataFrame(aa_rows)

    md = ["# Which-metal: embedding vs composition baseline", "",
          f"{args.model} L{layer}. Discriminate task (ion-X vs other-metal, coordinating "
          "residues), GroupKFold on cluster30.", "",
          "**If embedding ≈ composition, 'which metal' is residue identity, not representation.**", "",
          "| ion | n_pos | embedding | composition | Δ (emb−comp) |", "|---|--:|--:|--:|--:|"]
    for _, x in res.iterrows():
        md.append(f"| {x['ion']} | {x['n_pos']} | {x['embedding_auroc']} | "
                  f"{x['composition_auroc']} | {x['delta']} |")
    md += ["", "## Coordinating-residue AA makeup per ion (the soft↔hard chemistry)",
           "His/Cys = soft (N/S); Asp/Glu = hard (O). This is what the composition baseline uses.", "",
           "| ion | H | C | D | E | other |", "|---|--:|--:|--:|--:|--:|"]
    for _, x in aa_df.iterrows():
        md.append(f"| {x['ion']} | {x['H']} | {x['C']} | {x['D']} | {x['E']} | {x['other']} |")
    valid = res.dropna(subset=["delta"])
    md += ["", f"Mean Δ (embedding − composition) over ions with data: "
           f"**{valid['delta'].mean():+.3f}**. "
           + ("Embedding adds real which-metal signal beyond identity."
              if valid["delta"].mean() > 0.05 else
              "Embedding barely beats composition → which-metal is largely residue identity.")]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}/ (report.md, composition_vs_embedding.csv)")


if __name__ == "__main__":
    main()
