#!/usr/bin/env python3
"""
Does representing residues by the multi-ion MLP's HIDDEN layer (a supervised,
metal-shaped reprojection) collapse the microprotein "island" that the raw-embedding
site-atlas (14) suffered from?

Builds the top-k mean-pool site-atlas twice — once on RAW ESM residue embeddings,
once on the MLP hidden layer — and quantifies how far microproteins sit from the
metal-binder cloud in each. Headline = does the island collapse (microproteins move
toward the metal centroid) under the supervised representation?

Honesty notes:
- The hidden layer can't add information (its ceiling is the discriminate-AUROC);
  this is about *de-confounding domain shift*, not manufacturing ion signal.
- Study A residues are scored in-sample by the final model, Study B out-of-sample,
  so microproteins are if anything DISADVANTAGED — an island collapse here is a
  conservative result. Knowns ion-silhouette is reported out-of-fold (honest).

Usage: python 16_hidden_atlas.py --model 650M --layer 33 --kmain 3
Outputs: results/hidden_atlas/{report.md, site_atlas_raw_vs_hidden.png}
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, normalize

from embed_lib import probe, store

ROOT = Path(__file__).resolve().parent
IONS = ["Zn", "Cu", "Fe", "Mn", "Co", "Ni"]
CANDIDATE = set("HCDE")
# TOP-IDP-style composition proxy for intrinsic disorder (no external predictor):
DISORDER_PROMOTING = set("ARGQSEKP")
ORDER_PROMOTING = set("WCFIYVLN")


def disorder_proxy(seq):
    """Composition heuristic: (disorder-promoting - order-promoting) / length.
    Higher = more disorder-prone. A rough proxy to label an atlas axis, not a predictor."""
    if not seq:
        return 0.0
    d = sum(c in DISORDER_PROMOTING for c in seq)
    o = sum(c in ORDER_PROMOTING for c in seq)
    return (d - o) / len(seq)


def build_labels(meta, coord_ions):
    n = len(meta); Y = np.zeros((n, 6), np.float32); M = np.zeros((n, 6), np.float32)
    pids = meta["protein_id"].to_numpy(); pos = meta["position"].to_numpy()
    for r in range(n):
        s = coord_ions.get((pids[r], int(pos[r])), ())
        isc = len(s) > 0
        for i, ion in enumerate(IONS):
            ins = ion in s
            Y[r, i] = 1.0 if ins else 0.0
            M[r, i] = 1.0 if (ins or not isc) else 0.0
    return Y, M


def site_vectors(meta, feat, score, cand, k):
    """Per protein: mean of top-k candidate residues' feature vectors (by score)."""
    idx = np.where(cand)[0]
    df = pd.DataFrame({"protein_id": meta["protein_id"].to_numpy()[idx],
                       "row": idx, "score": score[idx]})
    pids, vecs = [], []
    for pid, g in df.groupby("protein_id", sort=False):
        sel = g.nlargest(k, "score")["row"].to_numpy()
        pids.append(pid); vecs.append(feat[sel].mean(0))
    return pids, np.vstack(vecs)


def island_stats(site, pids, cls_of, ci):
    """median cosine dist of micro / control to the metal centroid, and fraction
    of each closer to the micro centroid than the metal centroid."""
    S = normalize(site)
    cls = np.array([cls_of[p] for p in pids])
    metal = np.isin(cls, ["Zn", "Cu", "other_transition"])
    mc = normalize(S[metal].mean(0, keepdims=True))[0]
    uc = normalize(S[cls == "microprotein"].mean(0, keepdims=True))[0]
    out = {}
    for grp in ("microprotein", "size_matched_short"):
        g = S[cls == grp]
        dmetal = 1 - g @ mc
        dmicro = 1 - g @ uc
        out[grp] = (float(np.median(dmetal)), float((dmicro < dmetal).mean()))
    return out, cls, metal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="650M")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--kmain", type=int, default=3)
    ap.add_argument("--neg-ratio", type=float, default=5.0)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    catalog = pd.read_parquet(ROOT / cfg["catalog"])
    labels = pd.read_parquet(ROOT / "catalog" / "residue_labels.parquet")
    h5 = ROOT / cfg["embed_dir"] / f"esm2_{args.model}_res.h5"
    layer = args.layer or max(store.residue_layers(h5))
    out = ROOT / "results" / "hidden_atlas"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    r = store.read_residues(h5, layer)
    pid = np.array(r["proteins"])[r["res_protein"]]
    meta = pd.DataFrame({"protein_id": pid, "position": r["position"], "aa": r["aa"]})
    cat = catalog.set_index("protein_id")
    for c in ["study", "class", "cluster30"]:
        meta[c] = meta["protein_id"].map(cat[c])
    emb = r["emb"]
    coord_ions = labels.groupby(["protein_id", "position"])["ion"].agg(set).to_dict()
    Y, M = build_labels(meta, coord_ions)
    is_coord = Y.sum(1) > 0
    cand = np.isin(meta["aa"].to_numpy(), list(CANDIDATE))
    A = (meta["study"] == "A").to_numpy()

    # training set: all coord + subsampled non-coord (Study A candidate residues)
    coordA = np.where(A & is_coord & cand)[0]
    ncA = np.where(A & ~is_coord & cand)[0]
    keepn = min(len(ncA), int(args.neg_ratio * len(coordA)))
    tr = np.concatenate([coordA, rng.choice(ncA, keepn, replace=False)]); rng.shuffle(tr)
    scaler = StandardScaler().fit(emb[tr])
    Xtr = scaler.transform(emb[tr])
    groups = meta["cluster30"].astype(str).to_numpy()[tr]

    # OOF hidden for knowns ion-silhouette (honest)
    k = min(args.splits, len(np.unique(groups)))
    oof_h = None
    Ytr, Mtr = Y[tr], M[tr]
    for a, b in GroupKFold(k).split(Xtr, groups=groups):
        pr = probe.train_ion_mlp(Xtr[a], Ytr[a], Mtr[a])
        _, hh = pr(Xtr[b])
        if oof_h is None:
            oof_h = np.zeros((len(tr), hh.shape[1]))
        oof_h[b] = hh

    # final model -> probs + hidden for ALL candidate residues
    pr = probe.train_ion_mlp(Xtr, Y[tr], M[tr])
    probs = np.zeros((len(meta), 6)); hid = None
    p_all, h_all = pr(scaler.transform(emb[cand]))
    probs[cand] = p_all
    hid = np.zeros((len(meta), h_all.shape[1])); hid[cand] = h_all
    sel_score = np.where(cand, probs.max(1), -np.inf)   # coordination-ness for top-k

    # build site-vectors: raw vs hidden (same residues selected)
    pids, site_raw = site_vectors(meta, emb, sel_score, cand, args.kmain)
    _, site_hid = site_vectors(meta, hid, sel_score, cand, args.kmain)
    cls_of = cat["class"].to_dict()
    hl_of = (cat["highlight"].fillna(False).to_dict() if "highlight" in cat.columns else {})
    seq_of = cat["seq"].to_dict()

    # per-protein axis covariates (to label what the hidden PCA axes are)
    dfp = pd.DataFrame({"protein_id": meta["protein_id"].to_numpy()[np.where(cand)[0]],
                        "score": sel_score[cand]})
    mp_map = dfp.groupby("protein_id")["score"].apply(
        lambda s: s.nlargest(args.kmain).mean()).to_dict()
    prot_len = np.array([len(seq_of[p]) for p in pids], float)
    prot_dis = np.array([disorder_proxy(seq_of[p]) for p in pids], float)
    prot_mp = np.array([mp_map[p] for p in pids], float)

    # island stats + knowns ion silhouette (OOF hidden vs raw)
    dom_ion = labels.groupby("protein_id")["ion"].agg(lambda s: s.value_counts().idxmax())
    rows = []
    for name, site in [("raw", site_raw), ("hidden", site_hid)]:
        isl, cls, metal = island_stats(site, pids, cls_of, None)
        rows.append({"repr": name,
                     "micro_dist_to_metal": round(isl["microprotein"][0], 3),
                     "micro_frac_closer_to_micro": round(isl["microprotein"][1], 3),
                     "control_dist_to_metal": round(isl["size_matched_short"][0], 3),
                     "control_frac_closer_to_micro": round(isl["size_matched_short"][1], 3)})
    isldf = pd.DataFrame(rows)

    # knowns ion silhouette: raw (top-k raw site) vs hidden (OOF)
    known = pd.DataFrame({"protein_id": pids}).assign(cls=[cls_of[p] for p in pids])
    known["ion"] = known["protein_id"].map(dom_ion)
    kmask = known["cls"].isin(["Zn", "Cu", "other_transition"]).to_numpy() & known["ion"].notna().to_numpy()
    sil_raw = silhouette_score(StandardScaler().fit_transform(site_raw[kmask]), known["ion"][kmask])
    sil_hid = silhouette_score(StandardScaler().fit_transform(site_hid[kmask]), known["ion"][kmask])

    cls = np.array([cls_of[p] for p in pids])
    metal = np.isin(cls, ["Zn", "Cu", "other_transition"])
    hlmask = np.array([hl_of.get(p, False) for p in pids])
    xy_raw = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(site_raw))
    xy_hid = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(site_hid))

    # figure: raw vs hidden PCA, microproteins overlaid
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, name, xy in [(axes[0], "raw embeddings", xy_raw), (axes[1], "MLP hidden", xy_hid)]:
        ax.scatter(xy[cls == "non_metal", 0], xy[cls == "non_metal", 1], s=7, c="#dddddd", alpha=0.5, label="non-metal", linewidths=0)
        ax.scatter(xy[metal, 0], xy[metal, 1], s=9, c="#1f77b4", alpha=0.6, label="metal binder", linewidths=0)
        ax.scatter(xy[cls == "microprotein", 0], xy[cls == "microprotein", 1], s=20, marker="x", c="#d62728", alpha=0.8, label="microprotein", linewidths=0.8)
        if hlmask.any():
            ax.scatter(xy[hlmask, 0], xy[hlmask, 1], s=150, marker="*", facecolor="none",
                       edgecolor="black", linewidths=1.4, label="MetalNet2", zorder=6)
        ax.set_title(f"site-atlas on {name}"); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7)
    fig.suptitle(f"Top-{args.kmain} site-atlas: does the microprotein island collapse under the supervised representation?")
    fig.tight_layout(); fig.savefig(out / "site_atlas_raw_vs_hidden.png", dpi=150); plt.close(fig)

    # what ARE the hidden-atlas axes? colour by metal-propensity, length, disorder
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, vals, title in [(axes[0], prot_mp, "metal-propensity (top-k score)"),
                            (axes[1], np.log10(prot_len), "log10(length)"),
                            (axes[2], prot_dis, "disorder proxy")]:
        sc = ax.scatter(xy_hid[:, 0], xy_hid[:, 1], c=vals, s=10, cmap="viridis", linewidths=0)
        if hlmask.any():
            ax.scatter(xy_hid[hlmask, 0], xy_hid[hlmask, 1], s=150, marker="*", facecolor="none", edgecolor="red", linewidths=1.4)
        ax.set_title(f"hidden atlas by {title}"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax)
    fig.tight_layout(); fig.savefig(out / "hidden_atlas_axes.png", dpi=150); plt.close(fig)

    # quantify: which covariate does each PC track?
    axis_corr = {}
    for lab, v in [("metal_propensity", prot_mp), ("length", prot_len), ("disorder", prot_dis)]:
        axis_corr[lab] = (round(spearmanr(xy_hid[:, 0], v).statistic, 3),
                          round(spearmanr(xy_hid[:, 1], v).statistic, 3))

    md = ["# Hidden-representation site-atlas (raw vs MLP hidden)", "",
          f"{args.model} L{layer}, top-{args.kmain} mean-pool. Headline: does the microprotein "
          "island collapse (move toward the metal centroid) under the supervised hidden repr?", "",
          "## Microprotein-island metric (cosine)", "",
          "| repr | micro→metal dist | micro frac-closer-to-micro | control→metal dist | control frac |",
          "|---|--:|--:|--:|--:|"]
    for _, x in isldf.iterrows():
        md.append(f"| {x['repr']} | {x['micro_dist_to_metal']} | {x['micro_frac_closer_to_micro']} "
                  f"| {x['control_dist_to_metal']} | {x['control_frac_closer_to_micro']} |")
    md += ["", "Island collapses if `hidden` LOWERS micro→metal dist and micro frac-closer-to-micro "
           "vs `raw` (microproteins pulled into the metal cloud). Study A is in-sample / Study B "
           "out-of-sample, so this is a conservative test.", "",
           "## Knowns ion-type silhouette (bounded by discriminate-AUROC)",
           f"- raw site-vectors: {sil_raw:.3f}", f"- hidden (OOF): {sil_hid:.3f}",
           "(near 0 = no clean ion clusters; hidden can't exceed what discrimination allows.)",
           "", "## What are the hidden-atlas axes? (Spearman |corr| with PC1 / PC2)",
           "The metal-propensity axis is the robust signal; a length/disorder axis is the confound to regress out.", ""]
    md += ["| covariate | PC1 | PC2 |", "|---|--:|--:|"]
    for lab, (c1, c2) in axis_corr.items():
        md.append(f"| {lab} | {c1} | {c2} |")
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}/ (report.md, site_atlas_raw_vs_hidden.png)")


if __name__ == "__main__":
    main()
