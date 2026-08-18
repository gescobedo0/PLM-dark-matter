#!/usr/bin/env python3
"""
Site-atlas: represent each protein by the mean of its top-k metal-coordinating-
scored residue embeddings (a length-independent, site-focused vector), then ask
whether metal SITES organize by ion type and whether microproteins land among
canonical ion clusters despite no alignable sequence similarity.

Pass-gate on KNOWNS first (Study A) before trusting any microprotein position:
  Panel A  ion-type separation  : leakage-safe multiclass ion probe (GroupKFold
                                  on cluster) + silhouette, per k=1..4.
  Panel B  anti-circularity     : (i) ion clusters orthogonal to the coordinating
                                  score; (ii) top-k vs RANDOM-CHED site-vectors —
                                  is the structure from selection or from CHED
                                  embeddings generally?
  Panel C  convergence          : same-ion / DIFFERENT-cluster nearest-neighbour
                                  purity (alignment-free co-location on knowns).
Only then: project Study B, assign each microprotein its nearest ion centroid in
full site-vector space, compare to the length-matched control, mark MetalNet2 ORFs.

Representation = MEAN-POOL top-k (no concat: order variance). PCA fit on all
points (visual only; the claim is the full-space nearest-ion metric).

Usage: python 14_site_atlas.py --model 650M --layer 33 --kmain 3
Outputs: results/site_atlas/{panel_stats.md, microprotein_ion.csv, *.png}
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, normalize

from embed_lib import probe, store

ROOT = Path(__file__).resolve().parent
CANDIDATE = set("HCDE")
IONS = ["Zn", "Cu", "Fe", "Mn", "Co", "Ni"]
ION_COLOR = {"Zn": "#1f77b4", "Cu": "#17becf", "Fe": "#8c564b", "Mn": "#2ca02c",
             "Co": "#9467bd", "Ni": "#e377c2"}


def build_residue_table(h5, layer, catalog, labels):
    r = store.read_residues(h5, layer)
    pid = np.array(r["proteins"])[r["res_protein"]]
    meta = pd.DataFrame({"protein_id": pid, "position": r["position"], "aa": r["aa"]})
    cat = catalog.set_index("protein_id")
    for c in ["study", "label", "class", "cluster30", "length"]:
        meta[c] = meta["protein_id"].map(cat[c])
    meta["highlight"] = (meta["protein_id"].map(cat["highlight"].fillna(False))
                         if "highlight" in cat.columns else False)
    coord = set(zip(labels["protein_id"], labels["position"]))
    meta["y"] = [(p, q) in coord for p, q in zip(meta["protein_id"], meta["position"])]
    return meta, r["emb"], cat


def topk_site_vectors(meta, emb, score, k, rng=None):
    """Per protein: mean of the k highest-scoring H/C/D/E residue embeddings.
    rng given -> pick k RANDOM candidate residues instead (control)."""
    cand = meta["aa"].isin(CANDIDATE).to_numpy()
    df = meta.loc[cand, ["protein_id"]].copy()
    df["row"] = np.where(cand)[0]
    df["score"] = score[cand]
    vecs, pids, ncand = [], [], []
    for pid, g in df.groupby("protein_id", sort=False):
        if rng is not None:
            sel = g.iloc[rng.permutation(len(g))[:k]]
        else:
            sel = g.nlargest(k, "score")
        vecs.append(emb[sel["row"].to_numpy()].mean(axis=0))
        pids.append(pid); ncand.append(len(g))
    return pd.DataFrame({"protein_id": pids, "n_candidate": ncand}), np.vstack(vecs)


def ion_probe(site, ions, groups):
    """Leakage-safe multiclass ion probe -> balanced accuracy + Zn-vs-rest AUROC."""
    k = min(5, len(np.unique(groups)))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    pred = cross_val_predict(clf, site, ions, groups=groups, cv=GroupKFold(k))
    bacc = balanced_accuracy_score(ions, pred)
    # Zn one-vs-rest AUROC via CV probabilities
    yz = (ions == "Zn").astype(int)
    proba = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"),
                              site, yz, groups=groups, cv=GroupKFold(k), method="predict_proba")[:, 1]
    return bacc, roc_auc_score(yz, proba)


def same_ion_diff_cluster_purity(site, ions, clusters):
    """For each protein, nearest neighbour from a DIFFERENT cluster; fraction
    sharing the same ion (alignment-free convergence). Compares to ion base-rate."""
    S = normalize(site)
    sim = S @ S.T
    n = len(ions)
    same = 0
    for i in range(n):
        order = np.argsort(-sim[i])
        for j in order:
            if j != i and clusters[j] != clusters[i]:
                same += int(ions[j] == ions[i])
                break
    base = sum((np.array(ions) == ion).mean() ** 2 for ion in set(ions))  # chance
    return same / n, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="650M")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--kmain", type=int, default=3)
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    catalog = pd.read_parquet(ROOT / cfg["catalog"])
    labels = pd.read_parquet(ROOT / "catalog" / "residue_labels.parquet")
    h5 = ROOT / cfg["embed_dir"] / f"esm2_{args.model}_res.h5"
    layer = args.layer or max(store.residue_layers(h5))
    out = ROOT / "results" / "site_atlas"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    meta, emb, cat = build_residue_table(h5, layer, catalog, labels)

    # --- train coordinating scorer on Study A aa_matched, score all H/C/D/E ---
    A_aa = (meta["study"] == "A").to_numpy() & meta["aa"].isin(CANDIDATE).to_numpy()
    ia = np.where(A_aa)[0]; ya = meta["y"].to_numpy()[ia].astype(int)
    pos, neg = ia[ya == 1], ia[ya == 0]
    if len(neg) > args.neg_ratio * len(pos):
        neg = rng.choice(neg, int(args.neg_ratio * len(pos)), replace=False)
    tr = np.concatenate([pos, neg])
    scaler = StandardScaler().fit(emb[tr])
    predict = probe.fit_mlp(scaler.transform(emb[tr]), meta["y"].to_numpy()[tr].astype(int))
    score = np.full(len(meta), np.nan)
    cand = meta["aa"].isin(CANDIDATE).to_numpy()
    score[cand] = predict(scaler.transform(emb[cand]))

    # per-protein dominant ion (knowns)
    dom_ion = labels.groupby("protein_id")["ion"].agg(lambda s: s.value_counts().idxmax())

    # --- k ablation: ion-separation panel (A) + random-CHED control (B) ---
    panel = []
    site_by_k = {}
    for k in (1, 2, 3, 4):
        info, site = topk_site_vectors(meta, emb, score, k)
        info["site_idx"] = np.arange(len(info))
        info = info.merge(catalog[["protein_id", "study", "class", "cluster30"]], on="protein_id")
        info["ion"] = info["protein_id"].map(dom_ion)
        site_std = StandardScaler().fit_transform(site)
        site_by_k[k] = (info, site, site_std)

        known = info[(info["study"] == "A") & info["class"].isin(["Zn", "Cu", "other_transition"])
                     & info["ion"].notna()]
        idx = known["site_idx"].to_numpy()
        bacc, znauc = ion_probe(site_std[idx], known["ion"].to_numpy(),
                                known["cluster30"].astype(str).to_numpy())
        sil = silhouette_score(site_std[idx], known["ion"].to_numpy())
        # random-CHED control
        _, rsite = topk_site_vectors(meta, emb, score, k, rng=np.random.default_rng(1))
        rstd = StandardScaler().fit_transform(rsite)
        rbacc, _ = ion_probe(rstd[idx], known["ion"].to_numpy(), known["cluster30"].astype(str).to_numpy())
        # convergence (panel C)
        purity, chance = same_ion_diff_cluster_purity(
            site_std[idx], known["ion"].to_numpy(), known["cluster30"].astype(str).to_numpy())
        panel.append({"k": k, "ion_bacc": round(bacc, 3), "ion_bacc_randomCHED": round(rbacc, 3),
                      "Zn_vs_rest_auroc": round(znauc, 3), "silhouette_ion": round(sil, 3),
                      "same_ion_diffcluster_purity": round(purity, 3), "chance_purity": round(chance, 3)})
    panel = pd.DataFrame(panel)

    # --- microprotein projection at kmain (full-space nearest ion centroid) ---
    info, site, site_std = site_by_k[args.kmain]
    known = info[(info["study"] == "A") & info["ion"].notna()]
    Sn = normalize(site_std)
    centroids = {ion: Sn[known[known["ion"] == ion]["site_idx"]].mean(0) for ion in IONS
                 if (known["ion"] == ion).sum() > 0}
    cen_ions = list(centroids); cen_mat = normalize(np.vstack([centroids[i] for i in cen_ions]))

    B = info[info["study"] == "B"].copy()
    bs = Sn[B["site_idx"].to_numpy()]
    sims = bs @ cen_mat.T
    B["nearest_ion"] = [cen_ions[j] for j in sims.argmax(1)]
    B["nearest_cos_dist"] = 1 - sims.max(1)
    B["highlight"] = B["protein_id"].map(cat["highlight"].fillna(False)) if "highlight" in cat.columns else False

    micro = B[B["class"] == "microprotein"]
    ctrl = B[B["class"] == "size_matched_short"]
    u, p = mannwhitneyu(micro["nearest_cos_dist"], ctrl["nearest_cos_dist"], alternative="less")
    micro.sort_values("nearest_cos_dist")[
        ["protein_id", "highlight", "n_candidate", "nearest_ion", "nearest_cos_dist"]
    ].to_csv(out / "microprotein_ion.csv", index=False)

    # --- domain-shift diagnostic: where do the human controls land? ---
    Sn_all = normalize(site_std)
    pdb_mask = (info["study"] == "A").to_numpy()
    pdb_cen = normalize(Sn_all[pdb_mask].mean(0)[None])[0]
    micro_cen = normalize(Sn_all[(info["class"] == "microprotein").to_numpy()].mean(0)[None])[0]
    ds_groups = {
        "PDB metal": (info["study"] == "A").to_numpy() & info["class"].isin(["Zn", "Cu", "other_transition"]).to_numpy(),
        "PDB non-metal": (info["study"] == "A").to_numpy() & (info["class"] == "non_metal").to_numpy(),
        "microprotein": (info["class"] == "microprotein").to_numpy(),
        "size_matched_short": (info["class"] == "size_matched_short").to_numpy(),
        "general_background": (info["class"] == "general_background").to_numpy()}
    ds_rows = []
    for name, m in ds_groups.items():
        if not m.any():
            continue
        d_pdb = 1 - Sn_all[m] @ pdb_cen
        d_mic = 1 - Sn_all[m] @ micro_cen
        ds_rows.append({"group": name, "n": int(m.sum()),
                        "med_dist_to_PDB": round(float(np.median(d_pdb)), 3),
                        "med_dist_to_micro": round(float(np.median(d_mic)), 3),
                        "frac_closer_to_micro": round(float((d_mic < d_pdb).mean()), 3)})
    ds = pd.DataFrame(ds_rows)

    # shared PCA coords for both atlas + domain-shift figures
    xy = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(site))

    # --- figures ---
    # atlas PCA (all points), knowns by ion, microproteins overlaid
    fig, ax = plt.subplots(figsize=(7, 6))
    nm = info["class"] == "non_metal"
    ax.scatter(xy[nm, 0], xy[nm, 1], s=8, c="#dddddd", alpha=0.5, label="non-metal", linewidths=0)
    for ion in IONS:
        m = (info["ion"] == ion).to_numpy() & (info["study"] == "A").to_numpy()
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=10, c=ION_COLOR[ion], alpha=0.7, label=ion, linewidths=0)
    mp = (info["class"] == "microprotein").to_numpy()
    ax.scatter(xy[mp, 0], xy[mp, 1], s=22, marker="x", c="black", alpha=0.8, label="microprotein", linewidths=0.8)
    hl = info["protein_id"].map(cat["highlight"].fillna(False)).to_numpy() if "highlight" in cat.columns else np.zeros(len(info), bool)
    if hl.any():
        ax.scatter(xy[hl, 0], xy[hl, 1], s=160, marker="*", facecolor="none", edgecolor="black", linewidths=1.4, label="MetalNet2", zorder=6)
    ax.set_title(f"Site-atlas (top-{args.kmain} mean-pool, {args.model} L{layer})")
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7, markerscale=1.5)
    fig.tight_layout(); fig.savefig(out / "site_atlas_pca.png", dpi=150); plt.close(fig)

    # domain-shift diagnostic: all five populations, distinctly coloured
    fig, ax = plt.subplots(figsize=(7, 6))
    gcol = {"PDB metal": "#1f77b4", "PDB non-metal": "#bbbbbb", "microprotein": "#d62728",
            "size_matched_short": "#ff7f0e", "general_background": "#2ca02c"}
    for name, m in ds_groups.items():
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=10, c=gcol[name], alpha=0.5, label=name, linewidths=0)
    ax.set_title("Domain-shift check: where do the human controls land?")
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "domain_shift.png", dpi=150); plt.close(fig)

    # ablation: ion separation vs k (top-k vs random-CHED)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(panel["k"], panel["ion_bacc"], "o-", label="top-k (selected)")
    ax.plot(panel["k"], panel["ion_bacc_randomCHED"], "s--", label="random CHED (control)")
    ax.axhline(1 / len(cen_ions), ls=":", c="gray", label="chance")
    ax.set_xlabel("k (residues pooled)"); ax.set_ylabel("ion balanced accuracy (knowns)")
    ax.set_xticks([1, 2, 3, 4]); ax.legend(); ax.set_title("Panel A/B: ion-type separation vs k")
    fig.tight_layout(); fig.savefig(out / "ion_separation_vs_k.png", dpi=150); plt.close(fig)

    # microprotein nearest-ion distance vs control
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(micro["nearest_cos_dist"], bins=25, density=True, alpha=0.6, color="#d62728", label="microprotein")
    ax.hist(ctrl["nearest_cos_dist"], bins=25, density=True, alpha=0.6, color="#ff7f0e", label="size_matched_short")
    ax.set_xlabel("cosine distance to nearest ion centroid"); ax.set_ylabel("density"); ax.legend()
    ax.set_title("Microprotein proximity to ion clusters vs length-matched control")
    fig.tight_layout(); fig.savefig(out / "microprotein_nearest_ion.png", dpi=150); plt.close(fig)

    # --- report ---
    md = ["# Site-atlas — pass-gate panel (knowns) + microprotein projection", "",
          f"Representation: mean-pool top-k coordinating residues. {args.model} L{layer}. "
          f"Main k={args.kmain}. PCA on all points (visual only).", "",
          "## Panel A/B/C (Study A knowns)", "",
          "| k | ion bal-acc | random-CHED | Zn-vs-rest AUROC | silhouette | same-ion/diff-cluster NN | chance |",
          "|--:|--:|--:|--:|--:|--:|--:|"]
    for _, r in panel.iterrows():
        md.append(f"| {r['k']} | {r['ion_bacc']} | {r['ion_bacc_randomCHED']} | {r['Zn_vs_rest_auroc']} "
                  f"| {r['silhouette_ion']} | {r['same_ion_diffcluster_purity']} | {r['chance_purity']} |")
    md += ["", "- **Panel A** passes if ion bal-acc / Zn-AUROC >> chance.",
           "- **Panel B** informative if top-k > random-CHED (selection helps) — either way not circular for ion type.",
           "- **Panel C** passes if same-ion/diff-cluster NN purity >> chance (alignment-free convergence).",
           "", "## Microprotein projection (gated on A+C)", "",
           f"Microprotein vs length-matched control, cosine distance to nearest ion centroid "
           f"(closer = more site-like): micro median {micro['nearest_cos_dist'].median():.3f}, "
           f"control median {ctrl['nearest_cos_dist'].median():.3f}, MWU p(micro<control)={p:.2e}.",
           f"Nearest-ion assignments in microprotein_ion.csv ({len(micro)} microproteins)."]
    md += ["", "## Domain-shift diagnostic",
           "Median cosine distance of each population to the PDB (Study A) centroid vs the "
           "microprotein centroid. If `size_matched_short` (human, short, canonical) sits "
           "with the microproteins, the island is a short/human shift; if it sits with PDB, "
           "the shift is microprotein-specific (Ribo-seq novelty/disorder).", "",
           "| group | n | dist→PDB | dist→micro | frac closer to micro |",
           "|---|--:|--:|--:|--:|"]
    for _, r in ds.iterrows():
        md.append(f"| {r['group']} | {r['n']} | {r['med_dist_to_PDB']} | "
                  f"{r['med_dist_to_micro']} | {r['frac_closer_to_micro']} |")
    hlrep = micro[micro["highlight"] == True].sort_values("nearest_cos_dist")
    if len(hlrep):
        md += ["", f"MetalNet2-highlighted ORFs ({len(hlrep)}):"]
        for _, x in hlrep.iterrows():
            md.append(f"- {x['protein_id']}: nearest {x['nearest_ion']} (cos-dist {x['nearest_cos_dist']:.3f})")
    (out / "panel_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}/ (panel_stats, microprotein_ion.csv, 3 figures)")


if __name__ == "__main__":
    main()
