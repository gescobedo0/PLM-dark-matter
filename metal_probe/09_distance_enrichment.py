#!/usr/bin/env python3
"""
Study B enrichment: are human microproteins nearer metal-binder clusters (esp.
Zn) than LENGTH-MATCHED controls, in full embedding space?

Motivation: the class PCA separates microproteins mostly along PC1, which is
length (see 08's length-coloured panel) -> proximity read off PCA is a length
artifact. This script works in full D and uses `size_matched_short` as the
length-matched null, so a positive result means "metal-relevant", not "short".

Two independent lenses, each microprotein vs the length-matched control:
  (A) unsupervised: cosine kNN distance to metal binders, overall and per ion
      (Zn / Cu / other_transition); plus a relative score
      (sim to metal centroid - sim to non-metal centroid).
  (B) supervised: project Study B onto the metal-vs-non-metal probe direction
      trained on Study A (decision_function) -> a "metal-ness" score.

Stats: Mann-Whitney U (microprotein vs each null) + Cliff's delta effect size.
A score-vs-length Spearman check reports residual length dependence.

Usage: python 09_distance_enrichment.py --model 650M --layer 33 --pooling mean
Outputs: results/enrichment/{enrichment_stats.md,csv, candidates.csv, *.png}
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize

from embed_lib import store

ROOT = Path(__file__).resolve().parent
METAL_CLASSES = ["Zn", "Cu", "other_transition"]
NULLS = ["size_matched_short", "general_background"]


def cliffs_delta(a, b):
    """Effect size from Mann-Whitney U: 2U/(n*m) - 1, in [-1, 1]."""
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return 2.0 * u / (len(a) * len(b)) - 1.0


def knn_dist(Q, R, k=10):
    """Mean cosine DISTANCE from each row of Q to its k nearest rows in R."""
    sim = cosine_similarity(Q, R)                 # (|Q|, |R|)
    k = min(k, R.shape[0])
    topk = np.partition(sim, -k, axis=1)[:, -k:]  # k largest sims
    return 1.0 - topk.mean(axis=1)                # distance, lower = closer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="650M")
    ap.add_argument("--layer", type=int, default=None, help="default: deepest stored layer")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    cat = pd.read_parquet(ROOT / cfg["catalog"]).set_index("protein_id")
    h5 = ROOT / cfg["embed_dir"] / f"esm2_{args.model}.h5"
    if args.layer is None:
        args.layer = max(store.available(h5)[0])
    ids, arr = store.read(h5, args.layer, args.pooling)
    keep = [i for i in ids if i in cat.index]
    pos = {i: k for k, i in enumerate(ids)}
    E = arr[[pos[i] for i in keep]].astype(np.float64)
    meta = cat.loc[keep].copy()
    cls = meta["class"].to_numpy()
    hl_ids = (set(meta.index[meta["highlight"].fillna(False)])
              if "highlight" in meta.columns else set())
    En = normalize(E)                              # unit vectors -> cosine

    def rows(name_or_pred):
        m = name_or_pred(cls) if callable(name_or_pred) else (cls == name_or_pred)
        return En[m], meta[m]

    metal_mask = np.isin(cls, METAL_CLASSES)
    R_metal = En[metal_mask]
    R_nonmetal = En[cls == "non_metal"]
    ref = {"metal_all": R_metal, "non_metal": R_nonmetal,
           **{ion: En[cls == ion] for ion in METAL_CLASSES}}

    # ---- (A) unsupervised distances for every Study B protein ----
    query_classes = ["microprotein"] + NULLS
    recs = {}
    for qc in query_classes:
        Q, qm = rows(qc)
        d = {"length": qm["length"].to_numpy(), "protein_id": qm.index.to_numpy(),
             "highlight": qm.index.isin(hl_ids)}
        for rname, R in ref.items():
            if R.shape[0] == 0:
                continue
            d[f"knn_{rname}"] = knn_dist(Q, R, args.k)
        # relative metal-ness (unsupervised): closer to metal than to non-metal
        d["rel_metalness"] = (cosine_similarity(Q, R_metal).mean(1)
                              - cosine_similarity(Q, R_nonmetal).mean(1))
        recs[qc] = pd.DataFrame(d)

    # ---- (B) supervised probe-direction score ----
    A = meta["class"].isin(METAL_CLASSES + ["non_metal"]).to_numpy()
    yA = np.isin(cls[A], METAL_CLASSES).astype(int)
    scaler = StandardScaler().fit(E[A])
    clf = LogisticRegression(C=cfg["probe"]["C"], max_iter=cfg["probe"]["max_iter"],
                             class_weight="balanced").fit(scaler.transform(E[A]), yA)
    for qc in query_classes:
        Q_raw = E[cls == qc]
        recs[qc]["probe_score"] = clf.decision_function(scaler.transform(Q_raw))

    # ---- stats: microprotein vs each null, per score ----
    score_cols = ["knn_Zn", "knn_Cu", "knn_other_transition", "knn_metal_all",
                  "rel_metalness", "probe_score"]
    # note: for knn_* LOWER = closer (enriched); for rel/probe HIGHER = more metal
    mp = recs["microprotein"]
    stat_rows = []
    for null in NULLS:
        nl = recs[null]
        for sc in score_cols:
            if sc not in mp or sc not in nl:
                continue
            a, b = mp[sc].to_numpy(), nl[sc].to_numpy()
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            stat_rows.append({
                "score": sc, "vs_null": null,
                "microprotein_median": float(np.median(a)),
                "null_median": float(np.median(b)),
                "cliffs_delta": round(cliffs_delta(a, b), 3),
                "mannwhitney_p": p,
                "direction": "lower=closer" if sc.startswith("knn") else "higher=metal",
            })
    stats = pd.DataFrame(stat_rows)

    # length-confound check on the microprotein score itself
    len_checks = []
    for sc in score_cols:
        if sc in mp:
            r, p = spearmanr(mp[sc], mp["length"])
            len_checks.append({"score": sc, "spearman_score_vs_length": round(r, 3),
                               "p": p})
    lenc = pd.DataFrame(len_checks)

    out = ROOT / "results" / "enrichment"
    out.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out / "enrichment_stats.csv", index=False)
    for qc, df in recs.items():
        df.to_csv(out / f"scores_{qc}.csv", index=False)

    # ranked microprotein candidates: most metal-like (probe) + nearest ion
    ion_cols = [f"knn_{i}" for i in METAL_CLASSES if f"knn_{i}" in mp]
    mp2 = mp.copy()
    mp2["nearest_ion"] = mp2[ion_cols].idxmin(axis=1).str.replace("knn_", "")
    mp2 = mp2.sort_values("probe_score", ascending=False)
    mp2[["protein_id", "highlight", "length", "probe_score", "rel_metalness",
         "nearest_ion"] + ion_cols].to_csv(out / "candidates.csv", index=False)

    # where do the highlighted ORFs land among microproteins?
    hmask = mp["highlight"].to_numpy()
    hl_report = []
    if hmask.any():
        order = mp["probe_score"].rank(pct=True)   # percentile within microproteins
        for _, r in mp[hmask].sort_values("probe_score", ascending=False).iterrows():
            pct = float(order[mp["protein_id"] == r["protein_id"]].iloc[0]) * 100
            hl_report.append(f"  {r['protein_id']}: probe_score={r['probe_score']:.2f} "
                             f"({pct:.0f}th pct of microproteins), nearest={mp2.loc[mp2.protein_id==r['protein_id'],'nearest_ion'].iloc[0]}")

    # ---- figures ----
    # 1) probe score by class (incl. reference metal/non-metal for context)
    fig, ax = plt.subplots(figsize=(7, 4))
    groups, data = [], []
    ref_scores = {"metal (ref)": clf.decision_function(scaler.transform(E[metal_mask])),
                  "non_metal (ref)": clf.decision_function(scaler.transform(E[cls == "non_metal"]))}
    for name, v in ref_scores.items():
        groups.append(name); data.append(v)
    for qc in query_classes:
        groups.append(qc); data.append(recs[qc]["probe_score"].to_numpy())
    ax.boxplot(data, labels=groups, showfliers=False)
    if hmask.any():
        xi = groups.index("microprotein") + 1      # boxplot positions are 1-based
        ax.scatter([xi] * int(hmask.sum()), mp["probe_score"][hmask], marker="*",
                   s=130, color="black", zorder=5, label="highlighted")
        ax.legend(fontsize=7)
    ax.set_ylabel("probe metal-direction score"); ax.axhline(0, ls="--", c="gray", lw=0.8)
    ax.set_title(f"Metal-ness along probe axis ({args.model} L{args.layer} {args.pooling})")
    plt.xticks(rotation=25, ha="right"); fig.tight_layout()
    fig.savefig(out / "probe_score_by_class.png", dpi=150); plt.close(fig)

    # 2) kNN distance to Zn: microprotein vs length-matched null
    if "knn_Zn" in mp:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(mp["knn_Zn"], bins=30, alpha=0.6, density=True, label="microprotein", color="#d62728")
        ax.hist(recs["size_matched_short"]["knn_Zn"], bins=30, alpha=0.6, density=True,
                label="size_matched_short (length null)", color="#ff7f0e")
        if hmask.any():
            ax.scatter(mp["knn_Zn"][hmask], np.zeros(int(hmask.sum())), marker="*",
                       s=130, color="black", zorder=5, label="highlighted")
        ax.set_xlabel(f"cosine kNN distance to Zn binders (k={args.k}); lower=closer")
        ax.set_ylabel("density"); ax.legend(); ax.set_title("Microproteins vs length-matched control: proximity to Zn")
        fig.tight_layout(); fig.savefig(out / "knn_Zn_microprotein_vs_control.png", dpi=150); plt.close(fig)

    # 3) score vs length (confound visual)
    fig, ax = plt.subplots(figsize=(6, 4))
    for qc, c in [("microprotein", "#d62728"), ("size_matched_short", "#ff7f0e"),
                  ("general_background", "#c7c7c7")]:
        ax.scatter(recs[qc]["length"], recs[qc]["probe_score"], s=10, alpha=0.5, c=c, label=qc)
    if hmask.any():
        ax.scatter(mp["length"][hmask], mp["probe_score"][hmask], marker="*", s=150,
                   facecolor="none", edgecolor="black", linewidths=1.4,
                   label="highlighted", zorder=6)
    ax.set_xlabel("length (aa)"); ax.set_ylabel("probe metal score"); ax.legend()
    ax.set_title("Probe score vs length — residual length dependence check")
    fig.tight_layout(); fig.savefig(out / "score_vs_length.png", dpi=150); plt.close(fig)

    # ---- report ----
    md = ["# Study B — microprotein metal-proximity enrichment", "",
          f"Embedding: {args.model} L{args.layer} {args.pooling}, full {E.shape[1]}-D, "
          f"cosine, k={args.k}. Null = length-matched `size_matched_short`.", "",
          "Primary test: microprotein vs size_matched_short. For `knn_*` lower=closer "
          "(enriched); for `rel_metalness`/`probe_score` higher=more metal-like. "
          "Cliff's delta sign follows the raw median difference.", "",
          "| score | vs null | microprotein med | null med | Cliff's δ | MWU p |",
          "|---|---|--:|--:|--:|--:|"]
    for _, r in stats.iterrows():
        md.append(f"| {r['score']} | {r['vs_null']} | {r['microprotein_median']:.4f} | "
                  f"{r['null_median']:.4f} | {r['cliffs_delta']} | {r['mannwhitney_p']:.2e} |")
    md += ["", "Length-confound check (Spearman of score vs length, within microproteins):"]
    for _, r in lenc.iterrows():
        md.append(f"- {r['score']}: rho={r['spearman_score_vs_length']} (p={r['p']:.2e})")
    if hl_report:
        md += ["", f"Highlighted ORFs ({len(hl_report)}) — where they land among microproteins:"]
        md += hl_report
    (out / "enrichment_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\nwrote {out}/ (stats, candidates.csv, 3 figures, per-class scores)")


if __name__ == "__main__":
    main()
