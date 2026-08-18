#!/usr/bin/env python3
"""
Study B' — scan microproteins for metal-coordinating-like residues.

Train the residue scorer on Study A (H/C/D/E, coordinating vs non-coordinating —
the aa_matched setting that showed real site signal), then score every candidate
(H/C/D/E) residue in the microproteins and controls. Two length-safe tests:

  per-residue : distribution of P(coordinating) over candidate residues,
                microprotein vs length-matched `size_matched_short` (never per-
                protein hit counts -> "more residues" isn't a confound).
  per-protein : best-residue score per protein, microprotein vs size_matched
                (length-matched by construction, so this comparison is controlled).

Both logistic and MLP scorers (MLP primary; it was strongest). Outputs a ranked
candidate list with highlighted ORFs marked, plus figures.

Usage: python 12_microprotein_residue_scan.py --model 650M --layer 33
Outputs: results/microprotein_scan/{scan_stats.md, candidates.csv, *.png}
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from embed_lib import store

ROOT = Path(__file__).resolve().parent
CANDIDATE = set("HCDE")
NULLS = ["size_matched_short", "general_background"]


def cliffs_delta(a, b):
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return round(float(2.0 * u / (len(a) * len(b)) - 1.0), 3)


def train_mlp(X, y, epochs=50, hidden=128, dropout=0.2, lr=1e-3, seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    pw = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], dtype=torch.float32)
    net = nn.Sequential(nn.LayerNorm(X.shape[1]), nn.Linear(X.shape[1], hidden),
                        nn.LeakyReLU(), nn.Dropout(dropout),
                        nn.LayerNorm(hidden), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    net.train()
    for _ in range(epochs):
        opt.zero_grad(); lossf(net(Xt).squeeze(1), yt).backward(); opt.step()
    net.eval()

    def predict(Xn):
        import torch
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(Xn, dtype=torch.float32)).squeeze(1)).numpy()
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="650M")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    catalog = pd.read_parquet(ROOT / cfg["catalog"])
    labels = pd.read_parquet(ROOT / "catalog" / "residue_labels.parquet")
    h5 = ROOT / cfg["embed_dir"] / f"esm2_{args.model}_res.h5"
    layer = args.layer or max(store.residue_layers(h5))

    r = store.read_residues(h5, layer)
    pid = np.array(r["proteins"])[r["res_protein"]]
    meta = pd.DataFrame({"protein_id": pid, "position": r["position"], "aa": r["aa"]})
    cat = catalog.set_index("protein_id")
    for c in ["study", "class", "cluster30", "length"]:
        meta[c] = meta["protein_id"].map(cat[c])
    if "highlight" in cat.columns:
        meta["highlight"] = meta["protein_id"].map(cat["highlight"].fillna(False)).fillna(False)
    else:
        meta["highlight"] = False
    coord = set(zip(labels["protein_id"], labels["position"]))
    meta["y"] = [(p, q) in coord for p, q in zip(meta["protein_id"], meta["position"])]
    emb = r["emb"]

    cand = meta["aa"].isin(CANDIDATE).to_numpy()
    A = (meta["study"] == "A").to_numpy() & cand
    B = (meta["study"] == "B").to_numpy() & cand
    print(f"train residues (A, H/C/D/E): {A.sum()} ({meta['y'][A].sum()} coordinating)")
    print(f"score residues  (B, H/C/D/E): {B.sum()}")

    # --- train scorers on Study A aa_matched (subsample negatives) ---
    rng = np.random.default_rng(0)
    ia = np.where(A)[0]
    ya = meta["y"].to_numpy()[ia].astype(int)
    pos, neg = ia[ya == 1], ia[ya == 0]
    if len(neg) > args.neg_ratio * len(pos):
        neg = rng.choice(neg, int(args.neg_ratio * len(pos)), replace=False)
    tr = np.concatenate([pos, neg]); ytr = meta["y"].to_numpy()[tr].astype(int)
    scaler = StandardScaler().fit(emb[tr])
    Xtr = scaler.transform(emb[tr])
    logit = LogisticRegression(C=cfg["probe"]["C"], max_iter=cfg["probe"]["max_iter"],
                               class_weight="balanced").fit(Xtr, ytr)
    mlp = train_mlp(Xtr, ytr)

    # --- score Study B candidate residues ---
    ib = np.where(B)[0]
    Xb = scaler.transform(emb[ib])
    dfb = meta.iloc[ib].copy()
    dfb["score_mlp"] = mlp(Xb)
    dfb["score_logit"] = logit.predict_proba(Xb)[:, 1]

    def by_class(col, cls):
        return dfb[dfb["class"] == cls][col].to_numpy()

    out = ROOT / "results" / "microprotein_scan"; out.mkdir(parents=True, exist_ok=True)
    stat_rows = []
    for scorer in ["score_mlp", "score_logit"]:
        mp = by_class(scorer, "microprotein")
        for null in NULLS:
            nl = by_class(scorer, null)
            u, p = mannwhitneyu(mp, nl, alternative="two-sided")
            stat_rows.append({"level": "per_residue", "scorer": scorer, "vs": null,
                              "micro_median": float(np.median(mp)), "null_median": float(np.median(nl)),
                              "cliffs_delta": round(cliffs_delta(mp, nl), 3), "p": p})

    # --- per-protein best-residue score ---
    best = (dfb.sort_values("score_mlp", ascending=False)
            .groupby("protein_id")
            .agg(best_score_mlp=("score_mlp", "max"),
                 best_score_logit=("score_logit", "max"),
                 best_aa=("aa", "first"), best_pos=("position", "first"),
                 n_candidate=("aa", "size")).reset_index())
    best["class"] = best["protein_id"].map(cat["class"])
    best["length"] = best["protein_id"].map(cat["length"])
    best["highlight"] = best["protein_id"].map(cat["highlight"].fillna(False)) if "highlight" in cat.columns else False
    for null in NULLS:
        a = best[best["class"] == "microprotein"]["best_score_mlp"].to_numpy()
        b = best[best["class"] == null]["best_score_mlp"].to_numpy()
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        stat_rows.append({"level": "per_protein_best", "scorer": "score_mlp", "vs": null,
                          "micro_median": float(np.median(a)), "null_median": float(np.median(b)),
                          "cliffs_delta": round(cliffs_delta(a, b), 3), "p": p})
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(out / "scan_stats.csv", index=False)

    # candidate ranking (microproteins), percentile of best score vs size_matched control
    ctrl = best[best["class"] == "size_matched_short"]["best_score_mlp"].to_numpy()
    mp_best = best[best["class"] == "microprotein"].copy()
    mp_best["pctile_vs_control"] = mp_best["best_score_mlp"].apply(
        lambda s: round(100 * (ctrl < s).mean(), 1))
    mp_best = mp_best.sort_values("best_score_mlp", ascending=False)
    mp_best[["protein_id", "highlight", "length", "best_score_mlp", "best_aa",
             "best_pos", "n_candidate", "pctile_vs_control"]].to_csv(
        out / "candidates.csv", index=False)

    # --- figures ---
    # 1. per-residue score distribution by class
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, c in [("microprotein", "#d62728"), ("size_matched_short", "#ff7f0e"),
                   ("general_background", "#c7c7c7")]:
        v = by_class("score_mlp", cls)
        if len(v):
            ax.hist(v, bins=30, density=True, alpha=0.55, label=cls, color=c)
    ax.set_xlabel("P(metal-coordinating) per H/C/D/E residue (MLP)"); ax.set_ylabel("density")
    ax.legend(); ax.set_title("Per-residue coordination score by class")
    fig.tight_layout(); fig.savefig(out / "per_residue_scores.png", dpi=150); plt.close(fig)

    # 2. per-protein best score: microprotein vs length-matched control
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(best[best["class"] == "microprotein"]["best_score_mlp"], bins=25, density=True,
            alpha=0.6, color="#d62728", label="microprotein")
    ax.hist(best[best["class"] == "size_matched_short"]["best_score_mlp"], bins=25, density=True,
            alpha=0.6, color="#ff7f0e", label="size_matched_short")
    hb = mp_best[mp_best["highlight"] == True]
    if len(hb):
        ax.scatter(hb["best_score_mlp"], np.zeros(len(hb)), marker="*", s=130,
                   color="black", zorder=5, label="highlighted")
    ax.set_xlabel("best-residue coordination score (per protein)"); ax.set_ylabel("density")
    ax.legend(); ax.set_title("Best putative site: microproteins vs length-matched control")
    fig.tight_layout(); fig.savefig(out / "per_protein_best.png", dpi=150); plt.close(fig)

    # 3. best score vs length (confound check)
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, c in [("microprotein", "#d62728"), ("size_matched_short", "#ff7f0e")]:
        s = best[best["class"] == cls]
        ax.scatter(s["length"], s["best_score_mlp"], s=12, alpha=0.5, c=c, label=cls)
    if len(hb):
        ax.scatter(hb["length"], hb["best_score_mlp"], marker="*", s=150, facecolor="none",
                   edgecolor="black", linewidths=1.4, label="highlighted", zorder=6)
    ax.set_xlabel("length (aa)"); ax.set_ylabel("best-residue score"); ax.legend()
    ax.set_title("Best score vs length — confound check")
    fig.tight_layout(); fig.savefig(out / "best_vs_length.png", dpi=150); plt.close(fig)

    # --- report ---
    md = ["# Microprotein metal-coordination scan", "",
          f"Scorer trained on Study A H/C/D/E (coordinating vs not), {args.model} "
          f"L{layer}. Applied to Study B candidate residues. Null = length-matched "
          "`size_matched_short`.", "",
          "| level | scorer | vs null | micro med | null med | Cliff's δ | MWU p |",
          "|---|---|---|--:|--:|--:|--:|"]
    for _, r in stats.iterrows():
        md.append(f"| {r['level']} | {r['scorer']} | {r['vs']} | {r['micro_median']:.3f} | "
                  f"{r['null_median']:.3f} | {r['cliffs_delta']} | {r['p']:.2e} |")
    md += ["", f"Top candidate microproteins (by best-residue MLP score): "
           f"see candidates.csv ({len(mp_best)} microproteins)."]
    hl = mp_best[mp_best["highlight"] == True]
    if len(hl):
        md += ["", f"Highlighted ORFs ({len(hl)}):"]
        for _, x in hl.iterrows():
            md.append(f"- {x['protein_id']}: best score {x['best_score_mlp']:.3f} "
                      f"({x['best_aa']}{x['best_pos']}), {x['pctile_vs_control']}th pct vs control")
    (out / "scan_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}/ (scan_stats, candidates.csv, 3 figures)")


if __name__ == "__main__":
    main()
