#!/usr/bin/env python3
"""
Multi-label per-ion residue model (Study A' by ion) + characterization analyses.

Each candidate residue -> a vector [P_Zn,P_Cu,P_Fe,P_Mn,P_Co,P_Ni] from independent
sigmoid heads (a residue may be Cu-ish AND Fe-ish; softmax would forbid that).

Masked labels: for ion head X, positives = residues coordinating X, negatives =
NON-coordinating residues; residues coordinating a *different* metal are MASKED
(no loss on head X). So promiscuity is never punished and the metal-similarity
map emerges from held-out cross-scores.

Outputs (results/ion_residue/):
  1. per-ion held-out AUROC (+ aa_matched)         -> the honesty gate
  2. metal_similarity.png/.csv (true-ion x head)   -> chemical similarity
  3. residue_scores.parquet (6-D vector/residue)
  4. protein_coherence.csv  (do top residues agree on an ion = a site)
  5. microprotein_profile.csv + figure (+ MetalNet2 highlights)
  6. supervised hidden-layer atlas (out-of-fold; separability = the CV AUROC, not the picture)

Leakage-safe: GroupKFold on cluster30. Scalable via the residue embedding set
(more clusters -> better Cu/Fe/Mn/Co; Ni stays starved).

Usage: python 15_ion_residue_model.py --model 650M --layer 33
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from embed_lib import store

ROOT = Path(__file__).resolve().parent
IONS = ["Zn", "Cu", "Fe", "Mn", "Co", "Ni"]
ION_IDX = {ion: i for i, ion in enumerate(IONS)}
CANDIDATE = set("HCDE")


class IonMLP(nn.Module):
    def __init__(self, d, hidden=128, nion=6, dropout=0.2):
        super().__init__()
        self.norm_in = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.norm_h = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, nion)

    def forward(self, x, return_hidden=False):
        h = self.norm_h(self.drop(self.act(self.fc1(self.norm_in(x)))))
        logits = self.head(h)
        return (logits, h) if return_hidden else logits


def masked_bce_train(X, Y, M, epochs=60, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    Xt, Yt, Mt = (torch.tensor(a, dtype=torch.float32) for a in (X, Y, M))
    pw = torch.tensor([((M[:, i] * (1 - Y[:, i])).sum()) / max(1.0, (M[:, i] * Y[:, i]).sum())
                       for i in range(Y.shape[1])], dtype=torch.float32)
    net = IonMLP(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = (lossf(net(Xt), Yt) * Mt).sum() / Mt.sum()
        loss.backward(); opt.step()
    net.eval()
    return net


def predict(net, X):
    with torch.no_grad():
        logits, h = net(torch.tensor(X, dtype=torch.float32), return_hidden=True)
        return torch.sigmoid(logits).numpy(), h.numpy()


def build_labels(meta, coord_ions):
    n = len(meta); Y = np.zeros((n, 6), np.float32); Mask = np.zeros((n, 6), np.float32)
    for r in range(n):
        s = coord_ions.get((meta["protein_id"].iat[r], int(meta["position"].iat[r])), ())
        is_coord = len(s) > 0
        for i, ion in enumerate(IONS):
            in_s = ion in s
            Y[r, i] = 1.0 if in_s else 0.0
            Mask[r, i] = 1.0 if (in_s or not is_coord) else 0.0
    return Y, Mask


def protein_coherence(scores, meta, mask, topm=4):
    """Per protein over its candidate residues: dominant ion, agreement, strength."""
    rows = []
    idx = np.where(mask)[0]
    df = pd.DataFrame({"protein_id": meta["protein_id"].to_numpy()[idx], "row": idx})
    for pid, g in df.groupby("protein_id", sort=False):
        rs = scores[g["row"].to_numpy()]            # (ncand, 6)
        order = rs.max(1).argsort()[::-1][:topm]
        top = rs[order]
        agg = top.mean(0)
        dom = int(agg.argmax())
        agree = float(np.mean(top.argmax(1) == dom))
        rows.append({"protein_id": pid, "n_candidate": len(g),
                     "dominant_ion": IONS[dom], "coherence": round(agree, 3),
                     "strength": round(float(agg[dom]), 3)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="650M")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--neg-ratio", type=float, default=5.0)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--topm", type=int, default=4)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    catalog = pd.read_parquet(ROOT / cfg["catalog"])
    labels = pd.read_parquet(ROOT / "catalog" / "residue_labels.parquet")
    h5 = ROOT / cfg["embed_dir"] / f"esm2_{args.model}_res.h5"
    layer = args.layer or max(store.residue_layers(h5))
    out = ROOT / "results" / "ion_residue"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    r = store.read_residues(h5, layer)
    pid = np.array(r["proteins"])[r["res_protein"]]
    meta = pd.DataFrame({"protein_id": pid, "position": r["position"], "aa": r["aa"]})
    cat = catalog.set_index("protein_id")
    for c in ["study", "class", "cluster30"]:
        meta[c] = meta["protein_id"].map(cat[c])
    meta["highlight"] = (meta["protein_id"].map(cat["highlight"].fillna(False))
                         if "highlight" in cat.columns else False)
    emb = r["emb"]
    coord_ions = labels.groupby(["protein_id", "position"])["ion"].agg(set).to_dict()
    Y, Mask = build_labels(meta, coord_ions)
    is_coord = Y.sum(1) > 0

    # --- Study A training set: all coord + subsampled non-coord ---
    A = (meta["study"] == "A").to_numpy()
    coordA = np.where(A & is_coord)[0]
    noncoordA = np.where(A & ~is_coord)[0]
    keepn = min(len(noncoordA), int(args.neg_ratio * len(coordA)))
    tr_idx = np.concatenate([coordA, rng.choice(noncoordA, keepn, replace=False)])
    rng.shuffle(tr_idx)
    scaler = StandardScaler().fit(emb[tr_idx])
    Xtr = scaler.transform(emb[tr_idx])
    groups = meta["cluster30"].astype(str).to_numpy()[tr_idx]

    # --- GroupKFold OOF on Study A ---
    k = min(args.splits, len(np.unique(groups)))
    oof = np.full((len(tr_idx), 6), np.nan); oof_h = None
    Ytr, Mtr = Y[tr_idx], Mask[tr_idx]
    for tr, te in GroupKFold(k).split(Xtr, groups=groups):
        net = masked_bce_train(Xtr[tr], Ytr[tr], Mtr[tr])
        p, hh = predict(net, Xtr[te])
        oof[te] = p
        if oof_h is None:
            oof_h = np.zeros((len(tr_idx), hh.shape[1]))
        oof_h[te] = hh

    # (1) two honesty metrics per ion:
    #   detect     = ion-X ligands vs NON-coordinating  (mostly "is it coordinating")
    #   discriminate = ion-X ligands vs OTHER-metal ligands  (the real "which metal";
    #                  this is the residue-level analog of the atlas ion-separation)
    Ytr2 = Y[tr_idx]
    is_coord_tr = Ytr2.sum(1) > 0
    noncoord_tr = ~is_coord_tr

    def _au(sel, i):
        y = (Ytr2[sel, i] > 0).astype(int)
        return round(roc_auc_score(y, oof[sel, i]), 3) if len(np.unique(y)) == 2 else float("nan")

    auroc_rows = []
    for i, ion in enumerate(IONS):
        pos = Ytr2[:, i] > 0
        other = is_coord_tr & ~pos
        auroc_rows.append({"ion": ion, "n_pos": int(pos.sum()),
                           "detect_auroc": _au(pos | noncoord_tr, i),
                           "discriminate_auroc": _au(pos | other, i)})
    auroc = pd.DataFrame(auroc_rows)

    # (2) metal-similarity: background-CALIBRATED (z above non-coordinating), so the
    # shared "is-coordinating" signal and noisy/data-starved heads can't look
    # universally high. Off-diagonal now reflects ion-specific chemistry only.
    nc_oof = Y[tr_idx].sum(1) == 0
    bmo, bso = oof[nc_oof].mean(0), oof[nc_oof].std(0) + 1e-6
    oof_cal = (oof - bmo) / bso
    sim = np.zeros((6, 6))
    for t in range(6):
        sel = (Y[tr_idx][:, t] > 0)
        if sel.any():
            sim[t] = oof_cal[sel].mean(axis=0)
    simdf = pd.DataFrame(sim, index=[f"true_{i}" for i in IONS], columns=[f"score_{i}" for i in IONS])
    simdf.to_csv(out / "metal_similarity.csv")

    # --- final model on ALL Study A; score everything (incl. Study B) ---
    net = masked_bce_train(Xtr, Y[tr_idx], Mask[tr_idx])
    cand = np.isin(meta["aa"].to_numpy(), list(CANDIDATE))
    scores = np.full((len(meta), 6), np.nan)
    scores[cand], _ = predict(net, scaler.transform(emb[cand]))
    # background-calibrate: z above non-coordinating candidate residues (Study A),
    # so a data-starved head (e.g. Ni) can't win the dominant-ion argmax by noise.
    ncA = A & ~is_coord & cand
    bmean, bstd = np.nanmean(scores[ncA], axis=0), np.nanstd(scores[ncA], axis=0) + 1e-6
    scores_cal = (scores - bmean) / bstd

    # (3) store per-residue score vectors: raw P + background-calibrated z
    sc = meta.loc[cand, ["protein_id", "position", "aa", "study", "class"]].copy()
    for i, ion in enumerate(IONS):
        sc[f"P_{ion}"] = scores[cand, i]
        sc[f"z_{ion}"] = scores_cal[cand, i]
    sc.to_parquet(out / "residue_scores.parquet", index=False)

    # (4) per-protein coherence on CALIBRATED scores (all proteins w/ candidates)
    coh = protein_coherence(scores_cal, meta, cand, topm=args.topm)
    coh["study"] = coh["protein_id"].map(cat["study"])
    coh["class"] = coh["protein_id"].map(cat["class"])
    coh["highlight"] = coh["protein_id"].map(cat["highlight"].fillna(False)) if "highlight" in cat.columns else False
    coh.to_csv(out / "protein_coherence.csv", index=False)

    # (5) microprotein profile: rank by strength*coherence, vs length-matched control
    mp = coh[coh["class"] == "microprotein"].copy()
    mp["site_score"] = (mp["strength"] * mp["coherence"]).round(3)
    ctrl = coh[coh["class"] == "size_matched_short"].copy()
    ctrl["site_score"] = ctrl["strength"] * ctrl["coherence"]
    u, pmw = mannwhitneyu(mp["site_score"], ctrl["site_score"], alternative="greater")
    mp.sort_values("site_score", ascending=False)[
        ["protein_id", "highlight", "n_candidate", "dominant_ion", "coherence", "strength", "site_score"]
    ].to_csv(out / "microprotein_profile.csv", index=False)

    # --- figures ---
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    vmax = float(np.abs(sim).max()) or 1.0
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"→{i}" for i in IONS])
    ax.set_yticks(range(6)); ax.set_yticklabels([f"{i} ligands" for i in IONS])
    for a in range(6):
        for b in range(6):
            ax.text(b, a, f"{sim[a,b]:.1f}", ha="center", va="center", color="black", fontsize=7)
    ax.set_title("Metal-similarity (z above background): true-ion ligands → head")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out / "metal_similarity.png", dpi=150); plt.close(fig)

    # supervised hidden-layer atlas (out-of-fold), coord residues coloured by true ion
    coord_tr = Y[tr_idx].sum(1) > 0
    Hs = StandardScaler().fit_transform(oof_h[coord_tr])
    xy = PCA(2, random_state=0).fit_transform(Hs)
    true_ion = np.array(IONS)[Y[tr_idx][coord_tr].argmax(1)]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for ion in IONS:
        m = true_ion == ion
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.6, label=ion, linewidths=0)
    ax.set_title("Supervised hidden-layer view (OOF) — separability = CV AUROC, NOT this picture")
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "hidden_atlas.png", dpi=150); plt.close(fig)

    # microprotein site-score vs control
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(mp["site_score"], bins=25, density=True, alpha=0.6, color="#d62728", label="microprotein")
    ax.hist(ctrl["site_score"], bins=25, density=True, alpha=0.6, color="#ff7f0e", label="size_matched_short")
    hb = mp[mp["highlight"] == True]
    if len(hb):
        ax.scatter(hb["site_score"], np.zeros(len(hb)), marker="*", s=130, color="black", zorder=5, label="MetalNet2")
    ax.set_xlabel("site score (strength × coherence)"); ax.set_ylabel("density"); ax.legend()
    ax.set_title("Microprotein coherent-site score vs length-matched control")
    fig.tight_layout(); fig.savefig(out / "microprotein_site_score.png", dpi=150); plt.close(fig)

    # --- report ---
    md = ["# Multi-label per-ion residue model", "",
          f"{args.model} L{layer}. Masked multi-label (other-metal ligands masked per head). "
          f"GroupKFold on cluster30, {k} folds. neg-ratio {args.neg_ratio}.", "",
          "## (1) Per-ion held-out AUROC — the honesty gate", "",
          "`detect` = vs non-coordinating (mostly coordination detection). "
          "**`discriminate` = vs other-metal ligands = the real 'which metal' — trust this one.**", "",
          "| ion | n_pos | detect | discriminate |", "|---|--:|--:|--:|"]
    for _, x in auroc.iterrows():
        md.append(f"| {x['ion']} | {x['n_pos']} | {x['detect_auroc']} | {x['discriminate_auroc']} |")
    md += ["", "## (2) Metal-similarity — z above background (rows=true-ion ligands, cols=head)",
           "Diagonal should be highest; large off-diagonal = chemical similarity/promiscuity.", "",
           "| true \\ head | " + " | ".join(IONS) + " |",
           "|---|" + "|".join(["--:"] * 6) + "|"]
    for t, ion in enumerate(IONS):
        md.append(f"| {ion} | " + " | ".join(f"{sim[t,b]:.2f}" for b in range(6)) + " |")
    md += ["", "## (5) Microprotein coherent sites",
           f"site score (strength×coherence) micro median {mp['site_score'].median():.3f} vs "
           f"control {ctrl['site_score'].median():.3f}; MWU p(micro>control)={pmw:.2e}. "
           f"Ranked in microprotein_profile.csv ({len(mp)} microproteins).",
           "", "dominant-ion tally among microproteins: "
           + str(mp["dominant_ion"].value_counts().to_dict())]
    hlrep = mp[mp["highlight"] == True].sort_values("site_score", ascending=False)
    if len(hlrep):
        md += ["", f"MetalNet2 ORFs ({len(hlrep)}):"]
        for _, x in hlrep.iterrows():
            md.append(f"- {x['protein_id']}: {x['dominant_ion']} "
                      f"(coherence {x['coherence']}, strength {x['strength']}, site {x['site_score']})")
    (out / "ion_model_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}/ (report, similarity, residue_scores, coherence, microprotein_profile, 3 figs)")


if __name__ == "__main__":
    main()
