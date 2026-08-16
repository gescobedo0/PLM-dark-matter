#!/usr/bin/env python3
"""
Figures for the catalog in embedding space.

PCA is used for any figure that makes a clustering claim (a linear projection
cannot invent structure). UMAP is exploratory only and is labelled with its
n_neighbors / min_dist; inter-cluster distance and density are never read off it
(spec §5). Panels: colour by class, colour by sequence length (the length-confound
check), and microproteins highlighted against the metal-binder clusters (Study B).

Usage:  python 08_visualize.py --model 650M --layer 24 --pooling mean
Outputs: results/figures/*.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from embed_lib import store

ROOT = Path(__file__).resolve().parent
CLASS_COLORS = {
    "Zn": "#1f77b4", "Cu": "#17becf", "other_transition": "#2ca02c",
    "non_metal": "#7f7f7f", "microprotein": "#d62728",
    "size_matched_short": "#ff7f0e", "general_background": "#c7c7c7",
}


def scatter(ax, xy, labels, title):
    for cls in [c for c in CLASS_COLORS if c in set(labels)]:
        m = labels == cls
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6, label=cls,
                   c=CLASS_COLORS[cls], linewidths=0)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=7, loc="best", framealpha=0.7)


def overlay_highlights(ax, xy, hl):
    """Mark user-highlighted ORFs (add_highlights.py) as black outlined stars."""
    if hl is None or not hl.any():
        return
    ax.scatter(xy[hl, 0], xy[hl, 1], s=150, marker="*", facecolor="none",
               edgecolor="black", linewidths=1.4, label="highlighted", zorder=6)
    ax.legend(markerscale=1, fontsize=7, loc="best", framealpha=0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=None, help="default: deepest stored layer")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-mindist", type=float, default=0.1)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    cat = pd.read_parquet(ROOT / cfg["catalog"]).set_index("protein_id")
    h5 = ROOT / cfg["embed_dir"] / f"esm2_{args.model}.h5"
    if not h5.exists():
        shards = list((ROOT / cfg["embed_dir"]).glob(f"esm2_{args.model}_shard*of*.h5"))
        raise SystemExit(
            f"\n[08_visualize] no embeddings at {h5}.\n"
            f"  Embeddings are produced by the Slurm job, which runs asynchronously.\n"
            f"  {'Found ' + str(len(shards)) + ' shard(s) — run: python 06_merge_embeddings.py --model ' + args.model if shards else 'No shards yet — the embed job has not finished (check: sacct -j <id>, cat logs/embed_*.out).'}\n")
    if args.layer is None:
        args.layer = max(store.available(h5)[0])
    ids, arr = store.read(h5, args.layer, args.pooling)
    keep = [i for i in ids if i in cat.index]
    pos = {i: k for k, i in enumerate(ids)}
    X = arr[[pos[i] for i in keep]]
    meta = cat.loc[keep]
    classes = meta["class"].to_numpy()
    lengths = meta["length"].to_numpy()
    hl = (meta["highlight"].fillna(False).to_numpy()
          if "highlight" in meta.columns else np.zeros(len(meta), bool))
    if hl.any():
        print(f"overlaying {int(hl.sum())} highlighted ORF(s)")

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=cfg["seed"]).fit(Xs)
    xy = pca.transform(Xs)
    ev = pca.explained_variance_ratio_ * 100

    outdir = ROOT / "results" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    base = f"{args.model}_L{args.layer}_{args.pooling}"

    # PCA: by class
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter(ax, xy, classes, f"PCA by class ({args.model} L{args.layer} {args.pooling})")
    overlay_highlights(ax, xy, hl)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    fig.tight_layout(); fig.savefig(outdir / f"pca_class_{base}.png", dpi=150); plt.close(fig)

    # PCA: by length (confound check)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.6,
                    c=np.log10(lengths), cmap="viridis", linewidths=0)
    ax.set_title("PCA coloured by log10(length) — length-confound check")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, label="log10(length)")
    overlay_highlights(ax, xy, hl)
    fig.tight_layout(); fig.savefig(outdir / f"pca_length_{base}.png", dpi=150); plt.close(fig)

    # PCA: microproteins vs metal clusters (Study B question)
    fig, ax = plt.subplots(figsize=(6, 5))
    metal = np.isin(classes, ["Zn", "Cu", "other_transition"])
    ax.scatter(xy[metal, 0], xy[metal, 1], s=8, alpha=0.35, c="#1f77b4", label="metal binders", linewidths=0)
    ax.scatter(xy[classes == "non_metal", 0], xy[classes == "non_metal", 1], s=8,
               alpha=0.2, c="#cccccc", label="non-metal", linewidths=0)
    mp = classes == "microprotein"
    ax.scatter(xy[mp, 0], xy[mp, 1], s=16, alpha=0.9, c="#d62728", label="microproteins", linewidths=0)
    overlay_highlights(ax, xy, hl)
    ax.set_title("Microproteins vs metal-binder clusters (PCA)")
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / f"pca_microprotein_{base}.png", dpi=150); plt.close(fig)

    # UMAP (exploration only) if available
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=args.umap_neighbors, min_dist=args.umap_mindist,
                            random_state=cfg["seed"])
        uxy = reducer.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter(ax, uxy, classes,
                f"UMAP (n_neighbors={args.umap_neighbors}, min_dist={args.umap_mindist}) "
                "— exploratory; do NOT read inter-cluster distance")
        overlay_highlights(ax, uxy, hl)
        fig.tight_layout(); fig.savefig(outdir / f"umap_class_{base}.png", dpi=150); plt.close(fig)
        print("wrote UMAP figure")
    except ImportError:
        print("umap-learn not installed; skipped UMAP (PCA figures written)")

    print(f"wrote PCA figures to {outdir} (PC1 {ev[0]:.1f}%, PC2 {ev[1]:.1f}%)")


if __name__ == "__main__":
    main()
