#!/usr/bin/env python3
"""
Standalone interactive PCA (no Jupyter needed) + per-point neighbour report.

Writes a self-contained HTML you can open in any browser and hover to read each
point's identity (protein_id, class, ion, length, EC, organism). Highlighted
ORFs (add_highlights.py) show as stars. Also writes, for every highlighted ORF,
its full-D nearest metal binders — the rigorous "what is this near?" view that
PCA can't give.

CPU-only; consumes the precomputed embeddings. Needs plotly (pip install plotly).

Usage:
  python 10_interactive_pca.py --model 650M --layer 6      # early-peak layer
  python 10_interactive_pca.py --model 650M                # deepest layer
"""
import argparse
from pathlib import Path

from embed_lib import explore

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="650M")
    ap.add_argument("--layer", type=int, default=None, help="default: deepest stored")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--color", default="class", help="catalog column to colour by")
    ap.add_argument("--k", type=int, default=15, help="neighbours per highlighted ORF")
    args = ap.parse_args()

    meta, E, ids = explore.load_embeddings(args.model, args.layer, args.pooling)
    layer = int(meta["_layer"].iloc[0])
    coords, ev = explore.pca2d(E)
    out = ROOT / "results" / "interactive"
    out.mkdir(parents=True, exist_ok=True)

    title = (f"ESM-2 {args.model} L{layer} {args.pooling} — PCA "
             f"(PC1 {ev[0]:.1f}%, PC2 {ev[1]:.1f}%). Hover to identify.")
    fig = explore.make_figure(coords, meta, color=args.color, title=title)
    html = out / f"pca_{args.model}_L{layer}_{args.pooling}_{args.color}.html"
    explore.write_html(fig, html)
    print(f"wrote {html}  (open in a browser; hover any point)")

    # per-highlighted-ORF nearest metal binders (full-D)
    if "highlight" in meta.columns and meta["highlight"].fillna(False).any():
        hl = meta.index[meta["highlight"].fillna(False)]
        report = []
        for pid in hl:
            prof, med = explore.metal_ion_profile(pid, ids, E, meta, k=args.k)
            nb = explore.neighbors(pid, ids, E, meta, k=args.k, restrict="metal")
            nb.insert(0, "query", pid)
            report.append(nb)
            top = ", ".join(f"{k}:{v}" for k, v in prof.items())
            print(f"  {pid}: nearest {args.k} metal binders -> ions [{top}], "
                  f"median cos-dist {med:.3f}")
        import pandas as pd
        pd.concat(report, ignore_index=True).to_csv(
            out / "highlighted_metal_neighbors.csv", index=False)
        print(f"wrote {out/'highlighted_metal_neighbors.csv'}")
    else:
        print("  (no highlighted ORFs; run add_highlights.py to spotlight some)")


if __name__ == "__main__":
    main()
