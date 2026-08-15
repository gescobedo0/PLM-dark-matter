#!/usr/bin/env python
"""Compute the go/no-go diagnostics (handoff sections 1-4) and figures.

Reads 03 outputs + the subsample + the HF feature-description table, writes
results.json and figures under paths.report_dir. Makes no decision — 05 maps
these numbers onto the handoff decision table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sae_lib.config import load_config
from sae_lib import diagnostics, features, markers


def _plot_fvu(proteins: pd.DataFrame, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for g, sub in proteins.groupby("group"):
        v = np.sort(sub["protein_fvu"].values)
        ax.plot(v, np.linspace(0, 1, len(v)), label=f"{g} (n={len(v)})")
    ax.set_xlabel("per-protein FVU"); ax.set_ylabel("ECDF")
    ax.set_title("SAE reconstruction FVU by group"); ax.legend()
    ax.set_xlim(0, min(2.0, proteins["protein_fvu"].quantile(0.99)))
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-feature-table", action="store_true",
                    help="skip HF feature-table steps (informativeness, markers)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    feat_dir = Path(cfg["paths"]["sae_features"])
    rep_dir = Path(cfg["paths"]["report_dir"]); rep_dir.mkdir(parents=True, exist_ok=True)

    proteins = pd.read_parquet(feat_dir / "proteins.parquet")
    sub = pd.read_parquet(cfg["paths"]["subsample"])[["id", "tm_flag", "length"]]
    proteins = proteins.merge(sub, on="id", how="left")
    pooled_npz = np.load(feat_dir / "pooled.npz", allow_pickle=True)
    pooled = pooled_npz["pooled"].astype(np.float32)
    pooled_ids = list(pooled_npz["ids"])
    gstats = np.load(feat_dir / "group_stats.npz", allow_pickle=True)

    results = {"n_proteins": int(len(proteins))}

    # --- Section 1: FVU -------------------------------------------------
    micro = proteins["group"] == "microprotein"
    fvu_sum = diagnostics.summarize_fvu(
        proteins["protein_fvu"].values, proteins["group"].values,
        np.where(micro, proteins["subclass"].values, None))
    # TM/soluble breakout within microproteins
    tm_rows = []
    for tmf, s in proteins[micro].groupby("tm_flag"):
        tm_rows.append(diagnostics._fvu_row("microprotein", f"tm:{tmf}",
                                            s["protein_fvu"].values))
    fvu_sum = pd.concat([fvu_sum, pd.DataFrame(tm_rows)], ignore_index=True)
    results["fvu_summary"] = fvu_sum.to_dict(orient="records")
    results["fvu_verdict"] = diagnostics.fvu_verdict(
        fvu_sum, cfg["decision"]["fvu_comparable_ratio"],
        cfg["decision"]["fvu_elevated_ratio"])
    _plot_fvu(proteins, rep_dir / "fvu_ecdf.png")

    # --- Section 2: magnitude + entropy ---------------------------------
    groups = list(gstats["groups"])
    usage = gstats["usage"]
    results["magnitude"] = {g: diagnostics.activation_magnitude_stats(gstats[f"mag_{g}"])
                            for g in groups if f"mag_{g}" in gstats}
    results["entropy_bits"] = {g: diagnostics.entropy_from_usage(usage[i])
                               for i, g in enumerate(groups)}
    results["max_entropy_bits"] = float(np.log2(cfg["sae"]["codebook"]))

    # --- Section 2: description informativeness + Section 4: markers -----
    if not args.no_feature_table:
        ft = features.tag_informative(features.load_feature_table(
            cfg["hf"]["feature_table_repo"], cfg["hf"]["feature_table_file"]))
        results["informative"] = {}
        idpos = {pid: i for i, pid in enumerate(pooled_ids)}
        for g in groups:
            rows = [idpos[p] for p in proteins.loc[proteins.group == g, "id"]
                    if p in idpos]
            if rows:
                results["informative"][g] = diagnostics.top_feature_informativeness(
                    pooled[rows], ft)

        # PRE-REGISTER expected features per concept, then test recovery
        concept_ids = {c: features.search_features(ft, kw)
                       for c, kw in features.MARKER_CONCEPTS.items()}
        results["preregistered_feature_counts"] = {c: len(v) for c, v in concept_ids.items()}
        bg_rows = [idpos[p] for p in proteins.loc[proteins.group == "background", "id"]
                   if p in idpos]
        results["marker_recovery"] = {}
        for name, (_q, concept) in markers.MARKER_QUERIES.items():
            m_rows = [idpos[p] for p in proteins.loc[proteins.subclass == name, "id"]
                      if p in idpos]
            exp = concept_ids.get(concept, [])
            if m_rows and bg_rows and exp:
                results["marker_recovery"][name] = {
                    "concept": concept,
                    **diagnostics.marker_recovery(pooled[m_rows], exp, pooled[bg_rows])}
            else:
                results["marker_recovery"][name] = {
                    "concept": concept, "skipped": True,
                    "reason": f"m_rows={len(m_rows)} bg={len(bg_rows)} exp={len(exp)}"}

        # Disorder NEGATIVE control: structured-domain features must NOT fire on
        # disordered proteins (handoff section 4). Expect LOW fraction-above-bg.
        struct_ids = sorted(set(concept_ids.get("metal_coordination", [])) |
                            set(concept_ids.get("fe_s_coordination", [])) |
                            set(concept_ids.get("membrane", [])))
        dis_rows = [idpos[p] for p in
                    proteins.loc[proteins.subclass == "disordered_small", "id"]
                    if p in idpos]
        if dis_rows and bg_rows and struct_ids:
            results["disorder_negative_control"] = diagnostics.marker_recovery(
                pooled[dis_rows], struct_ids, pooled[bg_rows])
    else:
        print("skipping feature-table steps (--no-feature-table)")

    (rep_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {rep_dir/'results.json'} and {rep_dir/'fvu_ecdf.png'}")
    # console summary
    print("\nFVU verdict vs background (median):")
    for r in results["fvu_verdict"]["rows"]:
        print(f"  {r['group']:>14} / {r['subclass']:<22} "
              f"ratio={r['ratio_vs_bg']:.2f}  [{r['label']}]")


if __name__ == "__main__":
    main()
