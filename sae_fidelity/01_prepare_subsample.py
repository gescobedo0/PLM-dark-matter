#!/usr/bin/env python
"""Build the three diagnostic groups and write subsample.parquet.

  A microprotein     : stratified ~5k ORFs (subclass x length bin)
  B length_matched   : canonical N-terminal fragments matched to A's lengths
  C background        : random full-length canonical proteins

Usage:
  python 01_prepare_subsample.py                 # full sizes from config
  python 01_prepare_subsample.py --smoke 100     # 100 per group, for a fast run
  python 01_prepare_subsample.py --no-controls   # ORFs only (controls need network)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sae_lib.config import load_config
from sae_lib import catalog, controls, markers


def stratified_sample(df: pd.DataFrame, key_cols, n: int, seed: int) -> pd.DataFrame:
    """Proportional stratified sample of ~n rows over the given key columns."""
    rng = np.random.default_rng(seed)
    groups = list(df.groupby(key_cols, observed=True))
    total = len(df)
    picks = []
    for _, g in groups:
        take = int(round(n * len(g) / total))
        take = min(max(take, 0), len(g))
        if take == 0:
            continue
        picks.append(g.sample(n=take, random_state=int(rng.integers(1 << 31))))
    out = pd.concat(picks) if picks else df.iloc[:0]
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def add_biophysics(df: pd.DataFrame, tm_window: int, tm_threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["length"] = df["seq"].str.len()
    df["hydropathy"] = df["seq"].map(catalog.kd_hydropathy)
    df["tm_score"] = df["seq"].map(lambda s: catalog.max_tm_window(s, tm_window))
    df["tm_flag"] = np.where(df["tm_score"] >= tm_threshold, "TM", "soluble")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="rows per group for a fast run")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--marker-limit", type=int, default=60,
                    help="max sequences per positive-control marker set")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ss = cfg["subsample"]
    tmw, tmt = cfg["tm_proxy"]["window"], cfg["tm_proxy"]["threshold"]
    seed = cfg["seed"]
    bins = ss["length_bins"]

    n_a = args.smoke or ss["n_microprotein"]
    n_b = args.smoke or ss["n_length_matched"]
    n_c = args.smoke or ss["n_background"]

    # --- Group A: microproteins -------------------------------------------
    orfs = catalog.load_catalog(cfg["paths"]["catalog"], cfg["subclass_map"],
                                ss["min_len"], tmw, tmt)
    orfs["lbin"] = catalog.length_bin(orfs["length"], bins)
    a = stratified_sample(orfs, ["subclass", "lbin"], n_a, seed)
    a = a.assign(group="microprotein")[
        ["id", "group", "subclass", "seq", "length", "hydropathy", "tm_score", "tm_flag"]]
    print(f"[A] microproteins: {len(a)} "
          f"(from {orfs.attrs['n_after_length']} eligible ORFs)")
    print(a["subclass"].value_counts().to_string())

    parts = [a]

    # --- Groups B & C: canonical controls ---------------------------------
    if not args.no_controls:
        data_dir = Path(cfg["paths"]["data_dir"])
        fasta = data_dir / "swissprot_human.fasta"
        print(f"[BC] fetching SwissProt human -> {fasta} ...")
        controls.download_swissprot_human(fasta)
        canonical = controls.parse_fasta(fasta)
        print(f"[BC] canonical pool: {len(canonical)} proteins")

        b = controls.build_length_matched(
            canonical, a["length"].sample(n=n_b, replace=n_b > len(a),
                                          random_state=seed).values,
            ss["min_len"], seed)
        b = add_biophysics(b, tmw, tmt)

        max_bg = int(np.percentile(canonical.seq.str.len(), 99))
        c = controls.build_background(canonical, n_c, ss["min_len"], max_bg, seed)
        c = add_biophysics(c, tmw, tmt)

        print(f"[B] length_matched: {len(b)} (median len {b.length.median():.0f})")
        print(f"[C] background: {len(c)} (median len {c.length.median():.0f}, "
              f"capped at {max_bg})")
        parts += [b[a.columns], c[a.columns]]

        # --- Positive-control markers (flow through the same pipeline) -----
        print("[M] fetching positive-control marker sets ...")
        m = markers.build_all_markers(data_dir / "markers",
                                      limit=args.marker_limit)
        m = m.rename(columns={"marker_set": "subclass"}).assign(group="marker")
        m = add_biophysics(m, tmw, tmt)
        print(f"[M] markers: {len(m)} across {m['subclass'].nunique()} sets")
        parts.append(m[a.columns])
    else:
        print("[BC] skipped (--no-controls)")

    out = pd.concat(parts, ignore_index=True)
    dest = Path(cfg["paths"]["subsample"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"\nwrote {len(out)} rows -> {dest}")
    print(out.groupby("group").agg(n=("id", "size"),
          median_len=("length", "median")).to_string())


if __name__ == "__main__":
    main()
