#!/usr/bin/env python3
"""
Assemble the final catalog from the three pools (01/02/03).

Study A is sampled to the locked targets with ONE sequence per 30% cluster
(leakage-safe; the per-cluster representative is the best-resolution structure):
  positives : Zn 400 / Cu 200 / other-transition 200
  negatives : 800  (drawn from the 1,200-cluster pool)
Study B is taken whole (exploratory; cluster30 null, not in the probe split).

`split_group` = cluster30 for Study A -> the GroupKFold key (guardrail #1).
Positive and negative clusters are disjoint by construction (02), so no metal /
non-metal pair can straddle a CV fold.

Outputs (metal_probe/catalog/):
  catalog.parquet          full provenance, one row per sequence
  catalog.fasta            all sequences (embedding input)
  protein_cluster_map.csv  protein_id -> cluster30 -> split_group (first-class map)
  CATALOG_SUMMARY.md
"""
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "catalog"
SEED = 0
TARGETS = {"Zn": 400, "Cu": 200, "other_transition": 200}
N_NEG = 800
rng = random.Random(SEED)


def one_per_cluster(df, k, used=None):
    """pick k clusters (globally unused), best-resolution rep sequence from each."""
    if used is None:
        used = set()
    clusters = [c for c in sorted(df["cluster30"].astype(str).unique()) if c not in used]
    rng.shuffle(clusters)
    reps = []
    for c in clusters:
        if len(reps) >= k:
            break
        g = df[df["cluster30"].astype(str) == c].sort_values("resolution", na_position="last")
        reps.append(g.iloc[0])
        used.add(c)
    return pd.DataFrame(reps)


COMMON = ["protein_id", "study", "role", "class", "label", "source_db",
          "pdb", "chain", "entity", "uniprot", "organism", "resolution",
          "evidence", "ions", "coord_residues", "ec", "go", "orf_type",
          "gene_name", "length", "cluster30", "split_group", "n_redundant_chains",
          "label_rule", "seq"]


def conform(df):
    for c in COMMON:
        if c not in df.columns:
            df[c] = None
    return df[COMMON]


def main():
    pos = pd.read_parquet(OUT / "positives_pool.parquet")
    neg = pd.read_parquet(OUT / "negatives_pool.parquet")
    b = pd.read_parquet(OUT / "studyB_pool.parquet")

    # --- Study A positives: sample per class, globally 1/cluster ---
    used = set()
    parts = []
    for cls, k in TARGETS.items():
        sub = pos[pos["class"] == cls]
        picked = one_per_cluster(sub, k, used)
        if len(picked) < k:
            print(f"  !! {cls}: only {len(picked)} clusters available (< {k})")
        parts.append(picked)
    posA = pd.concat(parts, ignore_index=True)
    posA["study"] = "A"; posA["role"] = "positive"

    # --- Study A negatives: 800 clusters, 1/cluster (pool already 1/cluster) ---
    negA = one_per_cluster(neg, N_NEG, used)
    negA["study"] = "A"; negA["role"] = "negative"

    # --- Study B: all ---
    b["study"] = "B"; b["role"] = b["class"]
    b["resolution"] = None

    # split_group: cluster for A, unique per-row for B (excluded from probe anyway)
    posA["split_group"] = posA["cluster30"].astype(str)
    negA["split_group"] = negA["cluster30"].astype(str)
    b["split_group"] = None

    cat = pd.concat([conform(posA), conform(negA), conform(b)], ignore_index=True)

    # --- final exact-sequence dedup across the whole catalog ---
    before = len(cat)
    cat = cat.drop_duplicates(subset="seq", keep="first").reset_index(drop=True)
    if before != len(cat):
        print(f"  dropped {before-len(cat)} cross-class exact-duplicate sequences")

    # leakage sanity: no cluster shared between positive and negative
    pc = set(cat[(cat.role == "positive")]["split_group"])
    nc = set(cat[(cat.role == "negative")]["split_group"])
    assert not (pc & nc), f"LEAKAGE: {len(pc&nc)} clusters shared pos/neg"

    cat.to_parquet(OUT / "catalog.parquet", index=False)
    with open(OUT / "catalog.fasta", "w") as f:
        for _, r in cat.iterrows():
            f.write(f">{r['protein_id']} study={r['study']} class={r['class']}\n{r['seq']}\n")
    cat[["protein_id", "study", "class", "cluster30", "split_group"]].to_csv(
        OUT / "protein_cluster_map.csv", index=False)

    # summary
    L = ["# Catalog summary", ""]
    L.append(f"total sequences: **{len(cat)}**  |  seed={SEED}")
    L.append("")
    L.append("| study | role / class | n | clusters | len min/med/max |")
    L.append("|---|---|--:|--:|---|")
    for (study, role), g in cat.groupby(["study", "role"]):
        nclu = g["split_group"].nunique() if study == "A" else "-"
        L.append(f"| {study} | {role} | {len(g)} | {nclu} | "
                 f"{g.length.min()}/{int(g.length.median())}/{g.length.max()} |")
    L.append("")
    A = cat[cat.study == "A"]
    L.append(f"Study A probe set: {len(A)} sequences, "
             f"{A['split_group'].nunique()} clusters "
             f"({(A.label==1).sum()} metal / {(A.label==0).sum()} non-metal)")
    L.append(f"positive/negative cluster overlap: {len(pc & nc)} (leakage check)")
    L.append("")
    L.append("Study A positive ion membership:")
    for ion in ["Zn", "Cu", "Fe", "Mn", "Co", "Ni"]:
        n = A[A.ions.notna() & A.ions.str.contains(ion, na=False)].shape[0]
        L.append(f"- {ion}: {n}")
    (OUT / "CATALOG_SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {OUT/'catalog.parquet'} ({len(cat)} rows)")
    print(f"wrote {OUT/'catalog.fasta'}, {OUT/'protein_cluster_map.csv'}")


if __name__ == "__main__":
    main()
