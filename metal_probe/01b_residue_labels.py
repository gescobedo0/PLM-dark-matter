#!/usr/bin/env python3
"""
Data fix for the per-residue phase: derive coordinating-residue positions as
0-based SEQUENCE indices (the catalog stored BioLiP col 8 = PDB numbering, which
does not align to the stored sequence).

BioLiP col 9 ("residues re-numbered starting from 1") aligns to the receptor
sequence, so we re-parse it for every positive-pool chain and validate that the
residue letter matches the stored sequence at that position.

Output: catalog/residue_labels.parquet  (long: protein_id, position, aa, ion)
Only coordinating residues are listed; everything else is a non-coordinating
(negative) residue by construction. Runs locally (BioLiP.txt.gz cached).
"""
import gzip
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BIOLIP = ROOT / "data_cache" / "BioLiP.txt.gz"
POOL = ROOT / "catalog" / "positives_pool.parquet"
OUT = ROOT / "catalog" / "residue_labels.parquet"

ION_CODES = {"Zn": {"ZN"}, "Cu": {"CU", "CU1"}, "Fe": {"FE", "FE2"},
             "Mn": {"MN"}, "Co": {"CO", "3CO"}, "Ni": {"NI"}}
CODE2ION = {c: ion for ion, codes in ION_CODES.items() for c in codes}
TOKEN = re.compile(r"^([A-Z])(\d+)$")


def main():
    pool = pd.read_parquet(POOL)
    # protein_id is representative pdb_chain; map -> (pdb, chain) and its sequence
    want = {}
    for _, r in pool.iterrows():
        pdb, chain = r["protein_id"].rsplit("_", 1)
        want[(pdb, chain)] = r["protein_id"]
    seqof = dict(zip(pool["protein_id"], pool["seq"]))
    print(f"positive-pool chains to label: {len(want)}")

    # one pass over BioLiP: collect col-9 renumbered residues per (pdb,chain,ion)
    col9 = defaultdict(lambda: defaultdict(set))   # (pdb,chain) -> ion -> {tokens}
    with gzip.open(BIOLIP, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 21:
                continue
            ion = CODE2ION.get(p[4].strip())
            if ion is None:
                continue
            key = (p[0].strip(), p[1].strip())
            if key not in want:
                continue
            try:
                res = float(p[2])
            except ValueError:
                res = -1.0
            if not (0.0 < res < 3.0):
                continue
            for tok in p[8].split():                # col 9 = renumbered
                col9[key][ion].add(tok)

    rows, n_ok, n_bad, no_label = [], 0, 0, 0
    for key, pid in want.items():
        seq = seqof[pid]
        ions = col9.get(key)
        if not ions:
            no_label += 1
            continue
        for ion, toks in ions.items():
            for tok in toks:
                m = TOKEN.match(tok)
                if not m:
                    continue
                aa, num = m.group(1), int(m.group(2))
                pos0 = num - 1
                if 0 <= pos0 < len(seq) and seq[pos0] == aa:
                    rows.append({"protein_id": pid, "position": pos0, "aa": aa, "ion": ion})
                    n_ok += 1
                else:
                    n_bad += 1

    df = pd.DataFrame(rows).drop_duplicates(["protein_id", "position", "ion"])
    df.to_parquet(OUT, index=False)
    print(f"coordinating residues written: {len(df)}  "
          f"(validated {n_ok}, seq-mismatch dropped {n_bad}, "
          f"{100*n_bad/max(1,n_ok+n_bad):.2f}% mismatch)")
    print(f"pool chains with >=1 label: {df['protein_id'].nunique()} "
          f"(no BioLiP rows for {no_label})")
    print("coordinating residues per ion:",
          df.groupby("ion").size().to_dict())
    print("coordinating-residue amino-acid makeup:",
          df["aa"].value_counts().head(8).to_dict())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
