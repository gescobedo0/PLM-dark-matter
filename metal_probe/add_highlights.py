#!/usr/bin/env python3
"""
Mark a hand-picked set of ORFs as `highlight=True` in the catalog so they can be
spotlighted on the embedding plots (08) and scored individually (09).

Any highlighted ORF that isn't already among the sampled microproteins is
APPENDED to the catalog as a microprotein row (with provenance from
neworfcatalog.csv) so it gets an embedding on the next embed run. catalog.fasta
is rewritten to include them.

Identify ORFs by releasev45_id (e.g. c12riboseqorf747) or by 0-based data-row
index into neworfcatalog.csv. Mix freely; digit-only tokens are treated as row
indices, everything else as an id.

Usage:
  python add_highlights.py --ids c12riboseqorf747,c3riboseqorf88
  python add_highlights.py --rows 41,102,3350
  python add_highlights.py --file my_hits.txt        # one token per line
Re-running is idempotent. NB: re-running 04_assemble_catalog.py resets the
highlight column, so run this after it.
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CAT = ROOT / "catalog" / "catalog.parquet"
FASTA = ROOT / "catalog" / "catalog.fasta"
ORF_CSV = REPO / "neworfcatalog.csv"


def load_orf_index():
    """row_index -> record; releasev45_id -> record."""
    by_row, by_id = [], {}
    with open(ORF_CSV, encoding="utf-8", errors="ignore") as f:
        for i, row in enumerate(csv.DictReader(f)):
            aa = (row.get("sequence_aa") or "").strip().rstrip("*")
            rec = {"id": row["releasev45_id"], "orf_type": (row.get("orf_type") or "").strip(),
                   "gene_name": (row.get("gene_name") or "").strip() or None,
                   "transcript": (row.get("transcript") or "").strip() or None, "seq": aa}
            by_row.append(rec)
            by_id[rec["id"]] = rec
    return by_row, by_id


def resolve(tokens, by_row, by_id):
    out, missing = [], []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t.isdigit():
            i = int(t)
            (out.append(by_row[i]) if 0 <= i < len(by_row) else missing.append(t))
        elif t in by_id:
            out.append(by_id[t])
        else:
            missing.append(t)
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--rows", default="")
    ap.add_argument("--file", default=None)
    ap.add_argument("--clear", action="store_true", help="reset all highlights to False first")
    args = ap.parse_args()

    tokens = []
    tokens += [x for x in args.ids.split(",") if x]
    tokens += [x for x in args.rows.split(",") if x]
    if args.file:
        tokens += Path(args.file).read_text().split()
    if not tokens and not args.clear:
        raise SystemExit("provide --ids / --rows / --file")

    cat = pd.read_parquet(CAT)
    if "highlight" not in cat.columns or args.clear:
        cat["highlight"] = False
    cat["highlight"] = cat["highlight"].fillna(False).astype(bool)

    by_row, by_id = load_orf_index()
    recs, missing = resolve(tokens, by_row, by_id)
    if missing:
        print(f"  WARNING: {len(missing)} token(s) not found: {missing[:10]}")

    have_ids = set(cat["protein_id"])
    seq2idx = {s: i for i, s in enumerate(cat["seq"])}
    marked, added = 0, 0
    new_rows = []
    for r in recs:
        if not r["seq"]:
            continue
        if r["id"] in have_ids:
            cat.loc[cat.protein_id == r["id"], "highlight"] = True
            marked += 1
        elif r["seq"] in seq2idx:                      # same seq under another id
            cat.iloc[seq2idx[r["seq"]], cat.columns.get_loc("highlight")] = True
            marked += 1
        else:
            row = {c: None for c in cat.columns}
            row.update({"protein_id": r["id"], "study": "B", "role": "microprotein",
                        "class": "microprotein", "source_db": "neworfcatalog_v45",
                        "organism": "Homo sapiens", "orf_type": r["orf_type"],
                        "gene_name": r["gene_name"], "length": len(r["seq"]),
                        "seq": r["seq"], "highlight": True})
            new_rows.append(row)
            added += 1
    if new_rows:
        cat = pd.concat([cat, pd.DataFrame(new_rows)], ignore_index=True)

    cat.to_parquet(CAT, index=False)
    with open(FASTA, "w") as f:
        for _, r in cat.iterrows():
            f.write(f">{r['protein_id']} study={r['study']} class={r['class']}\n{r['seq']}\n")

    print(f"highlighted: {int(cat['highlight'].sum())} total "
          f"({marked} already in catalog, {added} newly appended)")
    if added:
        print(f"  -> re-run the embed job so the {added} new ORF(s) get vectors.")
    print(f"catalog now {len(cat)} rows; wrote {CAT.name}, {FASTA.name}")


if __name__ == "__main__":
    main()
