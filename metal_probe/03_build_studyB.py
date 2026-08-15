#!/usr/bin/env python3
"""
Study B classes (exploratory / motivating) — all from data already in the repo.

  1. microprotein         : neworfcatalog.csv v45 Ribo-seq ORFs, 30-150 aa band,
                            stratified across orf_type.
  2. size_matched_short   : reviewed human Swiss-Prot, 30-150 aa, NON-metal
                            (UniProt KW-0479 excluded) -> separates "is short"
                            from "is a microprotein" (the control that makes a
                            microprotein cluster interpretable).
  3. general_background   : reviewed human Swiss-Prot across normal lengths.

Length band note: neworfcatalog median is ~20 aa, but the size-matched control
only exists where human proteins exist (>=~30 aa). We therefore sample
microproteins from the 30-150 aa band (spec's microcin band) so every
microprotein has a length-matched human control; sub-30-aa ORFs (the majority,
and the SAE workstream's domain) are out of scope for THIS comparison.

Study B is not part of the leakage-safe probe (guardrail #1 note: redundancy in
these classes is only cosmetic), so cluster30 is left null here.

Outputs: metal_probe/catalog/studyB_pool.parquet, studyB_pool.fasta
"""
import csv
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / "catalog"
CACHE = ROOT / "data_cache"
ORF_CSV = REPO / "neworfcatalog.csv"
SWISS = REPO / "data" / "swissprot_human.fasta"
METAL_ACC = CACHE / "uniprot_human_metalbinding_KW0479.tsv"

SEED = 0
BAND = (30, 150)                 # microprotein / size-matched band
N_MICRO, N_SHORT, N_BKG = 300, 300, 500
BKG_MIN = 30                     # "normal length" background lower bound
rng = random.Random(SEED)


def read_fasta(path):
    recs, acc, name, gene, desc, seq = [], None, None, None, None, []
    def flush():
        if acc:
            recs.append({"acc": acc, "name": name, "gene": gene,
                         "desc": desc, "seq": "".join(seq)})
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                flush()
                seq = []
                h = line[1:].strip()
                parts = h.split("|")
                acc = parts[1] if len(parts) > 2 else h.split()[0]
                rest = parts[2] if len(parts) > 2 else h
                name = rest.split()[0]
                gene = None
                for tok in rest.split():
                    if tok.startswith("GN="):
                        gene = tok[3:]
                desc = rest.split(" OS=")[0]
                if " " in desc:
                    desc = desc.split(" ", 1)[1]
            else:
                seq.append(line.strip())
        flush()
    return recs


def build_microproteins():
    band_rows = []
    with open(ORF_CSV, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            aa = (row.get("sequence_aa") or "").strip().rstrip("*")
            if not aa or not (BAND[0] <= len(aa) <= BAND[1]):
                continue
            band_rows.append({
                "protein_id": row["releasev45_id"], "source_db": "neworfcatalog_v45",
                "class": "microprotein", "label": None, "organism": "Homo sapiens",
                "orf_type": (row.get("orf_type") or "").strip(),
                "gene_name": (row.get("gene_name") or "").strip() or None,
                "transcript": (row.get("transcript") or "").strip() or None,
                "length": len(aa), "cluster30": None, "seq": aa})
    # stratified sample across orf_type, proportional to availability
    by_type = {}
    for r in band_rows:
        by_type.setdefault(r["orf_type"], []).append(r)
    total = len(band_rows)
    picked = []
    for t, rows in by_type.items():
        k = max(1, round(N_MICRO * len(rows) / total))
        picked.extend(rng.sample(rows, min(k, len(rows))))
    rng.shuffle(picked)
    picked = picked[:N_MICRO]
    return picked, {t: len(v) for t, v in by_type.items()}


def build_swiss(metal_acc, micro_lengths, binw=10):
    """size_matched_short is drawn to match the microprotein length HISTOGRAM
    (10-aa bins), not just the band, so it controls for length distribution."""
    recs = read_fasta(SWISS)
    short = [r for r in recs if BAND[0] <= len(r["seq"]) <= BAND[1]
             and r["acc"] not in metal_acc]
    bkg = [r for r in recs if len(r["seq"]) >= BKG_MIN]

    # bucket available short human proteins by length bin
    def binof(n):
        return (n // binw) * binw
    avail = {}
    for r in short:
        avail.setdefault(binof(len(r["seq"])), []).append(r)
    # target count per bin = microprotein count in that bin
    micro_bins = {}
    for L in micro_lengths:
        micro_bins[binof(L)] = micro_bins.get(binof(L), 0) + 1
    short_pick, shortfall = [], {}
    for b, want in sorted(micro_bins.items()):
        have = avail.get(b, [])
        take = min(want, len(have))
        short_pick.extend(rng.sample(have, take))
        if take < want:
            shortfall[b] = want - take
    bkg_pick = rng.sample(bkg, min(N_BKG, len(bkg)))

    def mk(r, cls):
        return {"protein_id": r["acc"], "source_db": "SwissProt_human",
                "class": cls, "label": None, "organism": "Homo sapiens",
                "orf_type": None, "gene_name": r["gene"], "transcript": None,
                "length": len(r["seq"]), "cluster30": None, "seq": r["seq"],
                "desc": r["desc"]}
    return ([mk(r, "size_matched_short") for r in short_pick],
            [mk(r, "general_background") for r in bkg_pick],
            len(short), len(bkg), shortfall)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metal_acc = set()
    if METAL_ACC.exists():
        for line in METAL_ACC.read_text().splitlines()[1:]:
            if line.strip():
                metal_acc.add(line.strip())
    print(f"human metal-binding accessions excluded from control: {len(metal_acc)}")

    micro, type_avail = build_microproteins()
    micro_lengths = [r["length"] for r in micro]
    short, bkg, n_short_avail, n_bkg_avail, shortfall = build_swiss(metal_acc, micro_lengths)
    print(f"microproteins : {len(micro)} (band {BAND[0]}-{BAND[1]} aa)")
    print(f"  orf_type availability in band: {type_avail}")
    print(f"size_matched  : {len(short)} (length-histogram matched to microproteins; "
          f"{n_short_avail} non-metal short available)")
    if shortfall:
        print(f"  !! length bins where human controls ran short: {shortfall}")
    print(f"background    : {len(bkg)} of {n_bkg_avail} available")

    all_rows = micro + short + bkg
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT / "studyB_pool.parquet", index=False)
    with open(OUT / "studyB_pool.fasta", "w") as f:
        for _, r in df.iterrows():
            f.write(f">{r['protein_id']} class={r['class']} len={r['length']}\n{r['seq']}\n")

    print("\nby class:")
    for cls, g in df.groupby("class"):
        print(f"  {cls:<20} n={len(g):>4}  len {g.length.min()}-{int(g.length.median())}-{g.length.max()}")
    mic = df[df["class"] == "microprotein"]
    print("\nmicroprotein orf_type breakdown:", mic["orf_type"].value_counts().to_dict())
    print(f"\nwrote {OUT/'studyB_pool.parquet'}")


if __name__ == "__main__":
    main()
