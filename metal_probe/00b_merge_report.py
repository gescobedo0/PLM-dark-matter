#!/usr/bin/env python3
"""
Phase 0 report — merge RCSB structure/cluster counts (counts.json, from
00_phase0_counts.py) with BioLiP per-metal biological-relevance counts
(biolip_lig_frequency_*.txt) into one table + a markdown report.

The BioLiP column is the spec-mandated cross-check. Units differ on purpose:
  RCSB entries          = STRUCTURES containing the metal (X-ray, res<=3.0)
  RCSB clusters@30      = 30% seq-identity clusters among those structures'
                          polymer entities (upper bound on binder clusters)
  BioLiP entries        = biologically-relevant binding SITES (additive-filtered)
A metal whose BioLiP count collapses relative to RCSB is largely adventitious
(buffer / His-tag / heavy-atom) rather than a genuine coordination site.
"""
import json
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
P0 = HERE / "phase0"

# ion -> BioLiP comp codes to sum (mirrors IONS grouping in 00_phase0_counts.py)
BIOLIP_GROUPS = {
    "Zn": ["ZN"], "Cu": ["CU", "CU1"], "Fe (ionic)": ["FE", "FE2"], "Mn": ["MN"],
    "Co": ["CO", "3CO"], "Ni": ["NI"], "Mo": ["MO", "4MO", "6MO", "2MO", "MOO"],
    "Ca": ["CA"], "Mg": ["MG"], "Na": ["NA"], "K": ["K"],
    "Fe (heme)": ["HEM", "HEC", "HEA", "HEB"], "Fe-S cluster": ["FES", "SF4", "F3S"],
    "Cd (adventitious)": ["CD"], "Hg (adventitious)": ["HG"],
}


def load_biolip():
    f = sorted(P0.glob("biolip_lig_frequency_*.txt"))
    if not f:
        return {}, None
    freq = {}
    for line in f[-1].read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("#", "Rank")) or not line.strip():
            continue
        p = line.split("\t")
        if len(p) >= 3:
            try:
                freq[p[1].strip()] = int(p[2])
            except ValueError:
                pass
    return freq, f[-1].name


def main():
    counts = json.loads((P0 / "counts.json").read_text())
    biolip, biolip_src = load_biolip()

    merged = []
    for r in counts["rows"]:
        ion = r["ion"]
        codes = BIOLIP_GROUPS.get(ion, [])
        bl = sum(biolip.get(c, 0) for c in codes) if biolip else None
        rcsb_str = r["entries_xray_res3.0"]
        # crude adventitious flag: biological sites far below structure count
        ratio = (bl / rcsb_str) if (bl and rcsb_str) else None
        merged.append({
            "ion": ion, "category": r["category"], "codes": r["codes"],
            "rcsb_structures": rcsb_str,
            "rcsb_clusters_30pct": r["clusters_30pct"],
            "biolip_sites": bl,
            "biolip_per_structure": round(ratio, 2) if ratio else None,
        })

    # write merged CSV
    with open(P0 / "phase0_merged.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)

    # markdown report
    lines = []
    lines.append("# Phase 0 — per-ion counts (metal-binding probe)\n")
    lines.append(f"- RCSB Search API, generated {counts['generated']}; "
                 f"X-ray only, resolution <= {counts['res_cutoff']} A.")
    lines.append(f"- Clusters = {counts['cluster_cutoff_pct']}% sequence-identity clusters "
                 "(RCSB precomputed) over polymer entities in metal-containing structures "
                 "-> **upper bound** on binder clusters (coordinating-chain annotation deferred).")
    lines.append(f"- BioLiP cross-check from `{biolip_src}` = biologically-relevant binding "
                 "*sites* (crystallization additives filtered).\n")
    lines.append("| Ion | Category | RCSB structures | Clusters@30% | BioLiP sites | BioLiP/structure |")
    lines.append("|-----|----------|----------------:|-------------:|-------------:|-----------------:|")
    for m in merged:
        lines.append(f"| {m['ion']} | {m['category']} | {m['rcsb_structures']:,} | "
                     f"{m['rcsb_clusters_30pct']:,} | "
                     f"{m['biolip_sites']:,} | {m['biolip_per_structure']} |"
                     if m['biolip_sites'] is not None else
                     f"| {m['ion']} | {m['category']} | {m['rcsb_structures']:,} | "
                     f"{m['rcsb_clusters_30pct']:,} | n/a | n/a |")
    (P0 / "PHASE0_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # echo to stdout
    print("\n".join(lines))
    print(f"\nwrote {P0/'phase0_merged.csv'}")
    print(f"wrote {P0/'PHASE0_REPORT.md'}")


if __name__ == "__main__":
    main()
