#!/usr/bin/env python3
"""
Study A positives — metal-binding protein chains.

Pipeline:
  1. Parse BioLiP.txt.gz -> keep the locked ion set {Zn,Cu,Fe,Mn,Co,Ni}, res<3.0.
     Aggregate per (PDB, chain): ions bound, coordinating residues per ion,
     best resolution, EC/GO/UniProt, receptor sequence.  (BioLiP's biological-
     relevance curation already removes His-tag Ni / buffer metals.)
  2. One cached RCSB GraphQL pass over unique PDBs -> experimental method
     (enforce X-ray, honoring the locked quality choice, since BioLiP's
     resolution field cannot distinguish X-ray from cryo-EM) + authoritative
     resolution + (auth_chain -> polymer entity) map.
  3. Join RCSB precomputed 30% sequence-identity clusters (clusters-by-entity-30)
     -> cluster_id per chain  (leakage-safe split key, guardrail #1).
  4. Collapse 100%-identical sequences (guardrail: dedup before clustering counts).
  5. Write the full annotated, clustered, deduped positive pool + FASTA.

Target sampling (Zn 400 / Cu 200 / other-transition 200, one seq per cluster)
is deferred to the catalog-assembly step so positives/negatives/Study-B can be
balanced together. This script produces the *pool*, with full provenance.

Outputs (metal_probe/catalog/):
  positives_pool.parquet, positives_pool.fasta, positives_pool_summary.txt
Requires: pandas, pyarrow (present). No aligner needed (RCSB precomputed clusters).
"""
import gzip
import json
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
OUT = ROOT / "catalog"
BIOLIP = CACHE / "BioLiP.txt.gz"
CLUSTERS = CACHE / "clusters-by-entity-30.txt"
GQL_CACHE = CACHE / "rcsb_entry_meta.json"
GRAPHQL = "https://data.rcsb.org/graphql"

RES_MAX = 3.0
XRAY = "X-RAY DIFFRACTION"
# ion label -> CCD codes; class priority Zn > Cu > other-transition
ION_CODES = {"Zn": {"ZN"}, "Cu": {"CU", "CU1"}, "Fe": {"FE", "FE2"},
             "Mn": {"MN"}, "Co": {"CO", "3CO"}, "Ni": {"NI"}}
CODE2ION = {c: ion for ion, codes in ION_CODES.items() for c in codes}


# ---------------------------------------------------------------- stage 1
def parse_biolip():
    """(pdb, chain) -> aggregated record."""
    rec = {}
    with gzip.open(BIOLIP, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 21:
                continue
            code = p[4].strip()
            ion = CODE2ION.get(code)
            if ion is None:
                continue
            try:
                res = float(p[2])
            except ValueError:
                res = -1.0
            if not (0.0 < res < RES_MAX):
                continue
            pdb, chain = p[0].strip(), p[1].strip()
            key = (pdb, chain)
            r = rec.get(key)
            if r is None:
                r = rec[key] = {"pdb": pdb, "chain": chain, "resolution": res,
                                "ions": set(), "coord_res": defaultdict(set),
                                "ec": p[11].strip(), "go": p[12].strip(),
                                "uniprot": p[17].strip(), "seq": p[20].strip()}
            r["ions"].add(ion)
            for tok in p[7].split():
                r["coord_res"][ion].add(tok)
            r["resolution"] = min(r["resolution"], res)
            if len(p[20].strip()) > len(r["seq"]):   # keep longest observed seq
                r["seq"] = p[20].strip()
    return rec


# ---------------------------------------------------------------- stage 2
def fetch_rcsb_meta(pdb_ids, batch=300):
    """pdb -> {method, resolution, chain2entity{auth_chain: 'PDB_n'}}. Cached."""
    meta = {}
    if GQL_CACHE.exists():
        meta = json.loads(GQL_CACHE.read_text())
    todo = [p for p in pdb_ids if p.upper() not in meta]
    print(f"  RCSB meta: {len(pdb_ids)-len(todo)} cached, {len(todo)} to fetch")
    for i in range(0, len(todo), batch):
        chunk = [c.upper() for c in todo[i:i + batch]]
        q = ('{entries(entry_ids:%s){rcsb_id exptl{method} '
             'rcsb_entry_info{resolution_combined} polymer_entities{rcsb_id '
             'rcsb_entity_source_organism{scientific_name} '
             'rcsb_polymer_entity_container_identifiers{auth_asym_ids entity_id}}}}'
             % json.dumps(chunk))
        data = json.dumps({"query": q}).encode()
        for attempt in range(4):
            try:
                req = urllib.request.Request(GRAPHQL, data=data,
                                             headers={"Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=90))
                break
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        for e in r["data"]["entries"] or []:
            method = e["exptl"][0]["method"] if e.get("exptl") else None
            rc = e.get("rcsb_entry_info") or {}
            reslist = rc.get("resolution_combined") or []
            c2e = {}
            e2org = {}
            for pe in e.get("polymer_entities") or []:
                ent = pe["rcsb_id"]                       # 'PDB_n'
                ci = pe["rcsb_polymer_entity_container_identifiers"]
                for ch in (ci.get("auth_asym_ids") or []):
                    c2e[ch] = ent
                orgs = pe.get("rcsb_entity_source_organism") or []
                e2org[ent] = orgs[0]["scientific_name"] if orgs else None
            meta[e["rcsb_id"]] = {"method": method,
                                  "resolution": reslist[0] if reslist else None,
                                  "chain2entity": c2e, "entity2org": e2org}
        if (i // batch) % 10 == 0:
            GQL_CACHE.write_text(json.dumps(meta))
            print(f"    fetched {min(i+batch,len(todo))}/{len(todo)}")
    GQL_CACHE.write_text(json.dumps(meta))
    return meta


# ---------------------------------------------------------------- stage 3
def load_entity_clusters():
    """entity rcsb_id ('PDB_n') -> cluster index (line number in the file)."""
    ent2clu = {}
    with open(CLUSTERS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            for tok in line.split():
                ent2clu[tok] = i
    return ent2clu


# ---------------------------------------------------------------- main
def primary_class(ions):
    if "Zn" in ions:
        return "Zn"
    if "Cu" in ions:
        return "Cu"
    return "other_transition"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("stage 1: parse BioLiP ...")
    rec = parse_biolip()
    pdbs = sorted({k[0] for k in rec})
    print(f"  {len(rec)} (pdb,chain) metal chains across {len(pdbs)} PDBs")

    print("stage 2: RCSB method/entity meta ...")
    meta = fetch_rcsb_meta(pdbs)

    print("stage 3: load 30% clusters ...")
    ent2clu = load_entity_clusters()

    print("stage 4: assemble + X-ray gate + cluster join ...")
    rows = []
    n_nonxray = n_noentity = 0
    for (pdb, chain), r in rec.items():
        m = meta.get(pdb.upper())
        if not m or m.get("method") != XRAY:
            n_nonxray += 1
            continue
        ent = (m.get("chain2entity") or {}).get(chain)
        clu = ent2clu.get(ent) if ent else None
        if clu is None:
            n_noentity += 1
            clu = f"singleton_{pdb}_{chain}"    # unmapped -> its own cluster
        ions = sorted(r["ions"])
        org = (m.get("entity2org") or {}).get(ent)
        ec = r["ec"]
        ec = None if ec in ("", "?") else ec
        rows.append({
            "protein_id": f"{pdb}_{chain}",
            "source_db": "BioLiP",
            "pdb": pdb, "chain": chain, "entity": ent,
            "uniprot": r["uniprot"] or None, "organism": org,
            "resolution": m.get("resolution") or r["resolution"],
            "evidence": f"X-ray {m.get('resolution') or r['resolution']}A",
            "label": 1, "class": primary_class(ions),
            "label_rule": "BioLiP metal binding site; ions=" + "+".join(ions),
            "ions": "+".join(ions),
            "coord_residues": json.dumps({k: sorted(v) for k, v in r["coord_res"].items()}),
            "ec": ec, "go": r["go"] or None,
            "length": len(r["seq"]), "seq": r["seq"], "cluster30": clu,
        })
    df = pd.DataFrame(rows)
    print(f"  kept {len(df)} X-ray chains  (dropped {n_nonxray} non-X-ray; "
          f"{n_noentity} chains had no entity->cluster map -> singleton)")

    print("stage 5: collapse 100%-identical sequences ...")
    df = df[df["seq"].str.len() > 0].copy()
    agg = []
    for seq, g in df.groupby("seq", sort=False):
        g = g.sort_values("resolution")            # best-resolution representative
        rep = g.iloc[0].to_dict()
        all_ions = sorted({i for s in g["ions"] for i in s.split("+")})
        rep["ions"] = "+".join(all_ions)
        rep["class"] = primary_class(set(all_ions))
        rep["n_redundant_chains"] = len(g)
        rep["cluster30"] = sorted(g["cluster30"].astype(str))[0]  # stable pick
        rep["redundant_pdb_chains"] = ";".join(sorted(g["protein_id"]))
        agg.append(rep)
    pool = pd.DataFrame(agg)
    print(f"  {len(df)} chains -> {len(pool)} unique sequences "
          f"-> {pool['cluster30'].nunique()} 30% clusters")

    # outputs
    cols = ["protein_id", "source_db", "pdb", "chain", "entity", "uniprot",
            "organism", "resolution", "evidence", "label", "class", "ions",
            "coord_residues", "ec", "go", "length", "cluster30",
            "n_redundant_chains", "redundant_pdb_chains", "label_rule", "seq"]
    pool = pool[cols]
    pool.to_parquet(OUT / "positives_pool.parquet", index=False)
    with open(OUT / "positives_pool.fasta", "w") as f:
        for _, r in pool.iterrows():
            f.write(f">{r['protein_id']} class={r['class']} ions={r['ions']} "
                    f"cluster={r['cluster30']}\n{r['seq']}\n")

    # summary
    lines = ["Study A positives pool", "=" * 40,
             f"unique sequences : {len(pool)}",
             f"30% clusters     : {pool['cluster30'].nunique()}",
             f"median length    : {int(pool['length'].median())} aa",
             "", "by primary class (unique seqs / clusters):"]
    for cls, g in pool.groupby("class"):
        lines.append(f"  {cls:<18} {len(g):>6} seqs  {g['cluster30'].nunique():>6} clusters")
    lines.append("")
    lines.append("ion membership (unique seqs containing each ion):")
    for ion in ION_CODES:
        n = pool["ions"].str.contains(ion).sum()
        lines.append(f"  {ion:<4} {n:>6}")
    (OUT / "positives_pool_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT/'positives_pool.parquet'}")
    print(f"wrote {OUT/'positives_pool.fasta'}")


if __name__ == "__main__":
    main()
