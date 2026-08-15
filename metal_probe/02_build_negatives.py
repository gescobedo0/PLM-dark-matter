#!/usr/bin/env python3
"""
Study A negatives — non-metal protein chains from the SAME universe (guardrail #2).

Same universe as positives = PDB, X-ray, resolution <= 3.0. "Characterized" =
has a Pfam family assignment (not a hypothetical/uncharacterized protein), per
the spec's "studied, non-metal" intent. Metal-free is enforced twice:
  (a) exclude any 30% cluster that contains a positive (family-level: no
      metalloprotein relatives leak in), and
  (b) per-entry check: drop if the structure contains ANY metal ligand (a broad
      CCD list incl. heme / Fe-S cofactors), so negatives are truly metal-free
      rather than merely lacking the six probe ions.

Clusters come from RCSB's 30% grouping (same scheme as positives), so the
protein->cluster map is consistent across the whole Study A universe -> a single
GroupKFold split key with no positive/negative cluster overlap by construction.

Outputs: metal_probe/catalog/negatives_pool.parquet, negatives_pool.fasta
"""
import json
import random
import time
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
OUT = ROOT / "catalog"
CLUSTERS = CACHE / "clusters-by-entity-30.txt"
NEG_CACHE = CACHE / "rcsb_negative_meta.json"
SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL = "https://data.rcsb.org/graphql"

RES_MAX = 3.0
POOL_TARGET = 1200          # collect margin; final 800 sampled at assembly
SEED = 0
LEN_MIN, LEN_MAX = 30, 2000

# Broad metal / metallocofactor CCD codes -> any presence disqualifies a negative.
METALS = {
    "ZN", "CU", "CU1", "CU2", "CU3", "CUA", "CUB", "CUZ",
    "FE", "FE2", "FE3", "FES", "SF4", "F3S", "FES", "FCO", "FC6", "NFS", "NFU", "CFM", "CFN", "CLF", "ICS", "ICF",
    "MN", "MN3", "MN5", "MN6", "OMO", "OEC",
    "CO", "3CO", "NCO", "CON", "NI", "3NI", "NIK",
    "CA", "MG", "NA", "K", "CS", "RB", "LI", "BA", "SR", "BE",
    "CD", "HG", "MO", "MOO", "MOS", "2MO", "4MO", "6MO", "MSS", "MGD",
    "PT", "PT4", "PTN", "AU", "AU3", "AG", "PD", "PB", "PB1",
    "AL", "GA", "IN", "TL", "SN", "SB", "V", "VN3", "VO", "VO4",
    "W", "4W", "WO4", "CR", "CR3", "RU", "RH", "RH3", "OS", "IR", "IR3", "RE", "TC",
    "Y1", "SC", "ZR", "TI", "HF", "TA", "NB",
    "LA", "CE", "CE3", "PR", "ND", "SM", "SM3", "EU", "EU3", "GD", "GD3", "TB", "TB3",
    "DY", "HO", "HO3", "ER3", "TM", "YB", "YB2", "YB3", "LU", "U1", "UNL",
    "HEM", "HEC", "HEA", "HEB", "HEV", "HAS", "HDD", "DHE", "SRM", "HNI", "COH", "HFM",
    "1FH", "HIF", "HEO", "VER", "MH0", "FEO", "HDM", "HP5", "89R", "76R", "6HE", "HE5",
}


def _post(url, payload, retries=4, timeout=90):
    data = json.dumps(payload).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError) as e:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def T(attr, op, val):
    return {"type": "terminal", "service": "text",
            "parameters": {"attribute": attr, "operator": op, "value": val}}


def fetch_representatives():
    base = [T("rcsb_entry_info.experimental_method", "exact_match", "X-ray"),
            T("rcsb_entry_info.resolution_combined", "less_or_equal", RES_MAX),
            T("entity_poly.rcsb_entity_polymer_type", "exact_match", "Protein"),
            T("rcsb_polymer_entity_annotation.type", "exact_match", "Pfam")]
    reps, start, rows = [], 0, 10000
    while True:
        q = {"query": {"type": "group", "logical_operator": "and", "nodes": base},
             "return_type": "polymer_entity",
             "request_options": {
                 "group_by": {"aggregation_method": "sequence_identity", "similarity_cutoff": 30},
                 "group_by_return_type": "representatives",
                 "paginate": {"start": start, "rows": rows}}}
        r = _post(SEARCH, q)
        batch = [x["identifier"] for x in r.get("result_set", [])]
        reps.extend(batch)
        if len(batch) < rows or start + rows >= r.get("group_by_count", 0):
            break
        start += rows
    return reps


def load_entity_clusters():
    ent2clu = {}
    with open(CLUSTERS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            for tok in line.split():
                ent2clu[tok] = i
    return ent2clu


def fetch_entity_meta(entity_ids, batch=200):
    meta = json.loads(NEG_CACHE.read_text()) if NEG_CACHE.exists() else {}
    todo = [e for e in entity_ids if e not in meta]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        q = ("{polymer_entities(entity_ids:%s){rcsb_id "
             "entity_poly{pdbx_seq_one_letter_code_can} "
             "rcsb_polymer_entity{pdbx_ec} rcsb_entity_source_organism{scientific_name} "
             "entry{nonpolymer_entities{nonpolymer_comp{chem_comp{id}}}}}}" % json.dumps(chunk))
        r = _post(GRAPHQL, {"query": q})
        for e in (r.get("data", {}).get("polymer_entities") or []):
            seq = (e.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can") or ""
            org = [o["scientific_name"] for o in (e.get("rcsb_entity_source_organism") or [])]
            ligs = [c["nonpolymer_comp"]["chem_comp"]["id"]
                    for c in ((e.get("entry") or {}).get("nonpolymer_entities") or [])]
            meta[e["rcsb_id"]] = {
                "seq": seq.replace("\n", ""),
                "organism": org[0] if org else None,
                "ec": (e.get("rcsb_polymer_entity") or {}).get("pdbx_ec"),
                "ligands": ligs}
        if (i // batch) % 10 == 0:
            NEG_CACHE.write_text(json.dumps(meta))
            print(f"    fetched {min(i+batch,len(todo))}/{len(todo)}")
    NEG_CACHE.write_text(json.dumps(meta))
    return meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pos = pd.read_parquet(OUT / "positives_pool.parquet")
    pos_clusters = set(pos["cluster30"].astype(str))
    print(f"positive clusters to exclude: {len(pos_clusters)}")

    print("fetch RCSB 30% representatives (Pfam, X-ray, res<=3.0) ...")
    reps = fetch_representatives()
    print(f"  {len(reps)} cluster representatives")

    ent2clu = load_entity_clusters()
    # candidates = reps whose cluster is NOT a positive cluster
    cand = [e for e in reps if str(ent2clu.get(e, f"na_{e}")) not in pos_clusters]
    random.Random(SEED).shuffle(cand)
    print(f"  {len(cand)} candidates after excluding positive clusters")

    print("fetch sequences + ligands; select metal-free negatives ...")
    rows, seen_clu = [], set()
    # fetch in waves until POOL_TARGET reached
    idx = 0
    wave = 2000
    while len(rows) < POOL_TARGET and idx < len(cand):
        wave_ids = cand[idx:idx + wave]
        idx += wave
        meta = fetch_entity_meta(wave_ids)
        for ent in wave_ids:
            if len(rows) >= POOL_TARGET:
                break
            m = meta.get(ent)
            if not m:
                continue
            seq = m["seq"]
            if not (LEN_MIN <= len(seq) <= LEN_MAX):
                continue
            if any(l in METALS for l in m["ligands"]):     # (b) truly metal-free
                continue
            clu = str(ent2clu.get(ent, f"na_{ent}"))
            if clu in seen_clu or clu in pos_clusters:
                continue
            seen_clu.add(clu)
            pdb, eid = ent.split("_")
            rows.append({
                "protein_id": ent, "source_db": "RCSB", "pdb": pdb.lower(),
                "chain": None, "entity": ent, "uniprot": None,
                "resolution": None, "evidence": "X-ray <=3.0A (RCSB)",
                "label": 0, "class": "non_metal",
                "ions": None, "coord_residues": None,
                "ec": m["ec"], "go": None, "organism": m["organism"],
                "length": len(seq), "cluster30": clu,
                "n_redundant_chains": 1, "redundant_pdb_chains": ent,
                "label_rule": "RCSB Pfam protein, X-ray<=3A, no metal ligand in entry",
                "seq": seq})
        print(f"  collected {len(rows)}/{POOL_TARGET}")

    neg = pd.DataFrame(rows)
    neg.to_parquet(OUT / "negatives_pool.parquet", index=False)
    with open(OUT / "negatives_pool.fasta", "w") as f:
        for _, r in neg.iterrows():
            f.write(f">{r['protein_id']} class=non_metal cluster={r['cluster30']}\n{r['seq']}\n")
    print(f"\nnegatives: {len(neg)} unique seqs / {neg['cluster30'].nunique()} clusters")
    print(f"length min/median/max: {neg.length.min()}/{int(neg.length.median())}/{neg.length.max()}")
    print(f"cluster overlap with positives: {len(set(neg.cluster30) & pos_clusters)} (must be 0)")
    print(f"wrote {OUT/'negatives_pool.parquet'}")


if __name__ == "__main__":
    main()
