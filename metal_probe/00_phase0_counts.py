#!/usr/bin/env python3
"""
Phase 0 — counts before targets (ESM-2 metal-binding probe).

Pulls per-ion structure counts from the RCSB Search API (the authoritative
source MetalPDB itself is derived from), under the spec's experimental +
resolution filters, and RCSB's precomputed sequence-identity clusters so the
"estimated clusters" column is a *measured* number, not a division.

Why RCSB and not MetalPDB directly: MetalPDB (metalpdb.cerm.unifi.it) is not
reliably reachable from every network (verified unreachable from the dev box on
2026-08-15). MetalPDB adds coordinating-residue site annotation on top of PDB;
that annotation is needed at catalog-build time, not for target-setting. For
counts, RCSB is authoritative and precise. BioLiP is the independent cross-check
(see 00b_biolip_crosscheck).

Granularity note: an "entry" count is "structures containing this metal". The
true positive set is the subset of *chains* that actually coordinate the metal,
which requires MetalPDB/BioLiP site annotation (catalog-build step). Cluster
counts here are over all polymer entities in metal-containing structures, so
they are an UPPER BOUND on binder clusters — good for order-of-magnitude target
setting, to be tightened once sites are annotated.

Output: metal_probe/phase0/counts_table.csv  and  counts.json
No dependencies beyond the standard library.
"""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
LIG_ATTR = "rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id"
RES_CUTOFF = 3.0          # spec: start <3 A (M-Ionic)
CLUSTER_CUTOFF = 30       # spec §7: 20% vs 30%; RCSB offers 30 (not 20)
OUT = Path(__file__).resolve().parent / "phase0"

# ion label -> PDB nonpolymer comp_id(s). Grouped so, e.g., iron(II)+iron(III)
# count as one "Fe (ionic)" ion. Cofactor-bound and adventitious metals are
# reported separately (flagged) so the user can include/exclude deliberately.
IONS = [
    # label,            codes,                 category
    ("Zn",              ["ZN"],                "transition"),
    ("Cu",              ["CU", "CU1"],         "transition"),
    ("Fe (ionic)",      ["FE", "FE2"],         "transition"),
    ("Mn",              ["MN"],                "transition"),
    ("Co",              ["CO", "3CO"],         "transition"),
    ("Ni",              ["NI"],                "transition"),
    ("Mo",             ["MO", "4MO", "6MO", "2MO", "MOO"], "transition"),
    ("Ca",              ["CA"],                "alkaline_earth"),
    ("Mg",              ["MG"],                "alkaline_earth"),
    ("Na",              ["NA"],                "alkali"),
    ("K",               ["K"],                 "alkali"),
    # --- reported but flagged: not clean coordination positives ---
    ("Fe (heme)",       ["HEM", "HEC", "HEA", "HEB"], "cofactor"),
    ("Fe-S cluster",    ["FES", "SF4", "F3S"], "cofactor"),
    ("Cd (adventitious)", ["CD"],              "heavy_atom"),
    ("Hg (adventitious)", ["HG"],              "heavy_atom"),
]


def _post(query, retries=3):
    data = json.dumps(query).encode()
    req = urllib.request.Request(
        SEARCH_URL, data=data, headers={"Content-Type": "application/json"})
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code == 204:      # no hits -> empty result
                return {"total_count": 0, "group_by_count": 0}
            last = f"HTTP {e.code}: {body}"
        except Exception as e:
            last = repr(e)
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"RCSB query failed after {retries} tries: {last}")


def _terminal(attr, operator, value):
    return {"type": "terminal", "service": "text",
            "parameters": {"attribute": attr, "operator": operator, "value": value}}


def _filters(codes, xray=False, res=None):
    nodes = [_terminal(LIG_ATTR, "in", codes)]
    if xray:
        nodes.append(_terminal("rcsb_entry_info.experimental_method", "exact_match", "X-ray"))
    if res is not None:
        nodes.append(_terminal("rcsb_entry_info.resolution_combined", "less_or_equal", res))
    return {"type": "group", "logical_operator": "and", "nodes": nodes} if len(nodes) > 1 else nodes[0]


def count_entries(codes, xray=False, res=None):
    q = {"query": _filters(codes, xray, res), "return_type": "entry",
         "request_options": {"return_counts": True}}
    return _post(q).get("total_count", 0)


def count_clusters(codes, cutoff, xray=True, res=RES_CUTOFF):
    q = {"query": _filters(codes, xray, res), "return_type": "polymer_entity",
         "request_options": {
             "group_by": {"aggregation_method": "sequence_identity", "similarity_cutoff": cutoff},
             "group_by_return_type": "representatives", "return_counts": True}}
    r = _post(q)
    return r.get("total_count", 0), r.get("group_by_count", 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'ion':<20}{'all_entries':>12}{'xray_res3':>12}{'entities':>12}"
          f"{'clusters@'+str(CLUSTER_CUTOFF):>14}{'  category'}")
    print("-" * 92)
    for label, codes, category in IONS:
        all_entries = count_entries(codes, xray=False, res=None)
        xray_res3 = count_entries(codes, xray=True, res=RES_CUTOFF)
        entities, clusters = count_clusters(codes, CLUSTER_CUTOFF)
        row = {"ion": label, "codes": "+".join(codes), "category": category,
               "entries_all_methods": all_entries,
               "entries_xray_res3.0": xray_res3,
               "polymer_entities_xray_res3.0": entities,
               f"clusters_{CLUSTER_CUTOFF}pct": clusters}
        rows.append(row)
        print(f"{label:<20}{all_entries:>12}{xray_res3:>12}{entities:>12}"
              f"{clusters:>14}  {category}")

    # write outputs
    import csv
    csv_path = OUT / "counts_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "RCSB Search API v2", "res_cutoff": RES_CUTOFF,
            "cluster_cutoff_pct": CLUSTER_CUTOFF, "rows": rows,
            "notes": "clusters are over ALL polymer entities in metal-containing "
                     "structures -> upper bound on binder clusters (site annotation deferred)"}
    (OUT / "counts.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {csv_path}\nwrote {OUT/'counts.json'}")


if __name__ == "__main__":
    main()
