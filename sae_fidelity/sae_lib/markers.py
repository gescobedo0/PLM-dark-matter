"""Positive-control sequence sets (handoff section 4).

Fetched from UniProt so the diagnostic is reproducible. Each set maps to a
pre-registered SAE concept (features.MARKER_CONCEPTS): if the SAE space is
usable, the marker set should light up the expected features; the disordered set
should NOT fire structured-domain features.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .controls import UNIPROT_STREAM, parse_fasta

# UniProt query -> (marker set name, expected concept in MARKER_CONCEPTS)
MARKER_QUERIES = {
    "c2h2_zinc_finger": (
        "xref:pfam-PF00096 AND reviewed:true", "metal_coordination"),
    "ferredoxin": (
        "xref:pfam-PF00037 AND reviewed:true", "fe_s_coordination"),
    "rubredoxin": (
        "xref:pfam-PF00301 AND reviewed:true", "fe_s_coordination"),
    "tm_peptide": (
        "keyword:KW-0812 AND reviewed:true AND length:[1 TO 100]", "membrane"),
    "disordered_small": (
        "(ft_region:Disordered) AND reviewed:true AND length:[1 TO 200]", "disorder"),
}


def fetch_marker_set(query: str, cache_path: Path, limit: int = 60) -> pd.DataFrame:
    """Download up to `limit` sequences for a UniProt query, cached as FASTA."""
    cache_path = Path(cache_path)
    if not (cache_path.exists() and cache_path.stat().st_size > 0):
        import requests
        params = {"query": query, "format": "fasta", "size": min(limit, 500)}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(UNIPROT_STREAM.replace("/stream", "/search"),
                         params=params, timeout=120)
        r.raise_for_status()
        cache_path.write_text(r.text)
    df = parse_fasta(cache_path).head(limit)
    return df


def build_all_markers(cache_dir: Path, limit: int = 60) -> pd.DataFrame:
    """Fetch every marker set -> DataFrame(id, seq, marker_set, concept)."""
    cache_dir = Path(cache_dir)
    frames = []
    for name, (query, concept) in MARKER_QUERIES.items():
        df = fetch_marker_set(query, cache_dir / f"marker_{name}.fasta", limit)
        df = df.assign(marker_set=name, concept=concept)
        df["id"] = f"{name}_" + df["id"].astype(str)
        frames.append(df)
        print(f"  {name}: {len(df)} sequences")
    return pd.concat(frames, ignore_index=True)
