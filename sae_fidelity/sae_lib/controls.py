"""Canonical control sets from human SwissProt.

Group B (length-matched): N-terminal fragments of canonical proteins truncated
to match Group A's length distribution. Real canonical proteins as short as our
ORFs (median ~28 aa) barely exist, so truncation is how we isolate the *length*
effect from the *biology* effect. The tradeoff — a fragment loses C-terminal
context — is documented and is itself mild OOD, but it is the cleanest available
length control.

Group C (background): random canonical proteins at their natural full length —
the regime the SAE was trained on; the low-FVU anchor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def download_swissprot_human(cache_path: Path, organism_id: int = 9606) -> Path:
    """Download reviewed (SwissProt) proteins for an organism as FASTA, cached."""
    cache_path = Path(cache_path)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    import requests
    params = {
        "query": f"reviewed:true AND organism_id:{organism_id}",
        "format": "fasta",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(UNIPROT_STREAM, params=params, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return cache_path


def parse_fasta(path: Path) -> pd.DataFrame:
    """Parse FASTA -> DataFrame(id, seq). Skips entries with non-standard residues."""
    ids, seqs = [], []
    cur_id, cur = None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    _emit(ids, seqs, cur_id, "".join(cur))
                # UniProt header: >sp|P12345|NAME_HUMAN ...
                parts = line[1:].strip().split("|")
                cur_id = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                cur = []
            else:
                cur.append(line.strip())
        if cur_id is not None:
            _emit(ids, seqs, cur_id, "".join(cur))
    return pd.DataFrame({"id": ids, "seq": seqs})


def _emit(ids, seqs, cid, seq):
    seq = seq.upper()
    if seq and all(c in STANDARD_AA for c in seq):
        ids.append(cid)
        seqs.append(seq)


def build_background(canonical: pd.DataFrame, n: int, min_len: int,
                     max_len: int, seed: int) -> pd.DataFrame:
    """Group C: random full-length canonical proteins (length-capped for compute)."""
    rng = np.random.default_rng(seed)
    pool = canonical[(canonical.seq.str.len() >= min_len) &
                     (canonical.seq.str.len() <= max_len)].reset_index(drop=True)
    take = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    out = pool.iloc[take].copy()
    out["id"] = "bg_" + out["id"].astype(str)
    out["group"] = "background"
    out["subclass"] = "canonical"
    return out[["id", "group", "subclass", "seq"]].reset_index(drop=True)


def build_length_matched(canonical: pd.DataFrame, target_lengths: np.ndarray,
                         min_len: int, seed: int) -> pd.DataFrame:
    """Group B: N-terminal fragments truncated to match `target_lengths`.

    For each target length L (sampled from Group A), pick a canonical protein of
    length >= L and take its first L residues. Sampling without replacement over
    source proteins where possible, to avoid re-using the same N-terminus.
    """
    rng = np.random.default_rng(seed)
    src = canonical[canonical.seq.str.len() >= min_len].reset_index(drop=True)
    src_len = src.seq.str.len().values
    order = rng.permutation(len(src))
    ptr = 0
    ids, seqs, lens = [], [], []
    for L in target_lengths:
        L = int(L)
        # find next source (in shuffled order) long enough for this fragment
        found = None
        scanned = 0
        while scanned < len(order):
            i = order[ptr % len(order)]
            ptr += 1
            scanned += 1
            if src_len[i] >= L:
                found = i
                break
        if found is None:                  # no protein long enough (rare): skip
            continue
        row = src.iloc[found]
        ids.append(f"lm_{row['id']}_{L}")
        seqs.append(row["seq"][:L])
        lens.append(L)
    out = pd.DataFrame({"id": ids, "seq": seqs})
    out["group"] = "length_matched"
    out["subclass"] = "canonical_fragment"
    return out[["id", "group", "subclass", "seq"]].reset_index(drop=True)
