"""Parse and clean neworfcatalog.csv into a labelled microprotein table.

The catalog header is messy (blank column names at positions 32, 33, 37; a
newline inside the `sequence_aa_MS_ms` header). We select by name and ignore the
blanks. `sequence_aa` (col index 34) is the amino-acid ORF product, terminated by
a stop `*`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydropathy scale
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def clean_sequence(raw: str) -> str | None:
    """Uppercase, strip a single trailing stop, and validate residues.

    Returns None if the sequence has an internal stop or any non-standard
    residue (X, U, *, etc.) — such sequences are excluded from the diagnostic.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().upper()
    if s.endswith("*"):
        s = s[:-1]
    if not s:
        return None
    if any(c not in STANDARD_AA for c in s):
        return None
    return s


def kd_hydropathy(seq: str) -> float:
    """Mean Kyte-Doolittle hydropathy of a clean sequence."""
    return float(np.mean([KD[c] for c in seq]))


def max_tm_window(seq: str, window: int) -> float:
    """Max mean-hydropathy over any contiguous `window`-residue segment.

    For sequences shorter than the window, uses the whole sequence. This is a
    crude proxy for a TM helix (DeepTMHMM is out of scope for the diagnostic).
    """
    vals = np.array([KD[c] for c in seq], dtype=float)
    if len(vals) <= window:
        return float(vals.mean())
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    means = (csum[window:] - csum[:-window]) / window
    return float(means.max())


def load_catalog(path, subclass_map: dict, min_len: int,
                 tm_window: int, tm_threshold: float) -> pd.DataFrame:
    """Load the ORF catalog and return a clean, labelled DataFrame.

    Columns: id, orf_type, subclass, initiation_codon, seq, length,
             hydropathy, tm_score, tm_flag.
    Rows failing cleaning or the length floor are dropped.
    """
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    out = pd.DataFrame({
        "id": df["releasev45_id"].astype(str).str.strip(),
        "orf_type": df["orf_type"].astype(str).str.strip(),
        "initiation_codon": df["initiation_codon"].astype(str).str.strip(),
    })
    out["seq"] = df["sequence_aa"].map(clean_sequence)

    n_raw = len(out)
    out = out[out["seq"].notna()].copy()
    n_clean = len(out)

    out["length"] = out["seq"].str.len()
    out = out[out["length"] >= min_len].copy()
    n_len = len(out)

    out["subclass"] = out["orf_type"].map(subclass_map).fillna("other")
    out["hydropathy"] = out["seq"].map(kd_hydropathy)
    out["tm_score"] = out["seq"].map(lambda s: max_tm_window(s, tm_window))
    out["tm_flag"] = np.where(out["tm_score"] >= tm_threshold, "TM", "soluble")

    out = out.reset_index(drop=True)
    out.attrs["n_raw"] = n_raw
    out.attrs["n_after_clean"] = n_clean
    out.attrs["n_after_length"] = n_len
    return out


def length_bin(lengths, bins) -> np.ndarray:
    """Integer bin index for each length using the config `length_bins` edges."""
    return np.digitize(np.asarray(lengths), bins[1:-1])
