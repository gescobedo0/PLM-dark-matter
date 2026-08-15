"""SAE feature natural-language descriptions and informativeness tagging.

Loads biohub/ESMC-SAE-Features/uniref90_feature_table.parquet (16,384 rows x 13
cols, GPT-5-generated descriptions across 14 categories). Exact column names are
confirmed at load time by auto-detecting the id / description / category columns,
because the schema is not documented here — inspect `feat_table.columns` on first
run and pin the names if the heuristics pick wrong.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# descriptions that carry no usable biological meaning
_GENERIC_PATTERNS = re.compile(
    r"\b(no clear|unclear|uninformative|generic|mixed|ambiguous|unknown|"
    r"non-specific|nonspecific|no specific|difficult to|hard to|"
    r"heterogeneous|no consistent|weak|noisy)\b", re.I)


def load_feature_table(repo_id: str, filename: str) -> pd.DataFrame:
    """Download and load the feature-description parquet, normalizing columns to
    feature_id / description / category where detectable."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    df = pd.read_parquet(path)
    return _normalize_columns(df)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    # feature id
    id_col = next((cols[c] for c in cols
                   if c in ("feature_id", "feature", "id", "index", "latent")), None)
    if id_col is None:
        df = df.reset_index().rename(columns={"index": "feature_id"})
        id_col = "feature_id"
    # description = the text column with the longest average content
    text_cols = [c for c in df.columns if df[c].dtype == object]
    desc_col = max(text_cols,
                   key=lambda c: df[c].astype(str).str.len().mean(),
                   default=None)
    cat_col = next((cols[c] for c in cols if "categor" in c or c == "type"), None)

    out = df.rename(columns={id_col: "feature_id"})
    out["feature_id"] = out["feature_id"].astype(int)
    out["description"] = out[desc_col].astype(str) if desc_col else ""
    out["category"] = out[cat_col].astype(str) if cat_col else ""
    return out


def tag_informative(feat_table: pd.DataFrame, min_len: int = 25) -> pd.DataFrame:
    """Add is_informative: description is specific enough to label a cluster."""
    ft = feat_table.copy()
    d = ft["description"].fillna("")
    ft["is_informative"] = (d.str.len() >= min_len) & (~d.str.contains(_GENERIC_PATTERNS))
    return ft


def search_features(feat_table: pd.DataFrame, keywords) -> list[int]:
    """Feature ids whose description or category matches ANY keyword (regex, OR).

    Use this to PRE-REGISTER expected features for positive controls, before
    looking at any activation — search on concept ('zinc|metal coordination'),
    freeze the ids, then test recovery.
    """
    pat = re.compile("|".join(keywords), re.I)
    text = (feat_table["description"].fillna("") + " " +
            feat_table["category"].fillna(""))
    hits = feat_table.loc[text.str.contains(pat), "feature_id"]
    return sorted(int(x) for x in hits)


# concept -> regexes, for the pre-registered positive controls (handoff section 4)
MARKER_CONCEPTS = {
    "metal_coordination": [r"zinc", r"metal[- ]?bind", r"metal[- ]?coordinat",
                           r"\bZn\b", r"C2H2"],
    "fe_s_coordination": [r"iron[- ]?sulfur", r"\bFe-?S\b", r"ferredoxin",
                          r"rubredoxin", r"\b4Fe", r"\b2Fe"],
    "membrane": [r"transmembrane", r"membrane", r"\bTM\b", r"hydrophobic helix"],
    "disorder": [r"disorder", r"intrinsically disordered", r"low complexity",
                 r"unstructured"],
}
