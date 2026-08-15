"""Diagnostic statistics for the go/no-go decision (handoff sections 1-4).

None of these functions "decide" — they produce the numbers `05_report.py` maps
onto the handoff decision table. Keep them pure so they can be unit-tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --- Section 1: reconstruction error (FVU) ----------------------------------

def summarize_fvu(protein_fvu: np.ndarray, groups: np.ndarray,
                  subclasses: np.ndarray | None = None) -> pd.DataFrame:
    """Per-group (and optional per-subclass) FVU summary.

    protein_fvu : (P,) per-protein FVU (mean over residues)
    groups      : (P,) group label ('microprotein'/'length_matched'/'background')
    subclasses  : (P,) optional finer label; summarized within microprotein group
    """
    rows = []
    df = pd.DataFrame({"fvu": protein_fvu, "group": groups})
    if subclasses is not None:
        df["subclass"] = subclasses
    for g, sub in df.groupby("group"):
        rows.append(_fvu_row(g, "(all)", sub["fvu"].values))
    if subclasses is not None:
        micro = df[df["group"] == "microprotein"]
        for sc, sub in micro.groupby("subclass"):
            rows.append(_fvu_row("microprotein", sc, sub["fvu"].values))
    return pd.DataFrame(rows)


def _fvu_row(group, subclass, v):
    v = np.asarray(v, dtype=float)
    return {
        "group": group, "subclass": subclass, "n": len(v),
        "fvu_median": float(np.median(v)),
        "fvu_mean": float(np.mean(v)),
        "fvu_p25": float(np.percentile(v, 25)),
        "fvu_p75": float(np.percentile(v, 75)),
    }


def fvu_verdict(summary: pd.DataFrame, comparable_ratio: float,
                elevated_ratio: float) -> dict:
    """Compare each group's median FVU to the background group's median.

    Returns per-group ratio + a coarse label used to fill the decision table.
    """
    bg = summary[(summary.group == "background") & (summary.subclass == "(all)")]
    if bg.empty:
        raise ValueError("no background group in FVU summary")
    bg_med = float(bg.fvu_median.iloc[0])
    out = {"background_median_fvu": bg_med, "rows": []}
    for _, r in summary.iterrows():
        ratio = r.fvu_median / max(bg_med, 1e-12)
        if ratio <= comparable_ratio:
            label = "comparable"
        elif ratio >= elevated_ratio:
            label = "elevated"
        else:
            label = "intermediate"
        out["rows"].append({"group": r.group, "subclass": r.subclass,
                            "n": int(r.n), "fvu_median": float(r.fvu_median),
                            "ratio_vs_bg": float(ratio), "label": label})
    return out


# --- Section 2: magnitude & feature-identity entropy ------------------------

def activation_magnitude_stats(values: np.ndarray) -> dict:
    """Distribution of active-feature magnitudes (NOT counts; k is fixed).

    values : (n_residues, k) TopK activation magnitudes for one group.
    """
    v = np.asarray(values, dtype=float).ravel()
    return {
        "mag_median": float(np.median(v)),
        "mag_mean": float(np.mean(v)),
        "mag_p10": float(np.percentile(v, 10)),
        "mag_p90": float(np.percentile(v, 90)),
    }


def feature_usage_entropy(dense_or_pooled: np.ndarray) -> float:
    """Shannon entropy (bits) of the feature-usage distribution over a group.

    Input is any (rows, F) non-negative matrix (per-residue dense features or
    per-protein pooled). Usage per feature = summed activation; low entropy
    means the SAE routes everything into a few generic features on this group.
    """
    usage = np.asarray(dense_or_pooled, dtype=float).sum(axis=0)
    total = usage.sum()
    if total <= 0:
        return 0.0
    p = usage / total
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def usage_from_sparse(indices: np.ndarray, values: np.ndarray,
                      n_features: int) -> np.ndarray:
    """Total activation per feature id, from sparse (n_residues, k) codes."""
    usage = np.zeros(n_features, dtype=float)
    np.add.at(usage, indices.ravel(), values.ravel())
    return usage


def entropy_from_usage(usage: np.ndarray) -> float:
    total = usage.sum()
    if total <= 0:
        return 0.0
    p = usage[usage > 0] / total
    return float(-(p * np.log2(p)).sum())


# --- Section 2 (cont.): description informativeness -------------------------

def top_feature_informativeness(pooled: np.ndarray, feat_table: pd.DataFrame,
                                top_n: int = 10) -> dict:
    """Fraction of each protein's top-N pooled features that carry a meaningful
    (non-generic) natural-language description.

    feat_table must have columns 'feature_id' and 'is_informative' (bool);
    build it with features.tag_informative().
    """
    info = feat_table.set_index("feature_id")["is_informative"].to_dict()
    fracs = []
    for row in pooled:
        top = np.argpartition(row, -top_n)[-top_n:]
        top = top[row[top] > 0]
        if len(top) == 0:
            continue
        fracs.append(np.mean([bool(info.get(int(f), False)) for f in top]))
    fracs = np.asarray(fracs)
    return {"mean_informative_frac": float(fracs.mean()) if len(fracs) else 0.0,
            "n_proteins": int(len(fracs))}


# --- Section 4: positive-control marker recovery ----------------------------

def marker_recovery(pooled: np.ndarray, expected_feature_ids: list[int],
                    percentile_ref: np.ndarray) -> dict:
    """Do the pre-registered expected features fire strongly on the marker set?

    pooled          : (M, F) max-pooled features for the marker proteins
    expected_feature_ids : pre-registered feature ids for this concept
    percentile_ref  : (R, F) pooled features of a reference/background set, used
                      to express marker activation as a percentile (is it high
                      *relative to background*, not just nonzero?)
    """
    exp = np.asarray(expected_feature_ids, dtype=int)
    marker_max = pooled[:, exp].max(axis=1)          # best expected feature/protein
    ref_max = percentile_ref[:, exp].max(axis=1)
    ref_sorted = np.sort(ref_max)
    pct = np.searchsorted(ref_sorted, marker_max) / max(len(ref_sorted), 1)
    return {
        "expected_ids": exp.tolist(),
        "marker_activation_median": float(np.median(marker_max)),
        "fraction_above_bg_90pct": float(np.mean(pct >= 0.90)),
        "median_percentile_vs_bg": float(np.median(pct)),
    }
