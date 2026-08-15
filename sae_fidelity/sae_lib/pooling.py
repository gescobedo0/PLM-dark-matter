"""Per-residue -> per-protein pooling.

Handoff rule: use MAX over residues (optionally top-3 mean), never mean-pool.
Rationale: microprotein signals are local motifs (metal coordination fires on
~4 residues); mean-pooling both dilutes them and reintroduces the length
confound this feature space exists to avoid.
"""
from __future__ import annotations

import numpy as np


def max_pool(feat: np.ndarray) -> np.ndarray:
    """Max over residues. feat: (L, F) dense per-residue features -> (F,)."""
    if feat.ndim != 2:
        raise ValueError(f"expected (L, F), got {feat.shape}")
    return feat.max(axis=0)


def topk_mean_pool(feat: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean of the top-k residue values per feature (robust variant of max).

    For proteins with fewer than k residues, averages all available residues.
    """
    if feat.ndim != 2:
        raise ValueError(f"expected (L, F), got {feat.shape}")
    L = feat.shape[0]
    kk = min(k, L)
    # partition each column and average its kk largest entries
    part = np.partition(feat, L - kk, axis=0)[L - kk:]
    return part.mean(axis=0)


def max_pool_sparse(indices: np.ndarray, values: np.ndarray, n_features: int,
                    lengths: np.ndarray) -> np.ndarray:
    """Max-pool from a sparse per-residue representation.

    Sparse format (TopK SAE, k active per residue):
      indices : (n_residues, k) int   feature id active at each residue
      values  : (n_residues, k) float activation magnitude
      lengths : (n_proteins,) int      residues per protein (row groups, in order)
    Returns (n_proteins, n_features) dense max-pooled matrix. Features never
    active for a protein are 0 (TopK activations are non-negative).
    """
    n_prot = len(lengths)
    out = np.zeros((n_prot, n_features), dtype=values.dtype)
    start = 0
    for p, L in enumerate(lengths):
        idx = indices[start:start + L].ravel()
        val = values[start:start + L].ravel()
        # np.maximum.at accumulates max per feature id across all residues
        np.maximum.at(out[p], idx, val)
        start += L
    return out
