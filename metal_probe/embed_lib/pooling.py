"""Masked mean / max pooling over residue positions of ESM-2 representations.

fair-esm token layout per sequence of length L: position 0 is BOS (cls), the L
residues occupy positions 1..L, position L+1 is EOS, and the rest is padding.
We pool over residue positions ONLY (1..L) so BOS/EOS/pad never leak into the
per-protein vector. Pure-torch and dependency-light so it is unit-testable
without a GPU or fair-esm.
"""
import torch


def residue_mask(lengths, T):
    """(B, T) bool mask: True at residue positions 1..L_i for each sequence.

    lengths: 1-D LongTensor of true residue counts (excludes BOS/EOS).
    T: total token dim of the representation tensor.
    """
    lengths = torch.as_tensor(lengths, dtype=torch.long)
    idx = torch.arange(T).unsqueeze(0)               # (1, T)
    L = lengths.unsqueeze(1)                          # (B, 1)
    return (idx >= 1) & (idx <= L)


def pool(reps, lengths, mode):
    """Pool (B, T, D) token representations -> (B, D).

    mode: 'mean' (average over residues) or 'max' (elementwise max over residues).
    Masked positions are excluded exactly.
    """
    if mode not in ("mean", "max"):
        raise ValueError(f"unknown pooling mode: {mode}")
    B, T, D = reps.shape
    mask = residue_mask(lengths, T).to(reps.device)  # (B, T)
    m = mask.unsqueeze(-1)                            # (B, T, 1)
    if mode == "mean":
        summed = (reps * m).sum(dim=1)               # (B, D)
        counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B, 1)
        return summed / counts
    # max: push masked positions to -inf so they never win
    neg = torch.finfo(reps.dtype).min
    masked = reps.masked_fill(~m, neg)
    out = masked.max(dim=1).values
    # sequences with zero residues (shouldn't happen) -> zero rather than -inf
    empty = (mask.sum(dim=1) == 0)
    if empty.any():
        out[empty] = 0.0
    return out
