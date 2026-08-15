"""TopK sparse autoencoder: loading, forward pass, and reconstruction error.

The SAE (biohub/ESMC-6B-sae-layer60-k64-codebook16384) maps a 2560-dim layer-60
ESMC-6B residual vector to 16,384 features, keeping the top-64 by magnitude, then
reconstructs. This module is torch-optional: the numpy path lets us unit-test the
math and run FVU without a GPU; the torch path is used in the real pipeline.

Weight loading (`load_sae_weights`) intentionally probes several state_dict key
conventions because biohub has not published a single canonical loader; the actual
key names must be confirmed against the downloaded checkpoint on first run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SAEWeights:
    """Plain-array container so both numpy and torch paths share one source."""
    W_enc: np.ndarray      # (hidden, codebook)  x @ W_enc -> pre-activation
    b_enc: np.ndarray      # (codebook,)
    W_dec: np.ndarray      # (codebook, hidden)  z @ W_dec -> reconstruction
    b_dec: np.ndarray      # (hidden,) pre-encoder bias, added back on decode
    topk: int


def topk_encode(x: np.ndarray, w: SAEWeights) -> tuple[np.ndarray, np.ndarray]:
    """Encode (N, hidden) -> sparse top-k. Returns (idx, val), each (N, k).

    idx[i] = feature ids kept for row i; val[i] = their (ReLU'd) magnitudes.
    """
    pre = (x - w.b_dec) @ w.W_enc + w.b_enc     # (N, codebook)
    pre = np.maximum(pre, 0.0)                   # ReLU before top-k
    k = w.topk
    idx = np.argpartition(pre, -k, axis=1)[:, -k:]
    val = np.take_along_axis(pre, idx, axis=1)
    return idx, val


def decode(idx: np.ndarray, val: np.ndarray, w: SAEWeights) -> np.ndarray:
    """Reconstruct (N, hidden) from sparse codes."""
    n, hidden = idx.shape[0], w.W_dec.shape[1]
    recon = np.zeros((n, hidden), dtype=np.float64)
    for i in range(n):                            # k is tiny (64); this is fine
        recon[i] = val[i] @ w.W_dec[idx[i]]
    return recon + w.b_dec


def fvu(x: np.ndarray, x_hat: np.ndarray, baseline_mean: np.ndarray) -> np.ndarray:
    """Fraction of variance unexplained, per row (residue).

    FVU = ||x - x_hat||^2 / ||x - mean||^2, where `mean` is the dataset-level
    activation mean (the constant-prediction baseline). Values near 0 = good
    reconstruction; ~1 = no better than predicting the mean.
    """
    num = np.sum((x - x_hat) ** 2, axis=1)
    den = np.sum((x - baseline_mean) ** 2, axis=1)
    return num / np.maximum(den, 1e-12)


def to_dense(idx: np.ndarray, val: np.ndarray, codebook: int) -> np.ndarray:
    """Sparse (N,k) codes -> dense (N, codebook) feature matrix."""
    out = np.zeros((idx.shape[0], codebook), dtype=val.dtype)
    np.put_along_axis(out, idx, val, axis=1)
    return out


# --- weight loading (real pipeline) -----------------------------------------

_ENC_W_KEYS = ("W_enc", "encoder.weight", "encoder.W", "w_enc")
_ENC_B_KEYS = ("b_enc", "encoder.bias", "latent_bias", "b_latent")
_DEC_W_KEYS = ("W_dec", "decoder.weight", "decoder.W", "w_dec")
_DEC_B_KEYS = ("b_dec", "b_pre", "pre_bias", "decoder.bias")


def _first(sd: dict, keys, name: str):
    for k in keys:
        if k in sd:
            return sd[k]
    raise KeyError(
        f"Could not find {name} in SAE checkpoint. Tried {keys}. "
        f"Available keys: {sorted(sd)[:30]}... "
        f"Inspect the checkpoint and update the key lists in sae.py."
    )


def load_sae_weights(repo_id: str, topk: int, hidden: int, codebook: int,
                     filename: str | None = None) -> SAEWeights:
    """Download and load SAE weights from HuggingFace into a SAEWeights.

    Orientation of W_enc/W_dec is normalized to (hidden, codebook) /
    (codebook, hidden) using the known dims, so transposed conventions load
    correctly regardless of source layout.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if filename is None:
        # try safetensors then pytorch bin
        for fn in ("model.safetensors", "sae.safetensors", "pytorch_model.bin",
                   "sae.pt", "weights.pt"):
            try:
                path = hf_hub_download(repo_id, fn)
                filename = fn
                break
            except Exception:
                continue
        else:
            raise FileNotFoundError(
                f"No known weight file in {repo_id}; pass filename explicitly.")
    else:
        path = hf_hub_download(repo_id, filename)

    sd = load_file(path) if filename.endswith(".safetensors") else torch.load(
        path, map_location="cpu")
    sd = {k: (v.float().cpu().numpy() if hasattr(v, "numpy") else np.asarray(v))
          for k, v in sd.items()}

    W_enc = _orient(_first(sd, _ENC_W_KEYS, "W_enc"), (hidden, codebook))
    W_dec = _orient(_first(sd, _DEC_W_KEYS, "W_dec"), (codebook, hidden))
    b_enc = np.asarray(_first(sd, _ENC_B_KEYS, "b_enc")).reshape(-1)
    try:
        b_dec = np.asarray(_first(sd, _DEC_B_KEYS, "b_dec")).reshape(-1)
    except KeyError:
        b_dec = np.zeros(hidden)                  # some SAEs have no pre-bias
    return SAEWeights(W_enc=W_enc, b_enc=b_enc, W_dec=W_dec, b_dec=b_dec, topk=topk)


def _orient(mat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mat.shape == shape:
        return mat
    if mat.T.shape == shape:
        return mat.T
    raise ValueError(f"weight shape {mat.shape} not compatible with {shape}")
