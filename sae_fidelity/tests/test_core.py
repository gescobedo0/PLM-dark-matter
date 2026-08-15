"""Local, CPU-only unit tests for the math (no GPU, no network)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sae_lib import sae, pooling, diagnostics, catalog  # noqa: E402


def _invertible_sae(dim=64, topk=8, seed=0):
    """Exactly invertible TopK SAE: square orthonormal dictionary.

    With codebook == hidden and orthonormal decoder rows Q (Q Qᵀ = I), the tied
    encoder Qᵀ inverts the decoder, so a non-negative k-sparse combination of
    atoms reconstructs exactly. This tests the encode/decode/FVU math itself,
    not dictionary learning (a trained SAE supplies the inverting encoder).
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))     # orthonormal rows
    W_dec = Q                       # (codebook=dim, hidden=dim)
    W_enc = Q.T.copy()              # inverts the decoder
    b_enc = np.zeros(dim)
    b_dec = rng.standard_normal(dim) * 0.1
    return sae.SAEWeights(W_enc, b_enc, W_dec, b_dec, topk), rng


def _random_sae(hidden=32, codebook=128, topk=8, seed=0):
    """Overcomplete random dictionary (used only for the poor-reconstruction case)."""
    rng = np.random.default_rng(seed)
    W_dec = rng.standard_normal((codebook, hidden))
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
    return sae.SAEWeights(W_dec.T.copy(), np.zeros(codebook), W_dec,
                          rng.standard_normal(hidden) * 0.1, topk), rng


def test_topk_and_fvu_sparse_signal():
    w, rng = _invertible_sae()
    # Inputs are exact non-negative k-sparse combinations of orthonormal atoms;
    # the inverting encoder recovers them -> FVU ~ 0.
    codebook, hidden = w.W_dec.shape
    N = 200
    x = np.zeros((N, hidden))
    for i in range(N):
        atoms = rng.choice(codebook, size=w.topk, replace=False)
        coeffs = np.abs(rng.standard_normal(w.topk)) + 0.5
        x[i] = coeffs @ w.W_dec[atoms] + w.b_dec
    idx, val = sae.topk_encode(x, w)
    x_hat = sae.decode(idx, val, w)
    f = sae.fvu(x, x_hat, x.mean(axis=0))
    assert f.mean() < 1e-6, f"exact sparse signal should reconstruct, FVU={f.mean():.3e}"
    assert val.shape == (N, w.topk)
    assert (val >= 0).all(), "TopK activations must be non-negative (post-ReLU)"


def test_fvu_dense_noise_is_high():
    w, rng = _random_sae()
    hidden = w.W_dec.shape[1]
    x = rng.standard_normal((150, hidden)) * 3.0    # dense, not sparse in atoms
    idx, val = sae.topk_encode(x, w)
    x_hat = sae.decode(idx, val, w)
    f = sae.fvu(x, x_hat, x.mean(axis=0))
    assert f.mean() > 0.3, f"dense noise should reconstruct poorly, FVU={f.mean():.3f}"


def test_max_pool_dense_vs_sparse_agree():
    codebook = 64
    lengths = np.array([5, 3, 7])
    rng = np.random.default_rng(1)
    total = lengths.sum()
    k = 6
    idx = rng.integers(0, codebook, size=(total, k))
    val = np.abs(rng.standard_normal((total, k))) + 0.1
    sparse_pool = pooling.max_pool_sparse(idx, val, codebook, lengths)
    # dense reference
    start = 0
    for p, L in enumerate(lengths):
        dense = sae.to_dense(idx[start:start + L], val[start:start + L], codebook)
        ref = pooling.max_pool(dense)
        assert np.allclose(sparse_pool[p], ref), f"protein {p} mismatch"
        start += L


def test_topk_mean_pool_shapes():
    feat = np.abs(np.random.default_rng(2).standard_normal((4, 20)))
    assert pooling.topk_mean_pool(feat, k=3).shape == (20,)
    # fewer residues than k -> falls back to available
    assert pooling.topk_mean_pool(feat[:2], k=3).shape == (20,)


def test_feature_usage_entropy_bounds():
    # uniform usage -> entropy ~ log2(F); single feature -> 0
    F = 16
    uniform = np.ones((100, F))
    ent_u = diagnostics.feature_usage_entropy(uniform)
    assert abs(ent_u - np.log2(F)) < 1e-6
    concentrated = np.zeros((100, F)); concentrated[:, 0] = 1.0
    assert diagnostics.feature_usage_entropy(concentrated) < 1e-9


def test_catalog_clean_sequence():
    assert catalog.clean_sequence("MAAK*") == "MAAK"
    assert catalog.clean_sequence("MA*AK") is None       # internal stop
    assert catalog.clean_sequence("MAXK") is None         # non-standard X
    assert catalog.clean_sequence("") is None


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
