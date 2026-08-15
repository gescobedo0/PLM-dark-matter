"""ESM-2 model registry + loader (fair-esm).

fair-esm import is lazy so the rest of the pipeline (probe, plots, tests) runs
on machines without fair-esm / GPU.
"""

# key -> (fair-esm loader name, n transformer layers, embedding dim)
REGISTRY = {
    "8M":   ("esm2_t6_8M_UR50D",    6,  320),
    "35M":  ("esm2_t12_35M_UR50D",  12, 480),
    "150M": ("esm2_t30_150M_UR50D", 30, 640),
    "650M": ("esm2_t33_650M_UR50D", 33, 1280),
}


def default_layers(key):
    """Shallow/mid/deep sweep: ~0.5, ~0.75, last."""
    n = REGISTRY[key][1]
    return sorted({max(1, round(n * 0.5)), max(1, round(n * 0.75)), n})


def load(key, device="cuda"):
    """Load model + alphabet + batch converter. Returns (model, alphabet, bc)."""
    import esm  # lazy
    name = REGISTRY[key][0]
    model, alphabet = getattr(esm.pretrained, name)()
    model = model.eval().to(device)
    return model, alphabet, alphabet.get_batch_converter()
