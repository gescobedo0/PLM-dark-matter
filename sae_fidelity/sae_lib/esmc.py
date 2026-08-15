"""ESMC-6B layer-60 residue-activation extraction (GPU step).

Runs on Colab (L4/A100 recommended; 6B bf16 weights ~12 GB — do NOT quantize,
it corrupts the reconstruction signal we measure). CPU-import-safe: torch and
transformers are imported lazily inside functions so the rest of the library and
the unit tests run without them.

LOADER IS UNCONFIRMED. The precedent paper (arXiv 2606.12209) extracted layer-60
hidden states via HuggingFace Transformers. We default to AutoModel with
trust_remote_code and output_hidden_states, and expose an esm-SDK alternative.
First implementation task on the GPU box: confirm which one loads biohub/ESMC-6B,
and confirm the hidden_states index for "layer 60" (see extract_layer).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    n_hidden_states: int   # len of outputs.hidden_states (embeddings + each block)
    backend: str           # "hf" or "esm"


def load_esmc(base_model: str, dtype: str = "bfloat16",
              backend: str = "hf") -> LoadedModel:
    """Load ESMC-6B for hidden-state extraction. Returns a LoadedModel."""
    import torch
    torch_dtype = getattr(torch, dtype)

    if backend == "hf":
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            base_model, trust_remote_code=True, torch_dtype=torch_dtype,
            output_hidden_states=True)
        model.eval().to("cuda")
        # probe hidden-state count with a tiny forward
        with torch.no_grad():
            enc = tok("MAAA", return_tensors="pt").to("cuda")
            n_hs = len(model(**enc).hidden_states)
        return LoadedModel(model, tok, n_hs, "hf")

    elif backend == "esm":
        # EvolutionaryScale SDK path (may require Forge for 6B weights).
        from esm.models.esmc import ESMC
        model = ESMC.from_pretrained(base_model).to("cuda").eval()
        return LoadedModel(model, None, -1, "esm")

    raise ValueError(f"unknown backend {backend!r}")


def _bucket_by_length(items, max_batch_tokens: int):
    """Yield batches of (id, seq) grouped by similar length to limit padding."""
    ordered = sorted(items, key=lambda t: len(t[1]))
    batch, batch_tokens, cur_max = [], 0, 0
    for pid, seq in ordered:
        cur_max = max(cur_max, len(seq))
        if batch and (len(batch) + 1) * cur_max > max_batch_tokens:
            yield batch
            batch, cur_max = [], len(seq)
        batch.append((pid, seq))
        batch_tokens = len(batch) * cur_max
    if batch:
        yield batch


def extract_layer(lm: LoadedModel, items, layer_index: int,
                  max_batch_tokens: int, drop_special: bool = True):
    """Extract layer-`layer_index` residue activations for (id, seq) items.

    Yields (pid, acts) where acts is (L, hidden) float16, L = residue count
    (special tokens stripped). `layer_index` indexes outputs.hidden_states, where
    index 0 is the embedding layer and index i is the residual stream after block
    i — so layer 60 is index 60. VALIDATE against Group-C reconstruction before
    trusting any number.
    """
    import torch

    if lm.backend != "hf":
        raise NotImplementedError("extract_layer implemented for the HF backend; "
                                  "add the esm-SDK forward if that path is used.")
    if layer_index >= lm.n_hidden_states:
        raise IndexError(
            f"layer_index {layer_index} >= hidden_states count {lm.n_hidden_states}")

    for batch in _bucket_by_length(items, max_batch_tokens):
        ids = [pid for pid, _ in batch]
        seqs = [seq for _, seq in batch]
        enc = lm.tokenizer(seqs, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            hs = lm.model(**enc).hidden_states[layer_index]   # (B, T, hidden)
        hs = hs.float().cpu().numpy().astype(np.float16)
        mask = enc["attention_mask"].cpu().numpy().astype(bool)
        for i, pid in enumerate(ids):
            valid = np.where(mask[i])[0]
            if drop_special and len(valid) >= 2:
                valid = valid[1:-1]        # strip BOS/EOS (confirm tokenizer adds both)
            yield pid, hs[i, valid, :]
