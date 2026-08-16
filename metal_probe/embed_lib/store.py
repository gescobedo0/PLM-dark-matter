"""HDF5 storage for pooled per-protein embeddings.

Layout (one file per model, e.g. embeddings/esm2_650M.h5):
  /protein_ids            (N,)  utf-8 strings, the row order for all arrays
  /length                 (N,)  int  (true residue count used)
  /truncated              (N,)  bool (sequence longer than max_len -> truncated)
  /L{layer}/{mean|max}    (N, D) float32   pooled vectors
  attrs: model, dim, max_len, layers (json), poolings (json)

Keyed by protein_id x layer x pooling, exactly as the spec asks. Pooled vectors
only; no per-residue tensors are stored.
"""
import json

import h5py
import numpy as np


def write(path, protein_ids, lengths, truncated, arrays, *, model, dim,
          layers, poolings, max_len):
    """arrays: dict[(layer:int, pool:str)] -> np.ndarray (N, D)."""
    ids = np.array(list(protein_ids), dtype=object)
    with h5py.File(path, "w") as h:
        h.create_dataset("protein_ids", data=ids,
                         dtype=h5py.string_dtype("utf-8"))
        h.create_dataset("length", data=np.asarray(lengths, dtype=np.int32))
        h.create_dataset("truncated", data=np.asarray(truncated, dtype=bool))
        for (layer, pool), arr in arrays.items():
            h.create_dataset(f"L{layer}/{pool}",
                             data=np.asarray(arr, dtype=np.float32),
                             compression="gzip", compression_opts=4)
        h.attrs["model"] = model
        h.attrs["dim"] = dim
        h.attrs["max_len"] = max_len
        h.attrs["layers"] = json.dumps(list(layers))
        h.attrs["poolings"] = json.dumps(list(poolings))


def read_ids(path):
    with h5py.File(path, "r") as h:
        return [x.decode() if isinstance(x, bytes) else x
                for x in h["protein_ids"][:]]


def read(path, layer, pool):
    """-> (protein_ids: list[str], vectors: np.ndarray (N, D))."""
    with h5py.File(path, "r") as h:
        ids = [x.decode() if isinstance(x, bytes) else x
               for x in h["protein_ids"][:]]
        arr = h[f"L{layer}/{pool}"][:]
    return ids, arr


def available(path):
    """-> (layers, poolings) present in the file."""
    with h5py.File(path, "r") as h:
        return json.loads(h.attrs["layers"]), json.loads(h.attrs["poolings"])


# ---- per-residue (unpooled) store, for the residue-level phase ----------
# Flat residue table keyed by (model, layer). Embeddings are fp16 to halve size.
#   /proteins            (P,)   utf-8, unique protein ids (row index below refs these)
#   /res_protein         (R,)   int32  index into /proteins
#   /res_position        (R,)   int32  0-based sequence position
#   /res_aa              (R,)   S1     amino-acid letter
#   /L{layer}            (R, D) float16
def write_residues(path, proteins, res_protein_idx, res_pos, res_aa, arrays,
                   *, model, dim, layers, max_len):
    """arrays: dict[layer:int] -> np.ndarray (R, D)."""
    with h5py.File(path, "w") as h:
        h.create_dataset("proteins", data=np.array(list(proteins), dtype=object),
                         dtype=h5py.string_dtype("utf-8"))
        h.create_dataset("res_protein", data=np.asarray(res_protein_idx, np.int32))
        h.create_dataset("res_position", data=np.asarray(res_pos, np.int32))
        h.create_dataset("res_aa", data=np.asarray(res_aa, dtype="S1"))
        for layer, arr in arrays.items():
            h.create_dataset(f"L{layer}", data=np.asarray(arr, np.float16),
                             compression="gzip", compression_opts=4)
        h.attrs["model"] = model
        h.attrs["dim"] = dim
        h.attrs["layers"] = json.dumps(list(layers))
        h.attrs["max_len"] = max_len


def read_residues(path, layer):
    """-> dict with proteins(list), res_protein(int32 R), position(R), aa(list[str] R),
    emb(R,D float32)."""
    with h5py.File(path, "r") as h:
        proteins = [x.decode() if isinstance(x, bytes) else x for x in h["proteins"][:]]
        return {
            "proteins": proteins,
            "res_protein": h["res_protein"][:],
            "position": h["res_position"][:],
            "aa": [a.decode() if isinstance(a, bytes) else a for a in h["res_aa"][:]],
            "emb": h[f"L{layer}"][:].astype(np.float32),
        }


def residue_layers(path):
    with h5py.File(path, "r") as h:
        return json.loads(h.attrs["layers"])
