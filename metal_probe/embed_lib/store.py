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
