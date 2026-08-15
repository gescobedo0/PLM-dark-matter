"""Resumable sharded storage for per-residue activations and SAE codes.

A shard bundles many proteins' variable-length residue arrays into one flat array
plus a per-protein length index, so short microproteins don't create thousands of
tiny files. A JSON manifest tracks completed protein ids to make extraction
resumable across Colab disconnects.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_activation_shard(out_dir: Path, shard_idx: int, ids, lengths, acts):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"shard_{shard_idx:05d}.npz",
             ids=np.asarray(ids, dtype=object),
             lengths=np.asarray(lengths, dtype=np.int32),
             acts=np.asarray(acts, dtype=np.float16))


def load_activation_shard(path: Path):
    z = np.load(path, allow_pickle=True)
    return z["ids"], z["lengths"], z["acts"]


def iter_proteins(path: Path):
    """Yield (pid, acts[L, hidden]) from one activation shard."""
    ids, lengths, acts = load_activation_shard(path)
    start = 0
    for pid, L in zip(ids, lengths):
        yield str(pid), acts[start:start + int(L)]
        start += int(L)


def shard_paths(out_dir: Path):
    return sorted(Path(out_dir).glob("shard_*.npz"))


def read_manifest(out_dir: Path) -> dict:
    p = Path(out_dir) / "manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"done_ids": [], "n_shards": 0}


def write_manifest(out_dir: Path, manifest: dict):
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest))


def done_ids(out_dir: Path) -> set:
    return set(read_manifest(out_dir).get("done_ids", []))
