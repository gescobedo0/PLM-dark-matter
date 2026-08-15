"""Load the diagnostic config and resolve paths relative to the repo root."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import yaml

# repo root = parent of the sae_fidelity/ dir that contains this package
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def _resolve(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (root / q)


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Return the config dict with `paths.*` resolved to absolute Paths.

    Set env var SAE_DATA_DIR to override the data directory (e.g. a Colab Drive
    mount) without editing the file.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    root = REPO_ROOT
    paths = cfg.setdefault("paths", {})
    if os.environ.get("SAE_DATA_DIR"):
        paths["data_dir"] = os.environ["SAE_DATA_DIR"]
    data_dir = _resolve(root, paths.get("data_dir", "data"))

    resolved = {}
    for k, v in paths.items():
        if k == "catalog":
            resolved[k] = _resolve(root, v)
        elif k == "data_dir":
            resolved[k] = data_dir
        else:
            # other paths are given relative to repo root but conceptually live
            # under data_dir; honor whatever the file says, resolved from root.
            resolved[k] = _resolve(root, v)
    cfg["paths"] = resolved
    cfg["_root"] = root
    return cfg


def ns(cfg: dict) -> SimpleNamespace:
    """Dotted access for readability in scripts."""
    return SimpleNamespace(**cfg)
