"""CPU-only end-to-end plumbing test for 03 -> 04 -> 05.

No GPU, no network, no real ESMC/SAE. Synthesizes activations that are exact
sparse combinations of an orthonormal dictionary (so an invertible mock SAE
reconstructs them -> low FVU everywhere = 'comparable'), routes each marker set
onto reserved 'concept' atoms, and mocks the HF feature-description table so the
pre-registration search finds those atoms. Then asserts the report reaches a
coherent GO decision with markers validating.

Run: python tests/test_pipeline_mock.py
"""
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sae_lib import sae, store, features  # noqa: E402

DIM = 32
TOPK = 4
CONCEPT_ATOMS = {"metal_coordination": [0, 1], "fe_s_coordination": [2, 3],
                 "membrane": [4, 5], "disorder": [6, 7]}
GENERIC_ATOMS = list(range(8, DIM))
MARKER_CONCEPT = {"c2h2_zinc_finger": "metal_coordination",
                  "ferredoxin": "fe_s_coordination",
                  "rubredoxin": "fe_s_coordination",
                  "tm_peptide": "membrane",
                  "disordered_small": "disorder"}


def _Q(seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((DIM, DIM)))
    return q


def _mock_sae_weights(*a, **k):
    Q = _Q()
    return sae.SAEWeights(Q.T.copy(), np.zeros(DIM), Q, np.zeros(DIM), TOPK)


def _mock_feature_table(*a, **k):
    desc = []
    for i in range(DIM):
        if i in CONCEPT_ATOMS["metal_coordination"]:
            desc.append("zinc metal coordination site, C2H2 motif")
        elif i in CONCEPT_ATOMS["fe_s_coordination"]:
            desc.append("iron-sulfur cluster ferredoxin rubredoxin binding")
        elif i in CONCEPT_ATOMS["membrane"]:
            desc.append("transmembrane hydrophobic membrane helix segment")
        elif i in CONCEPT_ATOMS["disorder"]:
            desc.append("intrinsically disordered unstructured region")
        else:
            desc.append("generic mixed activation, no clear function")
    return pd.DataFrame({"feature_id": range(DIM), "description": desc,
                         "category": ["x"] * DIM})


def _residues(atoms_pool, n, rng, hi_atoms=None):
    """n residues, each a positive combo of TOPK atoms from atoms_pool."""
    Q = _Q()
    out = np.zeros((n, DIM), np.float32)
    for r in range(n):
        pool = list(atoms_pool)
        atoms = rng.choice(pool, size=min(TOPK, len(pool)), replace=False)
        coeff = np.abs(rng.standard_normal(len(atoms))) + 0.5
        if hi_atoms is not None:                      # force concept atoms high
            atoms = np.array(list(hi_atoms) + [a for a in atoms
                             if a not in hi_atoms])[:TOPK]
            coeff = np.array([3.0] * len(hi_atoms) +
                             list(coeff))[:len(atoms)]
        out[r] = coeff @ Q[atoms]
    return out


def build_fixture(tmp: Path):
    rng = np.random.default_rng(1)
    rows, shard_ids, shard_len, shard_acts = [], [], [], []

    def add(pid, group, subclass, L, hi=None, pool=GENERIC_ATOMS):
        rows.append({"id": pid, "group": group, "subclass": subclass,
                     "seq": "A" * L, "length": L, "hydropathy": 0.0,
                     "tm_score": 0.0, "tm_flag": "soluble"})
        shard_ids.append(pid); shard_len.append(L)
        shard_acts.append(_residues(pool, L, rng, hi_atoms=hi))

    for i in range(20):
        add(f"micro_{i}", "microprotein", "regulatory", 25)
    for i in range(20):
        add(f"bg_{i}", "background", "canonical", 60)
    for i in range(20):
        add(f"lm_{i}", "length_matched", "canonical_fragment", 25)
    for mset, concept in MARKER_CONCEPT.items():
        for i in range(10):
            add(f"{mset}_{i}", "marker", mset, 30,
                hi=CONCEPT_ATOMS[concept])

    sub = pd.DataFrame(rows)
    (tmp / "data").mkdir(parents=True, exist_ok=True)
    sub.to_parquet(tmp / "data" / "subsample.parquet", index=False)

    act_dir = tmp / "data" / "activations"
    store.save_activation_shard(act_dir, 0, shard_ids, shard_len,
                                np.concatenate(shard_acts, 0))
    store.write_manifest(act_dir, {"done_ids": shard_ids, "n_shards": 1})

    cfg = {
        "seed": 0,
        "paths": {"catalog": "neworfcatalog.csv", "data_dir": str(tmp / "data"),
                  "subsample": str(tmp / "data" / "subsample.parquet"),
                  "activations_dir": str(act_dir),
                  "sae_features": str(tmp / "data" / "sae_features"),
                  "report_dir": str(tmp / "data" / "report")},
        "hf": {"base_model": "x", "sae_repo": "x", "feature_table_repo": "x",
               "feature_table_file": "x"},
        "sae": {"hidden_dim": DIM, "codebook": DIM, "topk": TOPK, "layer_index": 1},
        "decision": {"fvu_comparable_ratio": 1.25, "fvu_elevated_ratio": 2.0},
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def run_script(name, cfg_path, extra=None):
    argv = [name, "--config", str(cfg_path)] + (extra or [])
    with mock.patch.object(sys, "argv", argv):
        runpy.run_path(str(ROOT / name), run_name="__main__")


def main():
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="sae_mock_"))
    cfg_path = build_fixture(tmp)

    with mock.patch.object(sae, "load_sae_weights", _mock_sae_weights), \
         mock.patch.object(features, "load_feature_table", _mock_feature_table):
        run_script("03_sae_forward.py", cfg_path)
        run_script("04_diagnostics.py", cfg_path)
        run_script("05_report.py", cfg_path)

    rep = tmp / "data" / "report"
    results = json.loads((rep / "results.json").read_text())
    dec = json.loads((rep / "decision.json").read_text())

    # FVU should be ~0 and comparable everywhere (exact reconstruction)
    fvu_micro = next(r for r in results["fvu_verdict"]["rows"]
                     if r["group"] == "microprotein" and r["subclass"] == "(all)")
    assert fvu_micro["label"] == "comparable", fvu_micro
    assert fvu_micro["fvu_median"] < 0.01, fvu_micro

    # structured markers must fire their expected features above background
    for mset in ("c2h2_zinc_finger", "ferredoxin", "tm_peptide"):
        mr = results["marker_recovery"][mset]
        assert mr["fraction_above_bg_90pct"] > 0.8, (mset, mr)

    # structured features must NOT fire on disordered proteins
    dn = results["disorder_negative_control"]
    assert dn["fraction_above_bg_90pct"] < 0.5, dn

    assert dec["decision"] == "GO", dec
    assert dec["markers_validate"] is True, dec
    print("\nPASS pipeline mock: decision =", dec["decision"],
          "| micro FVU label =", fvu_micro["label"],
          "| markers_validate =", dec["markers_validate"])


if __name__ == "__main__":
    main()
