"""End-to-end validation of the probe path on the REAL catalog with MOCK vectors.

Confirms (without a GPU / fair-esm): the catalog loads with valid Study A labels
and cluster groups; GroupKFold never leaks a cluster across folds; the probe
recovers a planted signal; the composition baseline runs; and the 07_probe CLI
wires embeddings -> results without error.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from embed_lib import probe, store


def _study_a():
    cat = pd.read_parquet(ROOT / "catalog" / "catalog.parquet")
    a = cat[cat["study"] == "A"].copy()
    a["label"] = a["label"].astype(int)
    return a


def test_catalog_labels_and_groups():
    a = _study_a()
    assert set(a["label"].unique()) == {0, 1}
    assert a["split_group"].notna().all()
    # leakage guarantee from construction: no cluster shared by both classes
    pos = set(a[a.label == 1]["split_group"])
    neg = set(a[a.label == 0]["split_group"])
    assert not (pos & neg)


def test_groupkfold_no_leakage():
    a = _study_a()
    groups = a["split_group"].to_numpy()
    X = np.zeros((len(a), 2))
    assert probe.assert_no_group_leakage(groups, X, a["label"].to_numpy(), 5)


def test_probe_recovers_planted_signal():
    a = _study_a()
    y = a["label"].to_numpy()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((len(a), 16)).astype("float32")
    X[y == 1, :4] += 1.5          # planted, row-level (independent of cluster)
    r = probe.cluster_cv(X, y, a["split_group"].to_numpy(), n_splits=5)
    assert r["auroc_mean"] > 0.7, r
    assert r["n_groups"] == a["split_group"].nunique()


def test_composition_baseline_runs():
    a = _study_a()
    Xc = probe.aa_composition(a["seq"].tolist())
    assert Xc.shape == (len(a), 20)
    r = probe.cluster_cv(Xc, a["label"].to_numpy(), a["split_group"].to_numpy())
    assert np.isfinite(r["auroc_mean"]) and np.isfinite(r["mcc_mean"])


def test_probe_cli_end_to_end(tmp_path=None):
    """Write a mock esm2 file the CLI can consume, run 07_probe, check outputs."""
    a = _study_a()
    y = a["label"].to_numpy()
    rng = np.random.default_rng(1)
    X = rng.standard_normal((len(a), 8)).astype("float32")
    X[y == 1, :2] += 1.0
    edir = ROOT / "embeddings"
    edir.mkdir(exist_ok=True)
    mock = edir / "esm2_MOCKTEST.h5"
    store.write(mock, a["protein_id"].tolist(), a["length"].tolist(),
                [False] * len(a), {(1, "mean"): X}, model="mock", dim=8,
                layers=[1], poolings=["mean"], max_len=1022)
    try:
        out = subprocess.run(
            [sys.executable, str(ROOT / "07_probe.py"), "--models", "MOCKTEST"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-1500:]
        assert (ROOT / "results" / "probe_results.csv").exists()
        res = pd.read_csv(ROOT / "results" / "probe_results.csv")
        assert (res["model"] == "baseline").any()
        assert (res["model"] == "MOCKTEST").any()
    finally:
        mock.unlink(missing_ok=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: passed")
    print("test_pipeline_mock: all passed")
