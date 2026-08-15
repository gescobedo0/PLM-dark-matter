"""HDF5 embedding store round-trips ids + vectors + metadata."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embed_lib import store


def test_roundtrip():
    ids = ["1abc_A", "2def_B", "microorf1"]
    lengths = [100, 250, 42]
    trunc = [False, False, False]
    arrays = {(24, "mean"): np.random.randn(3, 8).astype("float32"),
              (24, "max"): np.random.randn(3, 8).astype("float32")}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.h5"
        store.write(p, ids, lengths, trunc, arrays, model="esm2_test",
                    dim=8, layers=[24], poolings=["mean", "max"], max_len=1022)
        assert store.read_ids(p) == ids
        rids, arr = store.read(p, 24, "mean")
        assert rids == ids
        assert np.allclose(arr, arrays[(24, "mean")])
        layers, pools = store.available(p)
        assert layers == [24] and set(pools) == {"mean", "max"}


if __name__ == "__main__":
    test_roundtrip()
    print("test_store: passed")
