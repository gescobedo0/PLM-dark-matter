"""Masked mean/max pooling must exclude BOS, EOS and padding positions."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embed_lib.pooling import pool, residue_mask


def _reps():
    # T=6: pos0=BOS, pos1..pos3 residues, pos4=EOS, pos5=pad (for seq of len 3)
    r = torch.zeros(2, 6, 2)
    # seq 0, length 3
    r[0, 0] = torch.tensor([100., 100.])   # BOS  -> ignore
    r[0, 1] = torch.tensor([1., 10.])
    r[0, 2] = torch.tensor([2., 20.])
    r[0, 3] = torch.tensor([3., 30.])
    r[0, 4] = torch.tensor([-100., -100.]) # EOS  -> ignore
    r[0, 5] = torch.tensor([999., 999.])   # pad  -> ignore
    # seq 1, length 2
    r[1, 1] = torch.tensor([5., 5.])
    r[1, 2] = torch.tensor([7., 1.])
    r[1, 3] = torch.tensor([888., 888.])   # beyond length -> ignore
    r[1, 4] = torch.tensor([-9., -9.])
    return r, torch.tensor([3, 2])


def test_residue_mask():
    _, lengths = _reps()
    m = residue_mask(lengths, 6)
    assert m[0].tolist() == [False, True, True, True, False, False]
    assert m[1].tolist() == [False, True, True, False, False, False]


def test_mean_pool():
    r, L = _reps()
    out = pool(r, L, "mean")
    assert torch.allclose(out[0], torch.tensor([2., 20.]))   # mean of 1,2,3 / 10,20,30
    assert torch.allclose(out[1], torch.tensor([6., 3.]))    # mean of 5,7 / 5,1


def test_max_pool():
    r, L = _reps()
    out = pool(r, L, "max")
    assert torch.allclose(out[0], torch.tensor([3., 30.]))
    assert torch.allclose(out[1], torch.tensor([7., 5.]))    # ignores pos3 (888)


if __name__ == "__main__":
    test_residue_mask(); test_mean_pool(); test_max_pool()
    print("test_pooling: all passed")
