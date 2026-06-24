"""Tests for the H1/H7 research modules (single-thread)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mmsim.ingest.lob import load_lob
from mmsim.research.h1_replay import carve_holdout, stream_span_ns, rank_separation
from mmsim.research.perm_null import (
    permutation_null, _perm_null_reference, _perm_null_numba, tail_guarded_rrr,
)

HERE = Path(__file__).parent
SNAP_30S = HERE / "fixtures" / "lob_btcusdt_30sec_snapshots.parquet"
TRADE_30S = HERE / "fixtures" / "lob_btcusdt_30sec_trades.parquet"


def test_carve_holdout_disjoint():
    stream = load_lob(SNAP_30S, TRADE_30S)
    train, holdout, cut = carve_holdout(stream, holdout_minutes=0.2)  # 12s holdout
    assert all(ev.ts_ns <= cut for ev in train)
    assert all(ev.ts_ns > cut for ev in holdout)
    assert len(train) + len(holdout) == len(stream)


def test_perm_null_bit_identical():
    rng = np.random.default_rng(1)
    mk = rng.normal(0, 1e-4, 500)
    a = _perm_null_reference(mk, 100, 77)
    b = _perm_null_numba(mk, 100, np.uint64(77))
    assert np.array_equal(a, b)


def test_perm_null_calibrated_on_zero_signal():
    rng = np.random.default_rng(2)
    mk = rng.normal(0, 1e-4, 2000)  # zero-mean = no skill
    r = permutation_null(mk, m=500, seed=7)
    # null is centered near zero with a symmetric spread (calibrated).
    assert abs(r.null_mean) < 5e-5
    assert r.null_q05 < 0 < r.null_q95
    assert abs(r.null_q95 + r.null_q05) < 0.5 * (r.null_q95 - r.null_q05)
    # p-value is a valid probability
    assert 0.0 < r.p_value <= 1.0


def test_tail_guarded_rrr_ex_best():
    pnl = np.array([5.0, -1.0, -1.0, -1.0])  # one big win carries it
    g = tail_guarded_rrr(pnl)
    assert g["rrr"] > 1.0
    assert g["rrr_ex_best"] == 0.0  # removing the win => all losses


def test_rank_separation_symmetric():
    x = np.random.default_rng(4).normal(0, 1, 1000)
    # identical distributions -> ~0.5
    p = rank_separation(x, x.copy())
    assert abs(p - 0.5) < 0.05
