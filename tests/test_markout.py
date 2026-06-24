"""Tests for the markout engine + ledger (research layer H1/H3).

Includes the L4 load-bearing pollution test: appending snapshots beyond
fill_ts + 60s must not change any markout at a fill whose 60s horizon
ends before them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmsim.ingest.lob import load_lob, SnapshotEvent
from mmsim.sim.loop import run_sim
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.ledger.writer import build_mid_timeline, build_ledger
from mmsim.ledger.costs import DEFAULT_COST_MODEL
from mmsim.ledger.schema import LEDGER_COLUMNS
from mmsim.markout.engine import (
    compute_markout, compute_markout_reference, HORIZONS_NS,
)

HERE = Path(__file__).parent
SNAP_30S = HERE / "fixtures" / "lob_btcusdt_30sec_snapshots.parquet"
TRADE_30S = HERE / "fixtures" / "lob_btcusdt_30sec_trades.parquet"

# BracketQuoter lives in the sibling test module.
from test_sim_fills import BracketQuoter  # noqa: E402


@pytest.fixture(scope="module")
def smoke_run():
    stream = load_lob(SNAP_30S, TRADE_30S)
    res = run_sim(stream, BracketQuoter(taker_every=50), QueueAwareFillModel())
    snap_ts, snap_mid = build_mid_timeline(stream)
    return stream, res, snap_ts, snap_mid


def test_markout_numba_bit_identical_to_reference(smoke_run):
    _, res, snap_ts, snap_mid = smoke_run
    ref = compute_markout_reference(res.fills, snap_ts, snap_mid)
    nb = compute_markout(res.fills, snap_ts, snap_mid, use_numba=True)
    for c in ["mid0", "mid_1s", "mid_10s", "mid_60s", "markout_1s",
              "markout_10s", "markout_60s", "realised_spread_10s", "adverse_10s"]:
        assert np.array_equal(getattr(ref, c), getattr(nb, c), equal_nan=True), c


def test_decomposition_identity(smoke_run):
    _, res, snap_ts, snap_mid = smoke_run
    mo = compute_markout(res.fills, snap_ts, snap_mid)
    px = np.array([f.price for f in res.fills])
    side = np.array([f.side for f in res.fills])
    qhs = side * (px - mo.mid0) / mo.mid0
    ident = mo.realised_spread_10s + mo.adverse_10s
    assert np.nanmax(np.abs(qhs - ident)) < 1e-9


def test_ledger_schema_complete(smoke_run):
    stream, res, _, _ = smoke_run
    w = build_ledger(res, stream, cost_model=DEFAULT_COST_MODEL)
    assert len(w.collected) == len(res.fills)
    for row in w.collected:
        assert list(row.keys()) == LEDGER_COLUMNS


def test_ledger_costs_nonzero(smoke_run):
    stream, res, _, _ = smoke_run
    w = build_ledger(res, stream, cost_model=DEFAULT_COST_MODEL)
    # Never costless: every fill carries a non-zero exchange fee.
    assert all(r["fee"] > 0 for r in w.collected)
    # ECONOMICS: a passive (maker) fill rests at its limit and does NOT cross
    # the book -> NO slippage.  Only a taker (crossing) fill slips.
    for r in w.collected:
        if r["is_maker"]:
            assert r["slippage"] == 0.0
        else:
            assert r["slippage"] > 0.0
    # And the fixture exercises both paths (BracketQuoter crosses periodically).
    assert any(r["is_maker"] for r in w.collected)
    assert any(not r["is_maker"] for r in w.collected)


def test_markout_l4_no_lookahead_under_pollution(smoke_run):
    """L4: append snapshots far beyond the last fill's 60s horizon; the
    markout columns for all existing fills must be byte-identical."""
    stream, res, snap_ts, snap_mid = smoke_run
    clean = compute_markout(res.fills, snap_ts, snap_mid)

    # Pollute: add snapshots 10 minutes after the last fill (well past 60s).
    last_fill_ts = max(f.ts_ns for f in res.fills)
    far = last_fill_ts + 10 * 60 * 1_000_000_000
    poll_ts = np.append(snap_ts, [far, far + 1_000_000_000]).astype(np.int64)
    poll_mid = np.append(snap_mid, [snap_mid[-1] + 5000.0, snap_mid[-1] - 5000.0])
    polluted = compute_markout(res.fills, poll_ts, poll_mid)

    for c in ["mid0", "mid_1s", "mid_10s", "mid_60s", "markout_10s",
              "realised_spread_10s", "adverse_10s"]:
        assert np.array_equal(getattr(clean, c), getattr(polluted, c), equal_nan=True), c


def test_horizons_locked():
    assert HORIZONS_NS == (1_000_000_000, 10_000_000_000, 60_000_000_000)
