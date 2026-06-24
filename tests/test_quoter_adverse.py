"""Adverse-selection filter tests."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from mmsim.ingest.lob import Book, SnapshotEvent, TradeEvent, load_lob
from mmsim.quoter.adverse import (
    HybridAdverseFilter, MicropriceDevFilter, OFIFilter,
    QueueImbalanceFilter, TradeToxicityFilter, VolSurgeFilter,
)


HERE = Path(__file__).resolve().parent
SNAP_1H = HERE / "fixtures" / "lob_btcusdt_60min_snapshots.parquet"
TRADE_1H = HERE / "fixtures" / "lob_btcusdt_60min_trades.parquet"


def _book(bids, asks, ts=0):
    return Book(ts_ns=ts, bids=tuple(bids), asks=tuple(asks))


def _trade(ts, price, size, side=1):
    return TradeEvent(ts_ns=ts, recv_ns=ts, symbol="X", venue="v",
                       price=price, size=size, side=side)


# --------------------------------------------------------------------- #
# OFI
# --------------------------------------------------------------------- #

def test_ofi_balanced_no_adverse():
    f = OFIFilter(window_ns=1000, threshold=0.5)
    f.observe_trade(_trade(0, 100.0, 1.0, +1))
    f.observe_trade(_trade(100, 100.0, 1.0, -1))
    assert f.is_adverse(200) is False


def test_ofi_dominant_buy_adverse():
    f = OFIFilter(window_ns=1000, threshold=0.5)
    f.observe_trade(_trade(0, 100.0, 5.0, +1))
    f.observe_trade(_trade(100, 100.0, 1.0, -1))
    # OFI = (5-1)/6 = 0.667 ≥ 0.5
    assert f.is_adverse(200) is True


def test_ofi_window_evicts():
    f = OFIFilter(window_ns=100, threshold=0.5)
    f.observe_trade(_trade(0, 100.0, 5.0, +1))
    # At t=200: cutoff=100, ts=0 evicted (0 ≤ 100)
    assert f.is_adverse(200) is False


# --------------------------------------------------------------------- #
# TradeToxicity
# --------------------------------------------------------------------- #

def test_toxicity_below_threshold():
    f = TradeToxicityFilter(window_ns=1000, threshold=0.8)
    f.observe_trade(_trade(0, 100.0, 1.0, +1))
    f.observe_trade(_trade(100, 100.0, 1.0, -1))
    assert f.is_adverse(200) is False


def test_toxicity_above_threshold():
    f = TradeToxicityFilter(window_ns=1000, threshold=0.8)
    for i in range(10):
        f.observe_trade(_trade(i, 100.0, 1.0, +1))
    f.observe_trade(_trade(20, 100.0, 0.5, -1))
    # max_share = 10 / 10.5 ≈ 0.952 ≥ 0.8
    assert f.is_adverse(30) is True


def test_toxicity_rejects_bad_threshold():
    with pytest.raises(ValueError):
        TradeToxicityFilter(window_ns=1000, threshold=0.3)
    with pytest.raises(ValueError):
        TradeToxicityFilter(window_ns=1000, threshold=1.1)


# --------------------------------------------------------------------- #
# VolSurge
# --------------------------------------------------------------------- #

def test_volsurge_quiet_no_adverse():
    f = VolSurgeFilter(window_ns=10_000, threshold_bp=100.0)
    for i in range(5):
        f.observe_trade(_trade(i * 100, 100.0 + 0.001 * i, 1.0))
    assert f.is_adverse(1000) is False


def test_volsurge_loud_adverse():
    f = VolSurgeFilter(window_ns=10_000, threshold_bp=1.0)
    # Wild price swings
    prices = [100.0, 110.0, 95.0, 105.0, 90.0]
    for i, px in enumerate(prices):
        f.observe_trade(_trade(i * 100, px, 1.0))
    assert f.is_adverse(1000) is True


def test_volsurge_needs_3_obs():
    f = VolSurgeFilter(window_ns=10_000, threshold_bp=0.1)
    f.observe_trade(_trade(0, 100.0, 1.0))
    f.observe_trade(_trade(100, 200.0, 1.0))  # huge but only 2 obs
    assert f.is_adverse(200) is False


# --------------------------------------------------------------------- #
# MicropriceDev
# --------------------------------------------------------------------- #

def test_microprice_dev_balanced_no_adverse():
    f = MicropriceDevFilter(threshold_bp=1.0)
    f.observe_book(_book([(100.0, 5.0)], [(101.0, 5.0)]))
    # imb=0, dev=0
    assert f.is_adverse(0) is False


def test_microprice_dev_imbalanced_adverse():
    f = MicropriceDevFilter(threshold_bp=10.0)  # 10 bp
    f.observe_book(_book([(100.0, 9.0)], [(101.0, 1.0)]))
    # imb = 0.8, half_spread = 0.5, mid = 100.5
    # microprice = 100.9, dev = 0.4
    # dev_bp = 0.4 / 100.5 * 1e4 ≈ 39.8 ≥ 10
    assert f.is_adverse(0) is True


# --------------------------------------------------------------------- #
# QueueImbalance
# --------------------------------------------------------------------- #

def test_queue_imb_balanced_no_adverse():
    f = QueueImbalanceFilter(threshold=0.5)
    f.observe_book(_book([(100.0, 5.0)], [(101.0, 5.0)]))
    assert f.is_adverse(0) is False


def test_queue_imb_extreme_adverse():
    f = QueueImbalanceFilter(threshold=0.5)
    f.observe_book(_book([(100.0, 9.0)], [(101.0, 1.0)]))
    # |9-1|/10 = 0.8 ≥ 0.5
    assert f.is_adverse(0) is True


def test_queue_imb_rejects_bad_threshold():
    with pytest.raises(ValueError):
        QueueImbalanceFilter(threshold=-0.1)
    with pytest.raises(ValueError):
        QueueImbalanceFilter(threshold=1.1)


# --------------------------------------------------------------------- #
# Hybrid
# --------------------------------------------------------------------- #

def test_hybrid_any_activates_if_any_child():
    f = HybridAdverseFilter([
        QueueImbalanceFilter(threshold=0.5),
        MicropriceDevFilter(threshold_bp=10.0),
    ], mode="any")
    f.observe_book(_book([(100.0, 9.0)], [(101.0, 1.0)]))
    assert f.is_adverse(0) is True


def test_hybrid_all_requires_all_children():
    f = HybridAdverseFilter([
        QueueImbalanceFilter(threshold=0.5),
        MicropriceDevFilter(threshold_bp=10000.0),  # huge threshold, never fires
    ], mode="all")
    f.observe_book(_book([(100.0, 9.0)], [(101.0, 1.0)]))
    assert f.is_adverse(0) is False


def test_hybrid_observe_propagates_to_all_children():
    """When the hybrid observes a book, all children must see it."""
    qi = QueueImbalanceFilter(threshold=0.5)
    md = MicropriceDevFilter(threshold_bp=10.0)
    f = HybridAdverseFilter([qi, md], mode="any")
    f.observe_book(_book([(100.0, 9.0)], [(101.0, 1.0)]))
    assert qi.is_adverse(0) is True
    assert md.is_adverse(0) is True


# --------------------------------------------------------------------- #
# Leak invariant: future trades don't change a past is_adverse()
# --------------------------------------------------------------------- #

def test_filter_no_lookahead_under_future_trades():
    f = OFIFilter(window_ns=1000, threshold=0.5)
    f.observe_trade(_trade(0, 100.0, 5.0, +1))
    f.observe_trade(_trade(100, 100.0, 1.0, -1))
    v_at_500 = f.is_adverse(500)
    # Now feed a future trade
    f.observe_trade(_trade(800, 100.0, 100.0, -1))
    # is_adverse(500) re-queried: the future trade is filtered out
    # by ts <= t_ns; the answer doesn't change
    v_at_500_again = f.is_adverse(500)
    assert v_at_500 == v_at_500_again


# --------------------------------------------------------------------- #
# G3 — DS-LOB-1H baseline: activations at 99th-percentile threshold
# --------------------------------------------------------------------- #

@pytest.mark.skipif(
    not SNAP_1H.exists() or not TRADE_1H.exists(),
    reason="DS-LOB-1H fixture not present",
)
def test_filter_ds_lob_1h_baseline():
    """Drive an OFIFilter through DS-LOB-1H trade tape with a 99th-
    percentile-style threshold.  Expect a small number of activations
    (spec: 'a small number of activations; record baseline')."""
    stream = load_lob(SNAP_1H, TRADE_1H)
    trades = [e for e in stream if isinstance(e, TradeEvent)]

    f = OFIFilter(window_ns=1_000_000_000, threshold=0.95)  # 1s window, OFI ≥ 0.95
    activations = 0
    last_state = False
    for t in trades:
        f.observe_trade(t)
        cur = f.is_adverse(t.ts_ns)
        if cur and not last_state:
            activations += 1  # count rising edges
        last_state = cur

    # Baseline: bounded but non-zero.
    assert activations >= 1
    assert activations < len(trades) // 100  # < 1% of trades
