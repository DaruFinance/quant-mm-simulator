"""Maker/taker fill model tests.

Covers:
  - QueueAwareFillModel maker semantics (queue consumes first, then
    spillover fills us).
  - QueueAwareFillModel taker walks (book consumption, limit_px stop).
  - Bracket quoter on DS-LOB-1H: baseline maker/taker fill ratio.
  - Cross-event leak invariant (5 random T values; the 50-T HIGH-RISK
    battery already exists for the queue tracker, which is the underlying
    state — not duplicated here).
  - StatelessFillsAdapter back-compat with the stub fill model.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest

from mmsim.ingest.lob import (
    Book, Event, EventStream, SnapshotEvent, TradeEvent, load_lob,
)
from mmsim.sim.loop import (
    Fill, Order, QuoteRequest, SimResult, run_sim,
)
from mmsim.sim.fills import (
    FillModelProtocol, QueueAwareFillModel, StatelessFillsAdapter,
    TakerRequest,
)


HERE = Path(__file__).resolve().parent
SNAP_30S = HERE / "fixtures" / "lob_btcusdt_30sec_snapshots.parquet"
TRADE_30S = HERE / "fixtures" / "lob_btcusdt_30sec_trades.parquet"
SNAP_1H = HERE / "fixtures" / "lob_btcusdt_60min_snapshots.parquet"
TRADE_1H = HERE / "fixtures" / "lob_btcusdt_60min_trades.parquet"

DEFAULT_QUOTE_SIZE = 0.001


# --------------------------------------------------------------------- #
# Tiny constructors
# --------------------------------------------------------------------- #

def _book(ts, bids, asks):
    return Book(ts_ns=ts, bids=tuple(bids), asks=tuple(asks))


def _snap(ts, bids, asks):
    return SnapshotEvent(
        ts_ns=ts, recv_ns=ts, symbol="X", venue="v", depth=len(bids),
        bids=tuple(bids), asks=tuple(asks),
    )


def _trade(ts, price, size, side):
    return TradeEvent(
        ts_ns=ts, recv_ns=ts, symbol="X", venue="v",
        price=price, size=size, side=side,
    )


def _order(side, price, size, placed_at):
    return Order(
        order_id=0, side=side, price=price, size=size, placed_at_ns=placed_at,
    )


# --------------------------------------------------------------------- #
# Maker fill semantics — queue-aware
# --------------------------------------------------------------------- #

def test_queue_aware_no_fill_while_queue_pos_positive():
    """Trades smaller than our queue_pos consume from queue, not us."""
    book = _book(0, [(100.0, 5.0)], [(101.0, 3.0)])
    o = _order(+1, 100.0, 1.0, 0)
    o.order_id = 1
    model = QueueAwareFillModel()
    model.on_order_placed(o, book)
    # Trade of 2 < queue_pos (5) — no fill for us.
    hits = model.on_trade(_trade(10, 100.0, 2.0, -1), [o])
    assert hits == []


def test_queue_aware_fills_us_on_spillover():
    """Once queue_pos hits 0, the next trade's overflow fills us."""
    book = _book(0, [(100.0, 2.0)], [(101.0, 3.0)])
    o = _order(+1, 100.0, 1.0, 0)
    o.order_id = 1
    model = QueueAwareFillModel()
    model.on_order_placed(o, book)
    # First trade: 2 consumed from queue, 0 spills.
    assert model.on_trade(_trade(10, 100.0, 2.0, -1), [o]) == []
    # Second trade: queue is empty; 0.5 spills onto us.
    hits = model.on_trade(_trade(20, 100.0, 0.5, -1), [o])
    assert hits == [(1, 0.5)]


def test_queue_aware_partial_fill_caps_at_order_size():
    book = _book(0, [(100.0, 0.0)], [(101.0, 3.0)])
    # Queue starts at 0 — we're at the front immediately.
    o = _order(+1, 100.0, 0.001, 0)
    o.order_id = 1
    model = QueueAwareFillModel()
    model.on_order_placed(o, book)
    # Big sell trade — should cap at our 0.001 size.
    hits = model.on_trade(_trade(10, 100.0, 5.0, -1), [o])
    assert hits == [(1, 0.001)]


def test_taker_walks_top_of_book():
    book = _book(0, [(100.0, 1.0)], [(101.0, 0.5), (102.0, 0.5), (103.0, 1.0)])
    model = QueueAwareFillModel()
    rows = model.fill_taker(TakerRequest(side=+1, size=1.5), book, 100)
    # Buy 1.5: consume 0.5 @ 101 + 0.5 @ 102 + 0.5 @ 103.
    assert rows == [(101.0, 0.5), (102.0, 0.5), (103.0, 0.5)]


def test_taker_limit_px_stops_walk_early():
    book = _book(0, [(100.0, 1.0)], [(101.0, 0.5), (102.0, 0.5), (103.0, 1.0)])
    model = QueueAwareFillModel()
    rows = model.fill_taker(
        TakerRequest(side=+1, size=2.0, limit_px=101.5), book, 100)
    # Only consume 0.5 @ 101 — 102 > limit.
    assert rows == [(101.0, 0.5)]


def test_taker_size_zero_returns_empty():
    book = _book(0, [(100.0, 1.0)], [(101.0, 0.5)])
    model = QueueAwareFillModel()
    assert model.fill_taker(TakerRequest(side=+1, size=0.0), book, 0) == []


# --------------------------------------------------------------------- #
# Loop integration with QueueAwareFillModel
# --------------------------------------------------------------------- #

def quoter_tob_maker_only(book, _active, _t):
    if book is None or book.best_bid is None or book.best_ask is None:
        return []
    return [
        QuoteRequest(side=+1, price=book.best_bid, size=DEFAULT_QUOTE_SIZE),
        QuoteRequest(side=-1, price=book.best_ask, size=DEFAULT_QUOTE_SIZE),
    ]


def test_loop_with_queue_aware_model_fills_fewer_than_naive():
    """Sanity: queue-aware maker fills are a subset of naive fills.
    On the same fixture, the naive stub gets a bigger fill count
    because it ignores queue ahead of us."""
    stream = load_lob(SNAP_30S, TRADE_30S)
    model = QueueAwareFillModel()
    res_qa = run_sim(stream, quoter_tob_maker_only, model)

    # Compare against the naive stub (imported from test_sim_loop)
    import sys
    sys.path.insert(0, str(HERE))
    from test_sim_loop import stub_fills_naive
    res_naive = run_sim(stream, quoter_tob_maker_only, stub_fills_naive)

    # Queue-aware should produce strictly fewer (or equal) maker fills.
    qa_makers = sum(1 for f in res_qa.fills if f.is_maker)
    naive_makers = sum(1 for f in res_naive.fills if f.is_maker)
    assert qa_makers <= naive_makers
    # Both have zero takers (quoter is maker-only).
    assert res_qa.n_taker_fills == 0
    assert res_naive.n_taker_fills == 0


# --------------------------------------------------------------------- #
# Bracket quoter — TOB makers + occasional small taker fires.  The
# verification's "baseline maker-vs-taker fill ratio" target.
# --------------------------------------------------------------------- #

class BracketQuoter:
    """TOB maker on both sides + a 0.0001-BTC taker buy fired every
    ``taker_every`` snapshots.  Stateful (counter); deterministic."""

    def __init__(self, maker_size=0.001, taker_size=0.0001, taker_every=500):
        self.maker_size = maker_size
        self.taker_size = taker_size
        self.taker_every = taker_every
        self._snap_count = 0

    def __call__(self, book, _active, _t_ns):
        self._snap_count += 1
        if book is None or book.best_bid is None or book.best_ask is None:
            return []
        out = [
            QuoteRequest(side=+1, price=book.best_bid, size=self.maker_size),
            QuoteRequest(side=-1, price=book.best_ask, size=self.maker_size),
        ]
        if self._snap_count % self.taker_every == 0:
            # Buy a tiny amount across the spread.
            out.append(TakerRequest(side=+1, size=self.taker_size))
        return out


def test_bracket_quoter_fires_takers_on_30s_smoke():
    stream = load_lob(SNAP_30S, TRADE_30S)
    model = QueueAwareFillModel()
    res = run_sim(stream, BracketQuoter(taker_every=50), model)
    # 30-s smoke has ~290 snapshots; with taker_every=50 we fire ~5
    # takers, each producing >= 1 taker fill row.
    assert res.n_taker_fills >= 1
    assert res.n_taker_fills + res.n_maker_fills == len(res.fills)
    # All taker fills carry is_maker=False.
    taker_fills = [f for f in res.fills if not f.is_maker]
    assert all(f.side == +1 for f in taker_fills)


# --------------------------------------------------------------------- #
# Back-compat: stateless adapter still works (legacy callable path)
# --------------------------------------------------------------------- #

def test_stateless_fills_adapter_supports_legacy_callable():
    import sys
    sys.path.insert(0, str(HERE))
    from test_sim_loop import stub_fills_naive
    stream = load_lob(SNAP_30S, TRADE_30S)
    res = run_sim(stream, quoter_tob_maker_only, stub_fills_naive)
    assert res.n_taker_fills == 0
    assert res.n_maker_fills == len(res.fills)
    # Loop wraps the callable in the adapter; the existing
    # baseline behavior must match.


# --------------------------------------------------------------------- #
# Leak invariant: 5 random T values
# --------------------------------------------------------------------- #

def _pollute(e: Event, factor: float = 99.99) -> Event:
    if isinstance(e, SnapshotEvent):
        garbage = tuple((1.0, factor) for _ in e.bids)
        return dataclasses.replace(e, bids=garbage, asks=garbage)
    return dataclasses.replace(e, price=factor, size=factor, side=0)


@pytest.mark.parametrize("seed", [0, 7, 19, 42, 123])
def test_queue_aware_no_lookahead_under_pollution(seed):
    stream = load_lob(SNAP_30S, TRADE_30S)
    rng = np.random.default_rng(seed)
    pivot = int(rng.integers(len(stream) // 4, 3 * len(stream) // 4))
    T = stream[pivot].ts_ns

    prefix = [e for e in stream if e.ts_ns <= T]
    clean_res = run_sim(prefix, BracketQuoter(taker_every=50),
                          QueueAwareFillModel())

    polluted = [e if e.ts_ns <= T else _pollute(e) for e in stream]
    full_res = run_sim(polluted, BracketQuoter(taker_every=50),
                        QueueAwareFillModel())
    full_pre_T = [f for f in full_res.fills if f.ts_ns <= T]

    assert len(full_pre_T) == len(clean_res.fills)
    for fa, fb in zip(clean_res.fills, full_pre_T):
        assert fa.ts_ns == fb.ts_ns
        assert fa.price == fb.price
        assert fa.size == fb.size
        assert fa.side == fb.side
        assert fa.is_maker == fb.is_maker


# --------------------------------------------------------------------- #
# G3 — DS-LOB-1H baseline maker/taker ratio with bracket quoter
# --------------------------------------------------------------------- #

@pytest.mark.skipif(
    not SNAP_1H.exists() or not TRADE_1H.exists(),
    reason="DS-LOB-1H fixture not present",
)
def test_bracket_quoter_ds_lob_1h_baseline():
    stream = load_lob(SNAP_1H, TRADE_1H)
    model = QueueAwareFillModel()
    res = run_sim(stream, BracketQuoter(taker_every=500), model)

    # Pinned baselines on the bundled DS-LOB-1H fixture.  Failing
    # this test means the loop, the fill model, the queue tracker,
    # or the bracket quoter changed economically.
    expected_n_events = 535_876
    expected_n_quoter_calls = 35_989
    expected_n_maker_fills = 1_823
    expected_n_taker_fills = 71            # taker_every=500 * 35989 -> 71 fires
    expected_total_qty = 0.428150          # BTC; 0.421050 maker + 0.007100 taker

    assert res.n_events_processed == expected_n_events
    assert res.n_quoter_calls == expected_n_quoter_calls
    assert res.n_maker_fills == expected_n_maker_fills, (
        f"baseline regression: n_maker_fills got {res.n_maker_fills}, "
        f"expected {expected_n_maker_fills}")
    assert res.n_taker_fills == expected_n_taker_fills
    assert res.n_maker_fills + res.n_taker_fills == len(res.fills)
    total_qty = sum(f.size for f in res.fills)
    assert abs(total_qty - expected_total_qty) < 1e-6
