"""run_sim loop tests + DS-LOB-1H baseline.

The "1-tick-wide" quoter and the naive fill model used here are
**reference stubs** — they exist for loop verification only and
can be replaced by the formal Quoter contract and the
maker/taker fill model.  Their job in this file is to
let the loop exercise end-to-end on real data and emit a
reproducible baseline fill count (a regression target).
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

HERE = Path(__file__).resolve().parent
SNAP_30S = HERE / "fixtures" / "lob_btcusdt_30sec_snapshots.parquet"
TRADE_30S = HERE / "fixtures" / "lob_btcusdt_30sec_trades.parquet"
SNAP_1H = HERE / "fixtures" / "lob_btcusdt_60min_snapshots.parquet"
TRADE_1H = HERE / "fixtures" / "lob_btcusdt_60min_trades.parquet"

DEFAULT_QUOTE_SIZE = 0.001  # base-asset units (~$80 notional at $80k)


# --------------------------------------------------------------------- #
# Reference stubs.  These are explicitly NOT production: they exist to
# give the loop a self-contained verification path before the Protocol
# quoter and queue-aware fill model land.
# --------------------------------------------------------------------- #

def stub_quoter_top_of_book(
    book: Optional[Book],
    _current_orders: List[Order],
    _t_ns: int,
) -> List[QuoteRequest]:
    """Place one bid and one ask, joining TOB on each side, of fixed
    size ``DEFAULT_QUOTE_SIZE``.  Skip if the book isn't warmed up."""
    if book is None or book.best_bid is None or book.best_ask is None:
        return []
    return [
        QuoteRequest(side=+1, price=book.best_bid, size=DEFAULT_QUOTE_SIZE),
        QuoteRequest(side=-1, price=book.best_ask, size=DEFAULT_QUOTE_SIZE),
    ]


def stub_fills_naive(
    active_orders: List[Order],
    trade: TradeEvent,
) -> List[Tuple[int, float]]:
    """Naive maker fill: a sell-aggressor trade (side=-1) at price
    ``<=`` our bid hits the lowest-id matching bid; a buy-aggressor
    trade (side=+1) at price ``>=`` our ask hits the lowest-id
    matching ask.  At most one fill per trade — multi-level walks
    land in the queue-aware fill model.

    Order priority: lowest ``order_id`` first (deterministic).
    """
    if trade.size <= 0.0:
        return []
    if trade.side == 0:
        return []
    candidates = sorted(active_orders, key=lambda o: o.order_id)
    if trade.side == -1:
        # sell-aggressor crossing bids
        for o in candidates:
            if o.side == +1 and trade.price <= o.price:
                return [(o.order_id, min(trade.size, o.size))]
    elif trade.side == +1:
        # buy-aggressor crossing asks
        for o in candidates:
            if o.side == -1 and trade.price >= o.price:
                return [(o.order_id, min(trade.size, o.size))]
    return []


# --------------------------------------------------------------------- #
# Loop unit tests on the 30-s fixture.
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def stream_30s() -> EventStream:
    return load_lob(SNAP_30S, TRADE_30S)


def test_run_sim_returns_well_formed_result(stream_30s):
    res = run_sim(stream_30s, stub_quoter_top_of_book, stub_fills_naive)
    assert isinstance(res, SimResult)
    assert res.n_events_processed == len(stream_30s)
    assert res.n_snapshot_events + res.n_trade_events == len(stream_30s)
    # Quoter is called once per snapshot.
    assert res.n_quoter_calls == res.n_snapshot_events


def test_run_sim_emits_some_fills_on_real_data(stream_30s):
    res = run_sim(stream_30s, stub_quoter_top_of_book, stub_fills_naive)
    # The 30-s fixture has hundreds of trades; with a TOB-joining
    # quoter and naive fills, expect several fills.
    assert len(res.fills) > 0
    # Every fill is a maker (the stub doesn't do taker).
    assert all(f.is_maker for f in res.fills)
    # Side encoding: every fill side is +/- 1.
    assert all(f.side in (-1, +1) for f in res.fills)


def test_run_sim_reproducible_across_reruns(stream_30s):
    a = run_sim(stream_30s, stub_quoter_top_of_book, stub_fills_naive)
    b = run_sim(stream_30s, stub_quoter_top_of_book, stub_fills_naive)
    # Bit-identical fills given same inputs.
    assert len(a.fills) == len(b.fills)
    for fa, fb in zip(a.fills, b.fills):
        assert fa == fb


def test_run_sim_partial_fill_decrements_size():
    """Two trades against one bid: the first partial-fills, the
    second fully fills the remainder.  Verifies size accounting."""
    # Hand-built event stream — kept as a unit test (the leak +
    # reproducibility properties are the load-bearing real-data tests).
    snap = SnapshotEvent(
        ts_ns=100, recv_ns=100, symbol="BTC-USDT", venue="binance",
        depth=1, bids=((100.0, 1.0),), asks=((101.0, 1.0),),
    )
    t1 = TradeEvent(
        ts_ns=200, recv_ns=200, symbol="BTC-USDT", venue="binance",
        price=100.0, size=DEFAULT_QUOTE_SIZE / 2.0, side=-1,
    )
    t2 = TradeEvent(
        ts_ns=300, recv_ns=300, symbol="BTC-USDT", venue="binance",
        price=100.0, size=DEFAULT_QUOTE_SIZE * 2.0, side=-1,
    )
    res = run_sim([snap, t1, t2], stub_quoter_top_of_book, stub_fills_naive)
    # Two fill records, both on the bid side (+1), summing to the
    # original quote size.
    bid_fills = [f for f in res.fills if f.side == +1]
    assert len(bid_fills) == 2
    assert bid_fills[0].size == pytest.approx(DEFAULT_QUOTE_SIZE / 2.0)
    assert bid_fills[1].size == pytest.approx(DEFAULT_QUOTE_SIZE / 2.0)


# --------------------------------------------------------------------- #
# G2 — leak invariant.  Pollute events with ts_ns > T, assert the fill
# list up to T is bit-identical.
# --------------------------------------------------------------------- #

def _pollute(e: Event, factor: float = 99.99) -> Event:
    if isinstance(e, SnapshotEvent):
        garbage = tuple((1.0, factor) for _ in e.bids)
        return dataclasses.replace(e, bids=garbage, asks=garbage)
    return dataclasses.replace(e, price=factor, size=factor, side=0)


@pytest.mark.parametrize("seed", [0, 7, 19, 42, 123])
def test_run_sim_no_lookahead_under_pollution(stream_30s, seed):
    """For 5 random T points, polluting every event with ts_ns > T
    must not change the fills emitted up to T."""
    rng = np.random.default_rng(seed)
    pivot = int(rng.integers(len(stream_30s) // 4, 3 * len(stream_30s) // 4))
    T = stream_30s[pivot].ts_ns

    # Run on the prefix only — that's the "clean view at T" baseline.
    prefix = [e for e in stream_30s if e.ts_ns <= T]
    clean = run_sim(prefix, stub_quoter_top_of_book, stub_fills_naive)

    # Pollute the suffix and run on the full stream; fills with ts<=T
    # must match the clean prefix run exactly.
    polluted = [e if e.ts_ns <= T else _pollute(e) for e in stream_30s]
    full = run_sim(polluted, stub_quoter_top_of_book, stub_fills_naive)
    full_fills_pre_T = [f for f in full.fills if f.ts_ns <= T]

    assert len(full_fills_pre_T) == len(clean.fills), (
        f"leak: fill count up to T={T} differs (clean={len(clean.fills)}, "
        f"polluted-prefix={len(full_fills_pre_T)}, seed={seed})")
    for fa, fb in zip(clean.fills, full_fills_pre_T):
        # order_id and fill_id are state-counter-derived; in the
        # polluted run they may differ due to extra quoter calls
        # before T (they shouldn't — same prefix should mean same
        # counter trajectory — but to be robust we compare the
        # economically meaningful fields).
        assert fa.ts_ns == fb.ts_ns
        assert fa.price == fb.price
        assert fa.size == fb.size
        assert fa.side == fb.side


# --------------------------------------------------------------------- #
# G3 — DS-LOB-1H baseline fill count.  Run only when the fixture is
# present; otherwise skip (CI / fresh-clone friendly).
# --------------------------------------------------------------------- #

@pytest.mark.skipif(
    not SNAP_1H.exists() or not TRADE_1H.exists(),
    reason="DS-LOB-1H fixture not present (run scripts/capture_lob.py --minutes 60)",
)
def test_run_sim_ds_lob_1h_baseline_fill_count():
    """Regression-pin the baseline fill count on DS-LOB-1H.  Any
    change to the loop, the stub quoter, or the stub fill model
    that moves this number requires explicitly bumping the pin."""
    stream = load_lob(SNAP_1H, TRADE_1H)
    res = run_sim(stream, stub_quoter_top_of_book, stub_fills_naive)

    # The pin: filled in by docs/verification/item17.md when the
    # gate closes.  The number recorded here is what the bundled
    # 1h fixture produced under (stub_quoter_top_of_book +
    # stub_fills_naive); the test fails loudly if anything regresses.
    expected_fill_count = 65_483       # 35,755 bid + 29,728 ask
    expected_n_events = 535_876
    expected_n_quoter_calls = 35_989
    expected_total_filled_qty = 16.803350

    assert res.n_events_processed == expected_n_events
    assert res.n_quoter_calls == expected_n_quoter_calls
    assert res.fills, "expected at least one fill on DS-LOB-1H"
    assert len(res.fills) == expected_fill_count, (
        f"DS-LOB-1H baseline regression: got {len(res.fills)} fills, "
        f"expected {expected_fill_count}")
    total_qty = sum(f.size for f in res.fills)
    assert abs(total_qty - expected_total_filled_qty) < 1e-6, (
        f"DS-LOB-1H total-qty regression: got {total_qty:.6f}, "
        f"expected {expected_total_filled_qty}")
