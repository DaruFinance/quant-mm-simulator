"""Quoter Protocol contract tests + parity vs closure path."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest

from mmsim.ingest.lob import Book, SnapshotEvent, TradeEvent, load_lob
from mmsim.quoter import (
    BracketQuoter, ConstantQuoter, Quoter, TopOfBookQuoter,
)
from mmsim.sim.fills import QueueAwareFillModel, TakerRequest
from mmsim.sim.loop import QuoteRequest, run_sim


HERE = Path(__file__).resolve().parent
SNAP_30S = HERE / "fixtures" / "lob_btcusdt_30sec_snapshots.parquet"
TRADE_30S = HERE / "fixtures" / "lob_btcusdt_30sec_trades.parquet"
SNAP_1H = HERE / "fixtures" / "lob_btcusdt_60min_snapshots.parquet"
TRADE_1H = HERE / "fixtures" / "lob_btcusdt_60min_trades.parquet"


# --------------------------------------------------------------------- #
# Contract test: every reference quoter implements the Protocol.
# --------------------------------------------------------------------- #

def test_constant_quoter_implements_protocol():
    q = ConstantQuoter(bid_price=100.0, ask_price=101.0)
    assert isinstance(q, Quoter)


def test_top_of_book_quoter_implements_protocol():
    assert isinstance(TopOfBookQuoter(), Quoter)


def test_bracket_quoter_implements_protocol():
    assert isinstance(BracketQuoter(taker_every=500), Quoter)


# --------------------------------------------------------------------- #
# Direct invocation: Quoter.quote(book, inv, t) returns expected shapes.
# --------------------------------------------------------------------- #

def test_constant_quoter_returns_two_makers_regardless_of_book():
    q = ConstantQuoter(bid_price=100.0, ask_price=101.0)
    # No book at all (warmup) — still posts.
    out_no_book = q.quote(None, 0.0, 0)
    assert len(out_no_book) == 2
    # Book present — same output (constant quoter ignores book).
    book = Book(ts_ns=0, bids=((50.0, 1.0),), asks=((150.0, 1.0),))
    out_with_book = q.quote(book, 0.0, 1)
    assert out_with_book == out_no_book
    # Both items are QuoteRequest, sides ±1, prices match constants.
    assert all(isinstance(o, QuoteRequest) for o in out_with_book)
    assert sorted(o.side for o in out_with_book) == [-1, +1]


def test_top_of_book_quoter_returns_empty_before_book_warmup():
    q = TopOfBookQuoter()
    assert q.quote(None, 0.0, 0) == []


def test_top_of_book_quoter_joins_tob():
    q = TopOfBookQuoter(size=0.001)
    book = Book(ts_ns=0, bids=((100.0, 5.0),), asks=((101.0, 3.0),))
    out = q.quote(book, 0.0, 0)
    bid = next(o for o in out if o.side == +1)
    ask = next(o for o in out if o.side == -1)
    assert bid.price == 100.0
    assert ask.price == 101.0


def test_bracket_quoter_taker_fires_on_schedule():
    q = BracketQuoter(taker_every=3)
    book = Book(ts_ns=0, bids=((100.0, 5.0),), asks=((101.0, 3.0),))
    seen_takers = 0
    for i in range(10):
        out = q.quote(book, 0.0, i)
        if any(isinstance(o, TakerRequest) for o in out):
            seen_takers += 1
    # 10 calls, taker_every=3 => fires at calls 3, 6, 9 = 3 takers
    assert seen_takers == 3


# --------------------------------------------------------------------- #
# Loop integration parity: Protocol-wrapped BracketQuoter produces
# fills bit-identical to the closure-based BracketQuoter.
# This is the load-bearing parity verification — proves the new
# integration path doesn't change ANY economic outcome.
# --------------------------------------------------------------------- #

@pytest.mark.skipif(
    not SNAP_30S.exists() or not TRADE_30S.exists(),
    reason="30-s smoke fixture not present",
)
def test_protocol_quoter_produces_same_fills_as_legacy_callable():
    import sys
    sys.path.insert(0, str(HERE))
    from test_sim_fills import BracketQuoter as LegacyBracketQuoter

    stream = load_lob(SNAP_30S, TRADE_30S)

    # Protocol path: BracketQuoter from mmsim.quoter (Protocol impl).
    res_proto = run_sim(
        stream, BracketQuoter(taker_every=50), QueueAwareFillModel())
    # Legacy callable path: BracketQuoter from tests/test_sim_fills.py.
    res_legacy = run_sim(
        stream, LegacyBracketQuoter(taker_every=50), QueueAwareFillModel())

    assert len(res_proto.fills) == len(res_legacy.fills)
    assert res_proto.n_maker_fills == res_legacy.n_maker_fills
    assert res_proto.n_taker_fills == res_legacy.n_taker_fills
    for fa, fb in zip(res_proto.fills, res_legacy.fills):
        # fill_id may differ if one path emits state events the other
        # doesn't; the economically meaningful fields must match.
        assert fa.ts_ns == fb.ts_ns
        assert fa.price == fb.price
        assert fa.size == fb.size
        assert fa.side == fb.side
        assert fa.is_maker == fb.is_maker


# --------------------------------------------------------------------- #
# DS-LOB-1H baseline: Protocol-wrapped BracketQuoter run.  Same numbers
# as the closure-based baseline pin.  Proves the Protocol introduction is
# economically inert.
# --------------------------------------------------------------------- #

@pytest.mark.skipif(
    not SNAP_1H.exists() or not TRADE_1H.exists(),
    reason="DS-LOB-1H fixture not present",
)
def test_protocol_bracket_quoter_ds_lob_1h_baseline():
    stream = load_lob(SNAP_1H, TRADE_1H)
    res = run_sim(stream, BracketQuoter(taker_every=500),
                    QueueAwareFillModel())
    # Same numbers as the closure-based baseline pin.
    assert res.n_maker_fills == 1_823
    assert res.n_taker_fills == 71
    assert len(res.fills) == 1_894


# --------------------------------------------------------------------- #
# 5 quoter calls — direct invocation, dump inputs/outputs to
# reconcile the quoter's behaviour.
# --------------------------------------------------------------------- #

def test_quoter_5_calls_reconcile():
    """Drive a constant quoter directly through the Protocol on 5
    distinct (book, inv, t) inputs; confirm output matches the
    declared bid/ask both for shape and for content."""
    q = ConstantQuoter(bid_price=100.0, ask_price=101.0, size=0.001)
    cases = [
        (None, 0.0, 0),
        (Book(0, ((50.0, 5.0),), ((150.0, 5.0),)), 0.0, 1),
        (Book(1, ((75.0, 1.5),), ((125.0, 1.5),)), 0.5, 2),
        (Book(2, ((90.0, 0.001),), ((110.0, 0.001),)), -0.3, 3),
        (Book(3, ((100.0, 10.0),), ((100.5, 10.0),)), 1.0, 4),
    ]
    for book, inv, t in cases:
        out = q.quote(book, inv, t)
        assert len(out) == 2
        bids = [o for o in out if o.side == +1]
        asks = [o for o in out if o.side == -1]
        assert len(bids) == 1 and len(asks) == 1
        assert bids[0].price == 100.0
        assert asks[0].price == 101.0
        assert bids[0].size == 0.001
        assert asks[0].size == 0.001
