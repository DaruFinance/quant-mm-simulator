"""load_lob + reconstruct_book_at unit tests.

Uses the bundled 30-second BTCUSDT/binance fixture (real Binance
data; no synthetic stimuli).  The 1-hour DS-LOB-1H fixture is used
by the real-data verification suite; this file pins the
function-level behaviour at low cost.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from mmsim.ingest.lob import (
    Book, Event, EventStream, SnapshotEvent, TradeEvent,
    load_lob, reconstruct_book_at,
)

HERE = Path(__file__).resolve().parent
SNAP_FIX = HERE / "fixtures" / "lob_btcusdt_30sec_snapshots.parquet"
TRADE_FIX = HERE / "fixtures" / "lob_btcusdt_30sec_trades.parquet"


@pytest.fixture(scope="module")
def stream() -> EventStream:
    return load_lob(SNAP_FIX, TRADE_FIX)


def test_load_lob_returns_chronological_stream(stream):
    assert len(stream) > 100, "fixture should hold > 100 events"
    # ts_ns ascending (with snapshot < trade tiebreaker at ties).
    for prev, curr in zip(stream, stream[1:]):
        assert prev.ts_ns <= curr.ts_ns
        if prev.ts_ns == curr.ts_ns and prev.recv_ns == curr.recv_ns:
            # snapshot (rank 0) must come before trade (rank 1).
            prev_rank = 0 if isinstance(prev, SnapshotEvent) else 1
            curr_rank = 0 if isinstance(curr, SnapshotEvent) else 1
            assert prev_rank <= curr_rank


def test_load_lob_carries_both_event_kinds(stream):
    n_snap = sum(1 for e in stream if isinstance(e, SnapshotEvent))
    n_trade = sum(1 for e in stream if isinstance(e, TradeEvent))
    assert n_snap > 0
    assert n_trade > 0
    assert n_snap + n_trade == len(stream)


def test_load_lob_filters_by_symbol(stream):
    s2 = load_lob(SNAP_FIX, TRADE_FIX, symbol="BTC-USDT")
    assert len(s2) == len(stream)
    s3 = load_lob(SNAP_FIX, TRADE_FIX, symbol="ETH-USDT")
    assert len(s3) == 0


def test_snapshot_event_top_levels_well_formed(stream):
    """Best bid < best ask; sizes > 0; bids descending in price; asks
    ascending."""
    snaps = [e for e in stream if isinstance(e, SnapshotEvent)]
    for s in snaps[:20]:
        assert s.bids[0][0] < s.asks[0][0], (
            f"crossed book at ts={s.ts_ns}")
        # Bids descending, asks ascending.
        bid_pxs = [b[0] for b in s.bids]
        ask_pxs = [a[0] for a in s.asks]
        assert bid_pxs == sorted(bid_pxs, reverse=True)
        assert ask_pxs == sorted(ask_pxs)
        # Positive sizes.
        assert all(b[1] > 0 for b in s.bids)
        assert all(a[1] > 0 for a in s.asks)


def test_reconstruct_book_at_returns_most_recent_snapshot(stream):
    """At a t between two snapshots, reconstruction returns the
    earlier one — never peeks forward."""
    snaps = [e for e in stream if isinstance(e, SnapshotEvent)]
    assert len(snaps) >= 3
    s0, s1, _s2 = snaps[0], snaps[1], snaps[2]
    # A query between s0.ts and s1.ts must return s0.
    t_between = (s0.ts_ns + s1.ts_ns) // 2
    book = reconstruct_book_at(stream, t_between)
    assert book is not None
    assert book.ts_ns == s0.ts_ns
    assert book.bids == s0.bids
    assert book.asks == s0.asks


def test_reconstruct_book_at_returns_none_before_first_snapshot(stream):
    snaps = [e for e in stream if isinstance(e, SnapshotEvent)]
    first_snap_ts = snaps[0].ts_ns
    assert reconstruct_book_at(stream, first_snap_ts - 1) is None


def test_book_mid_matches_top_bid_ask(stream):
    snaps = [e for e in stream if isinstance(e, SnapshotEvent)]
    s = snaps[0]
    book = reconstruct_book_at(stream, s.ts_ns)
    assert book is not None
    assert book.best_bid == s.bids[0][0]
    assert book.best_ask == s.asks[0][0]
    assert book.mid == pytest.approx((s.bids[0][0] + s.asks[0][0]) / 2.0)


# --------------------------------------------------------------------- #
# G2 — leak invariant: pollute events strictly past T, assert
# reconstruct_book_at(stream, T) is unchanged.  Cornerstone
# no-lookahead invariant for the ingestion layer.
# --------------------------------------------------------------------- #

def _pollute_event(e: Event, factor: float = 99.99) -> Event:
    """Replace numeric fields with absurd values so any leak is loud."""
    if isinstance(e, SnapshotEvent):
        garbage_levels = tuple((1.0, factor) for _ in e.bids)
        return dataclasses.replace(e, bids=garbage_levels, asks=garbage_levels)
    if isinstance(e, TradeEvent):
        return dataclasses.replace(e, price=factor, size=factor, side=0)
    raise TypeError(f"unknown event kind: {type(e).__name__}")


@pytest.mark.parametrize("seed", [0, 7, 19, 42, 123])
def test_reconstruct_book_at_no_lookahead_under_pollution(stream, seed):
    """For 5 random query times T, polluting every event with
    ts_ns > T must not change reconstruct_book_at(stream, T)."""
    rng = np.random.default_rng(seed)
    snaps = [e for e in stream if isinstance(e, SnapshotEvent)]
    # Query T strictly between two snapshot timestamps.
    pivot = int(rng.integers(2, len(snaps) - 2))
    T = (snaps[pivot].ts_ns + snaps[pivot + 1].ts_ns) // 2

    clean = reconstruct_book_at(stream, T)

    polluted: EventStream = [
        _pollute_event(e) if e.ts_ns > T else e
        for e in stream
    ]
    poisoned = reconstruct_book_at(polluted, T)

    assert clean == poisoned, (
        f"leak: reconstruct_book_at({T}) changed after polluting "
        f"events strictly past T (seed={seed})")


def test_reconstruct_book_at_at_exactly_T_uses_event_at_T(stream):
    """An event at exactly T is consumed (the contract is <= T,
    not < T)."""
    snaps = [e for e in stream if isinstance(e, SnapshotEvent)]
    s = snaps[5]
    book = reconstruct_book_at(stream, s.ts_ns)
    assert book is not None
    assert book.ts_ns == s.ts_ns
