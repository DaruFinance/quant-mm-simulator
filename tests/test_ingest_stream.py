"""Streaming event source + streaming engine parity (bounded-RAM path).

Mirrors the parity script's guarantees as fast unit tests on the small
fixtures: the streamed event sequence equals load_lob's, the streamed engine
is bit-identical to run_sim, the streamed mid timeline equals
build_mid_timeline, and the top-k throttle is a trade-preserving subsequence
with no consecutive-equal kept books.
"""
from __future__ import annotations

import numpy as np
import pytest

pq = pytest.importorskip("pyarrow.parquet")

from mmsim.ingest.lob import load_lob, SnapshotEvent, TradeEvent
from mmsim.ingest.stream import (
    iter_events_streaming, iter_mid_timeline_streaming,
)
from mmsim.ledger.writer import build_mid_timeline
from mmsim.sim.loop import run_sim, QuoteRequest
from mmsim.sim.loop_stream import run_sim_streaming
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.quoter.builtin import TopOfBookQuoter


def _write_fixture(tmp_path):
    """Tiny two-sided book + trades parquet matching the lob reader schema."""
    import pyarrow as pa
    lvl = pa.list_(pa.struct([("px", pa.float64()), ("sz", pa.float64())]))
    schema = pa.schema([
        ("ts_ns", pa.int64()), ("recv_ns", pa.int64()),
        ("symbol", pa.string()), ("venue", pa.string()),
        ("depth", pa.int64()), ("bids", lvl), ("asks", lvl)])
    snaps = []
    # 6 snapshots; two of them have an unchanged top-1 book (throttle target)
    books = [
        ([(100.0, 5.0), (99.0, 3.0)], [(101.0, 4.0), (102.0, 2.0)]),
        ([(100.0, 5.0), (99.0, 3.0)], [(101.0, 4.0), (102.0, 2.0)]),  # dup top
        ([(100.5, 6.0), (99.5, 3.0)], [(101.5, 4.0), (102.5, 2.0)]),
        ([(100.5, 6.0), (99.5, 3.0)], [(101.5, 4.0), (102.5, 2.0)]),  # dup top
        ([(100.5, 6.0)], []),                                          # one-sided
        ([(101.0, 7.0)], [(102.0, 5.0)]),
    ]
    for i, (b, a) in enumerate(books):
        snaps.append({
            "ts_ns": 1000 + i * 10, "recv_ns": 1000 + i * 10,
            "symbol": "X", "venue": "CME", "depth": max(len(b), len(a)),
            "bids": [{"px": p, "sz": s} for p, s in b],
            "asks": [{"px": p, "sz": s} for p, s in a]})
    snap_path = tmp_path / "snap.parquet"
    pq.write_table(pa.Table.from_pylist(snaps, schema=schema), snap_path)
    trades = [
        {"ts_ns": 1015, "recv_ns": 1015, "symbol": "X", "venue": "CME",
         "price": 101.0, "size": 1.0, "side": 1},
        {"ts_ns": 1045, "recv_ns": 1045, "symbol": "X", "venue": "CME",
         "price": 100.5, "size": 2.0, "side": -1}]
    trade_path = tmp_path / "trades.parquet"
    pq.write_table(pa.Table.from_pylist(trades), trade_path)
    return str(snap_path), str(trade_path)


def _etuple(e):
    if isinstance(e, SnapshotEvent):
        return ("S", e.ts_ns, e.recv_ns, e.symbol, e.venue, e.depth,
                tuple(e.bids), tuple(e.asks))
    return ("T", e.ts_ns, e.recv_ns, e.symbol, e.venue, e.price, e.size, e.side)


def test_event_sequence_bit_identical(tmp_path):
    snap, trades = _write_fixture(tmp_path)
    ref = [_etuple(e) for e in load_lob(snap, trades)]
    got = [_etuple(e) for e in iter_events_streaming(snap, trades)]
    assert ref == got


def test_streamed_engine_bit_identical(tmp_path):
    snap, trades = _write_fixture(tmp_path)
    ref_stream = load_lob(snap, trades)
    r_ref = run_sim(ref_stream, TopOfBookQuoter(size=1.0), QueueAwareFillModel())
    r_str = run_sim_streaming(
        iter_events_streaming(snap, trades), TopOfBookQuoter(size=1.0),
        QueueAwareFillModel())
    assert len(r_ref.fills) == len(r_str.fills)
    for a, b in zip(r_ref.fills, r_str.fills):
        assert (a.fill_id, a.order_id, a.ts_ns, a.price, a.size, a.side,
                a.is_maker) == (b.fill_id, b.order_id, b.ts_ns, b.price,
                                b.size, b.side, b.is_maker)
    assert r_ref.n_events_processed == r_str.n_events_processed
    assert r_ref.n_snapshot_events == r_str.n_snapshot_events
    assert r_ref.n_trade_events == r_str.n_trade_events
    assert r_ref.n_maker_fills == r_str.n_maker_fills


def test_mid_timeline_bit_identical(tmp_path):
    snap, trades = _write_fixture(tmp_path)
    ts_ref, mid_ref = build_mid_timeline(load_lob(snap, trades))
    pairs = list(iter_mid_timeline_streaming(snap))
    ts_str = np.array([p[0] for p in pairs], dtype=np.int64)
    mid_str = np.array([p[1] for p in pairs], dtype=np.float64)
    assert np.array_equal(ts_ref, ts_str)
    assert np.array_equal(mid_ref, mid_str, equal_nan=True)


def test_throttle_preserves_trades_and_drops_unchanged(tmp_path):
    snap, trades = _write_fixture(tmp_path)
    full = list(iter_events_streaming(snap, trades))
    thr = list(iter_events_streaming(snap, trades, throttle_k=1))
    # all trades preserved
    ft = [e for e in full if isinstance(e, TradeEvent)]
    tt = [e for e in thr if isinstance(e, TradeEvent)]
    assert [_etuple(e) for e in ft] == [_etuple(e) for e in tt]
    # fewer or equal snapshots, and no two consecutive kept snaps share top-1
    snaps = [e for e in thr if isinstance(e, SnapshotEvent)]
    assert len(snaps) < len([e for e in full if isinstance(e, SnapshotEvent)])
    last = None
    for s in snaps:
        sig = (s.bids[:1], s.asks[:1])
        assert sig != last
        last = sig


def test_symbol_filter(tmp_path):
    snap, trades = _write_fixture(tmp_path)
    got = list(iter_events_streaming(snap, trades, symbol="NOPE"))
    assert got == []
    allmatch = list(iter_events_streaming(snap, trades, symbol="X"))
    assert len(allmatch) == 8  # 6 snaps + 2 trades
