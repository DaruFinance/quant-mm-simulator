"""Bounded-RAM streaming event source for the depth-10 engine.

The validated path (``mmsim.ingest.lob.load_lob`` and the futures ingester's
``snapshots_to_eventstream``) materialise the ENTIRE contract-day as a Python
list of frozen ``SnapshotEvent`` / ``TradeEvent`` objects, then sort it.  For a
real ES day (~9.5M depth snapshots, each carrying two 10-level tuple-of-tuple
books) that list is tens of GB of Python objects -> the 29 GB OOM.

This module yields the **bit-identical event sequence** ``load_lob`` would
produce, but lazily: it reads the snapshot and trade parquet files
row-group-by-row-group and merges the two (each already ts-sorted, as written
by ``fut_depth.write_parquet``) with a two-pointer merge.  At most a couple of
row-groups are decoded at a time, so peak RAM is bounded regardless of day
length.

ORDER CONTRACT (must match ``load_lob`` exactly):
    sort key = (ts_ns, recv_ns, kind_rank) with kind_rank = 0 for snapshot,
    1 for trade -> a snapshot sorts before a trade at an identical timestamp
    (book state is the pre-trade state).  Both source files are individually
    sorted ascending by (ts_ns, recv_ns) at write time, so a stable two-pointer
    merge that breaks ties in favour of the snapshot reproduces the global order
    byte-for-byte.  Bit-identical equivalence to ``load_lob`` is asserted in
    ``scripts/verify_stream_parity.py`` on the ES smoke slice.

The engine (``run_sim``) only ever iterates the stream once, so a one-shot
generator is sufficient.  The single place ``run_sim`` needed ``len(events)``
is handled by ``run_sim_streaming`` (see ``mmsim.sim.loop_stream``), which
counts events as it consumes them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Union

import pyarrow.parquet as pq

from mmsim.ingest.lob import SnapshotEvent, TradeEvent, Event


def _snap_from_row(r: dict) -> SnapshotEvent:
    return SnapshotEvent(
        ts_ns=int(r["ts_ns"]),
        recv_ns=int(r["recv_ns"]),
        symbol=str(r["symbol"]),
        venue=str(r["venue"]),
        depth=int(r["depth"]),
        bids=tuple((float(b["px"]), float(b["sz"])) for b in r["bids"]),
        asks=tuple((float(a["px"]), float(a["sz"])) for a in r["asks"]),
    )


def _trade_from_row(r: dict) -> TradeEvent:
    return TradeEvent(
        ts_ns=int(r["ts_ns"]),
        recv_ns=int(r["recv_ns"]),
        symbol=str(r["symbol"]),
        venue=str(r["venue"]),
        price=float(r["price"]),
        size=float(r["size"]),
        side=int(r["side"]),
    )


def _iter_parquet_rows(path: Union[str, Path], batch_size: int) -> Iterator[dict]:
    """Yield one python dict per row, decoding a row-group/batch at a time.

    ``ParquetFile.iter_batches`` already streams; ``to_pylist`` on each batch
    bounds the materialised window to ``batch_size`` rows.
    """
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row


def _topk_key(ev: SnapshotEvent, k: int):
    """Hashable top-k book signature for change detection."""
    return (ev.bids[:k], ev.asks[:k])


def iter_events_streaming(
    snap_path: Union[str, Path],
    trade_path: Union[str, Path],
    *,
    symbol: Optional[str] = None,
    venue: Optional[str] = None,
    batch_size: int = 100_000,
    throttle_k: Optional[int] = None,
) -> Iterator[Event]:
    """Lazily yield the merged, time-sorted event stream from parquet.

    With ``throttle_k=None`` (default) the event sequence is BIT-IDENTICAL to
    ``load_lob(snap_path, trade_path)`` but never materialises the whole day.
    ``symbol`` / ``venue`` filters drop non-matching rows (same semantics as
    ``load_lob``).  Tie-break: snapshot before trade at identical (ts_ns,
    recv_ns).

    With ``throttle_k=K`` set, a snapshot whose top-K book (prices + sizes,
    both sides) is unchanged vs the last KEPT snapshot is dropped -- the
    streaming analog of ``fut_depth.throttle_top_k_changed`` (the documented
    information-preserving MM reduction: identical consecutive books carry no
    new state).  ALL trades always pass through.  The first snapshot is always
    kept.  This is the canonical full-scale input; the un-throttled mode is the
    parity reference.
    """
    snap_rows = _iter_parquet_rows(snap_path, batch_size)
    trade_rows = _iter_parquet_rows(trade_path, batch_size)

    def _next_snap():
        for r in snap_rows:
            if symbol is not None and str(r["symbol"]) != symbol:
                continue
            if venue is not None and str(r["venue"]) != venue:
                continue
            return r
        return None

    def _next_trade():
        for r in trade_rows:
            if symbol is not None and str(r["symbol"]) != symbol:
                continue
            if venue is not None and str(r["venue"]) != venue:
                continue
            return r
        return None

    _last_kept = [None]  # last kept top-k signature (mutable closure cell)

    def _emit_snap(row):
        ev = _snap_from_row(row)
        if throttle_k is not None:
            sig = _topk_key(ev, throttle_k)
            if _last_kept[0] is not None and sig == _last_kept[0]:
                return None  # unchanged top-k book -> drop
            _last_kept[0] = sig
        return ev

    s = _next_snap()
    t = _next_trade()
    while s is not None or t is not None:
        if t is None:
            ev = _emit_snap(s); s = _next_snap()
            if ev is not None:
                yield ev
        elif s is None:
            yield _trade_from_row(t)
            t = _next_trade()
        else:
            sk = (int(s["ts_ns"]), int(s["recv_ns"]), 0)
            tk = (int(t["ts_ns"]), int(t["recv_ns"]), 1)
            if sk <= tk:
                ev = _emit_snap(s); s = _next_snap()
                if ev is not None:
                    yield ev
            else:
                yield _trade_from_row(t)
                t = _next_trade()


def iter_mid_timeline_streaming(
    snap_path: Union[str, Path],
    *,
    symbol: Optional[str] = None,
    venue: Optional[str] = None,
    batch_size: int = 200_000,
):
    """Stream the (ts, mid) snapshot timeline straight from the snap parquet
    without building SnapshotEvent objects -- a two-pass driver builds the mid
    arrays for markout once, cheaply, in bounded RAM.

    Yields (ts_ns:int, mid:float) for every two-sided snapshot, in file order
    (== ts order).  Mirrors ``build_mid_timeline`` exactly (best bid/ask mean,
    snapshots without both sides skipped).
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import numpy as np
    pf = pq.ParquetFile(str(snap_path))
    cols = ["ts_ns", "bids", "asks"]
    if symbol is not None:
        cols.append("symbol")
    if venue is not None:
        cols.append("venue")
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        # vectorised top-of-book px extraction.  Filter to rows with BOTH sides
        # non-empty FIRST (so list_element(0) is always in bounds — pyarrow
        # evaluates element extraction on every row before masking), then take
        # level-0 px per side.  Avoids materialising the full 10-level nested
        # book as Python dicts (that path peaked ~4 GB on a 0.5M-snap day).
        tbl = pa.table({"ts_ns": batch.column("ts_ns"),
                        "bids": batch.column("bids"),
                        "asks": batch.column("asks"),
                        **({"symbol": batch.column("symbol")} if symbol is not None else {}),
                        **({"venue": batch.column("venue")} if venue is not None else {})})
        n_b = pc.list_value_length(tbl["bids"])
        n_a = pc.list_value_length(tbl["asks"])
        keep_mask = pc.and_(pc.greater(n_b, 0), pc.greater(n_a, 0))
        if symbol is not None:
            keep_mask = pc.and_(keep_mask, pc.equal(tbl["symbol"], symbol))
        if venue is not None:
            keep_mask = pc.and_(keep_mask, pc.equal(tbl["venue"], venue))
        tbl = tbl.filter(keep_mask)
        if tbl.num_rows == 0:
            continue
        bid0 = pc.struct_field(pc.list_element(tbl["bids"], 0), "px")
        ask0 = pc.struct_field(pc.list_element(tbl["asks"], 0), "px")
        mid_np = pc.divide(pc.add(bid0, ask0), 2.0).to_numpy(zero_copy_only=False)
        ts_np = tbl["ts_ns"].to_numpy(zero_copy_only=False)
        for i in range(ts_np.shape[0]):
            yield int(ts_np[i]), float(mid_np[i])


__all__ = ["iter_events_streaming", "iter_mid_timeline_streaming"]
