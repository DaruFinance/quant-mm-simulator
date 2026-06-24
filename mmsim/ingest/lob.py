"""L2 book + trade-tape ingestion.

Loads L2 snapshots and the trade tape into a unified, chronologically
sorted ``EventStream`` that the downstream sim (loop, queue, fills)
consumes.

Design note:
    The design calls for "snapshots + deltas + tape".  The
    upstream ``crypto-data-aggregator`` (the canonical source for the
    DS-LOB-1H dataset) only ships *snapshots* — it does not
    materialize an L2 deltas table.  Each Binance partial-book
    snapshot is a complete top-N replacement of the L2 state, so we
    treat consecutive snapshots as the delta carrier.  Synthesizing
    deltas by diffing snapshots is therefore equivalent in
    reconstruction power; the deviation is naming, not capability.

Public surface (frozen for downstream consumers):
    - SnapshotEvent / TradeEvent (frozen dataclasses)
    - Event (union alias)
    - EventStream (list[Event])
    - load_lob(snapshots_path, trades_path, ...) -> EventStream
    - reconstruct_book_at(stream, t_ns) -> Book
    - Book (frozen dataclass)

Causality contract:
    Every consumer that asks for sim state at time ``t`` must observe
    only events with ``ts_ns <= t``.  ``reconstruct_book_at`` enforces
    this directly; the sim loop layers the same guarantee over
    the queue and fill modules.  The leak invariant test in
    ``tests/test_ingest_lob_leak.py`` pollutes the stream past T and
    asserts ``reconstruct_book_at(polluted, T) ==
    reconstruct_book_at(clean, T)`` byte-for-byte.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import pyarrow.parquet as pq


# --------------------------------------------------------------------- #
# Event types — frozen dataclasses, hash-stable so leak tests can use
# tuple-of-events equality without surprises.
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class SnapshotEvent:
    """An L2 partial-book snapshot at a single instant.  ``bids`` and
    ``asks`` are best-first; each level is ``(price, size)`` in base /
    quote units exactly as the venue published them."""
    ts_ns: int
    recv_ns: int
    symbol: str
    venue: str
    depth: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class TradeEvent:
    """A single executed trade from the tape.  ``side`` is the
    aggressor side (+1 buy, -1 sell, 0 unknown) per the
    ``crypto-data-aggregator`` data dictionary."""
    ts_ns: int
    recv_ns: int
    symbol: str
    venue: str
    price: float
    size: float
    side: int


Event = Union[SnapshotEvent, TradeEvent]
EventStream = List[Event]


@dataclass(frozen=True)
class Book:
    """Reconstructed L2 state at a logical instant.  ``bids`` /
    ``asks`` carry the same ``(price, size)`` levels the latest
    snapshot held; ``ts_ns`` is the snapshot's exchange timestamp."""
    ts_ns: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        return (b + a) / 2.0 if (b is not None and a is not None) else None


# --------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------- #

def _read_snapshots(path: Path) -> List[SnapshotEvent]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    out: List[SnapshotEvent] = []
    for r in rows:
        out.append(SnapshotEvent(
            ts_ns=int(r["ts_ns"]),
            recv_ns=int(r["recv_ns"]),
            symbol=str(r["symbol"]),
            venue=str(r["venue"]),
            depth=int(r["depth"]),
            bids=tuple((float(b["px"]), float(b["sz"])) for b in r["bids"]),
            asks=tuple((float(a["px"]), float(a["sz"])) for a in r["asks"]),
        ))
    return out


def _read_trades(path: Path) -> List[TradeEvent]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    out: List[TradeEvent] = []
    for r in rows:
        out.append(TradeEvent(
            ts_ns=int(r["ts_ns"]),
            recv_ns=int(r["recv_ns"]),
            symbol=str(r["symbol"]),
            venue=str(r["venue"]),
            price=float(r["price"]),
            size=float(r["size"]),
            side=int(r["side"]),
        ))
    return out


def load_lob(
    snapshots_path: Union[str, Path],
    trades_path: Union[str, Path],
    *,
    symbol: Optional[str] = None,
    venue: Optional[str] = None,
) -> EventStream:
    """Read parquet-backed L2 snapshots and trade tape, merge into a
    single ``EventStream`` ordered by ``(ts_ns, recv_ns, kind)``.

    The kind tiebreaker (``snapshot`` < ``trade`` when ts and recv tie)
    matches the order the venue would have processed them: a snapshot
    reflects book state immediately *before* a same-instant trade,
    not after.  In practice ts_ns ties are extraordinarily rare on
    nanosecond exchange timestamps; the rule is documented for
    reproducibility.

    ``symbol`` / ``venue`` filters drop rows that do not match (None
    keeps all rows as-loaded).  Useful when a future capture combines
    multiple symbols or venues into one parquet.
    """
    snapshots = _read_snapshots(Path(snapshots_path))
    trades = _read_trades(Path(trades_path))
    if symbol is not None:
        snapshots = [s for s in snapshots if s.symbol == symbol]
        trades = [t for t in trades if t.symbol == symbol]
    if venue is not None:
        snapshots = [s for s in snapshots if s.venue == venue]
        trades = [t for t in trades if t.venue == venue]

    # Merge with a stable sort.  Kind tiebreaker: snapshot (0) before
    # trade (1) at identical ts_ns + recv_ns.
    def _key(e: Event) -> tuple[int, int, int]:
        kind_rank = 0 if isinstance(e, SnapshotEvent) else 1
        return (e.ts_ns, e.recv_ns, kind_rank)

    merged: EventStream = sorted(
        [*snapshots, *trades], key=_key
    )
    return merged


# --------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------- #

def reconstruct_book_at(stream: EventStream, t_ns: int) -> Optional[Book]:
    """Most-recent snapshot at-or-before ``t_ns``.  Returns None if
    the stream contains no snapshot at-or-before ``t_ns`` (i.e. the
    book is not yet warmed up).

    Trades do not enter book reconstruction — in this simplified
    model, snapshots are the sole carrier of L2 state.  Trades
    influence queue position and fills; they do not
    rewrite the book here.

    Causality: never consults events with ``ts_ns > t_ns``.  Polluting
    those events does not change the return value (verified by
    ``test_ingest_lob_leak.py``).
    """
    last: Optional[SnapshotEvent] = None
    for ev in stream:
        if ev.ts_ns > t_ns:
            break
        if isinstance(ev, SnapshotEvent):
            last = ev
    if last is None:
        return None
    return Book(ts_ns=last.ts_ns, bids=last.bids, asks=last.asks)


__all__ = [
    "SnapshotEvent", "TradeEvent", "Event", "EventStream",
    "Book",
    "load_lob", "reconstruct_book_at",
]
