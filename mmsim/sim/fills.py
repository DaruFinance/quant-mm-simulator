"""Maker/taker fill model.

A queue-aware maker fill model with taker-fill support for
quoters that want to cross the spread.

## Maker fills (queue-aware)

For each resting order we keep a ``QueueTracker``.  When
a ``TradeEvent`` arrives at the order's price level:

1. The tracker consumes the front-of-queue: ``queue_pos -= min(
   trade.size, queue_pos)``.  If the trade was smaller than the
   queue ahead of us, **no fill** for us.
2. If the trade is **larger than ``queue_pos_before``**, the
   excess (``trade.size - queue_pos_before``) spills onto the front
   of the remaining queue — which is us, by the model.  We get a
   maker fill of ``min(spillover, our_order.size)``.

This is the "we are the front-most maker once queue_pos hits zero"
assumption — the same simplification used by the queue tracker.
It's reasonable in practice since real maker queues at TOB
are typically deep enough that the model only matters once we've
already aged to the front.

## Taker fills (immediate)

If the quoter returns a ``TakerRequest``, the loop fires it at the
current best bid (sell taker) or best ask (buy taker) immediately.
The `QueueAwareFillModel.fill_taker` walks the visible book if the
request size exceeds the top level, emitting one ``Fill`` record
per consumed level.  ``is_maker=False`` on every taker fill so the
downstream maker/taker ratio is accurate.

## Loop integration

`run_sim` accepts ``fills`` as either:
- a stateless callable (legacy) — wrapped
  internally by `StatelessFillsAdapter`;
- a `FillModelProtocol` object (new; queue-aware). The loop calls
  the protocol's lifecycle hooks (``on_order_placed``,
  ``on_orders_removed``, ``on_snapshot``, ``on_trade``,
  ``fill_taker``) at the right moments.

Backwards compatibility: existing callers passing a callable see no
behavior change.  The stub-fills tests still pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from mmsim.ingest.lob import Book, Event, SnapshotEvent, TradeEvent
from mmsim.sim.loop import Fill, Order, QuoteRequest
from mmsim.sim.queue import QueueTracker


# --------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class TakerRequest:
    """A request to immediately cross the spread (IOC/market order).

    The loop fires it against the visible book at the current event's
    timestamp; no resting order is created.  ``limit_px`` caps the
    walk: if a level's price is worse than ``limit_px`` (less than
    for sell, greater than for buy), the walk stops mid-fill (any
    remaining size is treated as cancelled).
    """
    side: int                       # +1 buy, -1 sell
    size: float                     # base-asset units to consume
    limit_px: Optional[float] = None


@runtime_checkable
class FillModelProtocol(Protocol):
    """Stateful fill model.  Implementations must:
      - on_order_placed(order, book): called once when a new resting
        order is added to the active set; passed the most-recent
        Book at placement time.
      - on_orders_removed(order_ids): called when active orders are
        cancelled or fully filled (the loop's cancel-on-replace
        semantics).
      - on_snapshot(snap): called for each SnapshotEvent BEFORE
        on_order_placed of any new orders posted in response to that
        snapshot.  Lets the model update its trackers' state.
      - on_trade(trade, active_orders): called for each TradeEvent;
        returns a list of (order_id, fill_size) hits to apply.
      - fill_taker(req, book, t_ns): resolve a taker request against
        the visible book; returns a list of (price, size) fill rows.
        Called at the same instant the request was emitted.
    """

    def on_order_placed(self, order: Order, book: Book) -> None: ...
    def on_orders_removed(self, order_ids: List[int]) -> None: ...
    def on_snapshot(self, snap: SnapshotEvent) -> None: ...
    def on_trade(
        self, trade: TradeEvent, active_orders: List[Order],
    ) -> List[Tuple[int, float]]: ...
    def fill_taker(
        self, req: TakerRequest, book: Book, t_ns: int,
    ) -> List[Tuple[float, float]]: ...


# --------------------------------------------------------------------- #
# Adapter for legacy stateless callables (back-compat)
# --------------------------------------------------------------------- #

class StatelessFillsAdapter:
    """Wraps a stateless ``(active_orders, trade) -> [(oid, size), ...]``
    callable as a ``FillModelProtocol``.  All non-trade hooks are
    no-ops; ``on_trade`` delegates to the wrapped callable; takers
    aren't supported (returns empty list — a stateless adapter
    cannot meaningfully walk the book without a state hook)."""

    def __init__(self, fn: Callable[[List[Order], TradeEvent], List[Tuple[int, float]]]):
        self._fn = fn

    def on_order_placed(self, order: Order, book: Book) -> None:
        pass

    def on_orders_removed(self, order_ids: List[int]) -> None:
        pass

    def on_snapshot(self, snap: SnapshotEvent) -> None:
        pass

    def on_trade(
        self, trade: TradeEvent, active_orders: List[Order],
    ) -> List[Tuple[int, float]]:
        return self._fn(active_orders, trade)

    def fill_taker(
        self, req: TakerRequest, book: Book, t_ns: int,
    ) -> List[Tuple[float, float]]:
        return []


# --------------------------------------------------------------------- #
# Queue-aware maker fill model
# --------------------------------------------------------------------- #

# Floating-point comparison tolerance for "same price level".  Same
# value the QueueTracker uses; kept local for readability.
_PX_EPSILON = 1e-9


def _trade_at_our_level(trade: TradeEvent, side: int, price: float) -> bool:
    """Mirror of mmsim.sim.queue's helper."""
    if trade.side == 0:
        return False
    if side == +1:
        return trade.side == -1 and abs(trade.price - price) <= _PX_EPSILON
    return trade.side == +1 and abs(trade.price - price) <= _PX_EPSILON


class QueueAwareFillModel:
    """The canonical maker/taker fill model.

    Maintains one ``QueueTracker`` per active resting order.  On each
    ``TradeEvent``, the trade volume first consumes the queue ahead
    of each potentially-impacted resting order; any spillover onto
    a queue with ``queue_pos == 0`` becomes a maker fill for that
    order.

    Taker fills walk the visible book greedily from the BBO.
    """

    def __init__(self):
        self._trackers: Dict[int, QueueTracker] = {}
        self._last_book: Optional[Book] = None

    def on_order_placed(self, order: Order, book: Book) -> None:
        self._trackers[order.order_id] = QueueTracker(order, book)
        self._last_book = book

    def on_orders_removed(self, order_ids: List[int]) -> None:
        for oid in order_ids:
            self._trackers.pop(oid, None)

    def on_snapshot(self, snap: SnapshotEvent) -> None:
        # Update the cached book for taker walks.
        self._last_book = Book(ts_ns=snap.ts_ns, bids=snap.bids, asks=snap.asks)
        # Push the snapshot to each active tracker so it can do
        # cancel attribution.
        for tr in self._trackers.values():
            tr.observe(snap)

    def on_trade(
        self, trade: TradeEvent, active_orders: List[Order],
    ) -> List[Tuple[int, float]]:
        hits: List[Tuple[int, float]] = []
        for o in active_orders:
            tr = self._trackers.get(o.order_id)
            if tr is None or tr.frozen:
                continue
            if not _trade_at_our_level(trade, o.side, o.price):
                # Tracker doesn't need to observe trades at other
                # levels; only ours.  But we DO push the trade so
                # snapshot reconciliation has accurate accounting
                # — actually no, _on_trade in the tracker no-ops
                # for non-matching trades anyway.  Keep observing
                # for symmetry with the snapshot path.
                tr.observe(trade)
                continue
            queue_pos_before = tr.queue_pos
            # Tracker consumes up to its queue_pos from this trade.
            tr.observe(trade)
            spillover = trade.size - queue_pos_before
            if spillover > 0.0:
                fill_size = min(spillover, o.size)
                if fill_size > 0.0:
                    hits.append((o.order_id, fill_size))
        return hits

    def fill_taker(
        self, req: TakerRequest, book: Book, t_ns: int,
    ) -> List[Tuple[float, float]]:
        """Walk the book's resting liquidity, consuming `req.size`
        from the BBO outward.  Returns ``[(price, size), ...]``
        fill rows; sum of sizes ≤ req.size (less if limit_px stops
        the walk early or the visible book is exhausted)."""
        if req.size <= 0.0:
            return []
        # Buy taker hits the asks; sell taker hits the bids.
        levels = book.asks if req.side == +1 else book.bids
        remaining = req.size
        rows: List[Tuple[float, float]] = []
        for px, sz in levels:
            if remaining <= 0.0:
                break
            if req.limit_px is not None:
                if req.side == +1 and px > req.limit_px:
                    break
                if req.side == -1 and px < req.limit_px:
                    break
            consume = min(remaining, sz)
            if consume > 0.0:
                rows.append((float(px), float(consume)))
                remaining -= consume
        return rows


__all__ = [
    "TakerRequest",
    "FillModelProtocol",
    "StatelessFillsAdapter",
    "QueueAwareFillModel",
]
