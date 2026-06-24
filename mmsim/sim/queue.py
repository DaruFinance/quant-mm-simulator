"""Queue-position tracker for resting orders.

Tracks how a single resting order's position in its price-level queue
evolves over the event stream.  Used by the maker/taker fill model
to determine whether an incoming taker trade hits our resting order
or only consumes the volume that's still ahead of it.

Causality is the load-bearing property here.  At every observed
event with ``ts_ns == t``, the tracker's state depends on **only**
events with ``ts_ns <= t``.  Polluting events strictly past T cannot
change ``queue_pos`` at T.  The leak battery in
``tests/test_sim_queue.py`` exercises this directly; it warrants
extra scrutiny because a stateful tracker is much easier to leak
through than a pure point-in-time function.

## Model

We track ``queue_pos``: the cumulative resting **size** (base-asset
units) ahead of our order at our price level.  ``queue_pos == 0`` means
we're at the front of the queue.  Once enough volume hits us to
consume ``queue_pos``, the next units fill us (handled by the fill model).

### Updates from `TradeEvent`

A trade aggressing **into** our level consumes from the front of the
queue.  Our ``queue_pos`` decreases by ``min(trade.size, queue_pos)``.

- Bid order (``side=+1``): a sell-aggressor (``trade.side=-1``) at
  ``trade.price == our_price`` consumes our level.
- Ask order (``side=-1``): a buy-aggressor (``trade.side=+1``) at
  ``trade.price == our_price`` consumes our level.

Trades at other levels (different prices) don't affect us in this
model — even though in reality a deep cross might cascade through
multiple levels, the trade tape's per-trade granularity already
splits a multi-level cross into per-level events.

### Updates from `SnapshotEvent`

Between consecutive snapshots, the size at our level changes by
some amount.  We attribute the change to a sum of trades + cancels:

```
size_new = size_old - trades_consumed_at_level + new_orders - cancels_at_level
```

We've observed ``trades_consumed_at_level`` directly from the tape.
The residual ``net_cancels = max(0, size_old - trades_consumed -
size_new)`` is the new cancels.  (If the residual is negative,
``net new orders`` joined the back of the queue and don't affect us.)

For cancels, we use the **pro-rata cancel model**: cancels are
uniformly distributed across queue positions.  Our share of the
cancels is ``net_cancels * (queue_pos / size_before_cancels)``,
where ``size_before_cancels = size_old - trades_consumed`` is the
queue size after fills but before the cancellation pulse.  This is
the mainstream choice in the queue-position-aware MM literature; the
two alternatives (FIFO with per-position uniform cancel and
pessimistic-cancels-go-behind) are documented in the verification log
with a short discussion of why pro-rata fits the verification
strategy.

### Edge case: our level drops out of the depth-N view

Snapshots are top-N (N=20 in DS-LOB-1H).  If our level was at TOB
and the book has since deepened (better prices on the same side),
our level may slip out of the visible top 20.  When that happens
``_size_at_level`` returns 0, which the cancel-attribution math
would mis-interpret as a complete level wipeout.

The tracker handles this by entering a ``frozen`` state: if the
level drops out of view after being non-empty, ``queue_pos`` stops
updating and the tracker emits a ``frozen=True`` flag in the trace.
Downstream consumers (the fill model) treat frozen tracks
conservatively — no fills until/unless the level reappears in view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from mmsim.ingest.lob import (
    Book, Event, EventStream, SnapshotEvent, TradeEvent,
)
from mmsim.sim.loop import Order


# Floating-point comparison tolerance for "same price level".  Bigger
# than f64 noise but tight enough to never collapse two truly
# different ticks (BTC tick is 0.01).
_PX_EPSILON = 1e-9


def _size_at_level(book: Book, side: int, price: float) -> Optional[float]:
    """Total resting size at the given side+price.  Returns ``None``
    if the level isn't visible in the book (depth-N truncation), so
    callers can distinguish "level empty" from "level out of view"."""
    levels = book.bids if side == +1 else book.asks
    for px, sz in levels:
        if abs(px - price) <= _PX_EPSILON:
            return float(sz)
    return None


def _trade_at_our_level(trade: TradeEvent, side: int, price: float) -> bool:
    """A trade aggressing into our specific level.  Per-event
    granularity in the tape: each TradeEvent is one matched
    maker/taker pair, which lands at exactly one price."""
    if trade.side == 0:
        return False
    if side == +1:
        return trade.side == -1 and abs(trade.price - price) <= _PX_EPSILON
    return trade.side == +1 and abs(trade.price - price) <= _PX_EPSILON


@dataclass(frozen=True)
class QueueSample:
    ts_ns: int
    queue_pos: float
    cause: str  # "placed", "fill_ahead", "cancel_ahead", "level_lost"


@dataclass(frozen=True)
class QueueTrace:
    """Result of running a tracker over an event stream."""
    samples: List[QueueSample]
    total_fills_ahead: float
    total_cancels_ahead: float
    final_queue_pos: float
    frozen: bool  # True if the level dropped out of depth-N view


class QueueTracker:
    """Stateful, event-by-event queue-position tracker.

    Causality contract: every state mutation depends only on events
    with ``ts_ns <= last observed event``.  Calling ``observe(ev)``
    with ``ev.ts_ns < self.last_ts_ns`` is a programming error and
    raises.
    """

    def __init__(self, order: Order, initial_book: Book):
        if order.size <= 0:
            raise ValueError("order.size must be positive")
        if initial_book.ts_ns > order.placed_at_ns:
            raise ValueError(
                "initial_book is from after order placement — "
                "callers must pass the book as it stood at "
                "order.placed_at_ns or earlier")
        self.order = order
        # queue_pos at placement = total resting size at our level
        # (we land behind everything currently there).  If the level
        # isn't visible, we cannot compute a meaningful initial pos.
        initial_size = _size_at_level(initial_book, order.side, order.price)
        if initial_size is None:
            raise ValueError(
                f"order.price={order.price} not visible at side={order.side} "
                f"in initial_book (depth-N truncation? ahead of best?)")
        self.queue_pos: float = float(initial_size)
        self._last_level_size: float = float(initial_size)
        self._pending_trade_volume_at_level: float = 0.0
        self._frozen: bool = False
        self._last_ts_ns: int = int(order.placed_at_ns)
        self.total_fills_ahead: float = 0.0
        self.total_cancels_ahead: float = 0.0
        self.samples: List[QueueSample] = [
            QueueSample(
                ts_ns=int(order.placed_at_ns),
                queue_pos=self.queue_pos,
                cause="placed",
            )
        ]

    def observe(self, event: Event) -> None:
        """Apply one event to the tracker's state."""
        if event.ts_ns < self._last_ts_ns:
            raise ValueError(
                f"out-of-order event: ev.ts_ns={event.ts_ns} < "
                f"last_ts_ns={self._last_ts_ns}")
        # Skip events strictly before placement (callers shouldn't
        # feed these, but be liberal).
        if event.ts_ns < self.order.placed_at_ns:
            return
        if self._frozen:
            self._last_ts_ns = int(event.ts_ns)
            return
        if isinstance(event, TradeEvent):
            self._on_trade(event)
        elif isinstance(event, SnapshotEvent):
            self._on_snapshot(event)
        else:
            raise TypeError(f"QueueTracker: unknown event {type(event).__name__}")
        self._last_ts_ns = int(event.ts_ns)

    def _on_trade(self, trade: TradeEvent) -> None:
        if not _trade_at_our_level(trade, self.order.side, self.order.price):
            return
        consumed_ahead = min(float(trade.size), self.queue_pos)
        if consumed_ahead > 0.0:
            self.queue_pos -= consumed_ahead
            self.total_fills_ahead += consumed_ahead
            self.samples.append(QueueSample(
                ts_ns=int(trade.ts_ns),
                queue_pos=self.queue_pos,
                cause="fill_ahead",
            ))
        # Track full trade volume at our level (including portions that
        # would fill us / fill behind us) so the snapshot reconciler
        # can subtract it from the size delta.
        self._pending_trade_volume_at_level += float(trade.size)

    def _on_snapshot(self, snap: SnapshotEvent) -> None:
        book = Book(ts_ns=snap.ts_ns, bids=snap.bids, asks=snap.asks)
        new_size = _size_at_level(book, self.order.side, self.order.price)
        if new_size is None:
            # Our level slipped out of the top-N view.  Freeze the
            # tracker — we can't meaningfully attribute future
            # changes.
            self._frozen = True
            self.samples.append(QueueSample(
                ts_ns=int(snap.ts_ns),
                queue_pos=self.queue_pos,
                cause="level_lost",
            ))
            return
        # Cancel attribution.
        size_after_trades = self._last_level_size - self._pending_trade_volume_at_level
        net_cancels = size_after_trades - new_size
        if net_cancels > 0.0 and size_after_trades > 0.0:
            # Pro-rata cancels: queue_pos shrinks by the same fraction
            # the queue itself shrinks.  Cap at queue_pos so we don't
            # go negative on f64 drift.
            fraction = self.queue_pos / size_after_trades
            cancels_ahead = min(net_cancels * fraction, self.queue_pos)
            if cancels_ahead > 0.0:
                self.queue_pos -= cancels_ahead
                self.total_cancels_ahead += cancels_ahead
                self.samples.append(QueueSample(
                    ts_ns=int(snap.ts_ns),
                    queue_pos=self.queue_pos,
                    cause="cancel_ahead",
                ))
        # Update level baseline for the next interval.
        self._last_level_size = float(new_size)
        self._pending_trade_volume_at_level = 0.0

    @property
    def frozen(self) -> bool:
        """Public read-only access to the freeze flag.  Other modules
        (the fill model) read this to short-circuit lookups
        on dead trackers."""
        return self._frozen

    def trace(self) -> QueueTrace:
        return QueueTrace(
            samples=list(self.samples),
            total_fills_ahead=self.total_fills_ahead,
            total_cancels_ahead=self.total_cancels_ahead,
            final_queue_pos=self.queue_pos,
            frozen=self._frozen,
        )


def track_queue_position(
    order: Order,
    events: EventStream,
    initial_book: Book,
) -> QueueTrace:
    """Run a fresh ``QueueTracker`` over ``events`` and return the
    trace.  Convenience wrapper for the one-shot verification path."""
    tracker = QueueTracker(order, initial_book)
    for ev in events:
        tracker.observe(ev)
    return tracker.trace()


__all__ = [
    "QueueSample", "QueueTrace", "QueueTracker",
    "track_queue_position",
]
