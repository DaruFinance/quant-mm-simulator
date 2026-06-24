"""Event-driven sim loop.

The single integration point for every sim run.  Consumes an
``EventStream``, advances sim time event-by-event, calls
the injected quoter on each snapshot to refresh the active order
book, and applies the injected fill model on each trade.

Causality contract:
    For every event consumed at sim time t, the quoter and fill
    callbacks observe only events with ts_ns <= t.  Polluting events
    with ts_ns > t cannot change the fills emitted up to t.  The
    cornerstone leak test (``test_run_sim_no_lookahead_under_pollution``)
    pollutes events past T and asserts the fill list up to T is
    bit-identical.

The loop is the substrate the richer components compose on top of:
the formal Quoter Protocol, quote shapes / triggers / refprices,
the adverse-selection filter, the concrete quoting models (AS, GLFT,
etc.), the inventory penalty, the hedge engine, the queue
position-aware fill model, maker/taker discrimination, and the
continuous fractional inventory tracker.

The simplest injected ``quoter`` and ``fills`` callables are
deliberately minimal — they exist so the loop can be exercised
end-to-end on real data and produce a baseline fill count.  Both
can be replaced by the formal protocols; the loop's
public surface stays stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING

from mmsim.ingest.lob import (
    Book, Event, EventStream, SnapshotEvent, TradeEvent,
)

if TYPE_CHECKING:
    # Avoid an import cycle: fills.py imports loop's Order/QuoteRequest
    # at module load; loop only needs fills' types for type hints.
    from mmsim.sim.fills import FillModelProtocol, TakerRequest


# --------------------------------------------------------------------- #
# Public types — frozen where natural; ``Order`` is mutable so the loop
# can shrink ``size`` on partial fills without copying the whole list.
# --------------------------------------------------------------------- #

@dataclass
class Order:
    """A resting order in our market.  ``side`` is +1 for a bid (our
    buy) and -1 for an ask (our sell).  ``size`` is mutated by the
    loop on partial fills; when it reaches zero the order is removed
    from the active list."""
    order_id: int
    side: int
    price: float
    size: float
    placed_at_ns: int


@dataclass(frozen=True)
class QuoteRequest:
    """What a quoter asks to post.  ``order_id`` is assigned by the
    loop on placement; quoters do not pick their own IDs.

    ``ttl_ns`` is an optional time-to-live in nanoseconds.
    When set, the loop is *expected* to auto-cancel the order at
    ``placed_at_ns + ttl_ns``.  The field is part of the quoter
    contract; runtime enforcement is provided by the refresh-trigger
    primitives.  Without those, the field is informational; quoters
    that set it observe no behavioural change."""
    side: int
    price: float
    size: float
    ttl_ns: Optional[int] = None


@dataclass(frozen=True)
class Fill:
    """A realised fill against one of our resting orders.  ``side`` is
    the resting order's side (+1 = our bid filled = we bought; -1 =
    our ask filled = we sold).  ``is_maker`` is always True for the
    maker-only path; the maker/taker fill model introduces taker fills
    and flips this."""
    fill_id: int
    order_id: int
    ts_ns: int
    price: float
    size: float
    side: int
    is_maker: bool


@dataclass(frozen=True)
class SimResult:
    """Output of one ``run_sim`` invocation.  Reproducible across
    reruns: same ``events``, same ``quoter``, same ``fills`` →
    bit-identical ``SimResult``.

    ``n_maker_fills`` + ``n_taker_fills`` == ``len(fills)``.  The
    queue-aware fill model introduces taker fills; with a stateless
    callable fills adapter ``n_taker_fills`` is always 0."""
    fills: List[Fill]
    n_events_processed: int
    n_snapshot_events: int
    n_trade_events: int
    n_quoter_calls: int
    n_maker_fills: int
    n_taker_fills: int
    final_orders: List[Order]


# Callback signatures.  Both replaceable per-run so the loop can host
# any quoter / fill model.

QuoterFn = Callable[
    [Optional[Book], List[Order], int],
    List["QuoteRequest | TakerRequest"],
]
"""Quoter signature: ``(book, current_active_orders, t_ns) ->
[QuoteRequest | TakerRequest, ...]``.

`QuoteRequest` items become **resting maker** orders that replace
the current active set (cancel-on-replace semantics).  `TakerRequest`
items fire **immediately** at the current event's
ts against the current visible book; they do not become resting
orders.  Mixing both kinds in one return is allowed.

Return an empty list to cancel everything (no makers re-posted, no
takers fired)."""

FillsFn = Callable[
    [List[Order], TradeEvent],
    List[Tuple[int, float]],
]
"""Fill model signature: ``(active_orders, trade_event) ->
[(order_id, fill_size), ...]``.  Returns which of our resting orders
got hit by this trade and how much filled.  The loop applies the
fills in the order returned (priority is the model's responsibility)
and updates ``active_orders`` in place."""


@dataclass
class _SimState:
    """Internal mutable state.  Not part of the public API."""
    active_orders: List[Order] = field(default_factory=list)
    fills: List[Fill] = field(default_factory=list)
    next_order_id: int = 0
    next_fill_id: int = 0
    last_book: Optional[Book] = None
    n_quoter_calls: int = 0


def _split_quoter_returns(
    items: List, t_ns: int,
) -> Tuple[List[QuoteRequest], List]:
    """Partition the quoter's return list into makers (QuoteRequest)
    and takers (TakerRequest from fills.py).  Done by isinstance,
    not by attribute sniffing, so the type discriminator is rigid."""
    # Lazy import to avoid the cycle (fills.py imports loop.py).
    from mmsim.sim.fills import TakerRequest
    makers: List[QuoteRequest] = []
    takers: List = []
    for it in items:
        if isinstance(it, TakerRequest):
            takers.append(it)
        elif isinstance(it, QuoteRequest):
            makers.append(it)
        else:
            raise TypeError(
                f"quoter returned {type(it).__name__} at t_ns={t_ns}; "
                f"expected QuoteRequest or TakerRequest")
    return makers, takers


def _replace_active_orders(
    state: _SimState,
    quotes: List[QuoteRequest],
    t_ns: int,
) -> Tuple[List[int], List[Order]]:
    """Replace ``state.active_orders`` with the quoter's returned set.
    Returns ``(removed_ids, new_orders)`` so the caller can notify
    a stateful FillModel."""
    removed_ids = [o.order_id for o in state.active_orders]
    new_orders: List[Order] = []
    for q in quotes:
        new_orders.append(Order(
            order_id=state.next_order_id,
            side=int(q.side),
            price=float(q.price),
            size=float(q.size),
            placed_at_ns=t_ns,
        ))
        state.next_order_id += 1
    state.active_orders = new_orders
    return removed_ids, new_orders


def _apply_fills(
    state: _SimState,
    hits: List[Tuple[int, float]],
    trade: TradeEvent,
) -> None:
    """Resolve ``(order_id, fill_size)`` hits returned by the fill
    model.  Each hit is appended as a Fill record and the matched
    order's size is decremented; orders that hit zero are removed
    from ``active_orders``."""
    if not hits:
        return
    by_id = {o.order_id: o for o in state.active_orders}
    for order_id, fill_size in hits:
        order = by_id.get(int(order_id))
        if order is None:
            # Fill model asked to fill an order that's no longer
            # active — caller bug.  Fail loud.
            raise KeyError(
                f"_apply_fills: order_id={order_id} not in active set "
                f"at trade ts={trade.ts_ns}")
        fill_size = float(fill_size)
        if fill_size <= 0.0:
            raise ValueError(
                f"_apply_fills: non-positive fill size {fill_size} "
                f"on order {order_id}")
        actual = min(fill_size, order.size)
        state.fills.append(Fill(
            fill_id=state.next_fill_id,
            order_id=order.order_id,
            ts_ns=int(trade.ts_ns),
            price=float(order.price),
            size=actual,
            side=int(order.side),
            is_maker=True,
        ))
        state.next_fill_id += 1
        order.size -= actual
    # Drop fully-filled orders.
    state.active_orders = [o for o in state.active_orders if o.size > 0.0]


def run_sim(
    events: EventStream,
    quoter: QuoterFn,
    fills,
) -> SimResult:
    """Drive an event stream through the loop.

    ``fills`` accepts either:
      - a stateless callable ``(active_orders, trade) -> [(oid, size), ...]``
        — the legacy path; gets wrapped in ``StatelessFillsAdapter``.
        Doesn't see snapshots, doesn't see order placements, doesn't
        handle takers.
      - a ``FillModelProtocol`` object with full lifecycle
        hooks: ``on_order_placed``, ``on_orders_removed``, ``on_snapshot``,
        ``on_trade``, ``fill_taker``.  Required for queue-aware maker
        fills and for any taker-fill processing.

    Reproducibility: deterministic quoter + deterministic fill model +
    same event stream ⇒ bit-identical ``SimResult`` across reruns.

    Causality: at each event with ``ts_ns == t``, the quoter and fill
    callbacks see only events at indices ``<= t`` in the stream.
    """
    # Lazy imports for the FillModel branch (avoid cycle with fills.py).
    from mmsim.sim.fills import (
        FillModelProtocol, StatelessFillsAdapter,
    )
    from mmsim.sim.inventory import InventoryTracker
    # Lazy import to avoid pulling quoter package on every loop run
    # (the user may pass a bare callable and never touch the package).
    try:
        from mmsim.quoter import Quoter as _QuoterProtocol
    except ImportError:
        _QuoterProtocol = None  # type: ignore

    if isinstance(fills, FillModelProtocol):
        model = fills
    elif callable(fills):
        model = StatelessFillsAdapter(fills)
    else:
        raise TypeError(
            f"run_sim: fills must be a callable or FillModelProtocol, "
            f"got {type(fills).__name__}")

    # Branch on quoter shape.  A Protocol-conforming
    # quoter is called as ``quoter.quote(book, inv, t_ns)`` and the
    # loop maintains an InventoryTracker fed by emitted fills.  A
    # legacy callable is called as ``quoter(book, active_orders, t_ns)``
    # — same as the stateless callable path; no inventory threading.
    use_protocol_quoter = (
        _QuoterProtocol is not None
        and not callable(quoter)
        and isinstance(quoter, _QuoterProtocol)
    ) or (
        _QuoterProtocol is not None
        and isinstance(quoter, _QuoterProtocol)
        and hasattr(quoter, "quote")
    )
    inv_tracker: Optional["InventoryTracker"] = (
        InventoryTracker() if use_protocol_quoter else None
    )

    state = _SimState()
    n_snap = 0
    n_trade = 0
    n_maker_fills = 0
    n_taker_fills = 0
    for ev in events:
        if isinstance(ev, SnapshotEvent):
            n_snap += 1
            state.last_book = Book(
                ts_ns=ev.ts_ns, bids=ev.bids, asks=ev.asks,
            )
            # 1) Notify the model so its trackers can do cancel
            #    attribution against this snapshot BEFORE any new
            #    orders are placed in response to it.
            model.on_snapshot(ev)
            # 2) Quoter decides what to post / take.
            if use_protocol_quoter:
                inv_now = inv_tracker.inv if inv_tracker is not None else 0.0
                items = quoter.quote(state.last_book, inv_now, ev.ts_ns)
            else:
                items = quoter(state.last_book, list(state.active_orders), ev.ts_ns)
            state.n_quoter_calls += 1
            makers, takers = _split_quoter_returns(items, ev.ts_ns)
            # 3) Replace active makers; notify the model.
            removed_ids, new_orders = _replace_active_orders(
                state, makers, ev.ts_ns)
            if removed_ids:
                model.on_orders_removed(removed_ids)
            for o in new_orders:
                model.on_order_placed(o, state.last_book)
            # 4) Fire any takers immediately against the current book.
            for req in takers:
                rows = model.fill_taker(req, state.last_book, ev.ts_ns)
                for px, sz in rows:
                    if sz <= 0.0:
                        continue
                    f = Fill(
                        fill_id=state.next_fill_id,
                        order_id=-1,  # taker fills aren't tied to a resting order
                        ts_ns=int(ev.ts_ns),
                        price=float(px),
                        size=float(sz),
                        side=int(req.side),
                        is_maker=False,
                    )
                    state.fills.append(f)
                    state.next_fill_id += 1
                    n_taker_fills += 1
                    if inv_tracker is not None:
                        inv_tracker.observe(f)
        elif isinstance(ev, TradeEvent):
            n_trade += 1
            if state.active_orders:
                hits = model.on_trade(ev, list(state.active_orders))
                if hits:
                    before = len(state.fills)
                    _apply_fills(state, hits, ev)
                    n_maker_fills += len(state.fills) - before
                    if inv_tracker is not None:
                        for f in state.fills[before:]:
                            inv_tracker.observe(f)
        else:
            raise TypeError(f"run_sim: unknown event kind {type(ev).__name__}")
    return SimResult(
        fills=list(state.fills),
        n_events_processed=len(events),
        n_snapshot_events=n_snap,
        n_trade_events=n_trade,
        n_quoter_calls=state.n_quoter_calls,
        n_maker_fills=n_maker_fills,
        n_taker_fills=n_taker_fills,
        final_orders=list(state.active_orders),
    )


__all__ = [
    "Order", "QuoteRequest", "Fill", "SimResult",
    "QuoterFn", "FillsFn",
    "run_sim",
]
