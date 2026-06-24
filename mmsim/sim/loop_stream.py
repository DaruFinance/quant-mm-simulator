"""Streaming driver for the frozen sim loop (bounded-RAM full-day runs).

``mmsim.sim.loop.run_sim`` is event-by-event already, but takes
``len(events)`` once at the end -- which forces a fully-materialised list and
so cannot consume a generator over a 9.5M-snapshot day without the OOM.

``run_sim_streaming`` is a line-for-line copy of ``run_sim``'s loop body that
instead **counts** events as it consumes them from any iterable / generator,
holding only the current event plus the (tiny) active-order set and the fill
model's per-order trackers.  Everything else -- quoter contract, fill model
lifecycle hook order, taker handling, inventory threading, the SnapshotEvent /
TradeEvent dispatch, the Fill records emitted -- is identical, so the
``SimResult`` is BIT-IDENTICAL to ``run_sim`` on the same event sequence.
Verified in ``scripts/verify_stream_parity.py``.

The only ``SimResult`` field that differs in *derivation* is
``n_events_processed`` (counted, not ``len``); its VALUE is identical.
"""
from __future__ import annotations

from typing import Iterable

from mmsim.ingest.lob import SnapshotEvent, TradeEvent, Book
from mmsim.sim.loop import Fill, SimResult, _SimState, _split_quoter_returns, \
    _replace_active_orders, _apply_fills


def run_sim_streaming(events: Iterable, quoter, fills) -> SimResult:
    """Identical semantics to ``mmsim.sim.loop.run_sim`` but consumes
    ``events`` as a one-shot iterable (generator-safe, bounded RAM)."""
    from mmsim.sim.fills import FillModelProtocol, StatelessFillsAdapter
    from mmsim.sim.inventory import InventoryTracker
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
            f"run_sim_streaming: fills must be a callable or FillModelProtocol, "
            f"got {type(fills).__name__}")

    use_protocol_quoter = (
        _QuoterProtocol is not None
        and not callable(quoter)
        and isinstance(quoter, _QuoterProtocol)
    ) or (
        _QuoterProtocol is not None
        and isinstance(quoter, _QuoterProtocol)
        and hasattr(quoter, "quote")
    )
    inv_tracker = InventoryTracker() if use_protocol_quoter else None

    state = _SimState()
    n_events = 0
    n_snap = 0
    n_trade = 0
    n_maker_fills = 0
    n_taker_fills = 0
    for ev in events:
        n_events += 1
        if isinstance(ev, SnapshotEvent):
            n_snap += 1
            state.last_book = Book(ts_ns=ev.ts_ns, bids=ev.bids, asks=ev.asks)
            model.on_snapshot(ev)
            if use_protocol_quoter:
                inv_now = inv_tracker.inv if inv_tracker is not None else 0.0
                items = quoter.quote(state.last_book, inv_now, ev.ts_ns)
            else:
                items = quoter(state.last_book, list(state.active_orders), ev.ts_ns)
            state.n_quoter_calls += 1
            makers, takers = _split_quoter_returns(items, ev.ts_ns)
            removed_ids, new_orders = _replace_active_orders(state, makers, ev.ts_ns)
            if removed_ids:
                model.on_orders_removed(removed_ids)
            for o in new_orders:
                model.on_order_placed(o, state.last_book)
            for req in takers:
                rows = model.fill_taker(req, state.last_book, ev.ts_ns)
                for px, sz in rows:
                    if sz <= 0.0:
                        continue
                    f = Fill(
                        fill_id=state.next_fill_id, order_id=-1,
                        ts_ns=int(ev.ts_ns), price=float(px), size=float(sz),
                        side=int(req.side), is_maker=False)
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
            raise TypeError(f"run_sim_streaming: unknown event kind {type(ev).__name__}")

    return SimResult(
        fills=list(state.fills),
        n_events_processed=n_events,
        n_snapshot_events=n_snap,
        n_trade_events=n_trade,
        n_quoter_calls=state.n_quoter_calls,
        n_maker_fills=n_maker_fills,
        n_taker_fills=n_taker_fills,
        final_orders=list(state.active_orders),
    )


__all__ = ["run_sim_streaming"]
