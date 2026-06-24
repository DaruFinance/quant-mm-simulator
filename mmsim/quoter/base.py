"""Quoter Protocol — the formal contract.

A typed Protocol in place of a loose ``QuoterFn`` callable. The single
shape every quoting model implements.

## Signature rationale

```
quote(book: Optional[Book], inv: float, t_ns: int) -> List[Decision]
```

- ``book``: the most-recent ``Book`` reconstructed at-or-before ``t_ns``
  (or ``None`` before warmup).  Built from the ingested snapshots.
- ``inv``: signed net inventory as observed up to ``t_ns``.
  The Protocol passes it explicitly rather than letting quoters
  consult mutable shared state — that keeps the leak invariant
  obvious by inspection.
- ``t_ns``: sim-time at the call.  Quoters that want past-only
  state must derive everything from arguments + their own
  internal state; the loop does not pass them future events.

Returns a list of ``Decision`` items, where each Decision is either:
  - ``QuoteRequest`` (resting maker order) — the loop replaces the
    current active set with the returned makers.
  - ``TakerRequest`` (immediate IOC) — the loop fires it against
    the current visible book at the same ``t_ns``.

Differences from the legacy ``QuoterFn`` callable:
  - Quoter is a Protocol (typed contract); QuoterFn was an opaque
    callable.
  - Quoter's signature is ``(book, inv, t_ns)`` per the spec; the
    legacy ``QuoterFn`` was ``(book, active_orders, t_ns)``.
  - Both paths still work in ``run_sim_with_model`` — branches on
    ``isinstance(quoter, Quoter)``.

## Causality contract (the leak rail this Protocol enforces by design)

Every input the Protocol passes is a snapshot of state at ``t_ns``
or earlier.  The Protocol does not pass the event stream, the
future event list, or any backstage handle that would let a quoter
peek past ``t_ns``.  A quoter that *wants* to look ahead can only
do so by storing future state across calls — an obvious code
smell and rejected by review (no automatic enforcement; the
property is a contract, not a runtime check).
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Union, runtime_checkable

from mmsim.ingest.lob import Book
from mmsim.sim.fills import TakerRequest
from mmsim.sim.loop import QuoteRequest


Decision = Union[QuoteRequest, TakerRequest]


@runtime_checkable
class Quoter(Protocol):
    """The formal quoter contract."""

    def quote(
        self,
        book: Optional[Book],
        inv: float,
        t_ns: int,
    ) -> List[Decision]: ...
