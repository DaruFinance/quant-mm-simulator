"""Refresh-trigger primitives.

Cancel-replace decision: given the current sim state at time ``t``,
should the quoter refresh its outstanding orders?  Five built-in
trigger families, each a small stateful class with a single
``step(book, inv, t_ns) -> bool`` method.

The triggers:
  - ``TimeTrigger(interval_ns)`` — every N ns elapsed since last fire
  - ``MidMoveTrigger(threshold_bp)`` — mid has moved ≥ threshold bp since last fire
  - ``InvChangeTrigger(threshold)`` — |inv - last_inv_at_fire| ≥ threshold
  - ``BookEventTrigger()`` — every call (every book event) fires
  - ``HybridTrigger(children, mode="any")`` — composition; "any" or "all"

Causality contract (`≤ t` only): each ``step()`` reads only the
arguments passed in (which the caller provides from sim state at-or-
before t) plus the trigger's own internal state (set by a previous
``step()`` at-or-before t).  No closure over future events, no time
peek.  The leak test in ``tests/test_quoter_triggers.py`` runs a
trigger through a clean prefix vs a polluted-suffix full stream and
asserts the prefix-fire sequence is identical.

Convention: ``step()`` returns True if the trigger fires AT this call.
When it fires, the trigger updates its internal state so the next call
re-bases against the new reference.  Callers that want to observe the
condition without committing to a refresh should not call ``step()``
twice on the same instant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from mmsim.ingest.lob import Book


class TimeTrigger:
    """Fires every ``interval_ns`` nanoseconds.  Initial call always
    fires (first refresh after instantiation); subsequent calls fire
    only after the interval elapses."""

    def __init__(self, interval_ns: int):
        if interval_ns <= 0:
            raise ValueError("interval_ns must be > 0")
        self.interval_ns = int(interval_ns)
        self._last_fire_ts: Optional[int] = None

    def step(self, book: Optional[Book], inv: float, t_ns: int) -> bool:
        if self._last_fire_ts is None:
            self._last_fire_ts = int(t_ns)
            return True
        if t_ns - self._last_fire_ts >= self.interval_ns:
            self._last_fire_ts = int(t_ns)
            return True
        return False


class MidMoveTrigger:
    """Fires when the mid has moved at least ``threshold_bp``
    basis-points away from the last fire's mid.  When no book is
    available, returns False (cannot evaluate).  First call with a
    book sets the baseline and fires."""

    def __init__(self, threshold_bp: float):
        if threshold_bp < 0:
            raise ValueError("threshold_bp must be >= 0")
        self.threshold_bp = float(threshold_bp)
        self._last_fire_mid: Optional[float] = None

    def step(self, book: Optional[Book], inv: float, t_ns: int) -> bool:
        if book is None:
            return False
        m = book.mid
        if m is None:
            return False
        if self._last_fire_mid is None:
            self._last_fire_mid = float(m)
            return True
        rel_bp = abs(m - self._last_fire_mid) / self._last_fire_mid * 1e4
        if rel_bp >= self.threshold_bp:
            self._last_fire_mid = float(m)
            return True
        return False


class InvChangeTrigger:
    """Fires when ``|inv - last_fire_inv| >= threshold``.  First
    call always fires; subsequent calls re-base on each fire."""

    def __init__(self, threshold: float):
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        self.threshold = float(threshold)
        self._last_fire_inv: Optional[float] = None

    def step(self, book: Optional[Book], inv: float, t_ns: int) -> bool:
        if self._last_fire_inv is None:
            self._last_fire_inv = float(inv)
            return True
        if abs(inv - self._last_fire_inv) >= self.threshold:
            self._last_fire_inv = float(inv)
            return True
        return False


class BookEventTrigger:
    """Fires on every call.  Useful when the caller already feeds the
    trigger on book events only — the "book-event" wiring is the
    caller's responsibility; this trigger has no internal predicate."""

    def step(self, book: Optional[Book], inv: float, t_ns: int) -> bool:
        return True


class HybridTrigger:
    """Composition of multiple triggers.

    ``mode="any"`` fires if ANY child fires (each child's step is
    invoked exactly once per call so state advances uniformly).
    ``mode="all"`` fires only if EVERY child fires.

    All children are stepped on every call regardless of mode — this
    keeps state advancement deterministic and avoids the surprise
    of a short-circuit boolean leaving a child out-of-sync.
    """

    def __init__(self, children: List, mode: str = "any"):
        if mode not in ("any", "all"):
            raise ValueError(f"mode must be 'any' or 'all', got {mode!r}")
        self.children = list(children)
        self.mode = mode

    def step(self, book: Optional[Book], inv: float, t_ns: int) -> bool:
        fires = [c.step(book, inv, t_ns) for c in self.children]
        if self.mode == "any":
            return any(fires)
        return all(fires)


__all__ = [
    "TimeTrigger", "MidMoveTrigger", "InvChangeTrigger",
    "BookEventTrigger", "HybridTrigger",
]
