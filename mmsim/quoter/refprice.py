"""Reference-price primitives.

Six families of reference-price computations.  Three are pure
functions of the current book; three are stateful trackers that
observe trade events to maintain a moving estimate.  All consume
only state at-or-before `t` per the spec's `≤ t` requirement.

Pure (book-only):
  - ``top_mid(book)``        — (best_bid + best_ask) / 2
  - ``weighted_mid(book)``   — size-weighted using TOB queue sizes
  - ``microprice(book)``     — Stoikov microprice using queue imbalance

Stateful (trade-tape):
  - ``VWAPTracker(window_ns)``  — trailing-window VWAP of trades
  - ``EWMAFairTracker(half_life_ns)`` — EWMA of mid (book-fed) or trade price
  - ``ModelPredictedTracker(callback)`` — generic; calls back into a
    user-provided function; ships a trivial deterministic example.

All stateful trackers expose ``observe(...)`` to ingest state and
``value(t_ns) -> Optional[float]`` to read the current estimate.
The leak property is: ``value(T)`` cannot change after polluting
events with ts > T (verified by ``test_refprice_no_lookahead_*``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple
from collections import deque

from mmsim.ingest.lob import Book, TradeEvent


# --------------------------------------------------------------------- #
# Pure book-only primitives
# --------------------------------------------------------------------- #

def top_mid(book: Optional[Book]) -> Optional[float]:
    """Simple mid = (best_bid + best_ask) / 2."""
    if book is None:
        return None
    return book.mid


def weighted_mid(book: Optional[Book]) -> Optional[float]:
    """Size-weighted mid using TOB queue sizes.  When the bid queue
    is heavy relative to the ask, the weighted mid leans toward the
    bid — the standard "imbalance-aware" mid."""
    if book is None or not book.bids or not book.asks:
        return None
    bid_px, bid_sz = book.bids[0]
    ask_px, ask_sz = book.asks[0]
    if bid_sz + ask_sz <= 0:
        return (bid_px + ask_px) / 2.0
    # When ask queue is heavier (sellers stacked), the price is more
    # likely to drift down: weight the bid_px higher.  That's
    # `(bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)`.
    return (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)


def microprice(book: Optional[Book]) -> Optional[float]:
    """Stoikov microprice: mid + (queue_imbalance) × half_spread,
    where queue_imbalance = (bid_sz - ask_sz) / (bid_sz + ask_sz).
    Equivalent to `weighted_mid` algebraically; kept as a separate
    primitive because the literature treats them as distinct
    constructions (the derivation differs even if the closed form
    coincides at the TOB level)."""
    if book is None or not book.bids or not book.asks:
        return None
    bid_px, bid_sz = book.bids[0]
    ask_px, ask_sz = book.asks[0]
    if bid_sz + ask_sz <= 0:
        return (bid_px + ask_px) / 2.0
    mid = (bid_px + ask_px) / 2.0
    half_spread = (ask_px - bid_px) / 2.0
    imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
    return mid + imb * half_spread


# --------------------------------------------------------------------- #
# Stateful trackers — VWAP, EWMA fair, model-predicted
# --------------------------------------------------------------------- #

@dataclass
class VWAPTracker:
    """Trailing-window VWAP of trade tape.  ``window_ns`` is the
    look-back in nanoseconds.  observe(trade) pushes a trade;
    value(t_ns) returns Σ(price·size) / Σ(size) over trades with
    `t_ns - window_ns < trade.ts_ns <= t_ns`.

    Implementation: a deque of (ts_ns, price, size).  On each
    value() call we evict entries older than the window.  Eviction
    is O(k) where k is the number of expired entries; for a 1-hour
    DS-LOB-1H run with 100ms windows this stays cheap."""

    window_ns: int

    def __post_init__(self):
        if self.window_ns <= 0:
            raise ValueError("window_ns must be > 0")
        self._buffer: Deque[Tuple[int, float, float]] = deque()

    def observe(self, trade: TradeEvent) -> None:
        self._buffer.append((trade.ts_ns, float(trade.price), float(trade.size)))

    def value(self, t_ns: int) -> Optional[float]:
        cutoff = t_ns - self.window_ns
        # Evict expired entries
        while self._buffer and self._buffer[0][0] <= cutoff:
            self._buffer.popleft()
        if not self._buffer:
            return None
        # Filter to entries with ts_ns <= t_ns (in-bounds; the deque
        # may hold entries with ts > t_ns if observe was called with
        # future trades — but the leak test forbids that; defensive).
        total_pv = 0.0
        total_v = 0.0
        for ts, px, sz in self._buffer:
            if ts > t_ns:
                continue
            total_pv += px * sz
            total_v += sz
        if total_v <= 0.0:
            return None
        return total_pv / total_v


@dataclass
class EWMAFairTracker:
    """Exponentially-weighted moving average of mid (when fed
    snapshots) or trade price (when fed trades).  Half-life form:
    decay factor per nanosecond α(Δt) = 0.5 ** (Δt / half_life_ns).

    First observation seeds the EWMA at that value.  Subsequent
    observations update as
      ewma <- α(Δt) * ewma + (1 - α(Δt)) * new_value
    """

    half_life_ns: int

    def __post_init__(self):
        if self.half_life_ns <= 0:
            raise ValueError("half_life_ns must be > 0")
        self._ewma: Optional[float] = None
        self._last_ts: Optional[int] = None

    def observe(self, t_ns: int, value: float) -> None:
        if self._ewma is None:
            self._ewma = float(value)
            self._last_ts = int(t_ns)
            return
        dt = t_ns - self._last_ts
        if dt < 0:
            # Out-of-order; ignore (defensive).
            return
        if dt == 0:
            # Same instant — replace with new value (more-recent
            # observation wins at the same timestamp).
            self._ewma = float(value)
            return
        alpha = 0.5 ** (dt / self.half_life_ns)
        self._ewma = alpha * self._ewma + (1.0 - alpha) * float(value)
        self._last_ts = int(t_ns)

    def value(self, t_ns: int) -> Optional[float]:
        return self._ewma


class ModelPredictedTracker:
    """Generic stateful tracker that defers to a user-supplied
    callback ``predict(state_dict) -> float | None``.  The state_dict
    accumulates whatever the user passes to ``observe(**kwargs)``.

    Ships with a built-in deterministic example: a hard-coded
    callback that returns mid + linear_drift_per_obs × n_obs, used
    by the verification log to anchor cross-language parity at a
    closed-form value."""

    def __init__(self, predict: Callable[[dict], Optional[float]]):
        self.predict = predict
        self.state: dict = {"n_obs": 0}

    def observe(self, **kwargs) -> None:
        self.state["n_obs"] = self.state.get("n_obs", 0) + 1
        self.state.update(kwargs)

    def value(self, t_ns: int) -> Optional[float]:
        return self.predict(self.state)


def linear_drift_predictor(state: dict) -> Optional[float]:
    """Reference deterministic predictor: returns
    ``mid + slope_per_obs * n_obs``.  Reads ``mid`` and
    ``slope_per_obs`` from state (set by the caller via observe())."""
    mid = state.get("mid")
    if mid is None:
        return None
    slope = state.get("slope_per_obs", 0.0)
    n = state.get("n_obs", 0)
    return mid + slope * n


__all__ = [
    "top_mid", "weighted_mid", "microprice",
    "VWAPTracker", "EWMAFairTracker", "ModelPredictedTracker",
    "linear_drift_predictor",
]
