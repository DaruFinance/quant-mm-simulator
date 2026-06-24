"""Adverse-selection filter primitives.

Six stateful filters that suppress quoting when the order book or
trade tape signals that the market is currently moving against
passive makers (us).  Each filter's ``is_adverse(t_ns) -> bool``
returns True iff the filter is currently activated; ``observe_*``
calls feed state.

The six:
  - ``OFIFilter`` — order-flow imbalance over trailing window
  - ``TradeToxicityFilter`` — aggressor-side imbalance (buy vs sell volume)
  - ``VolSurgeFilter`` — realized vol surge from trade-tape returns
  - ``MicropriceDevFilter`` — gap between microprice and mid (pure book)
  - ``QueueImbalanceFilter`` — TOB queue imbalance (pure book)
  - ``HybridAdverseFilter`` — any-of / all-of composition

All filters consume only `≤ t` state.  Trade-tape filters keep a
trailing-window deque of observations; the pure-book filters store
only the most-recent book passed via observe_book().
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

from mmsim.ingest.lob import Book, TradeEvent


# --------------------------------------------------------------------- #
# Trade-tape filters
# --------------------------------------------------------------------- #

class OFIFilter:
    """Order-flow imbalance over a trailing window of trades.
    OFI = (buy_volume - sell_volume) / total_volume.  Activates when
    ``|OFI| >= threshold``.  Aggressor side is taken from
    ``trade.side`` (+1 buy aggressor, -1 sell aggressor; 0 ignored)."""

    def __init__(self, window_ns: int, threshold: float):
        if window_ns <= 0:
            raise ValueError("window_ns must be > 0")
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        self.window_ns = int(window_ns)
        self.threshold = float(threshold)
        self._buffer: Deque[Tuple[int, int, float]] = deque()  # (ts, side, size)

    def observe_trade(self, trade: TradeEvent) -> None:
        if trade.side == 0:
            return
        self._buffer.append((int(trade.ts_ns), int(trade.side), float(trade.size)))

    def observe_book(self, book: Book) -> None:
        pass  # book-blind

    def _evict(self, t_ns: int) -> None:
        cutoff = t_ns - self.window_ns
        while self._buffer and self._buffer[0][0] <= cutoff:
            self._buffer.popleft()

    def is_adverse(self, t_ns: int) -> bool:
        self._evict(t_ns)
        if not self._buffer:
            return False
        buy_vol = sum(sz for ts, side, sz in self._buffer if ts <= t_ns and side == +1)
        sell_vol = sum(sz for ts, side, sz in self._buffer if ts <= t_ns and side == -1)
        total = buy_vol + sell_vol
        if total <= 0.0:
            return False
        ofi = (buy_vol - sell_vol) / total
        return abs(ofi) >= self.threshold


class TradeToxicityFilter:
    """Aggressor-side dominance over a trailing window.  Activates
    when one side's volume share exceeds ``threshold`` (a number in
    [0.5, 1.0] — 0.5 means perfectly balanced, 1.0 means all on one
    side)."""

    def __init__(self, window_ns: int, threshold: float):
        if window_ns <= 0:
            raise ValueError("window_ns must be > 0")
        if not (0.5 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0.5, 1.0]")
        self.window_ns = int(window_ns)
        self.threshold = float(threshold)
        self._buffer: Deque[Tuple[int, int, float]] = deque()

    def observe_trade(self, trade: TradeEvent) -> None:
        if trade.side == 0:
            return
        self._buffer.append((int(trade.ts_ns), int(trade.side), float(trade.size)))

    def observe_book(self, book: Book) -> None:
        pass

    def _evict(self, t_ns: int) -> None:
        cutoff = t_ns - self.window_ns
        while self._buffer and self._buffer[0][0] <= cutoff:
            self._buffer.popleft()

    def is_adverse(self, t_ns: int) -> bool:
        self._evict(t_ns)
        if not self._buffer:
            return False
        buy_vol = sum(sz for ts, side, sz in self._buffer if ts <= t_ns and side == +1)
        sell_vol = sum(sz for ts, side, sz in self._buffer if ts <= t_ns and side == -1)
        total = buy_vol + sell_vol
        if total <= 0.0:
            return False
        max_share = max(buy_vol, sell_vol) / total
        return max_share >= self.threshold


class VolSurgeFilter:
    """Realized-vol surge from trailing trade prices.  Activates when
    the rolling std of log-returns exceeds ``threshold_bp`` (in
    basis-points).  Needs at least 3 trades in the window before
    it can fire."""

    def __init__(self, window_ns: int, threshold_bp: float):
        if window_ns <= 0:
            raise ValueError("window_ns must be > 0")
        if threshold_bp < 0:
            raise ValueError("threshold_bp must be >= 0")
        self.window_ns = int(window_ns)
        self.threshold_bp = float(threshold_bp)
        self._buffer: Deque[Tuple[int, float]] = deque()  # (ts, price)

    def observe_trade(self, trade: TradeEvent) -> None:
        self._buffer.append((int(trade.ts_ns), float(trade.price)))

    def observe_book(self, book: Book) -> None:
        pass

    def _evict(self, t_ns: int) -> None:
        cutoff = t_ns - self.window_ns
        while self._buffer and self._buffer[0][0] <= cutoff:
            self._buffer.popleft()

    def is_adverse(self, t_ns: int) -> bool:
        self._evict(t_ns)
        prices = [px for ts, px in self._buffer if ts <= t_ns and px > 0]
        if len(prices) < 3:
            return False
        # Log-returns
        from math import log
        rets = [log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        n = len(rets)
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / n
        std_bp = (var ** 0.5) * 1e4
        return std_bp >= self.threshold_bp


# --------------------------------------------------------------------- #
# Pure-book filters (no trade-tape state needed)
# --------------------------------------------------------------------- #

class MicropriceDevFilter:
    """Activates when ``|microprice - mid| / mid * 1e4 >= threshold_bp``.
    Pure book function on the most-recent observed book."""

    def __init__(self, threshold_bp: float):
        if threshold_bp < 0:
            raise ValueError("threshold_bp must be >= 0")
        self.threshold_bp = float(threshold_bp)
        self._last_book: Optional[Book] = None

    def observe_trade(self, trade: TradeEvent) -> None:
        pass

    def observe_book(self, book: Book) -> None:
        self._last_book = book

    def is_adverse(self, t_ns: int) -> bool:
        b = self._last_book
        if b is None or not b.bids or not b.asks:
            return False
        bid_px, bid_sz = b.bids[0]
        ask_px, ask_sz = b.asks[0]
        if bid_sz + ask_sz <= 0.0:
            return False
        mid = (bid_px + ask_px) / 2.0
        if mid <= 0.0:
            return False
        half_spread = (ask_px - bid_px) / 2.0
        imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
        microprice = mid + imb * half_spread
        return abs(microprice - mid) / mid * 1e4 >= self.threshold_bp


class QueueImbalanceFilter:
    """Activates when the absolute queue imbalance at TOB exceeds
    ``threshold``: ``|bid_sz - ask_sz| / (bid_sz + ask_sz) >= threshold``."""

    def __init__(self, threshold: float):
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0.0, 1.0]")
        self.threshold = float(threshold)
        self._last_book: Optional[Book] = None

    def observe_trade(self, trade: TradeEvent) -> None:
        pass

    def observe_book(self, book: Book) -> None:
        self._last_book = book

    def is_adverse(self, t_ns: int) -> bool:
        b = self._last_book
        if b is None or not b.bids or not b.asks:
            return False
        bid_sz = b.bids[0][1]
        ask_sz = b.asks[0][1]
        total = bid_sz + ask_sz
        if total <= 0.0:
            return False
        imb = abs(bid_sz - ask_sz) / total
        return imb >= self.threshold


# --------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------- #

class HybridAdverseFilter:
    """Composes multiple adverse filters via any-of or all-of."""

    def __init__(self, children: List, mode: str = "any"):
        if mode not in ("any", "all"):
            raise ValueError(f"mode must be 'any' or 'all', got {mode!r}")
        self.children = list(children)
        self.mode = mode

    def observe_trade(self, trade: TradeEvent) -> None:
        for c in self.children:
            c.observe_trade(trade)

    def observe_book(self, book: Book) -> None:
        for c in self.children:
            c.observe_book(book)

    def is_adverse(self, t_ns: int) -> bool:
        # No short-circuit: every child gets a chance to evict / advance
        # state (irrelevant here since is_adverse() is read-only, but
        # we preserve the convention for symmetry with triggers).
        votes = [c.is_adverse(t_ns) for c in self.children]
        if self.mode == "any":
            return any(votes)
        return all(votes)


__all__ = [
    "OFIFilter", "TradeToxicityFilter", "VolSurgeFilter",
    "MicropriceDevFilter", "QueueImbalanceFilter", "HybridAdverseFilter",
]
