"""Shared rolling-volatility tracker for the closed-form quoting
models (Avellaneda-Stoikov, Cartea-Jaimungal, GLFT, Ho-Stoll).

The four AS-family models all need a per-snapshot vol estimate
`sigma`.  We don't want to pull a fully general vol model into the
quoter library, so this module ships a deterministic trailing-window
log-return std tracker that the four models share.

Implementation
--------------
`RollingSigma(window_ns)`:
  - `observe(t_ns, price)` pushes the price + ts into the trailing
    window (typically called once per snapshot with the current mid).
  - `value(t_ns) -> Optional[float]` returns the **per-nanosecond
    return standard deviation** — i.e. `std(log-returns) /
    sqrt(mean_dt_ns)` where `mean_dt_ns` is the average gap between
    observed prices in the window.  This yields a diffusion-coefficient
    figure such that ``σ² · τ`` (with τ in nanoseconds) has units of
    return-variance over the horizon τ — the dimensional convention
    Avellaneda-Stoikov, Cartea-Jaimungal, GLFT, and Ho-Stoll all
    assume.  Returns None when fewer than 2 valid prices fall inside
    the window.

Why mid-fed rather than trade-fed
---------------------------------
The Quoter Protocol gets called on snapshots; the model has no
direct hook to the trade stream from inside `quote()`.  Feeding the
sigma from snapshot mids keeps the model self-contained: every
input it needs is already on the Protocol's signature.  In
production a richer trade-tape vol would be wired in via an
adapter; for the in-tree library the snapshot-mid sigma is
deterministic and parity-friendly.

Leak property
-------------
`value(T)` filters its internal buffer to `ts <= T` before computing
the predicate, so polluting the buffer with `ts > T` observations
cannot change `value(T)`.

Determinism
-----------
Pure function of the observed `(ts, price)` sequence; bit-identical
across reruns given the same input sequence.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import log, sqrt
from typing import Deque, Optional, Tuple


@dataclass
class RollingSigma:
    """Trailing-window log-return std tracker (mid-fed).

    Returns ``std(log-returns) / sqrt(mean_dt_ns)`` — the
    per-nanosecond return std (a.k.a. diffusion coefficient).
    Multiplying ``σ² · τ`` with τ in nanoseconds gives a horizon
    variance estimate dimensionally consistent with what
    Avellaneda-Stoikov, Cartea-Jaimungal, GLFT, and Ho-Stoll
    expect on the right-hand-side of their spread formulas.
    """

    window_ns: int
    _buffer: Deque[Tuple[int, float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.window_ns <= 0:
            raise ValueError("window_ns must be > 0")

    def observe(self, t_ns: int, price: float) -> None:
        if price > 0.0:
            self._buffer.append((int(t_ns), float(price)))

    def _evict(self, t_ns: int) -> None:
        cutoff = t_ns - self.window_ns
        while self._buffer and self._buffer[0][0] <= cutoff:
            self._buffer.popleft()

    def value(self, t_ns: int) -> Optional[float]:
        self._evict(int(t_ns))
        # Keep only ts <= t_ns (defensive against out-of-order pollution).
        kept = [(ts, px) for ts, px in self._buffer if ts <= t_ns and px > 0.0]
        if len(kept) < 2:
            return None
        prices = [px for _, px in kept]
        timestamps = [ts for ts, _ in kept]
        rets = [log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        n = len(rets)
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / n
        std = sqrt(var)
        # Per-nanosecond std: divide by sqrt(mean inter-obs gap).
        total_dt = max(1, timestamps[-1] - timestamps[0])
        mean_dt = total_dt / n
        return std / sqrt(mean_dt)


__all__ = ["RollingSigma"]
