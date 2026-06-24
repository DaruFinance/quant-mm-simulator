"""Avellaneda-Stoikov quoting model.

Closed-form maker quotes derived in Avellaneda & Stoikov (2008),
"High-Frequency Trading in a Limit Order Book".  The model assumes:
  - mid-price evolves as a Brownian motion with volatility σ;
  - market-order arrivals follow a Poisson process whose intensity
    decays exponentially in the distance from mid (parameter k);
  - the market-maker has exponential utility with risk-aversion γ;
  - the trading horizon is T (in nanoseconds), time-to-go is (T - t).

Formulae
--------
Reservation (inventory-adjusted) price:

    r(s, q, t) = s - q · γ · σ² · (T - t)

Optimal half-spread (one side of the bid-ask):

    half_spread* = (γ · σ² · (T - t)) / 2 + (1/γ) · ln(1 + γ / k)

Quoted prices:

    bid_px = r - half_spread*
    ask_px = r + half_spread*

Sign conventions
----------------
`q` here is the signed net inventory (long positive, short negative).
Long inventory shifts the reservation DOWN — encouraging selling —
exactly matching the inventory-penalty primitive convention.

Time-to-go handling
-------------------
`(T - t)` is clamped to a small positive floor (1 ns) once `t > T`
so the spread doesn't collapse to zero or go negative.  The caller
sets `horizon_ns` (= T) on construction; `t_ns` flows in via the
Protocol.

Sigma
-----
Uses the shared `RollingSigma` tracker (mid-fed; ``vol_window_ns``
sets the trailing window).  Before sigma warms up (fewer than 2
mids observed in window), the model emits no quotes — it's not
willing to take a position based on a fake-zero vol.

Returns a single bid + ask pair (length-2 list).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import List, Optional

from mmsim.ingest.lob import Book
from mmsim.quoter.base import Decision
from mmsim.quoter.refprice import top_mid
from mmsim.sim.loop import QuoteRequest

from ._vol import RollingSigma


@dataclass
class AvellanedaStoikovQuoter:
    """Avellaneda-Stoikov closed-form market maker.

    Parameters
    ----------
    gamma:
        Risk aversion coefficient (> 0).  Higher gamma = more
        aggressive inventory skew + wider spread.
    k:
        Market-order intensity decay parameter (> 0).  Higher k =
        narrower spread (because order arrivals fall off more
        quickly with distance, so the optimal post is closer).
    sigma_floor:
        Lower bound on sigma estimate.  Guards against zero-vol
        intervals collapsing the spread.  Default 1e-9.
    horizon_ns:
        Trading horizon T in nanoseconds.  At t = 0 the time-to-go
        is `horizon_ns`; at t = horizon_ns it's 1 ns (floor).
    size:
        Quote size per side.
    vol_window_ns:
        RollingSigma window for the σ estimate.
    """

    gamma: float
    k: float
    horizon_ns: int
    size: float
    vol_window_ns: int
    sigma_floor: float = 1e-9
    _sigma: RollingSigma = field(init=False)
    _t_start_ns: Optional[int] = None

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")
        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.horizon_ns <= 0:
            raise ValueError("horizon_ns must be > 0")
        if self.size <= 0:
            raise ValueError("size must be > 0")
        if self.vol_window_ns <= 0:
            raise ValueError("vol_window_ns must be > 0")
        if self.sigma_floor < 0:
            raise ValueError("sigma_floor must be >= 0")
        self._sigma = RollingSigma(window_ns=int(self.vol_window_ns))

    def _time_to_go(self, t_ns: int) -> float:
        """Time-to-go in nanoseconds, floored at 1.0 to avoid the
        terminal-spread collapse pathology.

        Anchor `t_start` to the first observed timestamp so the
        horizon is measured from the first call, not from epoch.
        """
        if self._t_start_ns is None:
            self._t_start_ns = int(t_ns)
        elapsed = int(t_ns) - int(self._t_start_ns)
        remaining = float(self.horizon_ns) - float(elapsed)
        return max(remaining, 1.0)

    def quote(
        self,
        book: Optional[Book],
        inv: float,
        t_ns: int,
    ) -> List[Decision]:
        mid = top_mid(book)
        if mid is None:
            return []
        # Feed sigma off the snapshot mid (deterministic; no trade
        # tape hook in the Protocol).
        self._sigma.observe(int(t_ns), float(mid))
        sigma = self._sigma.value(int(t_ns))
        if sigma is None:
            return []
        sigma = max(float(sigma), self.sigma_floor)
        tau = self._time_to_go(int(t_ns))
        # Reservation: r = s - q · γ · σ² · (T - t)
        r = float(mid) - float(inv) * self.gamma * sigma * sigma * tau
        # Optimal half-spread: γσ²(T-t)/2 + (1/γ) ln(1 + γ/k)
        spread_half = (
            (self.gamma * sigma * sigma * tau) / 2.0
            + (1.0 / self.gamma) * log(1.0 + self.gamma / self.k)
        )
        bid_px = r - spread_half
        ask_px = r + spread_half
        return [
            QuoteRequest(side=+1, price=bid_px, size=self.size),
            QuoteRequest(side=-1, price=ask_px, size=self.size),
        ]


__all__ = ["AvellanedaStoikovQuoter"]
