"""Cartea-Jaimungal quoting model.

Close cousin of Avellaneda-Stoikov.  Cartea & Jaimungal's "Algorithmic
and High-Frequency Trading" derives optimal quotes under a different
utility / impact set-up; one of the simplest closed forms uses a
mean-reverting inventory target κ:

    r(s, q) = s + (κ - 2 · q) / (2 · γ)

where:
  - s is the mid;
  - q is signed inventory;
  - κ is a target inventory level (often 0);
  - γ is the risk-aversion / penalty coefficient.

With κ = 0 the reservation simplifies to ``r = s - q / γ`` — a linear
inventory skew proportional to 1/γ (smaller γ ⇒ stronger skew).

The optimal half-spread is the same Avellaneda-Stoikov "intensity +
risk" decomposition; we use the AS form so the two models are
directly comparable (and the difference is purely in the reservation
update):

    half_spread* = (γ · σ² · (T - t)) / 2 + (1/γ) · ln(1 + γ / k)

Quoted prices:

    bid_px = r - half_spread*
    ask_px = r + half_spread*

Sign / unit conventions match `AvellanedaStoikovQuoter`.  Sigma is
fed off the snapshot mid (same `RollingSigma` as AS).
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
class CarteaJaimungalQuoter:
    """Cartea-Jaimungal closed-form market maker.

    Parameters
    ----------
    gamma:
        Risk-aversion / penalty coefficient (> 0).
    k:
        Market-order intensity decay parameter for the AS-style
        spread term (> 0).
    kappa:
        Inventory target.  Typical default 0.0 (flat target).
    horizon_ns:
        Trading horizon T in nanoseconds.
    size:
        Quote size per side.
    vol_window_ns:
        RollingSigma window for σ.
    sigma_floor:
        Lower bound on σ estimate (default 1e-9).
    """

    gamma: float
    k: float
    kappa: float
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
        self._sigma.observe(int(t_ns), float(mid))
        sigma = self._sigma.value(int(t_ns))
        if sigma is None:
            return []
        sigma = max(float(sigma), self.sigma_floor)
        tau = self._time_to_go(int(t_ns))
        # Cartea-Jaimungal reservation:
        #   r = s + (kappa - 2q) / (2 gamma)
        r = float(mid) + (self.kappa - 2.0 * float(inv)) / (2.0 * self.gamma)
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


__all__ = ["CarteaJaimungalQuoter"]
