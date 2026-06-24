"""Ho-Stoll quoting model.

Classical 1981 inventory-driven dealer model from Ho & Stoll (1981),
"Optimal Dealer Pricing under Transactions and Return Uncertainty".

In contrast to Avellaneda-Stoikov (which derives a closed-form
spread + inventory skew jointly from a utility-optimization problem),
Ho-Stoll posts a wider spread when inventory is large (so a single
fill returns the inventory closer to flat) and a tighter spread
when inventory is small.  No volatility-scaling on the reservation
itself — just on the spread.

Formula
-------
Total spread is `α + β · |q| · σ²` where:
  - α is the constant baseline spread (≥ 0);
  - β > 0 scales the inventory penalty;
  - q is signed inventory;
  - σ² is the local return variance estimate.

The half-spread is half of the total; reservation centres on the
mid plus a linear inventory skew so the bid and ask end up at
asymmetric distances from mid (long inventory ⇒ both quotes
shifted DOWN):

    half_spread = (α + β · |q| · σ²) / 2
    skew         = -β · q · σ²          # signed; long inv ⇒ shift DOWN
    bid_px = s + skew - half_spread
    ask_px = s + skew + half_spread

Sigma is fed off the snapshot mid via the shared `RollingSigma`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from mmsim.ingest.lob import Book
from mmsim.quoter.base import Decision
from mmsim.quoter.refprice import top_mid
from mmsim.sim.loop import QuoteRequest

from ._vol import RollingSigma


@dataclass
class HoStollQuoter:
    """Ho-Stoll classical inventory dealer.

    Parameters
    ----------
    alpha:
        Baseline (constant) spread component, ≥ 0.
    beta:
        Inventory penalty coefficient, > 0.  Multiplies `|q| · σ²`
        in the spread term and (signed) `q · σ²` in the skew.
    size:
        Quote size per side.
    vol_window_ns:
        RollingSigma window for σ.
    sigma_floor:
        Lower bound on σ estimate (default 1e-9).
    """

    alpha: float
    beta: float
    size: float
    vol_window_ns: int
    sigma_floor: float = 1e-9
    _sigma: RollingSigma = field(init=False)

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")
        if self.size <= 0:
            raise ValueError("size must be > 0")
        if self.vol_window_ns <= 0:
            raise ValueError("vol_window_ns must be > 0")
        if self.sigma_floor < 0:
            raise ValueError("sigma_floor must be >= 0")
        self._sigma = RollingSigma(window_ns=int(self.vol_window_ns))

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
        var = sigma * sigma
        spread = self.alpha + self.beta * abs(float(inv)) * var
        half_spread = spread / 2.0
        skew = -self.beta * float(inv) * var
        center = float(mid) + skew
        bid_px = center - half_spread
        ask_px = center + half_spread
        return [
            QuoteRequest(side=+1, price=bid_px, size=self.size),
            QuoteRequest(side=-1, price=ask_px, size=self.size),
        ]


__all__ = ["HoStollQuoter"]
