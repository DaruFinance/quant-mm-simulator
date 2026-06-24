"""GLFT (Guéant-Lehalle-Fernandez-Tapia) quoting model.

Closed-form market-making quotes from Guéant, Lehalle & Fernandez-Tapia
(2013), "Dealing with the inventory risk: a solution to the
market-making problem".  The GLFT model extends Avellaneda-Stoikov
by giving the spread a market-order **arrival-intensity** term
(parameter A) in addition to the AS intensity-decay term (parameter
k).  This produces a slightly wider asymptotic spread that scales
with the volatility-to-intensity ratio.

Reservation
-----------
Same as Avellaneda-Stoikov:

    r(s, q, t) = s - q · γ · σ² · (T - t)

Spread
------
Asymptotic (long-horizon) GLFT half-spread:

    half_spread* = (1/γ) · ln(1 + γ/k)
                 + sqrt(σ² · γ / (2 · k · A)) · (1 + γ/k)^((1 + k/γ) / 2)

The first term is identical to the AS asymptotic spread term;
the second is the GLFT-specific addition that grows with the
σ²/(k·A) ratio.

Quoted prices
-------------
    bid_px = r - half_spread*
    ask_px = r + half_spread*

Sign / unit conventions match `AvellanedaStoikovQuoter`.

Notes
-----
This is the symmetric, infinite-horizon form.  The full GLFT paper
derives per-side asymptotic quotes that differ slightly between
bid and ask when |q| is large; the symmetric form here is the
common practitioner simplification and is sufficient for our
comparison library.  A later refinement could add the asymmetric
correction; until then this is documented as the chosen
simplification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from typing import List, Optional

from mmsim.ingest.lob import Book
from mmsim.quoter.base import Decision
from mmsim.quoter.refprice import top_mid
from mmsim.sim.loop import QuoteRequest

from ._vol import RollingSigma


@dataclass
class GLFTQuoter:
    """GLFT closed-form market maker.

    Parameters
    ----------
    gamma:
        Risk aversion (> 0).
    k:
        Market-order intensity decay parameter (> 0).
    A:
        Market-order baseline arrival intensity (> 0).
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
    A: float
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
        if self.A <= 0:
            raise ValueError("A must be > 0")
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
        # AS reservation
        r = float(mid) - float(inv) * self.gamma * sigma * sigma * tau
        # GLFT half-spread:
        #   (1/γ) ln(1 + γ/k) + sqrt(σ²γ / (2kA)) · (1+γ/k)^((1+k/γ)/2)
        as_term = (1.0 / self.gamma) * log(1.0 + self.gamma / self.k)
        glft_extra = (
            sqrt(sigma * sigma * self.gamma / (2.0 * self.k * self.A))
            * ((1.0 + self.gamma / self.k) ** ((1.0 + self.k / self.gamma) / 2.0))
        )
        spread_half = as_term + glft_extra
        bid_px = r - spread_half
        ask_px = r + spread_half
        return [
            QuoteRequest(side=+1, price=bid_px, size=self.size),
            QuoteRequest(side=-1, price=ask_px, size=self.size),
        ]


__all__ = ["GLFTQuoter"]
