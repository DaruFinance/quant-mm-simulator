"""Symmetric quoting model.

Fixed half-spread, ref-price-centred bid + ask of fixed size.  The
simplest non-trivial quoter in the library — a baseline that all
the other models compete against.

Formula
-------
Given a reference price s(t) (chosen by the caller's `ref_fn`):

    bid_px = s(t) - half_spread
    ask_px = s(t) + half_spread

with both sides at the same `size`.  No inventory skew, no adverse
filtering, no volatility scaling.  When the caller picks
``ref_fn = top_mid`` this collapses to a TOB-symmetric maker pair.

The model is stateless aside from holding the constructor params.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from mmsim.ingest.lob import Book
from mmsim.quoter.base import Decision
from mmsim.quoter.refprice import top_mid
from mmsim.sim.loop import QuoteRequest


@dataclass
class SymmetricQuoter:
    """Posts one bid + one ask at ``ref(book) ± half_spread``.

    Parameters
    ----------
    half_spread:
        Half-width of the quoted spread in price units.
    size:
        Quote size per side (same units as Book size).
    ref_fn:
        Pure book-only ref-price function.  Defaults to ``top_mid``.
    """

    half_spread: float
    size: float
    ref_fn: Callable[[Optional[Book]], Optional[float]] = top_mid

    def quote(
        self,
        book: Optional[Book],
        inv: float,
        t_ns: int,
    ) -> List[Decision]:
        ref = self.ref_fn(book)
        if ref is None:
            return []
        return [
            QuoteRequest(side=+1, price=ref - self.half_spread, size=self.size),
            QuoteRequest(side=-1, price=ref + self.half_spread, size=self.size),
        ]


__all__ = ["SymmetricQuoter"]
