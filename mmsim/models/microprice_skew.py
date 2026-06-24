"""Microprice-skew quoting model.

Uses the Stoikov `microprice()` as the reference price; bid/ask are
symmetric around it.  When the book is heavy on the bid (likely
upward drift), the microprice sits above the simple mid and our
quotes shift up — quoting slightly more aggressively to sell.

Formula
-------
    s(t) = microprice(book)
         = mid + ((bid_sz - ask_sz)/(bid_sz + ask_sz)) · half_book_spread

    bid_px = s(t) - half_spread
    ask_px = s(t) + half_spread

The model derives its short-term price view entirely from the queue
imbalance at TOB.  Stateless — every call computes the microprice
from the current book.

Falls back to None (no quotes) when the book has no liquidity on a
side or is missing entirely (warmup).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from mmsim.ingest.lob import Book
from mmsim.quoter.base import Decision
from mmsim.quoter.refprice import microprice
from mmsim.sim.loop import QuoteRequest


@dataclass
class MicropriceSkewQuoter:
    """Posts one bid + one ask at ``microprice ± half_spread``.

    Parameters
    ----------
    half_spread:
        Half-width of the quoted spread in price units.
    size:
        Quote size per side.
    """

    half_spread: float
    size: float

    def quote(
        self,
        book: Optional[Book],
        inv: float,
        t_ns: int,
    ) -> List[Decision]:
        ref = microprice(book)
        if ref is None:
            return []
        return [
            QuoteRequest(side=+1, price=ref - self.half_spread, size=self.size),
            QuoteRequest(side=-1, price=ref + self.half_spread, size=self.size),
        ]


__all__ = ["MicropriceSkewQuoter"]
