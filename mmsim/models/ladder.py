"""Ladder quoting model.

Multi-level ladder around a reference price.  Wraps the `ladder()`
shape primitive inside the Quoter Protocol.

Formula
-------
For each level ``k = 0..n_levels-1``:

    offset_k = half_spread + k · step
    bid_k_px = ref(t) - offset_k
    ask_k_px = ref(t) + offset_k

All levels carry the same `size_per_level`.  Returns ``2·n_levels``
QuoteRequest items.

Use cases
---------
A maker that wants deeper passive exposure (capture mid-reversion
trades that walk past the BBO) without committing the full inventory
at the TOB.  The ladder absorbs more flow than a single-level pair
but suffers more adverse selection per fill on average.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from mmsim.ingest.lob import Book
from mmsim.quoter.base import Decision
from mmsim.quoter.refprice import top_mid
from mmsim.quoter.shapes import LadderSpec, ladder
from mmsim.sim.loop import QuoteRequest


@dataclass
class LadderQuoter:
    """Posts an n-level evenly-spaced ladder per side around ``ref``.

    Parameters
    ----------
    half_spread:
        Distance from ref to the innermost level (level 0).
    step:
        Spacing between consecutive levels.
    n_levels:
        Number of levels per side.  Must be >= 1.
    size_per_level:
        Quote size at every level (same on both sides).
    ref_fn:
        Pure book-only ref-price function.  Defaults to ``top_mid``.
    """

    half_spread: float
    step: float
    n_levels: int
    size_per_level: float
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
        spec = LadderSpec(
            half_spread=self.half_spread,
            step=self.step,
            n_levels=self.n_levels,
            size_per_level=self.size_per_level,
        )
        return list(ladder(spec, ref, inv))


__all__ = ["LadderQuoter"]
