"""Fair-anchored quoting model.

Anchors on an `EWMAFairTracker`: the reference price is
the EWMA of the observed mid, fed once per quoter call.  The model
quotes symmetrically around the tracker.

Why EWMA-anchored
-----------------
Top-of-book mid jitters with every snapshot.  An EWMA smooths that
noise so the quoter doesn't churn cancel/replace on micro-moves;
the half-life sets the smoothing horizon.  A 1-second half-life
roughly tracks the trend-following mid; a 30-second half-life
fades quickly-mean-reverting wiggles.

Formula
-------
    f(t) = ewma(mid(t))                  # half-life decay updated each call
    bid_px = f(t) - half_spread
    ask_px = f(t) + half_spread

The tracker is fed with ``(t_ns, mid)`` every call.  Before the
tracker has seen any observation it returns None; before warmup
the quoter emits no orders.

Stateful: holds the EWMA tracker across calls.  Deterministic
given the same input sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from mmsim.ingest.lob import Book
from mmsim.quoter.base import Decision
from mmsim.quoter.refprice import EWMAFairTracker, top_mid
from mmsim.sim.loop import QuoteRequest


@dataclass
class FairAnchoredQuoter:
    """Posts one bid + one ask at ``ewma_fair(mid) ± half_spread``.

    Parameters
    ----------
    half_spread:
        Half-width of the quoted spread in price units.
    size:
        Quote size per side.
    half_life_ns:
        EWMA half-life in nanoseconds; controls smoothing horizon.
    """

    half_spread: float
    size: float
    half_life_ns: int
    _tracker: EWMAFairTracker = field(init=False)

    def __post_init__(self) -> None:
        self._tracker = EWMAFairTracker(half_life_ns=int(self.half_life_ns))

    def quote(
        self,
        book: Optional[Book],
        inv: float,
        t_ns: int,
    ) -> List[Decision]:
        mid = top_mid(book)
        if mid is not None:
            self._tracker.observe(int(t_ns), float(mid))
        f = self._tracker.value(int(t_ns))
        if f is None:
            return []
        return [
            QuoteRequest(side=+1, price=f - self.half_spread, size=self.size),
            QuoteRequest(side=-1, price=f + self.half_spread, size=self.size),
        ]


__all__ = ["FairAnchoredQuoter"]
