"""Reference Quoter implementations.

These are the canonical built-ins consumed by every sim
verification.  All three implement the ``Quoter`` Protocol; all
three are deterministic and stateless except for ``BracketQuoter``
which keeps a snapshot counter for its every-Nth-snapshot taker fire.

They live here (not in tests/) because the formal Protocol is the
integration point.  Tests import them
from this module to keep the production surface and the test
surface aligned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from mmsim.ingest.lob import Book
from mmsim.sim.fills import TakerRequest
from mmsim.sim.loop import QuoteRequest

from .base import Decision


@dataclass
class ConstantQuoter:
    """Posts a fixed bid / ask of fixed size, regardless of book.
    The verification's "trivial constant quoter" — useful for
    contract tests since its output is independent of book state."""
    bid_price: float
    ask_price: float
    size: float = 0.001

    def quote(
        self,
        book: Optional[Book],
        inv: float,
        t_ns: int,
    ) -> List[Decision]:
        return [
            QuoteRequest(side=+1, price=self.bid_price, size=self.size),
            QuoteRequest(side=-1, price=self.ask_price, size=self.size),
        ]


@dataclass
class TopOfBookQuoter:
    """TOB-joining maker pair."""
    size: float = 0.001

    def quote(
        self,
        book: Optional[Book],
        inv: float,
        t_ns: int,
    ) -> List[Decision]:
        if book is None or book.best_bid is None or book.best_ask is None:
            return []
        return [
            QuoteRequest(side=+1, price=book.best_bid, size=self.size),
            QuoteRequest(side=-1, price=book.best_ask, size=self.size),
        ]


class BracketQuoter:
    """TOB makers + a small taker buy fired every ``taker_every``
    snapshots.  The reference taker quoter, as a Protocol
    impl.  Stateful: keeps a snapshot counter."""

    def __init__(
        self,
        maker_size: float = 0.001,
        taker_size: float = 0.0001,
        taker_every: int = 500,
    ):
        self.maker_size = maker_size
        self.taker_size = taker_size
        self.taker_every = taker_every
        self._snap_count = 0

    def quote(
        self,
        book: Optional[Book],
        inv: float,
        t_ns: int,
    ) -> List[Decision]:
        self._snap_count += 1
        if book is None or book.best_bid is None or book.best_ask is None:
            return []
        out: List[Decision] = [
            QuoteRequest(side=+1, price=book.best_bid, size=self.maker_size),
            QuoteRequest(side=-1, price=book.best_ask, size=self.maker_size),
        ]
        if self._snap_count % self.taker_every == 0:
            out.append(TakerRequest(side=+1, size=self.taker_size))
        return out


__all__ = ["ConstantQuoter", "TopOfBookQuoter", "BracketQuoter"]
