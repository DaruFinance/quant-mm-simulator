"""Hedge engine.

Hedges MM inventory in a separate instrument (perp, spot, basket).
The engine is stateful, observes every Fill emitted by the primary
sim loop, and — when the absolute inventory crosses a configurable
threshold — fires a hedge taker trade against the supplied hedge
book that brings the *hedge-side* position back toward zero.

## What this exercises

The engine directly exercises the multi-leg ledger from the research
framework: every hedge fire emits a
``Fill`` on the *hedge instrument's* book with a distinct sentinel
``order_id`` (``HEDGE_ORDER_ID = -2``) so downstream PnL / risk
accounting can attribute hedge fills separately from primary
maker/taker fills.

## Causality

Every state mutation reads only the current ``Fill`` (for inventory
accumulation) or the current ``Book`` (for hedge decision + price).
Hedge decisions at time ``t`` use the inventory state ``inv_t``
which itself is a function of fills with ``ts_ns <= t``.  No future
fills, no future books.  Polluting fills or books past ``T`` cannot
change any hedge decision at ``T' <= T``.

## API

```
he = HedgeEngine(threshold=0.05, hedge_size_pct=1.0, instrument="perp")
for fill in primary_fills:
    he.observe_fill(fill)
    if he.should_hedge():
        hf = he.make_hedge(book=hedge_book, t_ns=fill.ts_ns)
        if hf is not None:
            hedge_log.append(hf)
```

Convention: ``hedge_size_pct=1.0`` flattens the full inventory in
one shot; ``hedge_size_pct=0.5`` shaves half.  Hedge fills are
emitted as taker fills (``is_maker=False``) against the hedge
instrument's BBO — this is the realistic operational model: you
cross the spread on the hedge venue because the hedge is a risk
operation, not an alpha-capture operation.

The engine tracks BOTH the primary inventory (driven by
``observe_fill``) AND the hedge-side cumulative position (driven by
``make_hedge``'s emitted fills), so the "net delta" any downstream
code wants is `engine.inv + engine.hedge_inv`.

## What's NOT here

- Hedge-instrument-specific BBO reconstruction — the caller passes
  in the hedge ``Book`` at the decision instant.
- Multi-leg basket hedging (one primary -> N hedge legs).  A single
  hedge leg is modelled here; multi-leg baskets are out of scope.
- Hedge cost modelling beyond the taker cross.  Funding rate etc
  is layered on by the PnL accounting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from mmsim.ingest.lob import Book
from mmsim.sim.loop import Fill


# Sentinel order_id for hedge fills (distinct from primary taker -1).
HEDGE_ORDER_ID: int = -2


@dataclass(frozen=True)
class HedgeDecision:
    """A record of one hedge fire decision.  Useful for reconciliation
    logs and the 5-event G3 dump."""
    t_ns: int
    pre_inv: float
    post_inv: float                  # primary inv unchanged by hedge fire
    hedge_size: float                # absolute size of the hedge cross
    hedge_side: int                  # +1 buy, -1 sell (the hedge fill's side)
    hedge_px: float
    pre_hedge_inv: float
    post_hedge_inv: float            # after this hedge fill is applied
    net_delta_pre: float             # pre_inv + pre_hedge_inv
    net_delta_post: float            # pre_inv + post_hedge_inv


class HedgeEngine:
    """Stateful hedge engine.

    Parameters
    ----------
    threshold : float
        Fire hedge when ``abs(inv) >= threshold``.  Units = primary
        instrument base units (BTC in DS-LOB-1H).
    hedge_size_pct : float
        Fraction of current ``|inv|`` to flatten on each fire.
        ``1.0`` flattens fully; ``0.5`` shaves half.  Must be in
        ``(0, 1]``.
    instrument : str
        Tag for the hedge instrument (``"perp"``, ``"spot"``, ...).
        Stored on the engine for log attribution; doesn't change
        engine behavior.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        hedge_size_pct: float = 1.0,
        instrument: str = "perp",
    ):
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if not (0.0 < hedge_size_pct <= 1.0):
            raise ValueError(
                f"hedge_size_pct must be in (0, 1], got {hedge_size_pct}")
        self.threshold = float(threshold)
        self.hedge_size_pct = float(hedge_size_pct)
        self.instrument = str(instrument)

        # Primary-side cumulative inv (driven by observe_fill).
        self.inv: float = 0.0
        # Hedge-side cumulative inv (driven by emitted hedge fills).
        self.hedge_inv: float = 0.0
        # All hedge fills emitted so far.
        self.hedge_fills: List[Fill] = []
        # All decisions (one per fire); useful for reconciliation logs.
        self.decisions: List[HedgeDecision] = []
        # Monotonically-increasing hedge fill id.
        self._next_fill_id: int = 0

    # ------------------------------------------------------------------ #
    # Read-only convenience
    # ------------------------------------------------------------------ #

    @property
    def net_delta(self) -> float:
        """Combined primary + hedge inventory (the risk we still carry)."""
        return self.inv + self.hedge_inv

    @property
    def n_hedge_fires(self) -> int:
        return len(self.hedge_fills)

    # ------------------------------------------------------------------ #
    # State updates
    # ------------------------------------------------------------------ #

    def observe_fill(self, fill: Fill) -> None:
        """Accumulate a primary-side fill into ``self.inv``.

        Bid fill (``side=+1``) → +size.  Ask fill (``side=-1``) → -size.
        Hedge fills SHOULD NOT be fed back in here — they're tracked
        in ``self.hedge_inv`` separately.  The engine has no way to
        distinguish them from primary fills based on the Fill record
        alone (other than the ``HEDGE_ORDER_ID`` sentinel), so the
        caller is responsible for not double-counting.
        """
        if fill.order_id == HEDGE_ORDER_ID:
            # Defensive: silently ignore — caller wired things wrong.
            return
        self.inv += float(fill.size) * float(fill.side)

    def should_hedge(self) -> bool:
        """True if the engine's net (primary + hedge) inventory has
        crossed the threshold.  This is the operational definition:
        we hedge what's left after prior hedge fills, not the gross
        primary inventory.  Re-using gross-inv here would cause the
        engine to keep firing even after the position is already
        hedged-flat."""
        return abs(self.net_delta) >= self.threshold

    def make_hedge(
        self,
        book: Optional[Book],
        t_ns: int,
    ) -> Optional[Fill]:
        """Emit a hedge fill that drives ``net_delta`` back toward zero.

        Returns the emitted ``Fill`` (and appends to ``self.hedge_fills``
        + ``self.decisions``) — or ``None`` when:
          - ``should_hedge()`` is False (nothing to do);
          - the supplied ``book`` is missing the relevant best price.

        Hedge semantics: if ``net_delta > 0`` (we are net long), we
        SELL the hedge (hedge side = -1) at the hedge book's best
        bid.  If ``net_delta < 0`` (net short), we BUY (hedge side =
        +1) at the hedge book's best ask.

        Taker semantics: ``is_maker=False`` always.  The hedge is a
        cross-the-spread operation by design.
        """
        if not self.should_hedge():
            return None
        if book is None:
            return None

        nd = self.net_delta
        size = abs(nd) * self.hedge_size_pct
        if size <= 0.0:
            return None

        if nd > 0.0:
            # Net long -> sell hedge.  Hedge fill side = -1; price = best bid.
            hedge_side = -1
            px = book.best_bid
        else:
            hedge_side = +1
            px = book.best_ask
        if px is None:
            return None

        # Build the fill record.
        fill = Fill(
            fill_id=self._next_fill_id,
            order_id=HEDGE_ORDER_ID,
            ts_ns=int(t_ns),
            price=float(px),
            size=float(size),
            side=int(hedge_side),
            is_maker=False,
        )
        # Update hedge-side state.
        pre_hedge_inv = self.hedge_inv
        signed = float(size) * float(hedge_side)
        self.hedge_inv += signed
        self.hedge_fills.append(fill)
        self._next_fill_id += 1

        self.decisions.append(HedgeDecision(
            t_ns=int(t_ns),
            pre_inv=self.inv,
            post_inv=self.inv,
            hedge_size=float(size),
            hedge_side=int(hedge_side),
            hedge_px=float(px),
            pre_hedge_inv=float(pre_hedge_inv),
            post_hedge_inv=float(self.hedge_inv),
            net_delta_pre=float(self.inv + pre_hedge_inv),
            net_delta_post=float(self.inv + self.hedge_inv),
        ))
        return fill


__all__ = [
    "HEDGE_ORDER_ID",
    "HedgeDecision",
    "HedgeEngine",
]
