"""Locked cost model.

Crypto defaults: taker 5 bp, maker 2 bp, slippage 2 bp, funding 1 bp/8h/leg.
All rates are fractions of notional (price * size). Funding accrues on
inventory carried across 8-hour funding boundaries; for a spot fixture
funding_rate = 0.

Maker vs taker economics (ECONOMICS FIX 2026-06-05)
---------------------------------------------------
A passive (maker) fill rests at a quoted price and is hit by an aggressor; it
fills AT ITS LIMIT PRICE.  Therefore a maker:
  * captures the quoted half-spread (priced into gross mark-to-mid PnL — the
    maker buys below / sells above mid),
  * pays a (small) exchange/clearing fee, optionally net of a venue rebate,
  * does NOT incur slippage (slippage is the cost of CROSSING the book, a
    taker concept), and
  * still bears adverse selection (the mid drifts against the fill), which is
    captured by the markout, not by an explicit cost line.
Only a taker (crossing) fill pays slippage.  ``slippage_for`` enforces this:
it returns 0 for maker fills unless ``maker_pays_slippage`` is set (which
reproduces the old, incorrect maker-as-taker accounting).

CME futures fee reality
-----------------------
CME charges a small per-CONTRACT fee (member/clearing schedules; broadly
~$0.10-0.40 all-in per side for liquid index/Treasury futures) and pays NO
maker rebate.  On these products' large per-contract notionals that fee is
sub-bp.  Example (ES, ~$0.25/side, multiplier 50, index ~4435 ->
notional ~ $221,750/contract): 0.25 / 221_750 = 1.13e-6 = 0.011 bp.  Treasuries
(ZN ~ $0.20/side on ~$110k notional) ~ 0.018 bp.  The locked default below uses
a deliberately conservative (high) 0.02 bp maker fee and NO rebate; the knob
``--maker-fee-bp`` / ``--maker-rebate-bp`` (see scripts/run_mm_full.py) lets a
run tighten it to the per-product reality.

Never run a costless backtest: a fill with zero cost is rejected at
construction unless the caller explicitly passes a zero model.
"""
from __future__ import annotations

from dataclasses import dataclass

_FUNDING_PERIOD_NS = 8 * 3600 * 1_000_000_000  # 8h in ns


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.0005      # 5 bp
    maker_fee: float = 0.0002      # 2 bp
    slippage: float = 0.0002       # 2 bp per fill (TAKER concept only)
    funding_per_8h: float = 0.0001  # 1 bp / 8h / leg
    # Spot capture has no funding; set funding_per_8h=0 for spot.
    is_perp: bool = False
    # Maker rebate (fraction of notional) CREDITED on a passive fill, e.g. on
    # crypto/equity venues that pay for liquidity.  CME futures pay NO rebate,
    # so the default is 0.  When non-zero, the maker net fee is
    # (maker_fee - maker_rebate); a rebate larger than the fee is a net credit.
    maker_rebate: float = 0.0
    # Whether a passive (maker) fill incurs slippage.  ECONOMICS FIX: a resting
    # limit order fills AT ITS LIMIT PRICE — it does NOT cross the book, so it
    # incurs NO slippage.  Slippage is a TAKER concept (you slip when you
    # aggress).  Default False (the correct maker economics); set True only to
    # reproduce the old (incorrect) maker-as-taker accounting.
    maker_pays_slippage: bool = False

    def fee_for(self, notional: float, is_maker: bool) -> float:
        """Exchange fee on a fill.  Maker fills net the rebate against the
        maker fee (CME rebate = 0 -> just the maker fee); a rebate exceeding
        the fee yields a negative fee (a net credit)."""
        if is_maker:
            return abs(notional) * (self.maker_fee - self.maker_rebate)
        return abs(notional) * self.taker_fee

    def slippage_for(self, notional: float, is_maker: bool = False) -> float:
        """Slippage cost.  Zero for a passive (maker) fill unless
        ``maker_pays_slippage`` is explicitly set — a resting order fills at
        its limit and never crosses the book."""
        if is_maker and not self.maker_pays_slippage:
            return 0.0
        return abs(notional) * self.slippage

    def funding_for(self, inv: float, price: float, dt_ns: int) -> float:
        """Funding paid on |inv| over dt_ns at the per-8h rate. Zero for
        spot (is_perp False or funding_per_8h 0)."""
        if not self.is_perp or self.funding_per_8h == 0.0 or dt_ns <= 0:
            return 0.0
        periods = dt_ns / _FUNDING_PERIOD_NS
        return abs(inv) * abs(price) * self.funding_per_8h * periods


# Locked default for the current spot fixture: real costs, no funding.
DEFAULT_COST_MODEL = CostModel(
    taker_fee=0.0005, maker_fee=0.0002, slippage=0.0002,
    funding_per_8h=0.0, is_perp=False,
)

__all__ = ["CostModel", "DEFAULT_COST_MODEL"]
