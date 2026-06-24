"""H2 — MM-objective walk-forward adapter (Avellaneda-Stoikov vs symmetric).

Compares the inventory-skewing Avellaneda-Stoikov quoter against a
symmetric quoter on costed replay, under a walk-forward geometry that
follows the qrf conventions (rolling in-sample window, disjoint forward
out-of-sample, step = OOS length, no overlap). The MM objective is:

    minimize OOS inventory CVaR-95  subject to  statistically-equal mean PnL.

NEW code = the MM-objective adapter and the WFO window driver. The qrf
harness (quant-research-framework/backtester:_walk_forward_impl,
run_robustness_tests) supplies the window-geometry + robustness-scenario
discipline this adapter mirrors; the full qrf-harness invocation with the
5-scenario robustness pass is run separately as a heavy run.

The model sims themselves come from the frozen engine
(mmsim.models.avellaneda_stoikov, mmsim.models.symmetric) wrapped as
Quoter-protocol objects; this module does NOT reimplement them.

The heavy multi-window sweep is run separately. The window driver +
objective are unit-smoke-able on the 60-min fixture (1-2 windows).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from mmsim.ledger.writer import build_ledger, build_mid_timeline
from mmsim.ledger.costs import CostModel, DEFAULT_COST_MODEL


class LevelSnappingQuoter:
    """Wrap a Quoter-protocol model so its quoted prices snap to the
    nearest VISIBLE book level on the correct side. Required because the
    queue-aware fill model (frozen engine) only tracks orders resting on a
    visible level; AS / symmetric models emit ref +/- half_spread at
    arbitrary prices. Snapping is a quoting-realism choice (we can only
    join an existing price level), documented and applied identically to
    both models so the A-S-vs-symmetric comparison stays fair."""

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _snap(price, side, book):
        levels = book.bids if side == +1 else book.asks
        if not levels:
            return None
        # nearest visible level to the requested price on this side
        best = min(levels, key=lambda lv: abs(lv[0] - price))
        return best[0]

    def quote(self, book, inv, t_ns):
        from mmsim.sim.loop import QuoteRequest
        items = self.model.quote(book, inv, t_ns)
        if book is None:
            return []
        out = []
        for it in items:
            # only QuoteRequests carry a price to snap; pass others through
            price = getattr(it, "price", None)
            side = getattr(it, "side", None)
            if price is None or side is None:
                out.append(it)
                continue
            snapped = self._snap(price, side, book)
            if snapped is None:
                continue
            out.append(QuoteRequest(side=side, price=snapped, size=it.size))
        return out


def wfo_windows(t0: int, t1: int, is_ns: int, oos_ns: int) -> List[Tuple[int, int, int, int]]:
    """qrf-style rolling windows: each window is (is_start, is_end, oos_start,
    oos_end) with oos_start == is_end and step == oos_ns (disjoint forward
    OOS, no overlap)."""
    windows = []
    is_start = t0
    while is_start + is_ns + oos_ns <= t1:
        is_end = is_start + is_ns
        oos_start = is_end
        oos_end = oos_start + oos_ns
        windows.append((is_start, is_end, oos_start, oos_end))
        is_start += oos_ns
    return windows


def slice_stream(stream, lo: int, hi: int):
    return [ev for ev in stream if lo <= ev.ts_ns < hi]


def inventory_cvar(ledger_rows: List[dict], alpha: float = 0.95) -> float:
    """CVaR-95 of the signed inventory path magnitude (tail inventory
    exposure). Higher = worse inventory control."""
    if not ledger_rows:
        return 0.0
    inv = np.array([abs(r["inv_after"]) for r in ledger_rows], dtype=np.float64)
    if inv.size == 0:
        return 0.0
    q = np.quantile(inv, alpha)
    tail = inv[inv >= q]
    return float(np.mean(tail)) if tail.size else float(q)


def mean_net_pnl(ledger_rows: List[dict]) -> float:
    if not ledger_rows:
        return 0.0
    return float(np.sum([r["net_pnl"] for r in ledger_rows]))


@dataclass
class WFOComparison:
    n_windows: int
    cvar_skew: List[float]       # A-S inventory CVaR per OOS window
    cvar_sym: List[float]        # symmetric inventory CVaR per OOS window
    pnl_skew: List[float]
    pnl_sym: List[float]
    mean_cvar_reduction: float   # mean(sym_cvar - skew_cvar); >0 favors A-S
    pnl_equal: bool              # |mean pnl diff| within tolerance
    verdict: str


def run_wfo_compare(stream, skew_quoter_factory: Callable, sym_quoter_factory: Callable,
                    *, is_minutes: float = 30.0, oos_minutes: float = 10.0,
                    cost_model: CostModel = DEFAULT_COST_MODEL,
                    pnl_tol: float = 0.5) -> WFOComparison:
    """Walk-forward A-S vs symmetric. SCREENING scale by default (short
    windows on the 60-min fixture). The locked verdict additionally
    requires a permutation p below threshold; that gate is part of the
    heavy run."""
    from mmsim.sim.loop import run_sim
    from mmsim.sim.fills import QueueAwareFillModel

    ts = [ev.ts_ns for ev in stream]
    t0, t1 = min(ts), max(ts)
    is_ns = int(is_minutes * 60 * 1e9)
    oos_ns = int(oos_minutes * 60 * 1e9)
    windows = wfo_windows(t0, t1, is_ns, oos_ns)

    cvar_s, cvar_y, pnl_s, pnl_y = [], [], [], []
    for (is_start, is_end, oos_start, oos_end) in windows:
        oos = slice_stream(stream, oos_start, oos_end)
        if not oos:
            continue
        # NOTE: IS window would tune the quoter params in the heavy run;
        # here the factories return fixed pre-set quoters (screening).
        res_skew = run_sim(oos, LevelSnappingQuoter(skew_quoter_factory()),
                           QueueAwareFillModel())
        res_sym = run_sim(oos, LevelSnappingQuoter(sym_quoter_factory()),
                          QueueAwareFillModel())
        led_skew = build_ledger(res_skew, oos, cost_model=cost_model).collected
        led_sym = build_ledger(res_sym, oos, cost_model=cost_model).collected
        cvar_s.append(inventory_cvar(led_skew))
        cvar_y.append(inventory_cvar(led_sym))
        pnl_s.append(mean_net_pnl(led_skew))
        pnl_y.append(mean_net_pnl(led_sym))

    reduction = float(np.mean(np.array(cvar_y) - np.array(cvar_s))) if cvar_s else 0.0
    pnl_diff = abs(float(np.mean(pnl_s)) - float(np.mean(pnl_y))) if pnl_s else 0.0
    pnl_equal = pnl_diff <= pnl_tol
    passed = reduction > 0 and pnl_equal
    return WFOComparison(
        n_windows=len(cvar_s), cvar_skew=cvar_s, cvar_sym=cvar_y,
        pnl_skew=pnl_s, pnl_sym=pnl_y,
        mean_cvar_reduction=reduction, pnl_equal=pnl_equal,
        verdict="SCREENING (perm gate held)" if passed else "SCREENING-NULL",
    )


__all__ = [
    "wfo_windows", "slice_stream", "inventory_cvar", "mean_net_pnl",
    "run_wfo_compare", "WFOComparison", "LevelSnappingQuoter",
]
