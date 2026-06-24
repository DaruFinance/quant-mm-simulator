"""H3 — markout-gated quote-pull policy.

A thin quoter wrapper that pulls (cancels) quotes when an adverse-selection
filter (mmsim.quoter.adverse) signals the market is currently toxic. The
research claim (H3): gating on adverse signals improves OUT-OF-SAMPLE net
realised spread (net of the fills it forgoes) vs an always-on quoter.

This module provides:
  - ``GatedQuoter``  : wraps any base quoter + an adverse filter; returns
    [] (pull all quotes) when the filter fires, else the base quotes.
  - ``net_realised_spread`` : the H3 objective on a SimResult's fills,
    using the markout decomposition. "Net of forgone fills" is captured
    naturally: pulling quotes removes the fills, so the objective is the
    SUM of realised_spread over the fills that actually happened (more
    fills at good realised spread beats fewer; toxic fills drag it down).

The heavy WFO sweep that tunes the filter thresholds is run separately;
this module is the policy + objective, unit-smoke-able on the fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from mmsim.ingest.lob import Book, SnapshotEvent, TradeEvent
from mmsim.markout.engine import compute_markout


class GatedQuoter:
    """Wrap a base quoter with an adverse-selection gate.

    The base quoter is any callable ``(book, active_orders, t_ns) -> items``.
    The ``filt`` is any object with ``observe_book(book)``,
    ``observe_trade(trade)`` and ``is_adverse(t_ns) -> bool`` (the
    mmsim.quoter.adverse filters all satisfy this).

    Causal: the filter only consumes <=t state, exactly as the loop feeds
    it snapshots before the quoter call. Trades are fed via ``feed_trade``
    by a wrapping harness when available; in the bare sim loop the quoter
    sees only snapshots, so book-based filters (MicropriceDev, QueueImbalance)
    are the directly-wireable gates and trade-tape filters need the
    trade-feeding harness.
    """

    def __init__(self, base_quoter: Callable, filt):
        self.base = base_quoter
        self.filt = filt
        self.n_pulled = 0
        self.n_quoted = 0

    def __call__(self, book: Optional[Book], active_orders: List, t_ns: int):
        if book is not None:
            self.filt.observe_book(book)
        if self.filt.is_adverse(t_ns):
            self.n_pulled += 1
            return []  # pull all quotes
        self.n_quoted += 1
        return self.base(book, active_orders, t_ns)


def net_realised_spread(fills, snap_ts, snap_mid, horizon: str = "10s") -> float:
    """Sum of per-fill realised spread (the H3 objective). Higher is
    better. Net-of-forgone-fills is implicit: gating that removes toxic
    (negative realised-spread) fills raises the sum; gating that removes
    good fills lowers it."""
    if not fills:
        return 0.0
    mo = compute_markout(fills, snap_ts, snap_mid)
    col = getattr(mo, f"realised_spread_{horizon}")
    col = col[~np.isnan(col)]
    return float(np.sum(col)) if col.size else 0.0


def mean_realised_spread(fills, snap_ts, snap_mid, horizon: str = "10s") -> float:
    if not fills:
        return float("nan")
    mo = compute_markout(fills, snap_ts, snap_mid)
    col = getattr(mo, f"realised_spread_{horizon}")
    col = col[~np.isnan(col)]
    return float(np.mean(col)) if col.size else float("nan")


@dataclass
class GatingResult:
    n_fills_ungated: int
    n_fills_gated: int
    net_rs_ungated: float
    net_rs_gated: float
    mean_rs_ungated: float
    mean_rs_gated: float
    improvement: float          # gated - ungated (net realised spread)
    verdict: str                # informational until the WFO+perm gate runs


def compare_gating(stream, base_quoter_factory: Callable, filt_factory: Callable,
                   horizon: str = "10s") -> GatingResult:
    """Run base vs gated quoter on ``stream`` and report the H3 objective.

    NOTE: this is the in-sample policy comparison used for smoke + as the
    statistic the WFO/permutation gate evaluates out-of-sample. A
    bare-engine in-sample improvement is NOT the verdict; the locked
    verdict requires OOS + a permutation p below threshold."""
    from mmsim.sim.loop import run_sim
    from mmsim.sim.fills import QueueAwareFillModel
    from mmsim.ledger.writer import build_mid_timeline

    res_base = run_sim(stream, base_quoter_factory(), QueueAwareFillModel())
    gated = GatedQuoter(base_quoter_factory(), filt_factory())
    res_gated = run_sim(stream, gated, QueueAwareFillModel())

    snap_ts, snap_mid = build_mid_timeline(stream)
    net_u = net_realised_spread(res_base.fills, snap_ts, snap_mid, horizon)
    net_g = net_realised_spread(res_gated.fills, snap_ts, snap_mid, horizon)
    mean_u = mean_realised_spread(res_base.fills, snap_ts, snap_mid, horizon)
    mean_g = mean_realised_spread(res_gated.fills, snap_ts, snap_mid, horizon)
    return GatingResult(
        n_fills_ungated=len(res_base.fills), n_fills_gated=len(res_gated.fills),
        net_rs_ungated=net_u, net_rs_gated=net_g,
        mean_rs_ungated=mean_u, mean_rs_gated=mean_g,
        improvement=net_g - net_u,
        verdict="SCREENING (OOS+perm gate held)",
    )


__all__ = [
    "GatedQuoter", "net_realised_spread", "mean_realised_spread",
    "compare_gating", "GatingResult",
]
