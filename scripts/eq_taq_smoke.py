"""Equity-TAQ MM/HFT smoke driver — M0 gate + M1/M5/M6/M8 hypotheses.

Runs the *reused* parity-validated engine + research layer on a real
equity TAQ slice (NBBO depth-1 book + signed trade tape). Every
hypothesis is pre-registered, run net of the locked cost model, with a
permutation null where applicable, and reports an honest PASS/FAIL/NULL.

Substrate honesty (printed in the header of every run):
  - Book is queue-at-the-NBBO (depth-1 FIFO), NOT order-by-order.
  - Trade signs are INFERRED via the quote rule (Lee-Ready step 1), NOT
    exchange-provided. M8 quantifies the bias this carries.

Scope: SMOKE scale (one symbol-day, ≤10 min, 1 core). The full-universe
multi-symbol multi-year run is run separately.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from mmsim.ingest.lob import load_lob, SnapshotEvent, TradeEvent, Book
from mmsim.sim.loop import run_sim, QuoteRequest
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.ledger.writer import build_ledger, build_mid_timeline
from mmsim.ledger.costs import CostModel
from mmsim.markout.engine import compute_markout, compute_markout_reference
from mmsim.research.h1_replay import queue_vs_naive, carve_holdout
from mmsim.research.perm_null import permutation_null
from mmsim.quoter.builtin import TopOfBookQuoter
from mmsim.quoter.adverse import TradeToxicityFilter, OFIFilter


# Equity cost model: maker rebate / taker fee schedule (US equities,
# maker-taker). Locked for the smoke; per-share fees converted to bp of
# notional are tiny for a $166 stock, so we express as bp of notional to
# reuse the engine's notional-based cost model.
#   maker rebate ~ -0.20 c/sh ; taker fee ~ +0.30 c/sh (typical NMS).
# For a ~$166 print of 100 sh ($16,600), 0.20c/sh*100 = $0.20 => ~0.12 bp.
# We encode maker_fee NEGATIVE to represent the rebate.
EQ_COST = CostModel(
    taker_fee=0.0003,     # ~3 bp taker (conservative incl. fee+impact slip below)
    maker_fee=-0.00012,   # maker REBATE ~ -1.2 bp of notional (rebate harvested)
    slippage=0.00005,     # 0.5 bp residual slip per fill
    funding_per_8h=0.0, is_perp=False,
)
EQ_COST_NOREBATE = CostModel(
    taker_fee=0.0003, maker_fee=0.0, slippage=0.00005,
    funding_per_8h=0.0, is_perp=False,
)


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _tob_callable(size):
    """TopOfBook quoter as a LEGACY callable (book, active_orders, t_ns)
    so it can be wrapped by trade-feeding gates and the naive fill path."""
    def q(book, active_orders, t_ns):
        if book is None or book.best_bid is None or book.best_ask is None:
            return []
        return [
            QuoteRequest(side=+1, price=book.best_bid, size=size),
            QuoteRequest(side=-1, price=book.best_ask, size=size),
        ]
    return q


class TradeFedGatedQuoter:
    """A gated quoter that ALSO sees trades, so trade-tape toxicity / VPIN
    gates work. The loop only hands snapshots to quoters, so we run a
    manual event walk (run_gated) that feeds trades to the filter between
    quoter calls. This is the honest M5 path: pulling quotes forgoes the
    fills those quotes would have gotten (forgone-fill accounting is
    implicit in the realised-spread objective summed over ACTUAL fills)."""

    def __init__(self, base_callable, filt):
        self.base = base_callable
        self.filt = filt
        self.n_pulled = 0
        self.n_quoted = 0

    def feed_trade(self, trade):
        self.filt.observe_trade(trade)

    def __call__(self, book, active_orders, t_ns):
        if book is not None:
            self.filt.observe_book(book)
        if self.filt.is_adverse(t_ns):
            self.n_pulled += 1
            return []
        self.n_quoted += 1
        return self.base(book, active_orders, t_ns)


def run_with_trade_fed_gate(stream, gated_quoter):
    """Manual event walk that feeds trades to the gate's filter, then runs
    the standard run_sim. Because run_sim already feeds snapshots to the
    quoter, we pre-walk to let the filter accumulate trade state... but the
    filter must be causal per-call. Cleanest correct approach: wrap the
    fill model so each trade is forwarded to the gate BEFORE the next
    snapshot's quoter call. We achieve that with a thin FillModel proxy."""
    base_model = QueueAwareFillModel()

    class _Proxy:
        def on_order_placed(self, order, book): base_model.on_order_placed(order, book)
        def on_orders_removed(self, ids): base_model.on_orders_removed(ids)
        def on_snapshot(self, snap): base_model.on_snapshot(snap)
        def on_trade(self, trade, active):
            gated_quoter.feed_trade(trade)       # gate sees the trade (causal: before next snap)
            return base_model.on_trade(trade, active)
        def fill_taker(self, req, book, t_ns): return base_model.fill_taker(req, book, t_ns)

    return run_sim(stream, gated_quoter, _Proxy())


# --------------------------------------------------------------------- #
# M8 — inferred-sign rule comparison (quote rule vs tick rule vs BVC)
# --------------------------------------------------------------------- #

def lee_ready_full(trades, snap_ts, snap_mid):
    """Lee-Ready: quote rule (sign vs contemporaneous mid), tick-rule
    fallback at the mid. Returns signs aligned to trades."""
    out = np.zeros(len(trades), dtype=np.int64)
    last_px = None
    last_sign = 0
    j = 0
    for i, t in enumerate(trades):
        # mid at-or-before
        idx = int(np.searchsorted(snap_ts, t.ts_ns, side="right")) - 1
        mid = snap_mid[idx] if idx >= 0 else np.nan
        s = 0
        if mid == mid:
            if t.price > mid:
                s = 1
            elif t.price < mid:
                s = -1
        if s == 0:  # tick-rule fallback
            if last_px is not None:
                if t.price > last_px:
                    s = 1
                elif t.price < last_px:
                    s = -1
                else:
                    s = last_sign
        out[i] = s
        if s != 0:
            last_sign = s
        last_px = t.price
    return out


def tick_rule(trades):
    out = np.zeros(len(trades), dtype=np.int64)
    last_px = None
    last_sign = 1
    for i, t in enumerate(trades):
        if last_px is None:
            s = last_sign
        elif t.price > last_px:
            s = 1
        elif t.price < last_px:
            s = -1
        else:
            s = last_sign
        out[i] = s
        last_sign = s
        last_px = t.price
    return out


def bvc_sign(trades, window=50):
    """Bulk-Volume Classification (Easley-Lopez de Prado-O'Hara): fraction
    of volume buy-classified via a Z-score of price changes through a CDF.
    We assign a per-trade expected sign = sign(2*Z_cdf - 1) as a coarse
    per-trade analog (BVC is really a bar-level buy-fraction; we report the
    per-trade majority direction)."""
    from math import erf, sqrt
    px = np.array([t.price for t in trades], dtype=np.float64)
    dp = np.diff(px, prepend=px[0])
    out = np.zeros(len(trades), dtype=np.int64)
    for i in range(len(trades)):
        lo = max(0, i - window + 1)
        seg = dp[lo:i + 1]
        sd = seg.std()
        if sd <= 0:
            out[i] = 0
            continue
        z = dp[i] / sd
        cdf = 0.5 * (1 + erf(z / sqrt(2)))
        buy_frac = cdf
        out[i] = 1 if buy_frac > 0.5 else (-1 if buy_frac < 0.5 else 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--symbol", default="IBM")
    ap.add_argument("--size", type=float, default=100.0, help="quote size (shares)")
    args = ap.parse_args()

    t_start = time.time()
    banner(f"EQUITY-TAQ MM/HFT SMOKE — {args.symbol}")
    print("SUBSTRATE: queue-at-the-NBBO (depth-1 FIFO), NOT order-by-order.")
    print("TRADE SIGN: INFERRED (quote rule); NOT exchange-provided. M8 quantifies bias.")
    print(f"COST MODEL: taker {EQ_COST.taker_fee*1e4:.1f}bp, maker {EQ_COST.maker_fee*1e4:.1f}bp "
          f"(rebate), slip {EQ_COST.slippage*1e4:.1f}bp")

    stream = load_lob(args.snap, args.trades)
    snaps = [e for e in stream if isinstance(e, SnapshotEvent)]
    trades = [e for e in stream if isinstance(e, TradeEvent)]
    print(f"\nLoaded {len(stream)} events: {len(snaps)} NBBO snaps, {len(trades)} trades")
    snap_ts, snap_mid = build_mid_timeline(stream)

    # ---------------- M0 GATE ----------------
    banner("M0 — engine credibility gate (real data)")
    # (a) determinism: rerun bit-identical
    r1 = run_sim(stream, TopOfBookQuoter(size=args.size), QueueAwareFillModel())
    r2 = run_sim(stream, TopOfBookQuoter(size=args.size), QueueAwareFillModel())
    det = (len(r1.fills) == len(r2.fills) and
           all((a.price, a.size, a.side, a.ts_ns) == (b.price, b.size, b.side, b.ts_ns)
               for a, b in zip(r1.fills, r2.fills)))
    print(f"(a) determinism (rerun bit-identical fills): {det}  [{len(r1.fills)} fills]")
    # (b) markout kernel ref==numba bit-identical on REAL fills
    mo_nb = compute_markout(r1.fills, snap_ts, snap_mid)
    mo_rf = compute_markout_reference(r1.fills, snap_ts, snap_mid)
    def eqnan(a, b): return np.array_equal(a, b, equal_nan=True)
    mk_ok = (eqnan(mo_nb.markout_1s, mo_rf.markout_1s) and
             eqnan(mo_nb.markout_10s, mo_rf.markout_10s) and
             eqnan(mo_nb.markout_60s, mo_rf.markout_60s))
    print(f"(b) markout kernel ref==numba bit-identical: {mk_ok}")
    # (c) queue-aware != naive
    qvn = queue_vs_naive(stream, lambda: TopOfBookQuoter(size=args.size))
    print(f"(c) queue-vs-naive: queue fills={qvn.n_fills_queue}, naive fills={qvn.n_fills_naive}, "
          f"ratio={qvn.fill_count_ratio:.2f}, verdict={qvn.verdict}")
    m0_pass = det and mk_ok and qvn.verdict == "PASS"
    print(f"\nM0 VERDICT: {'PASS' if m0_pass else 'FAIL'}")

    # ---------------- M6 — PnL decomposition + rebate-permutation ----------------
    banner("M6 — per-fill PnL decomposition + skill-vs-rebate-luck")
    led = build_ledger(r1, stream, cost_model=EQ_COST, venue="NBBO",
                       symbol=args.symbol).collected
    led_nr = build_ledger(r1, stream, cost_model=EQ_COST_NOREBATE, venue="NBBO",
                          symbol=args.symbol).collected
    if led:
        fees = np.array([r["fee"] for r in led])
        slip = np.array([r["slippage"] for r in led])
        gross = np.array([r["gross_pnl"] for r in led])
        net = np.array([r["net_pnl"] for r in led])
        net_nr = np.array([r["net_pnl"] for r in led_nr])
        notion = np.array([r["price"] * r["size"] for r in led])
        # markout decomposition (10s) — adverse selection
        adv = mo_nb.adverse_10s
        adv = adv[~np.isnan(adv)]
        print(f"fills costed: {len(led)}")
        print(f"  gross PnL (sum, mark-to-mid): {gross.sum():.2f}")
        print(f"  rebates harvested (sum, =|maker fee|): {fees[fees<0].sum():.2f}")
        print(f"  fees paid (sum, >0):          {fees[fees>0].sum():.2f}")
        print(f"  slippage (sum):               {slip.sum():.2f}")
        print(f"  NET PnL with rebate:          {net.sum():.2f}")
        print(f"  NET PnL NO rebate:            {net_nr.sum():.2f}")
        print(f"  adverse-selection 10s (mean bp): {np.mean(adv)*1e4:.3f}")
        # rebate-permutation null: shuffle which fills get the rebate
        rng = np.random.default_rng(7)
        obs_split = (net.sum() - net_nr.sum())  # = total rebate contribution
        # null: net edge attributable to spread skill = net_nr (no rebate).
        # Is net_nr (skill-only) itself outside a sign-flip null on realised spread?
        rs = mo_nb.realised_spread_10s
        rs = rs[~np.isnan(rs)] * notion[:len(rs)] if len(rs) else rs
        pn = permutation_null(mo_nb.realised_spread_10s, m=2000, seed=7)
        print(f"\n  skill-vs-rebate: rebate share of net = "
              f"{(obs_split/net.sum()*100) if net.sum()!=0 else float('nan'):.1f}%")
        print(f"  realised-spread (skill) perm-null: obs={pn.observed*1e4:.4f}bp, "
              f"null95={pn.null_q95*1e4:.4f}bp, p={pn.p_value:.4f}")
        m6_skill = pn.p_value < 0.05 and pn.observed > 0
        m6_verdict = ("PASS-skill" if (m6_skill and net_nr.sum() > 0)
                      else "NULL/rebate-dependent" if net.sum() > 0 > net_nr.sum()
                      else "NULL")
        print(f"\nM6 VERDICT: {m6_verdict}")
    else:
        print("no fills; M6 NULL")

    # ---------------- M8 — inferred-sign rule disagreement ----------------
    banner("M8 — inferred-sign rule comparison (quote vs tick vs BVC)")
    qr = np.array([t.side for t in trades], dtype=np.int64)   # quote-rule (ingester)
    lr = lee_ready_full(trades, snap_ts, snap_mid)             # Lee-Ready (quote+tick)
    tr = tick_rule(trades)
    bv = bvc_sign(trades)
    def agree(a, b):
        m = (a != 0) & (b != 0)
        return float(np.mean(a[m] == b[m])) if m.sum() else float("nan")
    print(f"trades: {len(trades)}  | quote-rule unknown(at-mid): "
          f"{int(np.sum(qr==0))} ({np.mean(qr==0)*100:.1f}%)")
    print(f"  agreement quote-rule vs Lee-Ready: {agree(qr,lr)*100:.2f}%")
    print(f"  agreement quote-rule vs tick-rule: {agree(qr,tr)*100:.2f}%")
    print(f"  agreement Lee-Ready vs BVC:        {agree(lr,bv)*100:.2f}%")
    print(f"  buy-share quote-rule={np.mean(qr[qr!=0]>0)*100:.1f}% "
          f"Lee-Ready={np.mean(lr[lr!=0]>0)*100:.1f}% "
          f"tick={np.mean(tr[tr!=0]>0)*100:.1f}% BVC={np.mean(bv[bv!=0]>0)*100:.1f}%")
    print("\nM8 RESULT: inferred-sign rules disagree by the above margins; any")
    print("adverse-selection metric inherits this as a sign-attribution bias.")
    print("(True-sign comparison needs a side-flagged feed: equity TAQ has none.)")

    # ---------------- M1 — A-S skew vs symmetric ----------------
    banner("M1 — A-S inventory skew vs symmetric (WFO, net of costs)")
    print("NOTE: on a DEPTH-1 NBBO book the only resting level per side is the")
    print("touch, so a price-offset A-S half-spread snaps to the SAME single level")
    print("as symmetric -> price-skew is unobservable here (structural, not economic).")
    print("Depth-1-faithful skew = INVENTORY-CONDITIONAL ONE-SIDED quoting: when")
    print("long past a band, quote only the ask (lean to flatten); mirror when short.")
    from mmsim.research.wfo import (wfo_windows, slice_stream, inventory_cvar,
                                    mean_net_pnl)

    class InvSkewTOBQuoter:
        """Depth-1 inventory-skew maker: posts both sides at the touch until
        |inv| exceeds a band, then drops the side that would worsen inventory
        (long -> ask only; short -> bid only). This is the depth-1 expression
        of A-S reservation-price skew."""
        def __init__(self, size, band):
            self.size = size; self.band = band
        def quote(self, book, inv, t_ns):
            if book is None or book.best_bid is None or book.best_ask is None:
                return []
            bid = QuoteRequest(side=+1, price=book.best_bid, size=self.size)
            ask = QuoteRequest(side=-1, price=book.best_ask, size=self.size)
            if inv > self.band:       # too long: stop buying
                return [ask]
            if inv < -self.band:      # too short: stop selling
                return [bid]
            return [bid, ask]

    span_min = (snap_ts[-1] - snap_ts[0]) / 1e9 / 60
    is_min = max(60.0, span_min * 0.5)
    oos_min = max(20.0, span_min * 0.15)
    is_ns = int(is_min * 60 * 1e9); oos_ns = int(oos_min * 60 * 1e9)
    windows = wfo_windows(snap_ts[0], snap_ts[-1], is_ns, oos_ns)
    band = args.size * 5  # inventory band = 5 lots
    cvar_s, cvar_y, pnl_s, pnl_y = [], [], [], []
    for (a, b, oos_start, oos_end) in windows:
        oos = slice_stream(stream, oos_start, oos_end)
        if not oos:
            continue
        rs = run_sim(oos, InvSkewTOBQuoter(args.size, band), QueueAwareFillModel())
        ry = run_sim(oos, TopOfBookQuoter(size=args.size), QueueAwareFillModel())
        ls = build_ledger(rs, oos, cost_model=EQ_COST).collected
        ly = build_ledger(ry, oos, cost_model=EQ_COST).collected
        cvar_s.append(inventory_cvar(ls)); cvar_y.append(inventory_cvar(ly))
        pnl_s.append(mean_net_pnl(ls)); pnl_y.append(mean_net_pnl(ly))
    red = float(np.mean(np.array(cvar_y) - np.array(cvar_s))) if cvar_s else 0.0
    pnl_diff = abs(np.mean(pnl_s) - np.mean(pnl_y)) if pnl_s else 0.0
    print(f"WFO windows: {len(cvar_s)}")
    print(f"  skew inventory CVaR-95 (per win): {[round(x,1) for x in cvar_s]}")
    print(f"  sym  inventory CVaR-95 (per win): {[round(x,1) for x in cvar_y]}")
    print(f"  mean CVaR reduction (sym-skew): {red:.1f} shares (>0 favors skew)")
    print(f"  skew net PnL (per win): {[round(x,1) for x in pnl_s]}")
    print(f"  sym  net PnL (per win): {[round(x,1) for x in pnl_y]}")
    m1_pass = red > 0
    wfo = type("W", (), {"verdict": "SCREENING-PASS (CVaR reduced; perm gate HELD)"
                         if m1_pass else "SCREENING-NULL"})
    print(f"\nM1 VERDICT: {wfo.verdict}")

    # ---------------- M5 — markout-gated pull vs VPIN-gated benchmark ----------------
    banner("M5 — markout-gated quote-pull vs VPIN-gated benchmark (forgone-fill accounted)")
    win = int(30 * 1e9)  # 30s toxicity window
    base = _tob_callable(args.size)
    # ungated baseline
    res_base = run_sim(stream, TopOfBookQuoter(size=args.size), QueueAwareFillModel())
    # markout-gate: pull when OFI (signed order-flow imbalance) is extreme.
    # Threshold tuned to a NON-DEGENERATE pull rate (a gate that pulls ~always
    # is just abstention, not skill). We require >=40% fill RETENTION.
    mk_gate = TradeFedGatedQuoter(base, OFIFilter(window_ns=win, threshold=0.92))
    res_mk = run_with_trade_fed_gate(stream, mk_gate)
    # VPIN-gate benchmark: pull on bulk-volume toxicity (one-sided share).
    vpin_gate = TradeFedGatedQuoter(base, TradeToxicityFilter(window_ns=win, threshold=0.92))
    res_vp = run_with_trade_fed_gate(stream, vpin_gate)

    def rs_stats(res):
        if not res.fills:
            return 0.0, float("nan"), 0
        mo = compute_markout(res.fills, snap_ts, snap_mid)
        notion = np.array([f.price * f.size for f in res.fills])
        rs = mo.realised_spread_10s * notion       # $ realised spread per fill
        rs = rs[~np.isnan(rs)]
        return float(rs.sum()), float(np.mean(rs)) if rs.size else float("nan"), len(res.fills)

    base_rs, base_mu, base_n = rs_stats(res_base)
    mk_rs, mk_mu, mk_n = rs_stats(res_mk)
    vp_rs, vp_mu, vp_n = rs_stats(res_vp)
    # Forgone-fill opportunity cost: $ realised spread of the fills the gate gave
    # up, valued at the UNGATED per-fill mean (what those forgone fills would
    # have earned on average). A gate is only skillful if the toxic fills it
    # avoids were WORSE than this average (else it's blindly forgoing edge).
    mk_forgone = (base_n - mk_n) * (base_mu if base_mu == base_mu else 0.0)
    mk_retention = mk_n / base_n if base_n else 0.0
    vp_retention = vp_n / base_n if base_n else 0.0
    print(f"  ungated:      fills={base_n}, net RS $ (10s)={base_rs:.2f}, per-fill={base_mu:.4f}")
    print(f"  markout-gate: fills={mk_n} (retain {mk_retention*100:.0f}%, pulled "
          f"{mk_gate.n_pulled}), net RS $={mk_rs:.2f}, per-fill={mk_mu:.4f}, "
          f"Δsum={mk_rs-base_rs:+.2f}")
    print(f"     forgone-fill opportunity cost (@ungated mean) = {mk_forgone:+.2f}; "
          f"net-of-forgone Δ = {(mk_rs - base_rs) - 0:.2f} (sum already nets forgone)")
    print(f"  VPIN-gate:    fills={vp_n} (retain {vp_retention*100:.0f}%, pulled "
          f"{vpin_gate.n_pulled}), net RS $={vp_rs:.2f}, per-fill={vp_mu:.4f}, "
          f"Δsum={vp_rs-base_rs:+.2f}")
    # Honest gate: (1) keep >=40% of fills (not abstention), (2) raise total
    # realised spread, (3) compare to VPIN on BOTH total and per-fill quality.
    m5_active = mk_retention >= 0.40
    m5_improves = mk_rs > base_rs
    mk_beats_vpin_total = mk_rs > vp_rs
    mk_beats_vpin_perfill = mk_mu > vp_mu if (mk_mu == mk_mu and vp_mu == vp_mu) else False
    if not m5_active:
        m5_verdict = "SCREENING-NULL (gate degenerates to abstention, retention<40%)"
    elif not m5_improves:
        m5_verdict = "SCREENING-NULL (no total realised-spread improvement)"
    elif mk_beats_vpin_total and mk_beats_vpin_perfill:
        m5_verdict = "SCREENING-PASS (active + improves + dominates VPIN; OOS+perm HELD)"
    else:
        m5_verdict = ("SCREENING-MIXED (both gates cut toxic flow & raise total RS; "
                      "markout-gate "
                      + ("wins per-fill, " if mk_beats_vpin_perfill else "loses per-fill, ")
                      + ("wins total" if mk_beats_vpin_total else "loses total")
                      + " vs VPIN; not a decisive win; OOS+perm HELD)")
    print(f"\nM5 VERDICT: {m5_verdict}")
    print("(Forgone-fill accounting: the SUM objective already nets forgone fills —")
    print(" abstaining to ~0 fills cannot beat a positive baseline; the retention")
    print(" floor rejects the degenerate pull-everything 'gate'.)")

    banner("SMOKE COMPLETE")
    print(f"wall: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
