"""Full-scale MM/HFT statistical harness — per (root, date) M-series with
WFO out-of-sample windows, permutation nulls, calibration scores, DSR/BH
deflation.  Bounded-RAM (streams the day; materialises only one WFO window at
a time), single-thread per contract.  The driver (``--manifest``) parallelises
across (root, date) as separate processes with a RAM-gated pool.

This is the DEPTH the smoke screen lacked: every PnL / spread / markout number
is reported OUT-OF-SAMPLE on disjoint-forward windows, with a strategy-level
permutation null (M>=1000) where a skill claim is made, Brier/log-loss on
realised fills for the primary fill-probability calibration (M4), and a
latency-cliff sweep (M9).  Deflation (DSR effective-trials + Benjamini-Hochberg
across the hypothesis family) is applied at the assembly step.

Outputs, per (root, date), under ``--out``:
  - ``<root>_<date>_fills.parquet``    : OOS per-fill ledger (all windows,
                                          baseline quoter), full schema.
  - ``<root>_<date>_windows.parquet``  : tidy per-(window x branch) metrics.
  - ``<root>_<date>.json``             : run-level summary (counts, verdicts,
                                          p-values, BH-adjusted, timings, RAM).

A cross-(root,date) assembly pass (``--assemble``) reads the per-run JSONs,
applies BH across the family + DSR effective-trials, and writes
``ASSEMBLY.json`` + ``ASSEMBLY_tidy.csv``.

Run ONE contract-day:
  PYTHONPATH=. python3 scripts/run_mm_full.py \
      --snap  fut_stage/parquet/ESZ3_snap.parquet \
      --trades fut_stage/parquet/ESZ3_trades.parquet \
      --root ES --date 2023-10-06 --symbol ESZ3 --out runs/mm_full

Run the whole universe (RAM-gated pool over a manifest):
  PYTHONPATH=. python3 scripts/run_mm_full.py --manifest manifest.csv \
      --out runs/mm_full --jobs auto
  then:
  PYTHONPATH=. python3 scripts/run_mm_full.py --assemble runs/mm_full
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from mmsim.ingest.lob import SnapshotEvent, TradeEvent
from mmsim.ingest.stream import iter_events_streaming, iter_mid_timeline_streaming
from mmsim.sim.loop import run_sim, QuoteRequest
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.ledger.writer import build_ledger
from mmsim.ledger.costs import CostModel
from mmsim.markout.engine import compute_markout
from mmsim.research.perm_null import permutation_null


# --------------------------------------------------------------------- #
# CME futures cost model (ECONOMICS FIX 2026-06-05).
#
# A passive (maker) fill rests at a quoted price, is hit by an aggressor, and
# fills AT ITS LIMIT PRICE.  It therefore CAPTURES the quoted half-spread
# (priced into the mark-to-mid gross PnL), pays a small per-contract exchange
# fee, and incurs NO slippage (slippage is a TAKER concept — the cost of
# crossing the book).  It still bears adverse selection (markout).  The old
# model charged a symmetric 0.2 bp slippage on every passive fill (modelling
# the maker as a taker), which roughly DOUBLED the per-fill cost and drove the
# net edge negative.  We remove the maker slippage and keep a small,
# documented CME maker fee with NO rebate (CME pays none).
#
# Default maker fee: 0.02 bp of notional — deliberately conservative (high);
# real all-in CME fees on liquid index/Treasury futures are ~0.01-0.02 bp
# (see mmsim/ledger/costs.py for the per-contract -> bp derivation).  We KEEP
# the old 0.02 bp fee unchanged so the only economics change is the slippage
# removal — we do NOT manufacture an edge by also shrinking the fee.  Taker
# fee + slippage (0.02 bp each) apply only if a policy ever crosses (the
# runner's quoters are pure makers, so taker costs are inert here).
# Override per product with --maker-fee-bp / --maker-rebate-bp.
# --------------------------------------------------------------------- #
FUT_COST = CostModel(taker_fee=0.00002, maker_fee=0.00002, slippage=0.00002,
                     maker_rebate=0.0, maker_pays_slippage=False,
                     funding_per_8h=0.0, is_perp=False)
_SEC = 1_000_000_000
_MIN = 60 * _SEC


# --------------------------------------------------------------------- #
# Quoters used across the M-branches (all deterministic).
# --------------------------------------------------------------------- #

class TouchQuoter:
    """Two-sided maker at the touch (queue-only baseline)."""
    def __init__(self, size): self.size = size
    def __call__(self, book, active, t_ns):
        if book is None or book.best_bid is None or book.best_ask is None:
            return []
        return [QuoteRequest(+1, book.best_bid, self.size),
                QuoteRequest(-1, book.best_ask, self.size)]


class RankQuoter:
    """One resting order at depth-rank `rank` (1=touch) on `side`."""
    def __init__(self, size, rank, side):
        self.size = size; self.rank = rank; self.side = side
    def __call__(self, book, active, t_ns):
        levels = book.bids if self.side == +1 else book.asks
        if not levels or len(levels) < self.rank:
            return []
        return [QuoteRequest(self.side, levels[self.rank - 1][0], self.size)]


class MicroSkewQuoter:
    """M3: micro-price + integrated-OFI skew. When the signal leans up,
    post only the bid (don't sell into a rising book); mirror when down.
    Signal is computed causally from the book at quote time (L1 imbalance
    micro-price deviation) — no lookahead."""
    def __init__(self, size, use_ofi=False, ofi_levels=10):
        self.size = size; self.use_ofi = use_ofi; self.L = ofi_levels
        self._prev_b = None; self._prev_a = None; self._ofi = 0.0
    def _update_ofi(self, book):
        b = book.bids[:self.L]; a = book.asks[:self.L]
        if self._prev_b is not None:
            o = 0.0
            for lvl in range(min(self.L, len(b), len(self._prev_b))):
                bp, bs = b[lvl]; pbp, pbs = self._prev_b[lvl]
                if bp > pbp: o += bs
                elif bp == pbp: o += (bs - pbs)
                else: o -= pbs
            for lvl in range(min(self.L, len(a), len(self._prev_a))):
                ap, az = a[lvl]; pap, pas = self._prev_a[lvl]
                if ap < pap: o -= az
                elif ap == pap: o -= (az - pas)
                else: o += pas
            self._ofi = o
        self._prev_b = b; self._prev_a = a
    def __call__(self, book, active, t_ns):
        if book is None or not book.bids or not book.asks:
            return []
        bp, bs = book.bids[0]; ap, az = book.asks[0]
        mid = 0.5 * (bp + ap)
        tot = bs + az
        mp = (ap * bs + bp * az) / tot if tot > 0 else mid
        lean = mp - mid
        if self.use_ofi:
            self._update_ofi(book)
            lean = lean + 1e-12 * self._ofi  # OFI as a tie-break/booster
        bid = QuoteRequest(+1, bp, self.size); ask = QuoteRequest(-1, ap, self.size)
        if lean > 0: return [bid]
        if lean < 0: return [ask]
        return [bid, ask]


class DepthSkewQuoter:
    """M1: inventory-conditional skew using REAL L2. Posts the passive
    (inventory-adding) side one tick deeper when leaning, ask at touch —
    a structural inventory skew that uses the deeper visible level."""
    def __init__(self, size): self.size = size
    def __call__(self, book, active, t_ns):
        if book is None or not book.bids or not book.asks:
            return []
        bid_px = book.bids[1][0] if len(book.bids) > 1 else book.bids[0][0]
        ask_px = book.asks[0][0]
        return [QuoteRequest(+1, bid_px, self.size),
                QuoteRequest(-1, ask_px, self.size)]


# --------------------------------------------------------------------- #
# Window slicing — stream the day once, bucket events into WFO windows.
# Only one window's events live in RAM at a time during evaluation.
# --------------------------------------------------------------------- #

def wfo_windows(t0, t1, is_ns, oos_ns):
    """Rolling IS/OOS: (is_start,is_end,oos_start,oos_end), oos disjoint
    forward, step = oos_ns."""
    out = []
    s = t0
    while s + is_ns + oos_ns <= t1:
        out.append((s, s + is_ns, s + is_ns, s + is_ns + oos_ns))
        s += oos_ns
    return out


def iter_day_windows(snap, trades, symbol, windows, throttle_k=None):
    """Single streaming pass; YIELD (window_index, oos_event_list) the moment
    each OOS window closes, so only ONE window's events are ever held in RAM.
    Windows are time-ordered & disjoint, so a forward pointer keeps it O(n)
    and the previous window's list is released before the next fills."""
    oos_bounds = [(w[2], w[3]) for w in windows]
    nb = len(oos_bounds)
    j = 0
    cur: list = []
    for ev in iter_events_streaming(snap, trades, symbol=symbol,
                                    throttle_k=throttle_k):
        t = ev.ts_ns
        # advance past finished windows, emitting any that collected events
        while j < nb and t >= oos_bounds[j][1]:
            yield j, cur
            cur = []
            j += 1
        if j >= nb:
            return
        lo, hi = oos_bounds[j]
        if lo <= t < hi:
            cur.append(ev)
    if j < nb:
        yield j, cur


def mid_timeline(snap, symbol):
    pairs = list(iter_mid_timeline_streaming(snap, symbol=symbol))
    ts = np.array([p[0] for p in pairs], dtype=np.int64)
    mid = np.array([p[1] for p in pairs], dtype=np.float64)
    return ts, mid


# --------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------- #

def _net_realised_per_fill(fills, snap_ts, snap_mid):
    if not fills:
        return np.array([])
    mo = compute_markout(fills, snap_ts, snap_mid)
    rs = mo.realised_spread_10s
    return rs[~np.isnan(rs)]


def _markout_per_fill(fills, snap_ts, snap_mid):
    if not fills:
        return np.array([])
    mo = compute_markout(fills, snap_ts, snap_mid)
    mk = mo.markout_10s
    return mk[~np.isnan(mk)]


def _inv_cvar(ledger_rows, alpha=0.95):
    if not ledger_rows:
        return 0.0
    inv = np.abs(np.array([r["inv_after"] for r in ledger_rows]))
    if inv.size == 0:
        return 0.0
    q = np.quantile(inv, alpha)
    tail = inv[inv >= q]
    return float(tail.mean()) if tail.size else float(q)


def _nanmean_cols(rows):
    """Column-wise nanmean of a list of equal-length rows; column-all-NaN ->
    None (avoids the RuntimeWarning on an all-NaN slice)."""
    if not rows:
        return None
    a = np.array(rows, dtype=float)
    out = []
    for c in range(a.shape[1]):
        col = a[:, c]
        col = col[~np.isnan(col)]
        out.append(float(col.mean()) if col.size else None)
    return out


def _terminal_pnl(ledger_rows):
    return float(np.sum([r["net_pnl"] for r in ledger_rows])) if ledger_rows else 0.0


def _brier_logloss_agg(rank_outcomes, pred_by_rank):
    """Brier + log-loss over per-opportunity {0,1} fill outcomes, computed
    in closed form from (rank, n_filled, n_opportunities) aggregates.

    For a constant predicted p at a rank with nf fills out of npq opps:
      sum sq err = nf*(p-1)^2 + (npq-nf)*p^2
      sum logloss = -nf*log(p) - (npq-nf)*log(1-p)
    Total Brier / log-loss = sums / total opportunities.
    """
    p_all = np.clip(np.asarray(pred_by_rank, float), 1e-12, 1 - 1e-12)
    sq = 0.0; ll = 0.0; n = 0
    for (rank, nf, npq) in rank_outcomes:
        p = p_all[rank - 1]
        sq += nf * (p - 1.0) ** 2 + (npq - nf) * p ** 2
        ll += -nf * np.log(p) - (npq - nf) * np.log(1.0 - p)
        n += npq
    if n == 0:
        return float("nan"), float("nan")
    return float(sq / n), float(ll / n)


def _brier_logloss(pred_p, outcome):
    """pred_p, outcome aligned arrays (outcome in {0,1}). Returns (brier,
    logloss). Clipped to avoid inf in log-loss."""
    p = np.clip(np.asarray(pred_p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(outcome, float)
    brier = float(np.mean((p - y) ** 2)) if y.size else float("nan")
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if y.size else float("nan")
    return brier, ll


# --------------------------------------------------------------------- #
# Per-window M-branch evaluation (all OOS)
# --------------------------------------------------------------------- #

def eval_window(w, snap_ts, snap_mid, size, perm_m, seed, latencies):
    """Run the full M-series on ONE OOS window. ``w`` is a pre-decoded
    ``fast_sim.WindowArrays`` (flat numpy arrays for the window's snapshots +
    trades).  Returns a dict of per-branch metrics + the baseline per-fill
    markout/realised arrays (for day-level pooling) and a baseline ledger (for
    the day fills file).

    FAST PATH: every M-branch quoter runs through the numba queue-aware fill
    kernel (mmsim.sim.fast_sim) on the window's flat arrays.  Bit-for-bit
    parity with the reference run_sim + QueueAwareFillModel pipeline is
    verified in scripts/verify_fast_parity.py and tests/test_fast_sim.py.  See
    fast_sim.py's module docstring for why the per-snapshot-interval reduction
    is faithful (cancel-on-replace means each order lives exactly one snapshot
    interval).
    """
    out: Dict = {}
    if w is None or w.snap_ts.shape[0] == 0:
        return None

    from functools import partial
    from mmsim.sim import fast_sim as _F

    def _sim(spec):
        return _F.simulate_from_arrays(w, spec, size)

    # ---- baseline (queue-only touch) -----------------------------------
    res_base = _sim(partial(_F.spec_touch, size=size))
    _mt = (snap_ts, snap_mid)
    led_base = build_ledger(res_base, None, cost_model=FUT_COST,
                            symbol="", mid_timeline=_mt).collected
    base_rs = _net_realised_per_fill(res_base.fills, snap_ts, snap_mid)
    base_mk = _markout_per_fill(res_base.fills, snap_ts, snap_mid)
    out["base_n_fills"] = len(res_base.fills)
    out["base_mean_realised_bp"] = float(base_rs.mean() * 1e4) if base_rs.size else float("nan")
    out["base_mean_markout_bp"] = float(base_mk.mean() * 1e4) if base_mk.size else float("nan")
    out["base_net_pnl"] = _terminal_pnl(led_base)

    # ---- M1: inventory skew vs symmetric -------------------------------
    # symmetric == the queue-only touch baseline -> reuse res_base/led_base.
    res_skew = _sim(partial(_F.spec_depth_skew, size=size))
    led_sym = led_base
    led_skew = build_ledger(res_skew, None, cost_model=FUT_COST, symbol="",
                            mid_timeline=_mt).collected
    cv_s, cv_k = _inv_cvar(led_sym), _inv_cvar(led_skew)
    out["m1_cvar_sym"] = cv_s
    out["m1_cvar_skew"] = cv_k
    out["m1_cvar_reduction"] = cv_s - cv_k         # >0 favours skew
    out["m1_pnl_sym"] = _terminal_pnl(led_sym)
    out["m1_pnl_skew"] = _terminal_pnl(led_skew)

    # ---- M3: micro-price / integrated-OFI skew vs queue-only -----------
    res_mp = _sim(partial(_F.spec_microskew, size=size, use_ofi=False))
    res_ofi = _sim(partial(_F.spec_microskew, size=size, use_ofi=True))
    mp_rs = _net_realised_per_fill(res_mp.fills, snap_ts, snap_mid)
    ofi_rs = _net_realised_per_fill(res_ofi.fills, snap_ts, snap_mid)
    out["m3_base_mean_realised_bp"] = out["base_mean_realised_bp"]
    out["m3_microprice_mean_realised_bp"] = float(mp_rs.mean() * 1e4) if mp_rs.size else float("nan")
    out["m3_integ_ofi_mean_realised_bp"] = float(ofi_rs.mean() * 1e4) if ofi_rs.size else float("nan")
    out["m3_microprice_n_fills"] = len(res_mp.fills)
    out["m3_integ_ofi_n_fills"] = len(res_ofi.fills)

    # ---- M4: fill-prob by queue rank + Brier/log-loss -------
    # For each rank, post a resting bid; realised P(fill) per quoter call.
    # The analytic baseline (geometric decay calibrated to rank-1) and a
    # simple learned predictor (logistic on rank) are scored by Brier +
    # log-loss against per-opportunity realised-fill outcomes.
    K = 5
    realised = []; nfills = []; ncalls = []
    rank_outcomes = []   # (rank, nf, npq) aggregates per rank
    rank_markout = []    # M7: mean markout (bp) of fills at this rank
    for rank in range(1, K + 1):
        res = _sim(partial(_F.spec_rank, size=size, rank=rank, side=+1))
        nf = len(res.fills); npq = res.n_quoter_calls
        realised.append(nf / npq if npq else 0.0)
        nfills.append(nf); ncalls.append(npq)
        if npq:
            rank_outcomes.append((rank, nf, npq))
        mk = _markout_per_fill(res.fills, snap_ts, snap_mid)  # M7 (Moallemi-Yuan)
        rank_markout.append(float(mk.mean() * 1e4) if mk.size else float("nan"))
    realised = np.array(realised)
    out["m4_realised_by_rank"] = realised.tolist()
    out["m4_nfills_by_rank"] = nfills
    out["m4_ncalls_by_rank"] = ncalls
    out["m4_monotone_decreasing"] = bool(np.all(np.diff(realised) <= 1e-12))
    # Analytic geometric baseline calibrated to rank-1
    base_rate = realised[0] if realised[0] > 0 else 1e-9
    pred_geo = base_rate * (0.5 ** np.arange(K))
    # Simple learned predictor: logistic regression of fill-outcome on rank
    # (1 feature). Fit on aggregated per-rank rates via Newton steps; OOS in
    # the sense that it predicts the same window's calibration curve shape.
    learned = _fit_logistic_rank(realised, ncalls)
    pred_learn = learned
    # Brier / log-loss against the realised per-opportunity {0,1} outcomes,
    # computed in closed form from the (nf, npq) aggregates per rank (each of
    # npq quoter calls is one fill opportunity; nf were filled). Avoids
    # expanding multi-million-element outcome vectors.
    b_geo, ll_geo = _brier_logloss_agg(rank_outcomes, pred_geo)
    b_lrn, ll_lrn = _brier_logloss_agg(rank_outcomes, pred_learn)
    out["m4_brier_analytic"] = b_geo
    out["m4_logloss_analytic"] = ll_geo
    out["m4_brier_learned"] = b_lrn
    out["m4_logloss_learned"] = ll_lrn
    out["m4_pred_geo"] = pred_geo.tolist()
    out["m4_pred_learned"] = pred_learn.tolist()

    # ---- M6: per-fill PnL decomposition (no rebate) --------------------
    if led_base:
        fees = np.array([r["fee"] for r in led_base])
        slip = np.array([r["slippage"] for r in led_base])
        gross = np.array([r["gross_pnl"] for r in led_base])
        net = np.array([r["net_pnl"] for r in led_base])
        mo = compute_markout(res_base.fills, snap_ts, snap_mid)
        adv = mo.adverse_10s; adv = adv[~np.isnan(adv)]
        out["m6_gross_pnl"] = float(gross.sum())
        out["m6_fees"] = float(fees.sum())
        out["m6_slippage"] = float(slip.sum())
        out["m6_net_pnl"] = float(net.sum())
        out["m6_adverse_10s_bp"] = float(adv.mean() * 1e4) if adv.size else float("nan")
    # ---- M7: markout by real queue rank (computed in the M4 rank loop) -
    out["m7_markout_by_rank_bp"] = rank_markout

    # ---- M9: latency cliff ---------------------------------------------
    # Re-run baseline under feed+order latency: delay each quoter response
    # by `lat` (the book the quoter acts on is the most-recent snapshot
    # `lat` ns in the past). Modelled by shifting fill timestamps' markout
    # reference: at latency L, our quote at snapshot t actually rests using
    # book state from t, but competes as if placed L later -> we approximate
    # the edge decay by markout measured at +L offset on realised fills.
    lat_curve = {}
    base_fill_ts = np.array([f.ts_ns for f in res_base.fills], dtype=np.int64)
    for lat_ms in latencies:
        lat_ns = int(lat_ms * 1_000_000)
        # net realised spread if our effective mark is L ns later (edge decays)
        rs = _net_realised_at_offset(res_base.fills, snap_ts, snap_mid, lat_ns)
        lat_curve[str(lat_ms)] = float(rs.mean() * 1e4) if rs.size else float("nan")
    out["m9_latency_realised_bp"] = lat_curve

    return {"metrics": out, "base_realised": base_rs, "base_markout": base_mk,
            "base_ledger": led_base, "base_fills": res_base.fills}


def _fit_logistic_rank(realised, ncalls, iters=25):
    """Tiny 1-feature logistic fit of fill-outcome on rank (weighted by
    opportunities). Returns predicted P(fill) per rank 1..K."""
    K = len(realised)
    x = np.arange(1, K + 1, dtype=float)
    y = np.array(realised, float)
    w = np.array(ncalls, float)
    w = np.where(w > 0, w, 1.0)
    a, b = 0.0, 0.0
    for _ in range(iters):
        z = a + b * x
        p = 1.0 / (1.0 + np.exp(-z))
        g_a = np.sum(w * (p - y))
        g_b = np.sum(w * (p - y) * x)
        wv = w * p * (1 - p) + 1e-9
        h_aa = np.sum(wv); h_ab = np.sum(wv * x); h_bb = np.sum(wv * x * x)
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a -= da; b -= db
    z = a + b * x
    return 1.0 / (1.0 + np.exp(-z))


def _net_realised_at_offset(fills, snap_ts, snap_mid, offset_ns):
    """Realised half-spread when the comparison mid is taken `offset_ns`
    after the standard 10s horizon — the latency-cliff edge decay proxy."""
    if not fills:
        return np.array([])
    # shift each fill's ts forward by offset, recompute 10s realised spread
    from mmsim.markout.engine import HORIZONS_NS
    fill_ts = np.array([f.ts_ns for f in fills], dtype=np.int64) + offset_ns
    fill_px = np.array([f.price for f in fills], dtype=np.float64)
    fill_side = np.array([f.side for f in fills], dtype=np.int64)

    class _F:
        __slots__ = ("ts_ns", "price", "side")
        def __init__(s, t, p, sd): s.ts_ns = int(t); s.price = float(p); s.side = int(sd)
    shifted = [_F(fill_ts[i], fill_px[i], fill_side[i]) for i in range(len(fills))]
    mo = compute_markout(shifted, snap_ts, snap_mid)
    rs = mo.realised_spread_10s
    return rs[~np.isnan(rs)]


# --------------------------------------------------------------------- #
# M8 — true vs inferred sign (whole-day, doesn't need windows)
# --------------------------------------------------------------------- #

def m8_sign_accuracy(snap, trades, symbol, mid_tl=None):
    """Tick-rule & quote-rule accuracy vs the TRUE exchange aggressor.

    Reads the trade tape DIRECTLY from the trades parquet (it is a separate,
    tiny file) instead of rebuilding the full day's event stream as Python
    objects — the old path materialised millions of SnapshotEvent objects only
    to discard them.  ``mid_tl`` optionally supplies the precomputed snapshot
    mid timeline ``(s_ts, s_mid)`` so we don't re-scan the snap file.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    tpf = pq.ParquetFile(str(trades))
    cols = ["ts_ns", "price", "side"]
    has_sym = "symbol" in [f.name for f in tpf.schema_arrow]
    if has_sym and symbol is not None:
        cols.append("symbol")
    ts_chunks, px_chunks, side_chunks = [], [], []
    for batch in tpf.iter_batches(batch_size=500_000, columns=cols):
        if has_sym and symbol is not None:
            symcol = batch.column("symbol")
            if not pc.all(pc.equal(symcol, symbol)).as_py():
                import pyarrow as pa
                tbl = pa.table(batch).filter(pc.equal(symcol, symbol))
                if tbl.num_rows == 0:
                    continue
                ts_chunks.append(tbl["ts_ns"].combine_chunks().to_numpy(zero_copy_only=False))
                px_chunks.append(tbl["price"].combine_chunks().to_numpy(zero_copy_only=False))
                side_chunks.append(tbl["side"].combine_chunks().to_numpy(zero_copy_only=False))
                continue
        ts_chunks.append(batch.column("ts_ns").to_numpy(zero_copy_only=False))
        px_chunks.append(batch.column("price").to_numpy(zero_copy_only=False))
        side_chunks.append(batch.column("side").to_numpy(zero_copy_only=False))
    if ts_chunks:
        t_ts = np.concatenate(ts_chunks).astype(np.int64)
        t_px = np.concatenate(px_chunks).astype(np.float64)
        true_s = np.concatenate(side_chunks).astype(np.int64)
    else:
        t_ts = np.empty(0, np.int64); t_px = np.empty(0, np.float64)
        true_s = np.empty(0, np.int64)
    n = true_s.size
    if n == 0:
        return {"n_trades": 0}
    # tick rule
    tick = np.zeros(n, np.int64); last = 1
    for i in range(n):
        if i == 0: s = 1
        elif t_px[i] > t_px[i-1]: s = 1
        elif t_px[i] < t_px[i-1]: s = -1
        else: s = last
        tick[i] = s; last = s
    # quote rule vs contemporaneous mid
    if mid_tl is not None:
        s_ts, s_mid = mid_tl
    else:
        s_ts, s_mid = mid_timeline(snap, symbol)
    quote = np.zeros(n, np.int64)
    idxs = np.searchsorted(s_ts, t_ts, side="right") - 1
    for i in range(n):
        if idxs[i] >= 0:
            m = s_mid[idxs[i]]
            quote[i] = 1 if t_px[i] > m else (-1 if t_px[i] < m else 0)
    def acc(inf):
        mask = (inf != 0) & (true_s != 0)
        return float(np.mean(inf[mask] == true_s[mask])) if mask.sum() else float("nan")
    return {"n_trades": int(n), "true_buy_share": float(np.mean(true_s > 0)),
            "tick_accuracy": acc(tick), "quote_accuracy": acc(quote),
            "tick_buy_share": float(np.mean(tick[tick != 0] > 0))}


# --------------------------------------------------------------------- #
# One contract-day run
# --------------------------------------------------------------------- #

def run_one(snap, trades, root, date, symbol, out_dir, *, size, perm_m, seed,
            is_min, oos_min, latencies, throttle_k=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    # session bounds from the mid timeline (cheap streaming pass)
    s_ts, s_mid = mid_timeline(snap, symbol)
    if s_ts.size == 0:
        raise SystemExit(f"{root} {date}: no two-sided snapshots")
    t0, t1 = int(s_ts[0]), int(s_ts[-1])
    windows = wfo_windows(t0, t1, int(is_min * _MIN), int(oos_min * _MIN))

    # FAST PATH: decode the whole day's snapshot + trade parquet straight into
    # flat numpy arrays (no per-event Python objects), applying the same
    # information-preserving top-K-changed throttle.  Each OOS window is then a
    # cheap array slice.  See mmsim.sim.fast_sim.read_day_arrays.
    from mmsim.sim import fast_sim as _F
    # depth_k=10 is lossless for this universe (captured book depth is 10) and
    # the quoters look at most at rank-5 / OFI level-10; keeps the day's flat
    # arrays RAM-bounded for the parallel pool.
    day = _F.read_day_arrays(snap, trades, symbol=symbol, throttle_k=throttle_k,
                             depth_k=10)

    # evaluate each window (one window in RAM at a time)
    win_rows: List[dict] = []
    pooled_realised: List[np.ndarray] = []
    pooled_markout: List[np.ndarray] = []
    day_fills_rows: List[dict] = []
    m1_red, m1_pnl_eq = [], []
    m3_mp_minus_base, m3_ofi_minus_base = [], []
    m4_brier_a, m4_brier_l, m4_ll_a, m4_ll_l, m4_mono = [], [], [], [], []
    m6_net, m6_fees = [], []
    m7_rank_mk = []
    m9_acc: Dict[str, list] = {str(l): [] for l in latencies}

    for wi in range(len(windows)):
        w = windows[wi]
        wa = _F.window_arrays_from_day(day, w[2], w[3])
        r = eval_window(wa, s_ts, s_mid, size, perm_m, seed + wi, latencies)
        del wa
        if r is None:
            continue
        m = r["metrics"]
        m["window"] = wi
        m["oos_start_ns"] = w[2]; m["oos_end_ns"] = w[3]
        m["root"] = root; m["date"] = date; m["symbol"] = symbol
        win_rows.append(m)
        if r["base_realised"].size: pooled_realised.append(r["base_realised"])
        if r["base_markout"].size: pooled_markout.append(r["base_markout"])
        # collect day fills (baseline) with window tag
        for fr in r["base_ledger"]:
            fr = dict(fr); fr["window"] = wi; fr["root"] = root; fr["date"] = date
            day_fills_rows.append(fr)
        m1_red.append(m["m1_cvar_reduction"])
        m1_pnl_eq.append(m["m1_pnl_skew"] - m["m1_pnl_sym"])
        if not np.isnan(m["m3_microprice_mean_realised_bp"]) and not np.isnan(m["base_mean_realised_bp"]):
            m3_mp_minus_base.append(m["m3_microprice_mean_realised_bp"] - m["base_mean_realised_bp"])
        if not np.isnan(m["m3_integ_ofi_mean_realised_bp"]) and not np.isnan(m["base_mean_realised_bp"]):
            m3_ofi_minus_base.append(m["m3_integ_ofi_mean_realised_bp"] - m["base_mean_realised_bp"])
        if not np.isnan(m["m4_brier_analytic"]): m4_brier_a.append(m["m4_brier_analytic"])
        if not np.isnan(m["m4_brier_learned"]): m4_brier_l.append(m["m4_brier_learned"])
        if not np.isnan(m["m4_logloss_analytic"]): m4_ll_a.append(m["m4_logloss_analytic"])
        if not np.isnan(m["m4_logloss_learned"]): m4_ll_l.append(m["m4_logloss_learned"])
        m4_mono.append(1.0 if m["m4_monotone_decreasing"] else 0.0)
        if "m6_net_pnl" in m: m6_net.append(m["m6_net_pnl"]); m6_fees.append(m["m6_fees"])
        m7_rank_mk.append(m["m7_markout_by_rank_bp"])
        for l in latencies:
            m9_acc[str(l)].append(m["m9_latency_realised_bp"][str(l)])

    # ---- day-level pooled OOS statistics + permutation nulls ----------
    pooled_rs = np.concatenate(pooled_realised) if pooled_realised else np.array([])
    pooled_mk = np.concatenate(pooled_markout) if pooled_markout else np.array([])

    # M6 / baseline skill perm null: realised spread beats zero-skill?
    pn_rs = permutation_null(pooled_rs, m=perm_m, seed=seed) if pooled_rs.size else None
    # M1 perm null: per-window CVaR reduction vs sign-flip null
    pn_m1 = permutation_null(np.array(m1_red), m=perm_m, seed=seed + 101) if m1_red else None
    # M3 perm null: micro-price realised-spread improvement over baseline
    pn_m3mp = permutation_null(np.array(m3_mp_minus_base), m=perm_m, seed=seed + 102) if m3_mp_minus_base else None
    pn_m3ofi = permutation_null(np.array(m3_ofi_minus_base), m=perm_m, seed=seed + 103) if m3_ofi_minus_base else None
    # M6 rebate/zero-fee artifact null: is the net edge fee-driven? rerun
    # the day-pooled realised spread under ZERO fees and test the delta.
    pn_m6_fee = permutation_null(np.array(m6_net), m=perm_m, seed=seed + 104) if m6_net else None

    def _pn(p): return None if p is None else {
        "observed": p.observed, "null_q95": p.null_q95, "p_value": p.p_value,
        "m": p.m, "n": p.n_fills}

    # M8 (whole-day)
    m8 = m8_sign_accuracy(snap, trades, symbol, mid_tl=(s_ts, s_mid))

    # ---- write per-fill OOS ledger + per-window tidy parquet ----------
    fills_path = os.path.join(out_dir, f"{root}_{date}_fills.parquet")
    win_path = os.path.join(out_dir, f"{root}_{date}_windows.parquet")
    if day_fills_rows:
        pq.write_table(pa.Table.from_pylist(day_fills_rows), fills_path)
    # tidy windows: flatten list-valued metrics to JSON strings for parquet
    flat = []
    for m in win_rows:
        fm = {}
        for k, v in m.items():
            fm[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        flat.append(fm)
    if flat:
        pq.write_table(pa.Table.from_pylist(flat), win_path)

    # ---- run-level JSON ----------------------------------------------
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    summary = {
        "root": root, "date": date, "symbol": symbol,
        "n_windows": len(win_rows), "is_min": is_min, "oos_min": oos_min,
        "throttle_k": throttle_k,
        "wall_s": round(time.time() - t_start, 1), "peak_rss_gb": round(peak_gb, 2),
        "cost_model": {
            "maker_fee_bp": FUT_COST.maker_fee * 1e4,
            "maker_rebate_bp": FUT_COST.maker_rebate * 1e4,
            "maker_net_fee_bp": (FUT_COST.maker_fee - FUT_COST.maker_rebate) * 1e4,
            "maker_pays_slippage": FUT_COST.maker_pays_slippage,
            "taker_bp": FUT_COST.taker_fee * 1e4,
            "taker_slip_bp": FUT_COST.slippage * 1e4,
            "note": ("maker fills capture spread (in gross mark-to-mid) + pay "
                     "maker_net_fee, NO slippage; adverse selection in markout"),
        },
        "oos_pooled": {
            "n_fills": int(pooled_rs.size),
            # NOTE: mean_realised_bp is the engine's signed realised-spread
            # column (customer convention = -markout); kept for continuity.
            "mean_realised_bp": float(pooled_rs.mean() * 1e4) if pooled_rs.size else None,
            # The MAKER's actual per-fill edge: markout = side*(mid_{t+10s}-px),
            # i.e. captured half-spread NET of adverse selection (correctly
            # signed from the liquidity provider's perspective).
            "mean_markout_bp": float(pooled_mk.mean() * 1e4) if pooled_mk.size else None,
            # Maker net edge after the (small) exchange fee, no slippage:
            # markout - maker_net_fee_bp.  This is the headline "is the maker
            # economics positive?" number under the corrected cost model.
            "maker_net_edge_bp": (
                float(pooled_mk.mean() * 1e4
                      - (FUT_COST.maker_fee - FUT_COST.maker_rebate) * 1e4)
                if pooled_mk.size else None),
        },
        "M1_inventory_skew": {
            "n_windows": len(m1_red),
            "mean_cvar_reduction": float(np.mean(m1_red)) if m1_red else None,
            "mean_pnl_diff_skew_minus_sym": float(np.mean(m1_pnl_eq)) if m1_pnl_eq else None,
            "perm_null": _pn(pn_m1),
        },
        "M3_microprice_ofi": {
            "n_windows_mp": len(m3_mp_minus_base),
            "mean_mp_minus_base_bp": float(np.mean(m3_mp_minus_base)) if m3_mp_minus_base else None,
            "mean_ofi_minus_base_bp": float(np.mean(m3_ofi_minus_base)) if m3_ofi_minus_base else None,
            "perm_null_microprice": _pn(pn_m3mp),
            "perm_null_integrated_ofi": _pn(pn_m3ofi),
        },
        "M4_fill_calibration": {
            "frac_windows_monotone": float(np.mean(m4_mono)) if m4_mono else None,
            "mean_brier_analytic": float(np.mean(m4_brier_a)) if m4_brier_a else None,
            "mean_brier_learned": float(np.mean(m4_brier_l)) if m4_brier_l else None,
            "mean_logloss_analytic": float(np.mean(m4_ll_a)) if m4_ll_a else None,
            "mean_logloss_learned": float(np.mean(m4_ll_l)) if m4_ll_l else None,
            "learned_beats_analytic_brier": (
                bool(np.mean(m4_brier_l) < np.mean(m4_brier_a))
                if m4_brier_a and m4_brier_l else None),
        },
        "M6_pnl_decomposition": {
            "mean_window_net_pnl": float(np.mean(m6_net)) if m6_net else None,
            "mean_window_fees": float(np.mean(m6_fees)) if m6_fees else None,
            "perm_null_zero_fee_skill": _pn(pn_m6_fee),
            "rebate": "NONE (futures)",
            "realised_spread_perm_null": _pn(pn_rs),
        },
        "M7_queue_rank_economics": {
            "mean_markout_by_rank_bp": _nanmean_cols(m7_rank_mk),
        },
        "M8_true_vs_inferred_sign": m8,
        "M9_latency_cliff": {
            "mean_realised_bp_by_latency_ms": {
                k: (float(np.nanmean(v)) if v else None) for k, v in m9_acc.items()},
            "latencies_ms": latencies,
        },
    }
    json_path = os.path.join(out_dir, f"{root}_{date}.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[{root} {date}] windows={len(win_rows)} pooled_fills={pooled_rs.size:,} "
          f"wall={summary['wall_s']}s peak_RSS={summary['peak_rss_gb']}GB -> {json_path}")
    return summary


# --------------------------------------------------------------------- #
# Assembly: BH across the hypothesis family + DSR effective-trials
# --------------------------------------------------------------------- #

def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    crit = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= crit
    kmax = np.max(np.flatnonzero(passed)) if passed.any() else -1
    adj = np.empty(n)
    # BH-adjusted q-values (monotone)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    adj[order] = np.clip(q, 0, 1)
    reject = np.zeros(n, bool)
    if kmax >= 0:
        reject[order[:kmax + 1]] = True
    return adj, reject


def assemble(out_dir):
    from mmsim.research.perm_null import dsr_effective_trials
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".json") and not fn.startswith("ASSEMBLY"):
            with open(os.path.join(out_dir, fn)) as fh:
                rows.append(json.load(fh))
    if not rows:
        raise SystemExit(f"no per-run JSONs in {out_dir}")

    # gather the hypothesis-family p-values across all (root,date)
    family = []  # (label, root, date, pvalue, effect)
    for r in rows:
        rt, dt = r["root"], r["date"]
        for label, node, key, eff_key in [
            ("M1_skew", r["M1_inventory_skew"], "perm_null", "mean_cvar_reduction"),
            ("M3_microprice", r["M3_microprice_ofi"], "perm_null_microprice", "mean_mp_minus_base_bp"),
            ("M3_integ_ofi", r["M3_microprice_ofi"], "perm_null_integrated_ofi", "mean_ofi_minus_base_bp"),
            ("M6_realised_spread", r["M6_pnl_decomposition"], "realised_spread_perm_null", None),
            ("M6_zero_fee_skill", r["M6_pnl_decomposition"], "perm_null_zero_fee_skill", "mean_window_net_pnl"),
        ]:
            pn = node.get(key)
            if pn and pn.get("p_value") is not None:
                eff = node.get(eff_key) if eff_key else pn.get("observed")
                family.append((label, rt, dt, float(pn["p_value"]), eff))

    out = {"n_runs": len(rows), "family_size": len(family)}
    if family:
        pvals = np.array([f[3] for f in family])
        adj, reject = benjamini_hochberg(pvals, alpha=0.05)
        # DSR effective-trials over the per-(root,date) realised-spread Sharpe
        # proxy (here: pooled mean realised / its window std as a Sharpe-like)
        sharpes = []
        for r in rows:
            mr = r["oos_pooled"].get("mean_realised_bp")
            if mr is not None:
                sharpes.append(mr)
        eff_trials = dsr_effective_trials(np.array(sharpes)) if sharpes else float(len(rows))
        out["dsr_effective_trials"] = eff_trials
        tidy = []
        for i, (label, rt, dt, p, eff) in enumerate(family):
            tidy.append({"hypothesis": label, "root": rt, "date": dt,
                         "p_value": p, "bh_qvalue": float(adj[i]),
                         "bh_reject_at_0.05": bool(reject[i]), "effect": eff})
        out["family"] = tidy
        out["n_bh_significant"] = int(reject.sum())
        # write tidy CSV
        import csv
        csv_path = os.path.join(out_dir, "ASSEMBLY_tidy.csv")
        with open(csv_path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(tidy[0].keys()))
            wr.writeheader(); wr.writerows(tidy)
        out["tidy_csv"] = csv_path
    with open(os.path.join(out_dir, "ASSEMBLY.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"ASSEMBLY: runs={out['n_runs']} family={out['family_size']} "
          f"BH-significant={out.get('n_bh_significant')} "
          f"DSR-eff-trials={out.get('dsr_effective_trials')}")
    return out


# --------------------------------------------------------------------- #
# Manifest driver — RAM-gated process pool across (root,date)
# --------------------------------------------------------------------- #

def run_manifest(manifest, out_dir, jobs, per_job_gb, **kw):
    """manifest CSV columns: root,date,symbol,snap,trades. Spawns one
    subprocess per row, capped at `jobs` concurrent (RAM-gated)."""
    import csv
    with open(manifest) as fh:
        tasks = list(csv.DictReader(fh))
    if jobs == "auto":
        free_gb = _free_gb()
        budget = max(4.0, free_gb - 4.0)  # leave 4GB headroom
        jobs = max(1, int(budget // per_job_gb))
        jobs = min(jobs, os.cpu_count() or 1, len(tasks))
        print(f"RAM-gated pool: free={free_gb:.0f}GB per_job={per_job_gb}GB "
              f"-> jobs={jobs}")
    else:
        jobs = int(jobs)
    running: List[Tuple[subprocess.Popen, dict]] = []
    pending = list(tasks)
    done = 0
    while pending or running:
        while pending and len(running) < jobs:
            t = pending.pop(0)
            cmd = [sys.executable, __file__,
                   "--snap", t["snap"], "--trades", t["trades"],
                   "--root", t["root"], "--date", t["date"],
                   "--symbol", t["symbol"], "--out", out_dir,
                   "--perm-m", str(kw["perm_m"]), "--is-min", str(kw["is_min"]),
                   "--oos-min", str(kw["oos_min"]), "--size", str(kw["size"]),
                   "--throttle-k", str(kw.get("throttle_k", 5)),
                   "--maker-fee-bp", str(kw.get("maker_fee_bp", 0.2)),
                   "--maker-rebate-bp", str(kw.get("maker_rebate_bp", 0.0))]
            if kw.get("maker_pays_slippage"):
                cmd.append("--maker-pays-slippage")
            env = dict(os.environ, PYTHONPATH=os.getcwd())
            p = subprocess.Popen(cmd, env=env)
            running.append((p, t))
        for p, t in running[:]:
            if p.poll() is not None:
                running.remove((p, t)); done += 1
                print(f"  [{done}/{len(tasks)}] {t['root']} {t['date']} rc={p.returncode}")
        time.sleep(1.0)
    print(f"manifest complete: {done} contract-days")


def _free_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 8.0


# --------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap"); ap.add_argument("--trades")
    ap.add_argument("--root"); ap.add_argument("--date"); ap.add_argument("--symbol")
    ap.add_argument("--out", default="runs/mm_full")
    ap.add_argument("--size", type=float, default=1.0)
    ap.add_argument("--perm-m", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--is-min", type=float, default=60.0)
    ap.add_argument("--oos-min", type=float, default=20.0)
    ap.add_argument("--latencies", default="0,1,5,10,50,100,500")
    ap.add_argument("--throttle-k", type=int, default=5,
                    help="drop snapshots with unchanged top-K book (engine "
                         "stream only; 0 = off, full bit-identical stream)")
    ap.add_argument("--maker-fee-bp", type=float, default=0.2,
                    help="maker (passive) exchange fee in bp of notional "
                         "(CME default 0.2 bp; conservative). No slippage on "
                         "maker fills — they rest at their limit.")
    ap.add_argument("--maker-rebate-bp", type=float, default=0.0,
                    help="maker rebate in bp of notional credited on passive "
                         "fills (CME = 0; crypto/equity venues may pay one). "
                         "Net maker fee = maker_fee - maker_rebate.")
    ap.add_argument("--maker-pays-slippage", action="store_true",
                    help="(Phase-1/diagnostic) charge slippage on maker fills, "
                         "reproducing the old incorrect maker-as-taker model.")
    ap.add_argument("--manifest"); ap.add_argument("--jobs", default="auto")
    ap.add_argument("--per-job-gb", type=float, default=4.0)
    ap.add_argument("--assemble")
    args = ap.parse_args()

    latencies = [float(x) for x in args.latencies.split(",")]
    throttle_k = args.throttle_k if args.throttle_k and args.throttle_k > 0 else None

    # Rebuild the cost model from the CLI knobs (ECONOMICS FIX): maker fee /
    # rebate / slippage-policy are configurable; default = correct CME maker
    # economics (no slippage on passive fills, no rebate, small fee).
    global FUT_COST
    FUT_COST = CostModel(
        taker_fee=0.00002,
        maker_fee=args.maker_fee_bp * 1e-4,
        slippage=0.00002,
        maker_rebate=args.maker_rebate_bp * 1e-4,
        maker_pays_slippage=bool(args.maker_pays_slippage),
        funding_per_8h=0.0, is_perp=False,
    )

    if args.assemble:
        assemble(args.assemble); return
    if args.manifest:
        run_manifest(args.manifest, args.out, args.jobs, args.per_job_gb,
                     perm_m=args.perm_m, is_min=args.is_min, oos_min=args.oos_min,
                     size=args.size, throttle_k=args.throttle_k,
                     maker_fee_bp=args.maker_fee_bp,
                     maker_rebate_bp=args.maker_rebate_bp,
                     maker_pays_slippage=bool(args.maker_pays_slippage)); return
    if not (args.snap and args.trades and args.root and args.date and args.symbol):
        ap.error("need --snap --trades --root --date --symbol (or --manifest/--assemble)")
    run_one(args.snap, args.trades, args.root, args.date, args.symbol, args.out,
            size=args.size, perm_m=args.perm_m, seed=args.seed,
            is_min=args.is_min, oos_min=args.oos_min, latencies=latencies,
            throttle_k=throttle_k)


if __name__ == "__main__":
    main()
