"""Vectorised / numba fast path for the queue-aware MM fill simulation.

This module is a PERFORMANCE redesign of the ``run_sim`` + ``QueueAwareFillModel``
+ per-snapshot cancel-on-replace quoter pipeline used by
``scripts/run_mm_full.py``.  It produces the SAME economic result as the
reference path (``mmsim.sim.loop.run_sim`` driving ``QueueAwareFillModel``)
but operates on flat numpy arrays through a numba kernel instead of millions
of per-event Python object calls.

Why a clean kernel is faithful (not a shortcut)
-----------------------------------------------
The reference pipeline, for the runner's quoters, is *cancel-on-replace on
every snapshot*: the quoter is called on each ``SnapshotEvent`` and its
returned orders REPLACE the active set (old orders cancelled, new posted).
Therefore every resting order lives for exactly ONE snapshot interval — from
the snapshot that posted it until the next snapshot, which cancels and
replaces it before any further trade is processed.  (Event order is
``snapshot, trades-in-interval..., next snapshot``; the next snapshot's
``on_orders_removed`` fires before the next interval's trades.)

Within one interval the reference semantics are exactly:

  * At placement, ``queue_pos = size at our price level in the snapshot's
    book`` (the resting size we land behind).  If our price is not visible in
    the depth-K book, the order cannot be tracked -> no fills (reference
    raises in ``QueueTracker.__init__``; the runner's quoters only ever post
    at a visible level, so this never triggers there, but we guard it).
  * Each trade in the interval that aggresses INTO our level (sell-aggressor
    for a bid, buy-aggressor for an ask, at our exact price) consumes the
    queue ahead: ``consumed = min(trade.size, queue_pos); queue_pos -=
    consumed``.  The spillover ``trade.size - queue_pos_before`` (when
    positive) fills us, ``fill = min(spillover, remaining_order_size)``.
  * The order's remaining size shrinks per fill; once it hits zero the order
    is gone for the rest of the interval.

The snapshot-time cancel-attribution (pro-rata cancels) in the reference
tracker is computed and then immediately DISCARDED, because the order is
cancelled and replaced at that same snapshot before it can fill again.  It
has zero effect on the emitted fills, so the fast path omits it.  The
``frozen`` (level-dropped-out-of-view) branch likewise only matters across
snapshot boundaries, which never happens for one-interval orders.

This per-interval reduction is verified bit-for-bit against the reference
``run_sim`` pipeline in ``scripts/verify_fast_parity.py`` and in
``tests/test_fast_sim.py``.

Causality
---------
A fill at interval i depends only on the snapshot opening interval i and the
trades within interval i — all at ``ts <= fill_ts``.  No event past the
fill's timestamp is consulted.  The leak property is therefore structural.

Output
------
``simulate`` returns a ``FastResult`` whose ``fills`` is a list of
``mmsim.sim.loop.Fill`` (so the existing ledger / markout code consumes it
unchanged) plus the scalar counts ``run_sim`` reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from mmsim.ingest.lob import SnapshotEvent, TradeEvent
from mmsim.sim.loop import Fill, SimResult

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


# Price-equality tolerance — mirrors the reference (_PX_EPSILON = 1e-9).
_PX_EPSILON = 1e-9


# --------------------------------------------------------------------- #
# Window decode: event list -> flat arrays
# --------------------------------------------------------------------- #

@dataclass
class WindowArrays:
    """Flat-array view of one window's events.

    Snapshots and trades are interleaved in time but split into two arrays;
    the kernel walks them with a two-pointer merge preserving the original
    (ts, recv, kind) order (snapshot-before-trade on ties), exactly like the
    event stream the reference consumed.
    """
    # snapshots
    snap_ts: np.ndarray          # int64 [S]
    snap_recv: np.ndarray        # int64 [S]
    bid_px: np.ndarray           # float64 [S, K]   (NaN-padded)
    bid_sz: np.ndarray           # float64 [S, K]
    ask_px: np.ndarray           # float64 [S, K]
    ask_sz: np.ndarray           # float64 [S, K]
    n_bid: np.ndarray            # int32 [S]  number of valid bid levels
    n_ask: np.ndarray            # int32 [S]
    # trades
    trd_ts: np.ndarray           # int64 [T]
    trd_recv: np.ndarray         # int64 [T]
    trd_px: np.ndarray           # float64 [T]
    trd_sz: np.ndarray           # float64 [T]
    trd_side: np.ndarray         # int8 [T]
    K: int


def decode_window(events, depth_k: int = 20) -> WindowArrays:
    """Decode a window's event list (SnapshotEvent / TradeEvent) into flat
    arrays.  ``depth_k`` caps the number of book levels kept per side; the
    runner's quoters only look at the top few levels, so depth 20 is ample
    and matches the captured depth.
    """
    snaps = []
    trades = []
    for ev in events:
        if type(ev) is SnapshotEvent:
            snaps.append(ev)
        elif type(ev) is TradeEvent:
            trades.append(ev)
        else:
            # be liberal: dispatch by attribute (covers subclasses)
            if isinstance(ev, SnapshotEvent):
                snaps.append(ev)
            elif isinstance(ev, TradeEvent):
                trades.append(ev)
            else:
                raise TypeError(f"decode_window: unknown event {type(ev).__name__}")

    S = len(snaps)
    T = len(trades)
    K = depth_k

    snap_ts = np.empty(S, np.int64)
    snap_recv = np.empty(S, np.int64)
    bid_px = np.full((S, K), np.nan, np.float64)
    bid_sz = np.zeros((S, K), np.float64)
    ask_px = np.full((S, K), np.nan, np.float64)
    ask_sz = np.zeros((S, K), np.float64)
    n_bid = np.zeros(S, np.int32)
    n_ask = np.zeros(S, np.int32)

    for i, s in enumerate(snaps):
        snap_ts[i] = s.ts_ns
        snap_recv[i] = s.recv_ns
        b = s.bids
        a = s.asks
        nb = min(len(b), K)
        na = min(len(a), K)
        n_bid[i] = nb
        n_ask[i] = na
        for j in range(nb):
            bid_px[i, j] = b[j][0]
            bid_sz[i, j] = b[j][1]
        for j in range(na):
            ask_px[i, j] = a[j][0]
            ask_sz[i, j] = a[j][1]

    trd_ts = np.empty(T, np.int64)
    trd_recv = np.empty(T, np.int64)
    trd_px = np.empty(T, np.float64)
    trd_sz = np.empty(T, np.float64)
    trd_side = np.empty(T, np.int8)
    for i, t in enumerate(trades):
        trd_ts[i] = t.ts_ns
        trd_recv[i] = t.recv_ns
        trd_px[i] = t.price
        trd_sz[i] = t.size
        trd_side[i] = t.side

    return WindowArrays(
        snap_ts=snap_ts, snap_recv=snap_recv,
        bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz,
        n_bid=n_bid, n_ask=n_ask,
        trd_ts=trd_ts, trd_recv=trd_recv, trd_px=trd_px, trd_sz=trd_sz,
        trd_side=trd_side, K=K,
    )


# --------------------------------------------------------------------- #
# Array-native parquet decode (bypasses Python event objects entirely)
# --------------------------------------------------------------------- #
# The streaming SnapshotEvent/TradeEvent object path costs ~13s per 500k
# snapshots (per-row tuple-of-tuple construction).  For full-scale futures
# days (millions of snapshots) that dominates.  Here we decode the parquet
# directly into the flat (S,K) arrays the kernel needs, using pyarrow's
# vectorised list-offsets + struct-field extraction (~0.3s per 500k) and a
# numba gather, then apply the same information-preserving top-K throttle.

if _HAVE_NUMBA:
    @njit(cache=True)
    def _gather_levels(offs, px, sz, K):
        """Scatter a ragged list<struct{px,sz}> (flat values + offsets) into
        padded (S,K) px/sz matrices + a per-row valid-count vector."""
        S = offs.shape[0] - 1
        out_px = np.full((S, K), np.nan, np.float64)
        out_sz = np.zeros((S, K), np.float64)
        n = np.zeros(S, np.int32)
        for i in range(S):
            a = offs[i]
            b = offs[i + 1]
            m = b - a
            if m > K:
                m = K
            n[i] = m
            for j in range(m):
                out_px[i, j] = px[a + j]
                out_sz[i, j] = sz[a + j]
        return out_px, out_sz, n
else:  # pragma: no cover
    def _gather_levels(offs, px, sz, K):
        S = offs.shape[0] - 1
        out_px = np.full((S, K), np.nan, np.float64)
        out_sz = np.zeros((S, K), np.float64)
        n = np.zeros(S, np.int32)
        for i in range(S):
            a = int(offs[i]); b = int(offs[i + 1])
            m = min(b - a, K)
            n[i] = m
            for j in range(m):
                out_px[i, j] = px[a + j]
                out_sz[i, j] = sz[a + j]
        return out_px, out_sz, n


if _HAVE_NUMBA:
    @njit(cache=True)
    def _scatter_levels(offs, px, sz, K, out_px, out_sz, out_n, pos):
        """Scatter a ragged batch (flat values + offsets) into the pre-allocated
        full-day (cap,K) matrices starting at row ``pos``.  out_px is NaN-pre-
        filled and out_sz zero-pre-filled by the caller; we only write valid
        cells so unused tail columns stay padded."""
        S = offs.shape[0] - 1
        for i in range(S):
            a = offs[i]
            b = offs[i + 1]
            m = b - a
            if m > K:
                m = K
            r = pos + i
            out_n[r] = m
            for j in range(m):
                out_px[r, j] = px[a + j]
                out_sz[r, j] = sz[a + j]
else:  # pragma: no cover
    def _scatter_levels(offs, px, sz, K, out_px, out_sz, out_n, pos):
        S = offs.shape[0] - 1
        for i in range(S):
            a = int(offs[i]); b = int(offs[i + 1])
            m = min(b - a, K)
            r = pos + i
            out_n[r] = m
            for j in range(m):
                out_px[r, j] = px[a + j]
                out_sz[r, j] = sz[a + j]


if _HAVE_NUMBA:
    @njit(cache=True)
    def _throttle_mask_topk(bid_px, bid_sz, ask_px, ask_sz, n_bid, n_ask, k):
        """Boolean keep-mask for the top-k-changed throttle: drop a snapshot
        whose top-k book (prices+sizes both sides) is identical to the last
        KEPT snapshot.  First snapshot always kept.  Mirrors the streaming
        throttle in mmsim.ingest.stream (which compares ev.bids[:k]/asks[:k])."""
        S = bid_px.shape[0]
        keep = np.zeros(S, np.bool_)
        if S == 0:
            return keep
        keep[0] = True
        last = 0
        for i in range(1, S):
            same = True
            # compare top-k bids
            kb_last = n_bid[last]
            kb_cur = n_bid[i]
            if kb_last > k:
                kb_last = k
            if kb_cur > k:
                kb_cur = k
            if kb_last != kb_cur:
                same = False
            else:
                for j in range(kb_cur):
                    if bid_px[i, j] != bid_px[last, j] or bid_sz[i, j] != bid_sz[last, j]:
                        same = False
                        break
            if same:
                ka_last = n_ask[last]
                ka_cur = n_ask[i]
                if ka_last > k:
                    ka_last = k
                if ka_cur > k:
                    ka_cur = k
                if ka_last != ka_cur:
                    same = False
                else:
                    for j in range(ka_cur):
                        if ask_px[i, j] != ask_px[last, j] or ask_sz[i, j] != ask_sz[last, j]:
                            same = False
                            break
            if not same:
                keep[i] = True
                last = i
        return keep
else:  # pragma: no cover
    def _throttle_mask_topk(bid_px, bid_sz, ask_px, ask_sz, n_bid, n_ask, k):
        S = bid_px.shape[0]
        keep = np.zeros(S, np.bool_)
        if S == 0:
            return keep
        keep[0] = True
        last = 0
        for i in range(1, S):
            kb = min(int(n_bid[i]), k); ka = min(int(n_ask[i]), k)
            same = (kb == min(int(n_bid[last]), k) and ka == min(int(n_ask[last]), k))
            if same:
                for j in range(kb):
                    if bid_px[i, j] != bid_px[last, j] or bid_sz[i, j] != bid_sz[last, j]:
                        same = False; break
            if same:
                for j in range(ka):
                    if ask_px[i, j] != ask_px[last, j] or ask_sz[i, j] != ask_sz[last, j]:
                        same = False; break
            if not same:
                keep[i] = True; last = i
        return keep


def read_day_arrays(snap_path, trade_path, symbol=None, depth_k=20,
                    throttle_k=None, batch_size=500_000):
    """Decode a full contract-day's snapshot + trade parquet straight into
    flat arrays, applying the top-K-changed throttle in numpy.  Returns a
    dict of arrays for the WHOLE day (sorted by ts), plus the snapshot mid
    timeline.  No Python event objects are constructed.

    Returns ``(WholeDay)`` namespace-ish dict with keys:
      snap_ts, snap_recv, bid_px, bid_sz, ask_px, ask_sz, n_bid, n_ask,
      trd_ts, trd_recv, trd_px, trd_sz, trd_side
    all already throttled & sorted ascending by (ts, recv).
    """
    import pyarrow.parquet as pq

    K = depth_k
    # ---- snapshots ----
    # Pre-allocate the full-day arrays (num_rows is an upper bound under a
    # symbol filter; we track the actual written count) so the decode never
    # holds a per-batch list + a concatenated copy at the same time.  Peak RAM
    # is then ~one copy of the day's (S,K) matrices, halved again by K=10.
    pf = pq.ParquetFile(str(snap_path))
    cap = pf.metadata.num_rows
    cols = ["ts_ns", "recv_ns", "bids", "asks"]
    # Only filter on symbol when a non-empty symbol is given AND the file
    # actually carries a symbol column (some single-contract files don't).
    _snap_has_sym = "symbol" in [f.name for f in pf.schema_arrow]
    do_filter = symbol is not None and symbol != "" and _snap_has_sym
    if do_filter:
        cols.append("symbol")
    snap_ts = np.empty(cap, np.int64)
    snap_recv = np.empty(cap, np.int64)
    bid_px = np.full((cap, K), np.nan, np.float64)
    bid_sz = np.zeros((cap, K), np.float64)
    ask_px = np.full((cap, K), np.nan, np.float64)
    ask_sz = np.zeros((cap, K), np.float64)
    n_bid = np.empty(cap, np.int32)
    n_ask = np.empty(cap, np.int32)
    pos = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        if do_filter:
            import pyarrow.compute as pc
            import pyarrow as pa
            symcol = batch.column("symbol")
            # Fast path for single-symbol files (the futures-universe case): if
            # every row already matches, skip the filter/combine_chunks copy.
            if pc.all(pc.equal(symcol, symbol)).as_py():
                ts = batch.column("ts_ns"); rc = batch.column("recv_ns")
                bids = batch.column("bids"); asks = batch.column("asks")
            else:
                tbl = pa.table(batch).filter(pc.equal(symcol, symbol))
                if tbl.num_rows == 0:
                    continue
                ts = tbl["ts_ns"].combine_chunks()
                rc = tbl["recv_ns"].combine_chunks()
                bids = tbl["bids"].combine_chunks()
                asks = tbl["asks"].combine_chunks()
        else:
            ts = batch.column("ts_ns")
            rc = batch.column("recv_ns")
            bids = batch.column("bids")
            asks = batch.column("asks")
        m = len(ts)
        b_off = bids.offsets.to_numpy().astype(np.int64)
        b_off = b_off - b_off[0]
        b_px = bids.values.field("px").to_numpy(zero_copy_only=False)
        b_sz = bids.values.field("sz").to_numpy(zero_copy_only=False)
        a_off = asks.offsets.to_numpy().astype(np.int64)
        a_off = a_off - a_off[0]
        a_px = asks.values.field("px").to_numpy(zero_copy_only=False)
        a_sz = asks.values.field("sz").to_numpy(zero_copy_only=False)
        _scatter_levels(b_off, b_px, b_sz, K, bid_px, bid_sz, n_bid, pos)
        _scatter_levels(a_off, a_px, a_sz, K, ask_px, ask_sz, n_ask, pos)
        snap_ts[pos:pos + m] = ts.to_numpy(zero_copy_only=False).astype(np.int64)
        snap_recv[pos:pos + m] = rc.to_numpy(zero_copy_only=False).astype(np.int64)
        pos += m
    if pos < cap:
        snap_ts = snap_ts[:pos]; snap_recv = snap_recv[:pos]
        bid_px = bid_px[:pos]; bid_sz = bid_sz[:pos]
        ask_px = ask_px[:pos]; ask_sz = ask_sz[:pos]
        n_bid = n_bid[:pos]; n_ask = n_ask[:pos]

    # throttle (top-K changed) — applied on the full day, like the stream path
    if throttle_k is not None and snap_ts.shape[0] > 0:
        keep = _throttle_mask_topk(bid_px, bid_sz, ask_px, ask_sz,
                                   n_bid, n_ask, int(throttle_k))
        snap_ts = snap_ts[keep]; snap_recv = snap_recv[keep]
        bid_px = bid_px[keep]; bid_sz = bid_sz[keep]
        ask_px = ask_px[keep]; ask_sz = ask_sz[keep]
        n_bid = n_bid[keep]; n_ask = n_ask[keep]

    # ---- trades ----
    tpf = pq.ParquetFile(str(trade_path))
    tcols = ["ts_ns", "recv_ns", "price", "size", "side"]
    _trd_has_sym = "symbol" in [f.name for f in tpf.schema_arrow]
    do_filter_t = symbol is not None and symbol != "" and _trd_has_sym
    if do_filter_t:
        tcols.append("symbol")
    tts_l, trecv_l, tpx_l, tsz_l, tside_l = [], [], [], [], []
    for batch in tpf.iter_batches(batch_size=batch_size, columns=tcols):
        if do_filter_t:
            import pyarrow.compute as pc
            import pyarrow as pa
            symcol = batch.column("symbol")
            if pc.all(pc.equal(symcol, symbol)).as_py():
                getc = lambda c: batch.column(c).to_numpy(zero_copy_only=False)
            else:
                tbl = pa.table(batch).filter(pc.equal(symcol, symbol))
                if tbl.num_rows == 0:
                    continue
                getc = lambda c: tbl[c].combine_chunks().to_numpy(zero_copy_only=False)
        else:
            getc = lambda c: batch.column(c).to_numpy(zero_copy_only=False)
        tts_l.append(getc("ts_ns").astype(np.int64))
        trecv_l.append(getc("recv_ns").astype(np.int64))
        tpx_l.append(getc("price").astype(np.float64))
        tsz_l.append(getc("size").astype(np.float64))
        tside_l.append(getc("side").astype(np.int8))
    if tts_l:
        trd_ts = np.concatenate(tts_l); trd_recv = np.concatenate(trecv_l)
        trd_px = np.concatenate(tpx_l); trd_sz = np.concatenate(tsz_l)
        trd_side = np.concatenate(tside_l)
    else:
        trd_ts = np.empty(0, np.int64); trd_recv = np.empty(0, np.int64)
        trd_px = np.empty(0, np.float64); trd_sz = np.empty(0, np.float64)
        trd_side = np.empty(0, np.int8)

    return {
        "snap_ts": snap_ts, "snap_recv": snap_recv,
        "bid_px": bid_px, "bid_sz": bid_sz, "ask_px": ask_px, "ask_sz": ask_sz,
        "n_bid": n_bid, "n_ask": n_ask,
        "trd_ts": trd_ts, "trd_recv": trd_recv, "trd_px": trd_px,
        "trd_sz": trd_sz, "trd_side": trd_side, "K": K,
    }


def window_arrays_from_day(day: dict, oos_lo: int, oos_hi: int) -> "WindowArrays":
    """Slice a decoded day's arrays to one OOS window [oos_lo, oos_hi).

    Snapshots and trades whose ts lies in [lo, hi) are included.  This matches
    iter_day_windows' bucketing (lo <= t < hi) exactly.
    """
    st = day["snap_ts"]
    s_a = int(np.searchsorted(st, oos_lo, "left"))
    s_b = int(np.searchsorted(st, oos_hi, "left"))
    tt = day["trd_ts"]
    t_a = int(np.searchsorted(tt, oos_lo, "left"))
    t_b = int(np.searchsorted(tt, oos_hi, "left"))
    return WindowArrays(
        snap_ts=st[s_a:s_b], snap_recv=day["snap_recv"][s_a:s_b],
        bid_px=day["bid_px"][s_a:s_b], bid_sz=day["bid_sz"][s_a:s_b],
        ask_px=day["ask_px"][s_a:s_b], ask_sz=day["ask_sz"][s_a:s_b],
        n_bid=day["n_bid"][s_a:s_b], n_ask=day["n_ask"][s_a:s_b],
        trd_ts=tt[t_a:t_b], trd_recv=day["trd_recv"][t_a:t_b],
        trd_px=day["trd_px"][t_a:t_b], trd_sz=day["trd_sz"][t_a:t_b],
        trd_side=day["trd_side"][t_a:t_b], K=day["K"],
    )


# --------------------------------------------------------------------- #
# Quote-spec generation (per snapshot: bid/ask price+size; NaN px = no quote)
# --------------------------------------------------------------------- #
# The runner's quoters all reduce to "for each snapshot, decide a bid quote
# (price,size) and/or an ask quote (price,size), at a level that exists in
# the book".  We compute these spec arrays cheaply (vectorised where possible)
# and feed them to the single shared fill kernel below.

def _empty_spec(S):
    px = np.full(S, np.nan, np.float64)
    sz = np.zeros(S, np.float64)
    return px, sz


def spec_touch(w: WindowArrays, size: float):
    """TouchQuoter: both sides at the best level, when both sides exist."""
    S = w.snap_ts.shape[0]
    has_both = (w.n_bid > 0) & (w.n_ask > 0)
    bpx = np.where(has_both, w.bid_px[:, 0], np.nan)
    apx = np.where(has_both, w.ask_px[:, 0], np.nan)
    bsz = np.where(has_both, size, 0.0)
    asz = np.where(has_both, size, 0.0)
    return bpx, bsz, apx, asz


def spec_depth_skew(w: WindowArrays, size: float):
    """DepthSkewQuoter: bid at level-2 (or level-1 if only one), ask at touch;
    requires both sides present."""
    S = w.snap_ts.shape[0]
    has_both = (w.n_bid > 0) & (w.n_ask > 0)
    use_l2 = w.n_bid > 1
    bid_lvl2 = np.where(use_l2, w.bid_px[:, 1], w.bid_px[:, 0])
    bpx = np.where(has_both, bid_lvl2, np.nan)
    apx = np.where(has_both, w.ask_px[:, 0], np.nan)
    bsz = np.where(has_both, size, 0.0)
    asz = np.where(has_both, size, 0.0)
    return bpx, bsz, apx, asz


def spec_rank(w: WindowArrays, size: float, rank: int, side: int):
    """RankQuoter: one resting order at depth-rank `rank` (1=touch) on `side`.
    Requires that rank exists on that side."""
    S = w.snap_ts.shape[0]
    if side == +1:
        n = w.n_bid
        px_src = w.bid_px
    else:
        n = w.n_ask
        px_src = w.ask_px
    ok = n >= rank
    px = np.where(ok, px_src[:, rank - 1], np.nan)
    sz = np.where(ok, size, 0.0)
    bpx, bsz = _empty_spec(S)
    apx, asz = _empty_spec(S)
    if side == +1:
        return px, sz, apx, asz
    else:
        return bpx, bsz, px, sz


def spec_microskew(w: WindowArrays, size: float, use_ofi: bool, ofi_levels: int = 10):
    """MicroSkewQuoter: micro-price (+ optional integrated-OFI) lean.  When
    the lean is up, post only the bid; when down, only the ask; on a flat
    lean, post both.  Signal is computed causally from the book at each
    snapshot (level-1 imbalance micro-price deviation; OFI from consecutive
    top-L book changes), exactly mirroring the reference quoter.

    The reference OFI quoter carries ``_prev_b/_prev_a`` across *every*
    snapshot it is called on (i.e. every snapshot in the window), so we
    replicate that running state here over the snapshot sequence.
    """
    S = w.snap_ts.shape[0]
    has_both = (w.n_bid > 0) & (w.n_ask > 0)
    bp = w.bid_px[:, 0]
    bs = w.bid_sz[:, 0]
    ap = w.ask_px[:, 0]
    az = w.ask_sz[:, 0]
    mid = 0.5 * (bp + ap)
    tot = bs + az
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = np.where(tot > 0, (ap * bs + bp * az) / tot, mid)
    lean = mp - mid

    if use_ofi:
        ofi = _ofi_running(w, ofi_levels)
        lean = lean + 1e-12 * ofi

    # decide sides
    bpx, bsz = _empty_spec(S)
    apx, asz = _empty_spec(S)
    # default both when has_both and lean == 0
    post_bid = has_both & (lean >= 0)        # lean>0 -> bid only; lean==0 -> both
    post_ask = has_both & (lean <= 0)        # lean<0 -> ask only; lean==0 -> both
    bpx = np.where(post_bid, bp, np.nan)
    bsz = np.where(post_bid, size, 0.0)
    apx = np.where(post_ask, ap, np.nan)
    asz = np.where(post_ask, size, 0.0)
    return bpx, bsz, apx, asz


def _ofi_running(w: WindowArrays, L: int) -> np.ndarray:
    """Integrated OFI per snapshot, replicating MicroSkewQuoter._update_ofi.

    For snapshot i (i>=1), OFI vs snapshot i-1 over the top-L levels:
      bid level l: +bs if bp>pbp ; +(bs-pbs) if bp==pbp ; -pbs if bp<pbp
      ask level l: -az if ap<pap ; -(az-pas) if ap==pap ; +pas if ap>pap
    where the per-level comparison runs over min(L, len, prev_len) levels.
    The reference recomputes _ofi only and stores it; lean uses the latest.
    NaN-padded missing levels are treated as absent (loop bound by n_*).
    """
    S = w.snap_ts.shape[0]
    K = w.K
    Lk = min(L, K)
    return _ofi_kernel(w.bid_px, w.bid_sz, w.ask_px, w.ask_sz,
                       w.n_bid.astype(np.int64), w.n_ask.astype(np.int64), Lk)


if _HAVE_NUMBA:
    @njit(cache=True)
    def _ofi_kernel(bid_px, bid_sz, ask_px, ask_sz, n_bid, n_ask, L):
        S = bid_px.shape[0]
        ofi = np.zeros(S, np.float64)
        for i in range(1, S):
            o = 0.0
            nb = n_bid[i]
            if nb > L:
                nb = L
            pnb = n_bid[i - 1]
            mb = nb if nb < pnb else pnb
            for l in range(mb):
                bp = bid_px[i, l]; bs = bid_sz[i, l]
                pbp = bid_px[i - 1, l]; pbs = bid_sz[i - 1, l]
                if bp > pbp:
                    o += bs
                elif bp == pbp:
                    o += (bs - pbs)
                else:
                    o -= pbs
            na = n_ask[i]
            if na > L:
                na = L
            pna = n_ask[i - 1]
            ma = na if na < pna else pna
            for l in range(ma):
                ap = ask_px[i, l]; az = ask_sz[i, l]
                pap = ask_px[i - 1, l]; pas = ask_sz[i - 1, l]
                if ap < pap:
                    o -= az
                elif ap == pap:
                    o -= (az - pas)
                else:
                    o += pas
            ofi[i] = o
        return ofi
else:  # pragma: no cover
    def _ofi_kernel(bid_px, bid_sz, ask_px, ask_sz, n_bid, n_ask, L):
        S = bid_px.shape[0]
        ofi = np.zeros(S, np.float64)
        for i in range(1, S):
            o = 0.0
            nb = min(int(n_bid[i]), L)
            mb = min(nb, int(n_bid[i - 1]))
            for l in range(mb):
                bp = bid_px[i, l]; bs = bid_sz[i, l]
                pbp = bid_px[i - 1, l]; pbs = bid_sz[i - 1, l]
                if bp > pbp:
                    o += bs
                elif bp == pbp:
                    o += (bs - pbs)
                else:
                    o -= pbs
            na = min(int(n_ask[i]), L)
            ma = min(na, int(n_ask[i - 1]))
            for l in range(ma):
                ap = ask_px[i, l]; az = ask_sz[i, l]
                pap = ask_px[i - 1, l]; pas = ask_sz[i - 1, l]
                if ap < pap:
                    o -= az
                elif ap == pap:
                    o -= (az - pas)
                else:
                    o += pas
            ofi[i] = o
        return ofi


# --------------------------------------------------------------------- #
# The shared fill kernel
# --------------------------------------------------------------------- #
# Given per-snapshot bid/ask quote specs + the trade tape, produce fills.
# Two-pointer merge over (snap, trade) preserving (ts, recv, snap<trade)
# ordering.  An order rests for exactly one snapshot interval.

def _fill_kernel_py(
    snap_ts, snap_recv, bid_px_lvl, bid_sz_at_lvl,
    quote_bpx, quote_bsz, quote_apx, quote_asz,
    trd_ts, trd_recv, trd_px, trd_sz, trd_side,
):
    """Pure-Python reference of the fill kernel (used when numba is absent
    and as the parity oracle in tests)."""
    return _fill_core(
        snap_ts, snap_recv, bid_px_lvl, bid_sz_at_lvl,
        quote_bpx, quote_bsz, quote_apx, quote_asz,
        trd_ts, trd_recv, trd_px, trd_sz, trd_side, _PX_EPSILON)


def _fill_core(snap_ts, snap_recv, quote_b_level_sz, quote_a_level_sz,
               quote_bpx, quote_bsz, quote_apx, quote_asz,
               trd_ts, trd_recv, trd_px, trd_sz, trd_side, eps):
    # NOTE: signature mirrors the numba kernel; see _fill_kernel_njit.
    raise NotImplementedError


# The real kernel takes the LEVEL SIZE our quote lands behind (queue_pos at
# placement) precomputed per snapshot/side, plus the quote px/sz.  We do the
# level-size lookup in the spec stage so the kernel is branch-light.

def _build_kernel():
    def kernel(
        snap_ts, snap_recv,
        quote_bpx, quote_bsz, quote_b_q0,
        quote_apx, quote_asz, quote_a_q0,
        trd_ts, trd_recv, trd_px, trd_sz, trd_side,
        eps,
    ):
        S = snap_ts.shape[0]
        T = trd_ts.shape[0]
        # Outputs (over-allocate to max possible fills = up to 2 sides * S, but
        # trades can split a fill... cap generously at T+S+1 and grow logic
        # avoided by sizing to S*2 which is the max number of orders; each
        # order fills at most once per trade but total per order capped by size
        # -> number of fill ROWS <= number of trades that hit a live order,
        # bounded by T*2). Use T*2 + 2.
        cap = 2 * T + 2
        f_ts = np.empty(cap, np.int64)
        f_side = np.empty(cap, np.int8)
        f_px = np.empty(cap, np.float64)
        f_sz = np.empty(cap, np.float64)
        nf = 0

        ti = 0  # trade pointer
        for si in range(S):
            # interval [snap_ts[si], next snapshot). Determine the boundary.
            start_ts = snap_ts[si]
            start_recv = snap_recv[si]
            if si + 1 < S:
                end_ts = snap_ts[si + 1]
                end_recv = snap_recv[si + 1]
            else:
                end_ts = 9223372036854775807
                end_recv = 9223372036854775807

            # Skip any trade that sorts strictly BEFORE this snapshot under the
            # event order key (ts, recv, snapshot<trade): such a trade arrived
            # before our order was posted (the order is placed AT this snapshot)
            # so it cannot consume/fill it.  Normally only the pre-first-snapshot
            # trades hit this, but it is a correctness guard at every interval.
            while ti < T:
                tt0 = trd_ts[ti]
                if tt0 > start_ts or (tt0 == start_ts and trd_recv[ti] >= start_recv):
                    break
                ti += 1

            # set up the two orders for this interval
            bpx = quote_bpx[si]
            apx = quote_apx[si]
            has_bid = bpx == bpx  # not NaN
            has_ask = apx == apx
            # remaining order size and current queue ahead
            b_rem = quote_bsz[si] if has_bid else 0.0
            a_rem = quote_asz[si] if has_ask else 0.0
            b_q = quote_b_q0[si] if has_bid else 0.0   # queue ahead (level size)
            a_q = quote_a_q0[si] if has_ask else 0.0

            if not (has_bid or has_ask):
                # still must skip trades in this interval
                while ti < T:
                    tt = trd_ts[ti]; tr = trd_recv[ti]
                    # trade belongs to this interval if it sorts before next snap
                    if tt > end_ts or (tt == end_ts and tr >= end_recv):
                        break
                    ti += 1
                continue

            while ti < T:
                tt = trd_ts[ti]
                tr = trd_recv[ti]
                # stop if trade is at/after the next snapshot boundary
                if tt > end_ts or (tt == end_ts and tr >= end_recv):
                    break
                tside = trd_side[ti]
                tpx = trd_px[ti]
                tsz = trd_sz[ti]
                # bid order: sell-aggressor (tside==-1) at our bid price
                if has_bid and b_rem > 0.0 and tside == -1:
                    d = tpx - bpx
                    if d < 0.0:
                        d = -d
                    if d <= eps:
                        qb = b_q
                        consumed = tsz if tsz < qb else qb
                        b_q = qb - consumed
                        spill = tsz - qb
                        if spill > 0.0:
                            fill = spill if spill < b_rem else b_rem
                            if fill > 0.0:
                                f_ts[nf] = tt
                                f_side[nf] = 1
                                f_px[nf] = bpx
                                f_sz[nf] = fill
                                nf += 1
                                b_rem -= fill
                # ask order: buy-aggressor (tside==+1) at our ask price
                if has_ask and a_rem > 0.0 and tside == 1:
                    d = tpx - apx
                    if d < 0.0:
                        d = -d
                    if d <= eps:
                        qa = a_q
                        consumed = tsz if tsz < qa else qa
                        a_q = qa - consumed
                        spill = tsz - qa
                        if spill > 0.0:
                            fill = spill if spill < a_rem else a_rem
                            if fill > 0.0:
                                f_ts[nf] = tt
                                f_side[nf] = -1
                                f_px[nf] = apx
                                f_sz[nf] = fill
                                nf += 1
                                a_rem -= fill
                ti += 1
        return f_ts[:nf], f_side[:nf], f_px[:nf], f_sz[:nf]
    return kernel


_fill_kernel_pyimpl = _build_kernel()
if _HAVE_NUMBA:
    _fill_kernel_njit = njit(cache=True)(_build_kernel())
else:  # pragma: no cover
    _fill_kernel_njit = _fill_kernel_pyimpl


# --------------------------------------------------------------------- #
# Queue-ahead (level size at our quote price) lookup per snapshot
# --------------------------------------------------------------------- #

def _level_size_for_quotes(w: WindowArrays, quote_px: np.ndarray, side: int) -> np.ndarray:
    """For each snapshot, the resting size at the quote's price on `side`
    (the queue_pos our order lands behind).  NaN quote -> 0.  If the price
    is not present in the visible book, returns NaN (reference would fail to
    track -> no fills); the kernel treats NaN q0 as "no order"."""
    S = quote_px.shape[0]
    if side == +1:
        px_src = w.bid_px
        sz_src = w.bid_sz
        n = w.n_bid
    else:
        px_src = w.ask_px
        sz_src = w.ask_sz
        n = w.n_ask
    return _level_size_kernel(quote_px, px_src, sz_src, n.astype(np.int64),
                              w.K, _PX_EPSILON)


if _HAVE_NUMBA:
    @njit(cache=True)
    def _level_size_kernel(quote_px, px_src, sz_src, n, K, eps):
        S = quote_px.shape[0]
        out = np.full(S, np.nan, np.float64)
        for i in range(S):
            qp = quote_px[i]
            if qp != qp:  # NaN
                continue
            nn = n[i]
            found = False
            for j in range(nn):
                p = px_src[i, j]
                d = p - qp
                if d < 0.0:
                    d = -d
                if d <= eps:
                    out[i] = sz_src[i, j]
                    found = True
                    break
            # not found -> stays NaN
        return out
else:  # pragma: no cover
    def _level_size_kernel(quote_px, px_src, sz_src, n, K, eps):
        S = quote_px.shape[0]
        out = np.full(S, np.nan, np.float64)
        for i in range(S):
            qp = quote_px[i]
            if qp != qp:
                continue
            for j in range(int(n[i])):
                if abs(px_src[i, j] - qp) <= eps:
                    out[i] = sz_src[i, j]
                    break
        return out


# --------------------------------------------------------------------- #
# Public result + driver
# --------------------------------------------------------------------- #

@dataclass
class FastResult:
    fills: List[Fill]
    n_quoter_calls: int
    n_maker_fills: int
    n_taker_fills: int


def _run_specs(w: WindowArrays, bpx, bsz, apx, asz, base_fill_id: int = 0):
    """Given quote-spec arrays, run the fill kernel and build Fill records."""
    b_q0 = _level_size_for_quotes(w, bpx, +1)
    a_q0 = _level_size_for_quotes(w, apx, -1)
    # where q0 is NaN (price not visible), suppress the order by NaN-ing px
    bpx2 = np.where(np.isnan(b_q0), np.nan, bpx)
    apx2 = np.where(np.isnan(a_q0), np.nan, apx)
    b_q0f = np.where(np.isnan(b_q0), 0.0, b_q0)
    a_q0f = np.where(np.isnan(a_q0), 0.0, a_q0)

    f_ts, f_side, f_px, f_sz = _fill_kernel_njit(
        w.snap_ts, w.snap_recv,
        bpx2, bsz, b_q0f,
        apx2, asz, a_q0f,
        w.trd_ts, w.trd_recv, w.trd_px, w.trd_sz, w.trd_side,
        _PX_EPSILON,
    )
    fills: List[Fill] = []
    fid = base_fill_id
    for i in range(f_ts.shape[0]):
        fills.append(Fill(
            fill_id=fid, order_id=0, ts_ns=int(f_ts[i]),
            price=float(f_px[i]), size=float(f_sz[i]),
            side=int(f_side[i]), is_maker=True,
        ))
        fid += 1
    return fills


def simulate(events, quoter_spec, size: float, depth_k: int = 20) -> SimResult:
    """Run one quoter strategy over a window's events via the fast kernel.

    ``quoter_spec`` is one of the spec callables in this module (or a partial
    producing ``(bid_px, bid_sz, ask_px, ask_sz)`` from a ``WindowArrays``).
    Returns a ``SimResult`` (maker-only; takers are not used by the runner's
    quoters) compatible with the reference path's downstream consumers.
    """
    w = decode_window(events, depth_k=depth_k)
    bpx, bsz, apx, asz = quoter_spec(w)
    fills = _run_specs(w, bpx, bsz, apx, asz)
    S = int(w.snap_ts.shape[0])
    return SimResult(
        fills=fills,
        n_events_processed=int(w.snap_ts.shape[0] + w.trd_ts.shape[0]),
        n_snapshot_events=S,
        n_trade_events=int(w.trd_ts.shape[0]),
        n_quoter_calls=S,
        n_maker_fills=len(fills),
        n_taker_fills=0,
        final_orders=[],
    )


def simulate_from_arrays(w: WindowArrays, quoter_spec, size: float) -> SimResult:
    """Same as ``simulate`` but on a pre-decoded ``WindowArrays`` (so a single
    decode can be reused across all the M-branch quoters in a window)."""
    bpx, bsz, apx, asz = quoter_spec(w)
    fills = _run_specs(w, bpx, bsz, apx, asz)
    S = int(w.snap_ts.shape[0])
    return SimResult(
        fills=fills,
        n_events_processed=int(w.snap_ts.shape[0] + w.trd_ts.shape[0]),
        n_snapshot_events=S,
        n_trade_events=int(w.trd_ts.shape[0]),
        n_quoter_calls=S,
        n_maker_fills=len(fills),
        n_taker_fills=0,
        final_orders=[],
    )


__all__ = [
    "WindowArrays", "decode_window", "FastResult",
    "spec_touch", "spec_depth_skew", "spec_rank", "spec_microskew",
    "simulate", "simulate_from_arrays",
]
