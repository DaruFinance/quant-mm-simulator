"""CME futures **TAQ-only** (top-of-book) -> engine event stream.

Companion to :mod:`mmsim.ingest.fut_depth`.  Where ``fut_depth`` builds a
real 10-level book from a depth reconstruction, this builds a **depth-1**
(top-of-book) book from a top-of-book trades-and-quotes (TAQ) reconstruction
alone -- the QUOTE BUY / QUOTE SELL rows that the futures TAQ tape carries at
Level 0 -- while taking the **true exchange aggressor sign** from the same
tape's TRADE AGGRESSOR rows.

Why this exists
---------------
The depth-10 reconstruction underlies the calm 2023 headline; a
nanosecond-resolution TAQ reconstruction spans the additional volatility
regimes.  The paper's *central* result -- spread capture vs post-fill adverse
drift for a touch-quoter -- needs only the L1 mid at fill, the L1 mid at the
markout horizon, the true aggressor sign, and a touch-level fill model.  All
of that is in the TAQ tape.  This adapter therefore lets the *same*
decomposition engine run on stress-regime dates **on one consistent
reconstruction** -- subject to a same-dates TAQ-vs-depth commensurability
control on the 2023 overlap (the depth-10-dependent analyses -- queue-rank
economics, the 10-level Cont OFI -- cannot be reproduced here and are out of
scope for the TAQ path).

Source format (flat-file futures TAQ, 20 cols, header present)::

    UTCDate,UTCTime,LocalDate,LocalTime,SecurityID,Product,Group,Ticker,
    TypeMask,Info,Side,Level,Price,Quantity,Orders,Flags,PriceDecimals,
    MainFraction,SubFraction,PriceDisplayFormat

  - ``TypeMask`` (int, col 8): 161 = QUOTE BUY (bid), 97 = QUOTE SELL
    (ask), 162 = TRADE AGGRESSOR ON BUY (+1), 98 = TRADE AGGRESSOR ON
    SELL (-1), 172 = EMPTY BOOK BUY (bid cleared), 108 = EMPTY BOOK SELL
    (ask cleared).  We branch on the int, never the ``Info`` string.
  - ``Level`` (col 11) = 0 for top-of-book quotes (the only level TAQ
    carries).  ``Price`` col 12, ``Quantity`` col 13 (contracts).
  - ``UTCTime`` is the 15-digit ``HHMMSSnnnnnnnnn`` field decoded by
    :func:`mmsim.ingest.fut_depth._utctime_to_ns_scalar`.

The trade tape (true sign) is read by the *shared* parser
``fut_depth.parse_taq_trades_{fast,reference}`` -- identical extraction to
the depth pipeline, so when both pipelines read the same TAQ file the tape
is byte-identical and ONLY the book reconstruction differs.  That is
exactly what the commensurability control isolates.

Output is the snaps-dict shape of ``fut_depth.build_book_snapshots``
(``snap_ts``, ``bid_px``/``bid_sz``/``ask_px``/``ask_sz`` as ``(n,10)`` with
only level 0 populated, ``dbid``/``dask``), so ``fut_depth.write_parquet``,
``fut_depth.snapshots_to_eventstream`` and ``fut_depth.throttle_top_k_changed``
all consume it unchanged.

Causality
---------
Single forward pass; the L1 book at row ``i`` depends only on quote rows
``<= i``; a trade's sign is exchange-stamped (no contemporaneous-mid
lookup), so there is no quote-vs-trade ordering hazard.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from mmsim.ingest.fut_depth import (
    _MAX_LEVELS,
    _date_to_day_ns,
    _open_text,
    _utctime_to_ns_scalar,
    _utctime_to_ns_vec,
    parse_taq_trades_fast,
    parse_taq_trades_reference,
)

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


# TypeMask quote/empty codes.
_TM_QUOTE_BID = 161   # QUOTE BUY  -> bid side
_TM_QUOTE_ASK = 97    # QUOTE SELL -> ask side
_TM_EMPTY_BID = 172   # EMPTY BOOK BUY  -> bid cleared
_TM_EMPTY_ASK = 108   # EMPTY BOOK SELL -> ask cleared


# --------------------------------------------------------------------- #
# Parse: raw TAQ CSV -> quote-row columns (reference + fast, parity-checked)
# --------------------------------------------------------------------- #

def parse_taq_quotes_reference(path: Union[str, Path]) -> dict:
    """Pure-Python reference: keep only top-of-book quote / empty-book rows.

    Returns parallel arrays: ``ts_ns``, ``qside`` (+1 bid / -1 ask),
    ``px``, ``sz``, and ``empty`` (1 if this is an EMPTY-BOOK row clearing
    the side, else 0).  Source of truth the fast path is checked against.
    """
    ts: List[int] = []
    qside: List[int] = []
    px: List[float] = []
    sz: List[float] = []
    empty: List[int] = []
    day_ns = None
    with _open_text(path) as fh:
        fh.readline()  # header
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 15:
                continue
            try:
                tm = int(parts[8])
            except ValueError:
                continue
            if tm == _TM_QUOTE_BID:
                s, e = +1, 0
            elif tm == _TM_QUOTE_ASK:
                s, e = -1, 0
            elif tm == _TM_EMPTY_BID:
                s, e = +1, 1
            elif tm == _TM_EMPTY_ASK:
                s, e = -1, 1
            else:
                continue
            # top-of-book only (Level col 11); guard non-numeric
            try:
                if int(parts[11]) != 0:
                    continue
            except ValueError:
                continue
            if day_ns is None:
                day_ns = _date_to_day_ns(parts[0].strip())
            t = day_ns + _utctime_to_ns_scalar(int(parts[1]))
            p = 0.0 if e else float(parts[12])
            q = 0.0 if e else float(parts[13])
            ts.append(t); qside.append(s); px.append(p); sz.append(q); empty.append(e)
    return {
        "ts_ns": np.asarray(ts, dtype=np.int64),
        "qside": np.asarray(qside, dtype=np.int64),
        "px": np.asarray(px, dtype=np.float64),
        "sz": np.asarray(sz, dtype=np.float64),
        "empty": np.asarray(empty, dtype=np.int64),
    }


def parse_taq_quotes_fast(path: Union[str, Path]) -> dict:
    """pandas-C fast path, bit-identical to ``parse_taq_quotes_reference``.

    Reads UTCDate, UTCTime, TypeMask, Level, Price, Quantity; keeps
    TypeMask in {161,97,172,108} with Level == 0.
    """
    import pandas as pd
    usecols = [0, 1, 8, 11, 12, 13]  # UTCDate, UTCTime, TypeMask, Level, Price, Quantity
    df = pd.read_csv(
        path, usecols=usecols, header=0,
        dtype={0: str, 1: np.int64, 8: str, 11: str, 12: np.float64, 13: np.float64},
        engine="c", na_filter=False,
    )
    cb = {orig: df.columns[k] for k, orig in enumerate(usecols)}
    tm = pd.to_numeric(df[cb[8]], errors="coerce").to_numpy()
    lvl = pd.to_numeric(df[cb[11]], errors="coerce").to_numpy()
    is_bid = tm == _TM_QUOTE_BID
    is_ask = tm == _TM_QUOTE_ASK
    is_eb = tm == _TM_EMPTY_BID
    is_ea = tm == _TM_EMPTY_ASK
    keep = (is_bid | is_ask | is_eb | is_ea) & (lvl == 0)
    if not keep.any():
        z = lambda d: np.zeros(0, d)
        return {"ts_ns": z(np.int64), "qside": z(np.int64), "px": z(np.float64),
                "sz": z(np.float64), "empty": z(np.int64)}
    day_ns = _date_to_day_ns(str(df[cb[0]].iloc[0]))
    ts = (day_ns + _utctime_to_ns_vec(df[cb[1]].to_numpy().astype(np.int64)))[keep]
    qside = np.where(is_bid | is_eb, 1, -1).astype(np.int64)[keep]
    empty = np.where(is_eb | is_ea, 1, 0).astype(np.int64)[keep]
    px = df[cb[12]].to_numpy().astype(np.float64)[keep]
    sz = df[cb[13]].to_numpy().astype(np.float64)[keep]
    px = np.where(empty == 1, 0.0, px)
    sz = np.where(empty == 1, 0.0, sz)
    return {"ts_ns": ts, "qside": qside, "px": px, "sz": sz, "empty": empty}


# --------------------------------------------------------------------- #
# L1 book builder: per-side quote rows -> two-sided depth-1 snapshots
# (same output dict shape as fut_depth.build_book_snapshots, level-0 only)
# --------------------------------------------------------------------- #

def build_l1_book_reference(cols: dict) -> dict:
    """Walk quote rows forward, carry the unchanged side, emit one two-sided
    depth-1 snapshot per row.  Empty-book rows clear their side (px/sz 0)."""
    ts = cols["ts_ns"]; qs = cols["qside"]; px = cols["px"]; sz = cols["sz"]
    n = ts.shape[0]
    snap_ts = np.empty(n, np.int64)
    bid_px = np.zeros((n, _MAX_LEVELS)); bid_sz = np.zeros((n, _MAX_LEVELS))
    ask_px = np.zeros((n, _MAX_LEVELS)); ask_sz = np.zeros((n, _MAX_LEVELS))
    dbid = np.zeros(n, np.int64); dask = np.zeros(n, np.int64)
    cbp = 0.0; cbs = 0.0; cap = 0.0; cas = 0.0
    cdb = 0; cda = 0
    for i in range(n):
        if qs[i] == +1:
            cbp = px[i]; cbs = sz[i]; cdb = 1 if (px[i] > 0.0 and sz[i] > 0.0) else 0
        else:
            cap = px[i]; cas = sz[i]; cda = 1 if (px[i] > 0.0 and sz[i] > 0.0) else 0
        snap_ts[i] = ts[i]
        bid_px[i, 0] = cbp; bid_sz[i, 0] = cbs
        ask_px[i, 0] = cap; ask_sz[i, 0] = cas
        dbid[i] = cdb; dask[i] = cda
    return {"snap_ts": snap_ts, "bid_px": bid_px, "bid_sz": bid_sz,
            "ask_px": ask_px, "ask_sz": ask_sz, "dbid": dbid, "dask": dask}


if _HAVE_NUMBA:
    @njit(cache=True)
    def _build_l1_book_numba(ts, qs, px, sz):
        n = ts.shape[0]
        L = _MAX_LEVELS
        snap_ts = np.empty(n, np.int64)
        bid_px = np.zeros((n, L)); bid_sz = np.zeros((n, L))
        ask_px = np.zeros((n, L)); ask_sz = np.zeros((n, L))
        dbid = np.zeros(n, np.int64); dask = np.zeros(n, np.int64)
        cbp = 0.0; cbs = 0.0; cap = 0.0; cas = 0.0
        cdb = 0; cda = 0
        for i in range(n):
            if qs[i] == 1:
                cbp = px[i]; cbs = sz[i]
                cdb = 1 if (px[i] > 0.0 and sz[i] > 0.0) else 0
            else:
                cap = px[i]; cas = sz[i]
                cda = 1 if (px[i] > 0.0 and sz[i] > 0.0) else 0
            snap_ts[i] = ts[i]
            bid_px[i, 0] = cbp; bid_sz[i, 0] = cbs
            ask_px[i, 0] = cap; ask_sz[i, 0] = cas
            dbid[i] = cdb; dask[i] = cda
        return snap_ts, bid_px, bid_sz, ask_px, ask_sz, dbid, dask
else:  # pragma: no cover
    _build_l1_book_numba = None


def build_l1_book(cols: dict, *, use_numba: Optional[bool] = None) -> dict:
    """Dispatch: numba production path, else the reference. Bit-identical."""
    if use_numba is None:
        use_numba = _HAVE_NUMBA
    if cols["ts_ns"].shape[0] == 0:
        z2 = np.zeros((0, _MAX_LEVELS))
        return {"snap_ts": np.zeros(0, np.int64), "bid_px": z2, "bid_sz": z2,
                "ask_px": z2, "ask_sz": z2, "dbid": np.zeros(0, np.int64),
                "dask": np.zeros(0, np.int64)}
    if use_numba and _build_l1_book_numba is not None:
        out = _build_l1_book_numba(cols["ts_ns"], cols["qside"], cols["px"], cols["sz"])
        keys = ["snap_ts", "bid_px", "bid_sz", "ask_px", "ask_sz", "dbid", "dask"]
        return dict(zip(keys, out))
    return build_l1_book_reference(cols)


# --------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------- #

def ingest_taq_contract(
    taq_csv: Union[str, Path], *, symbol: str, venue: str = "CME",
    use_numba: Optional[bool] = None, fast: bool = True,
    throttle_k: Optional[int] = 1,
) -> Tuple[dict, dict]:
    """One contract's TAQ CSV -> (snaps_dict, trades_dict).

    ``snaps`` matches ``fut_depth.build_book_snapshots`` output (level-0
    only); ``trades`` carries the true exchange aggressor sign.  Pass the
    pair to ``fut_depth.write_parquet`` / ``snapshots_to_eventstream``.
    ``throttle_k`` (default 1) drops snapshots where the top-1 book is
    unchanged vs the previous KEPT snapshot; None disables throttling.
    """
    from mmsim.ingest.fut_depth import throttle_top_k_changed
    if fast:
        qcols = parse_taq_quotes_fast(taq_csv)
        trades = parse_taq_trades_fast(taq_csv)
    else:
        qcols = parse_taq_quotes_reference(taq_csv)
        trades = parse_taq_trades_reference(taq_csv)
    snaps = build_l1_book(qcols, use_numba=use_numba)
    if throttle_k is not None and snaps["snap_ts"].shape[0] > 0:
        snaps = throttle_top_k_changed(snaps, k=throttle_k)
    return snaps, trades


__all__ = [
    "parse_taq_quotes_reference", "parse_taq_quotes_fast",
    "build_l1_book", "build_l1_book_reference", "ingest_taq_contract",
]
