"""US equities TAQ (Trades + NBBO) -> engine event stream.

Adapts a "Level 1" trades-and-quotes CSV (consolidated SIP trades +
National-Best-Bid/Offer quotes) into the frozen engine event stream
defined in ``mmsim.ingest.lob`` (SnapshotEvent + TradeEvent).

Source format (flat-file equities trades-and-quotes), 8 comma-
delimited columns with leading spaces::

    Date, Timestamp, EventType, Ticker, Price, Quantity, Exchange, Conditions
    20150616, 04:00:00.026, QUOTE BID NB, IBM, 163.68, 100, ARCA, 00000001

EventType values we consume:
  - ``QUOTE BID NB`` / ``QUOTE ASK NB`` : National Best Bid/Offer
    updates. These carry the depth-1 (top-of-book) NBBO price + size.
  - ``TRADE NB``  : trade printed at the national best price.
  - ``TRADE``     : trade printed at a market-center best price.
  We IGNORE per-market-center ``QUOTE BID`` / ``QUOTE ASK`` (non-NB):
  the engine's equity book is the NBBO (depth-1), so only the NB
  quotes carry our book state. (They are the consolidated National
  Best; per-venue quotes would require an L2 view this feed does not
  provide.)

## The book is depth-1 (queue-at-the-NBBO), stated honestly

This feed is top-of-book NBBO only. There is NO order-by-order or
multi-level depth here. The engine's queue model on this substrate is
therefore **queue-at-the-NBBO (depth-1 FIFO)** -- we track our rank in
the size resting at the National Best, not a real per-order queue.
This is the honest equity-TAQ limitation; deep-queue (M3/M4/M7)
requires the futures depth-10 feed, not this.

## Trade sign is INFERRED, not exchange-provided

CRITICAL: Level-1 equity TAQ does NOT carry an aggressor
buy/sell flag. The ``Conditions`` field is a trade-condition bitmask
(settlement type, ISO, open/close prints, ...), not a side. We infer
the aggressor side with the **quote rule** (Lee-Ready first step)
against the contemporaneous NBBO mid known strictly at-or-before the
trade timestamp:

    price > nbbo_mid  -> buyer-initiated  (+1)
    price < nbbo_mid  -> seller-initiated (-1)
    price == nbbo_mid -> 0 (unknown; Lee-Ready would fall back to the
                            tick rule -- we keep it 0 here and let the
                            sign-rule module M8 own tick/BVC variants)

So ``TradeEvent.side`` here is an *inferred* sign. Any claim built on
it must say "inferred (quote rule)", never "true sign". This is the
exact bias the M8 hypothesis quantifies.

## Causality

The NBBO mid used to sign a trade is the most-recent NB quote pair at
or before the trade's timestamp -- never a future quote. The parser is
single-pass forward in time, so this is structural.

## Timestamps

Source resolution is milliseconds (``HH:MM:SS.mmm``). We promote to
nanoseconds (``* 1_000_000``) to match the engine's ns contract. The
date column anchors the day; ts_ns is ns since the Unix epoch (UTC is
assumed already-normalized at source; the absolute epoch only needs
to be monotone + consistent for markout horizons, which it is).
``recv_ns == ts_ns`` (this feed has no separate receive stamp).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from mmsim.ingest.lob import SnapshotEvent, TradeEvent, EventStream

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


_MS_NS = 1_000_000          # ms -> ns
_DAY_NS = 86_400 * 1_000_000_000

# EventType integer codes for the hot loop (avoid string compares in numba).
_ET_QBID_NB = 0
_ET_QASK_NB = 1
_ET_TRADE = 2
_ET_IGNORE = 3


def _ts_to_ns(date_str: str, time_str: str) -> int:
    """``YYYYMMDD`` + ``HH:MM:SS.mmm`` -> ns since a fixed epoch.

    We anchor to a synthetic epoch = the date's midnight expressed as
    (days-since-1970 * 86400e9) + intraday-ns. Exact calendar accuracy
    is unnecessary; what the engine needs is a globally monotone ns
    clock that preserves real gaps -- this gives it.
    """
    y = int(date_str[0:4]); m = int(date_str[4:6]); d = int(date_str[6:8])
    # days since 1970-01-01 via a cheap proleptic-Gregorian day count.
    days = _days_from_civil(y, m, d)
    hh = int(time_str[0:2]); mm = int(time_str[3:5]); ss = int(time_str[6:8])
    frac = time_str[9:] if len(time_str) > 9 else "0"
    ms = int((frac + "000")[:3])  # pad/truncate to 3 digits
    intraday_ns = ((hh * 3600 + mm * 60 + ss) * 1000 + ms) * _MS_NS
    return days * _DAY_NS + intraday_ns


def _days_from_civil(y: int, m: int, d: int) -> int:
    """Howard Hinnant's days_from_civil: days since 1970-01-01."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


# --------------------------------------------------------------------- #
# Pure-Python reference parser (source of truth for parity)
# --------------------------------------------------------------------- #

def _classify(event_type: str) -> int:
    et = event_type.strip()
    if et == "QUOTE BID NB":
        return _ET_QBID_NB
    if et == "QUOTE ASK NB":
        return _ET_QASK_NB
    if et == "TRADE NB" or et == "TRADE":
        return _ET_TRADE
    return _ET_IGNORE


def parse_taq_arrays_reference(
    path: Union[str, Path],
    *,
    regular_hours_only: bool = True,
) -> dict:
    """Pure-Python reference: read the TAQ CSV into flat numpy arrays.

    Returns a dict of column arrays. This is the slow, obviously-correct
    reference the numba path is checked bit-identical against.
    """
    ts: List[int] = []
    code: List[int] = []
    price: List[float] = []
    qty: List[float] = []
    with open(path, "r") as fh:
        header = fh.readline()  # skip header
        for line in fh:
            parts = line.split(",")
            if len(parts) < 6:
                continue
            et = _classify(parts[2])
            if et == _ET_IGNORE:
                continue
            date_s = parts[0].strip()
            time_s = parts[1].strip()
            if regular_hours_only:
                hh = int(time_s[0:2]); mm = int(time_s[3:5])
                t_min = hh * 60 + mm
                if t_min < 9 * 60 + 30 or t_min >= 16 * 60:
                    continue
            t = _ts_to_ns(date_s, time_s)
            p = float(parts[4])
            q = float(parts[5])
            ts.append(t); code.append(et); price.append(p); qty.append(q)
    return {
        "ts_ns": np.asarray(ts, dtype=np.int64),
        "code": np.asarray(code, dtype=np.int64),
        "price": np.asarray(price, dtype=np.float64),
        "qty": np.asarray(qty, dtype=np.float64),
    }


# --------------------------------------------------------------------- #
# numba hot loop: arrays -> signed-event columns + NBBO book columns
# --------------------------------------------------------------------- #

def build_event_columns_reference(cols: dict) -> dict:
    """Pure-Python reference for the event-builder. Walks the classified
    rows forward, maintaining the running NBBO, emitting:
      - snapshot rows (one per NB quote update) carrying (bid_px, bid_sz,
        ask_px, ask_sz),
      - trade rows carrying (price, size, inferred_sign).
    Returns flat parallel arrays; the parquet writer + EventStream
    builder consume these.
    """
    ts = cols["ts_ns"]; code = cols["code"]; price = cols["price"]; qty = cols["qty"]
    n = ts.shape[0]
    # outputs
    snap_ts = np.empty(n, np.int64); snap_bidpx = np.empty(n, np.float64)
    snap_bidsz = np.empty(n, np.float64); snap_askpx = np.empty(n, np.float64)
    snap_asksz = np.empty(n, np.float64); n_snap = 0
    tr_ts = np.empty(n, np.int64); tr_px = np.empty(n, np.float64)
    tr_sz = np.empty(n, np.float64); tr_side = np.empty(n, np.int64); n_tr = 0

    bid_px = np.nan; bid_sz = 0.0; ask_px = np.nan; ask_sz = 0.0
    for i in range(n):
        c = code[i]
        if c == _ET_QBID_NB:
            bid_px = price[i]; bid_sz = qty[i]
            snap_ts[n_snap] = ts[i]; snap_bidpx[n_snap] = bid_px
            snap_bidsz[n_snap] = bid_sz; snap_askpx[n_snap] = ask_px
            snap_asksz[n_snap] = ask_sz; n_snap += 1
        elif c == _ET_QASK_NB:
            ask_px = price[i]; ask_sz = qty[i]
            snap_ts[n_snap] = ts[i]; snap_bidpx[n_snap] = bid_px
            snap_bidsz[n_snap] = bid_sz; snap_askpx[n_snap] = ask_px
            snap_asksz[n_snap] = ask_sz; n_snap += 1
        else:  # trade
            # quote-rule sign vs the most-recent NBBO mid known so far
            if bid_px == bid_px and ask_px == ask_px:
                mid = (bid_px + ask_px) * 0.5
                if price[i] > mid:
                    s = 1
                elif price[i] < mid:
                    s = -1
                else:
                    s = 0
            else:
                s = 0
            tr_ts[n_tr] = ts[i]; tr_px[n_tr] = price[i]
            tr_sz[n_tr] = qty[i]; tr_side[n_tr] = s; n_tr += 1
    return {
        "snap_ts": snap_ts[:n_snap], "snap_bidpx": snap_bidpx[:n_snap],
        "snap_bidsz": snap_bidsz[:n_snap], "snap_askpx": snap_askpx[:n_snap],
        "snap_asksz": snap_asksz[:n_snap],
        "tr_ts": tr_ts[:n_tr], "tr_px": tr_px[:n_tr],
        "tr_sz": tr_sz[:n_tr], "tr_side": tr_side[:n_tr],
    }


if _HAVE_NUMBA:
    @njit(cache=True)
    def _build_event_columns_numba(ts, code, price, qty):
        n = ts.shape[0]
        snap_ts = np.empty(n, np.int64); snap_bidpx = np.empty(n, np.float64)
        snap_bidsz = np.empty(n, np.float64); snap_askpx = np.empty(n, np.float64)
        snap_asksz = np.empty(n, np.float64); n_snap = 0
        tr_ts = np.empty(n, np.int64); tr_px = np.empty(n, np.float64)
        tr_sz = np.empty(n, np.float64); tr_side = np.empty(n, np.int64); n_tr = 0
        bid_px = np.nan; bid_sz = 0.0; ask_px = np.nan; ask_sz = 0.0
        for i in range(n):
            c = code[i]
            if c == 0:  # QBID_NB
                bid_px = price[i]; bid_sz = qty[i]
                snap_ts[n_snap] = ts[i]; snap_bidpx[n_snap] = bid_px
                snap_bidsz[n_snap] = bid_sz; snap_askpx[n_snap] = ask_px
                snap_asksz[n_snap] = ask_sz; n_snap += 1
            elif c == 1:  # QASK_NB
                ask_px = price[i]; ask_sz = qty[i]
                snap_ts[n_snap] = ts[i]; snap_bidpx[n_snap] = bid_px
                snap_bidsz[n_snap] = bid_sz; snap_askpx[n_snap] = ask_px
                snap_asksz[n_snap] = ask_sz; n_snap += 1
            else:  # trade
                if bid_px == bid_px and ask_px == ask_px:
                    mid = (bid_px + ask_px) * 0.5
                    if price[i] > mid:
                        s = 1
                    elif price[i] < mid:
                        s = -1
                    else:
                        s = 0
                else:
                    s = 0
                tr_ts[n_tr] = ts[i]; tr_px[n_tr] = price[i]
                tr_sz[n_tr] = qty[i]; tr_side[n_tr] = s; n_tr += 1
        return (snap_ts[:n_snap], snap_bidpx[:n_snap], snap_bidsz[:n_snap],
                snap_askpx[:n_snap], snap_asksz[:n_snap],
                tr_ts[:n_tr], tr_px[:n_tr], tr_sz[:n_tr], tr_side[:n_tr])
else:  # pragma: no cover
    _build_event_columns_numba = None


def build_event_columns(cols: dict, *, use_numba: Optional[bool] = None) -> dict:
    """Dispatch: numba production path, else the reference. Bit-identical."""
    if use_numba is None:
        use_numba = _HAVE_NUMBA
    if use_numba and _build_event_columns_numba is not None:
        out = _build_event_columns_numba(
            cols["ts_ns"], cols["code"], cols["price"], cols["qty"])
        keys = ["snap_ts", "snap_bidpx", "snap_bidsz", "snap_askpx",
                "snap_asksz", "tr_ts", "tr_px", "tr_sz", "tr_side"]
        return dict(zip(keys, out))
    return build_event_columns_reference(cols)


# --------------------------------------------------------------------- #
# EventStream + parquet builders
# --------------------------------------------------------------------- #

def columns_to_eventstream(
    ev: dict, *, symbol: str = "EQ", venue: str = "NBBO",
) -> EventStream:
    """Build the merged, time-sorted EventStream the engine consumes.

    Snapshot kind sorts before trade at identical ts (matches lob.load_lob's
    tiebreaker: book state reflects pre-trade).
    """
    out: list = []
    st = ev["snap_ts"]; bp = ev["snap_bidpx"]; bs = ev["snap_bidsz"]
    ap = ev["snap_askpx"]; az = ev["snap_asksz"]
    for i in range(st.shape[0]):
        bids = ((float(bp[i]), float(bs[i])),) if bp[i] == bp[i] else ()
        asks = ((float(ap[i]), float(az[i])),) if ap[i] == ap[i] else ()
        out.append(SnapshotEvent(
            ts_ns=int(st[i]), recv_ns=int(st[i]), symbol=symbol, venue=venue,
            depth=1, bids=bids, asks=asks))
    tt = ev["tr_ts"]; tp = ev["tr_px"]; tsz = ev["tr_sz"]; tsd = ev["tr_side"]
    for i in range(tt.shape[0]):
        out.append(TradeEvent(
            ts_ns=int(tt[i]), recv_ns=int(tt[i]), symbol=symbol, venue=venue,
            price=float(tp[i]), size=float(tsz[i]), side=int(tsd[i])))
    out.sort(key=lambda e: (e.ts_ns, e.recv_ns, 0 if isinstance(e, SnapshotEvent) else 1))
    return out


def write_parquet(ev: dict, snap_path: Union[str, Path], trade_path: Union[str, Path],
                  *, symbol: str = "EQ", venue: str = "NBBO") -> Tuple[int, int]:
    """Write compact per-symbol parquet matching mmsim.ingest.lob's reader
    schema (so load_lob can read it back unchanged). Returns (n_snap, n_trade).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    st = ev["snap_ts"]; bp = ev["snap_bidpx"]; bs = ev["snap_bidsz"]
    ap = ev["snap_askpx"]; az = ev["snap_asksz"]
    bids_col = []; asks_col = []
    for i in range(st.shape[0]):
        bids_col.append([{"px": float(bp[i]), "sz": float(bs[i])}] if bp[i] == bp[i] else [])
        asks_col.append([{"px": float(ap[i]), "sz": float(az[i])}] if ap[i] == ap[i] else [])
    snap_tbl = pa.table({
        "ts_ns": pa.array(st.tolist(), pa.int64()),
        "recv_ns": pa.array(st.tolist(), pa.int64()),
        "symbol": pa.array([symbol] * st.shape[0]),
        "venue": pa.array([venue] * st.shape[0]),
        "depth": pa.array([1] * st.shape[0], pa.int64()),
        "bids": pa.array(bids_col),
        "asks": pa.array(asks_col),
    })
    pq.write_table(snap_tbl, str(snap_path))
    tt = ev["tr_ts"]
    trade_tbl = pa.table({
        "ts_ns": pa.array(tt.tolist(), pa.int64()),
        "recv_ns": pa.array(tt.tolist(), pa.int64()),
        "symbol": pa.array([symbol] * tt.shape[0]),
        "venue": pa.array([venue] * tt.shape[0]),
        "price": pa.array(ev["tr_px"].tolist(), pa.float64()),
        "size": pa.array(ev["tr_sz"].tolist(), pa.float64()),
        "side": pa.array(ev["tr_side"].tolist(), pa.int64()),
    })
    pq.write_table(trade_tbl, str(trade_path))
    return int(st.shape[0]), int(tt.shape[0])


def ingest_taq_csv(
    path: Union[str, Path], *, symbol: str = "EQ", venue: str = "NBBO",
    regular_hours_only: bool = True, use_numba: Optional[bool] = None,
) -> EventStream:
    """End-to-end: TAQ CSV -> EventStream (in memory). Convenience for smoke."""
    cols = parse_taq_arrays_reference(path, regular_hours_only=regular_hours_only)
    ev = build_event_columns(cols, use_numba=use_numba)
    return columns_to_eventstream(ev, symbol=symbol, venue=venue)


__all__ = [
    "parse_taq_arrays_reference", "build_event_columns",
    "build_event_columns_reference", "columns_to_eventstream",
    "write_parquet", "ingest_taq_csv",
]
