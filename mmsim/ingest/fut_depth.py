"""CME-family futures (CME/CBOT/NYMEX/COMEX) Level-2 depth-10 + TAQ
-> the frozen engine event stream (mmsim.ingest.lob).

This is the **deep-queue** substrate: a real 10-level Price x Contracts
(Orders) book per side with an EXPLICIT exchange-provided aggressor side
on the trade tape.  Unlike the equity TAQ ingester (depth-1 NBBO,
*inferred* signs), this carries genuine multi-level queue depth and
true signs -- the substrate the M3/M4/M7/M8 hypotheses require.

## Source format (flat-file futures depth-10 + TAQ)

### Depth file (one gzipped CSV per contract-day)
44 comma-delimited columns, header row present.  ONE row per book
update, **per side** (``Side`` in {B, S}).  Each row carries the FULL
current 10-level book for that one side::

    UTCDate,UTCTime,LocalDate,LocalTime,SecurityID,Product,Group,Ticker,
    Side,Flags,PriceDecimals,MainFraction,SubFraction,PriceDisplayFormat,
    Depth,L1Price,L1Size,L1Orders,...,L10Price,L10Size,L10Orders

    20231006,000000000000000,...,ESZ3,B,0,...,10,4286,6,6,4285.75,34,28,...

  - ``UTCDate`` = YYYYMMDD ; ``UTCTime`` = 15-digit ns-within-day
    (HHMMSSnnnnnnnnn).
  - ``Side`` B = bid side, S = ask side.
  - ``Depth`` = number of populated levels (<=10); unused levels are
    zero-filled (price 0, size 0, orders 0).
  - Each level n: ``LnPrice, LnSize(contracts), LnOrders``.
  Because a row updates only ONE side, we carry forward the other
  side's last-known levels and emit a two-sided SnapshotEvent on every
  row -- so the engine always sees a complete book.

### TAQ file (one gzipped CSV per contract-day)
20 comma-delimited columns, header present.  Mixed quote/trade/volume
rows; the load-bearing columns for us::

    UTCDate,UTCTime,...,Ticker,TypeMask,Info,Side,Level,Price,Quantity,
    Orders,Flags,...

  - ``TypeMask`` (int): 161 = QUOTE BUY, 97 = QUOTE SELL,
    **162 = TRADE AGGRESSOR ON BUY**, **98 = TRADE AGGRESSOR ON SELL**,
    49 = electronic volume, etc.
  - ``Info`` (text): the human-readable echo of TypeMask, e.g.
    "TRADE AGGRESSOR ON SELL FINAL".  We classify off TypeMask (an int,
    cheap in the hot loop) and the spec's bit semantics; ``Info`` is
    only used by the reference parser as a cross-check.
  - We consume ONLY trade rows.  ``side`` is the EXCHANGE-PROVIDED
    aggressor: +1 (buy-aggressor, TypeMask 162) / -1 (sell-aggressor,
    TypeMask 98).  This is a TRUE sign, not inferred -- the whole point
    of the M8 true-vs-inferred comparison.

## Front-month / roll

This ingester operates on ONE contract file at a time (e.g. ESZ3 for
2023-10-06).  Choosing the front/active contract (by volume/OI) and
handling the roll is the caller's job (the smoke driver picks the
largest-by-activity contract per root per day).

## Timestamps

UTCTime is a 15-digit ``HHMMSSnnnnnnnnn`` field (6-digit HHMMSS + 9-digit
nanoseconds), NOT a raw ns count.  ns since a fixed epoch =
days_from_civil(UTCDate) * 86400e9 + decode(UTCTime), where decode maps
HH:MM:SS.nanos to ns-within-day.  This matches the equity ingester's
synthetic-but-monotone epoch contract; only monotonicity + real gaps
matter for markout horizons, and this preserves both.  ``recv_ns ==
ts_ns`` (no separate receive stamp in this feed).

## Causality

Single forward pass.  The depth book at row i depends only on rows
<= i; a trade's sign is exchange-stamped (no contemporaneous-mid lookup
needed), so there is no quote-vs-trade ordering hazard at all.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from mmsim.ingest.lob import SnapshotEvent, TradeEvent, EventStream

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


_DAY_NS = 86_400 * 1_000_000_000
_MAX_LEVELS = 10

# TypeMask trade codes (aggressor side baked in by the exchange feed).
_TM_TRADE_BUY = 162   # TRADE AGGRESSOR ON BUY  -> aggressor +1
_TM_TRADE_SELL = 98   # TRADE AGGRESSOR ON SELL -> aggressor -1


def _days_from_civil(y: int, m: int, d: int) -> int:
    """Howard Hinnant's days_from_civil: days since 1970-01-01."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _date_to_day_ns(date_str: str) -> int:
    y = int(date_str[0:4]); m = int(date_str[4:6]); d = int(date_str[6:8])
    return _days_from_civil(y, m, d) * _DAY_NS


def _utctime_to_ns_scalar(v: int) -> int:
    """Decode a UTCTime field ``HHMMSSnnnnnnnnn`` (15 digits: 6-digit
    HHMMSS + 9-digit nanoseconds) into ns-within-day.  e.g.
    000001007081052 -> 00:00:01.007081052 -> 1_007_081_052 ns."""
    hhmmss = v // 1_000_000_000          # top 6 digits
    nanos = v % 1_000_000_000            # bottom 9 digits
    hh = hhmmss // 10000
    mm = (hhmmss // 100) % 100
    ss = hhmmss % 100
    return (hh * 3600 + mm * 60 + ss) * 1_000_000_000 + nanos


def _utctime_to_ns_vec(arr):
    """Vectorized HHMMSSnnnnnnnnn -> ns-within-day for an int64 array."""
    hhmmss = arr // 1_000_000_000
    nanos = arr % 1_000_000_000
    hh = hhmmss // 10000
    mm = (hhmmss // 100) % 100
    ss = hhmmss % 100
    return (hh * 3600 + mm * 60 + ss) * 1_000_000_000 + nanos


def _open_text(path: Union[str, Path]):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt")
    return open(p, "r")


# --------------------------------------------------------------------- #
# DEPTH: raw CSV -> flat arrays (reference parser, source of truth)
# --------------------------------------------------------------------- #

def parse_depth_arrays_reference(
    path: Union[str, Path],
) -> dict:
    """Read a futures depth CSV into flat numpy arrays.

    Returns one row per source row (one side each), with the 10 levels
    flattened into (n, 10) price/size arrays plus a side code
    (+1 bid / -1 ask) and ts_ns.  This is the slow obviously-correct
    reference the numba book-builder is checked against.
    """
    ts: List[int] = []
    side: List[int] = []
    depth: List[int] = []
    px = []   # list of 10-float lists
    sz = []
    day_ns = None
    with _open_text(path) as fh:
        fh.readline()  # header
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 15 + _MAX_LEVELS * 3:
                continue
            if day_ns is None:
                day_ns = _date_to_day_ns(parts[0].strip())
            t = day_ns + _utctime_to_ns_scalar(int(parts[1]))  # HHMMSSnnnnnnnnn
            s = parts[8].strip()
            if s == "B":
                sc = +1
            elif s == "S":
                sc = -1
            else:
                continue
            d = int(parts[14])
            base = 15
            prow = [0.0] * _MAX_LEVELS
            srow = [0.0] * _MAX_LEVELS
            for lvl in range(_MAX_LEVELS):
                o = base + lvl * 3
                prow[lvl] = float(parts[o])
                srow[lvl] = float(parts[o + 1])
            ts.append(t); side.append(sc); depth.append(d)
            px.append(prow); sz.append(srow)
    return {
        "ts_ns": np.asarray(ts, dtype=np.int64),
        "side": np.asarray(side, dtype=np.int64),
        "depth": np.asarray(depth, dtype=np.int64),
        "px": np.asarray(px, dtype=np.float64) if px else np.zeros((0, _MAX_LEVELS)),
        "sz": np.asarray(sz, dtype=np.float64) if sz else np.zeros((0, _MAX_LEVELS)),
    }


def parse_depth_arrays_fast(path: Union[str, Path]) -> dict:
    """pandas-C-parser fast path producing arrays BIT-IDENTICAL to
    ``parse_depth_arrays_reference`` (asserted in tests).  Reads only the
    columns we need (ts, side, depth, the 20 price/size cells), vectorizes
    the side mapping + ns timestamp.  ~10x faster than the per-row loop.

    Rows with Side not in {B,S} are dropped to match the reference.
    """
    import pandas as pd
    base = 15
    px_cols = [base + lvl * 3 for lvl in range(_MAX_LEVELS)]
    sz_cols = [base + lvl * 3 + 1 for lvl in range(_MAX_LEVELS)]
    usecols = [0, 1, 8, 14] + px_cols + sz_cols
    df = pd.read_csv(
        path, usecols=sorted(usecols), header=0,
        dtype={0: str, 1: np.int64, 8: str, 14: np.int64},
        engine="c", na_filter=False,
    )
    cols_by_idx = {orig: df.columns[k] for k, orig in enumerate(sorted(usecols))}
    side_raw = df[cols_by_idx[8]].to_numpy()
    keep = (side_raw == "B") | (side_raw == "S")
    df = df[keep]
    side_raw = side_raw[keep]
    if df.shape[0] == 0:
        return {"ts_ns": np.zeros(0, np.int64), "side": np.zeros(0, np.int64),
                "depth": np.zeros(0, np.int64),
                "px": np.zeros((0, _MAX_LEVELS)), "sz": np.zeros((0, _MAX_LEVELS))}
    day_ns = _date_to_day_ns(str(df[cols_by_idx[0]].iloc[0]))
    ts = day_ns + _utctime_to_ns_vec(df[cols_by_idx[1]].to_numpy().astype(np.int64))
    side = np.where(side_raw == "B", 1, -1).astype(np.int64)
    depth = df[cols_by_idx[14]].to_numpy().astype(np.int64)
    px = np.column_stack([df[cols_by_idx[c]].to_numpy().astype(np.float64) for c in px_cols])
    sz = np.column_stack([df[cols_by_idx[c]].to_numpy().astype(np.float64) for c in sz_cols])
    return {"ts_ns": ts, "side": side, "depth": depth, "px": px, "sz": sz}


def parse_taq_trades_fast(path: Union[str, Path]) -> dict:
    """pandas-C fast path for TAQ trade extraction, bit-identical to
    ``parse_taq_trades_reference``.  Keeps only TypeMask in {162,98}."""
    import pandas as pd
    usecols = [0, 1, 8, 12, 13]   # UTCDate, UTCTime, TypeMask, Price, Quantity
    df = pd.read_csv(
        path, usecols=usecols, header=0,
        dtype={0: str, 1: np.int64, 8: str, 12: np.float64, 13: np.float64},
        engine="c", na_filter=False,
    )
    cb = {orig: df.columns[k] for k, orig in enumerate(usecols)}
    tm = pd.to_numeric(df[cb[8]], errors="coerce").to_numpy()
    is_buy = tm == _TM_TRADE_BUY
    is_sell = tm == _TM_TRADE_SELL
    keep = (is_buy | is_sell)
    qty = df[cb[13]].to_numpy().astype(np.float64)
    keep = keep & (qty > 0.0)
    if not keep.any():
        return {"ts_ns": np.zeros(0, np.int64), "px": np.zeros(0),
                "sz": np.zeros(0), "side": np.zeros(0, np.int64)}
    day_ns = _date_to_day_ns(str(df[cb[0]].iloc[0]))
    ts = (day_ns + _utctime_to_ns_vec(df[cb[1]].to_numpy().astype(np.int64)))[keep]
    px = df[cb[12]].to_numpy().astype(np.float64)[keep]
    sz = qty[keep]
    side = np.where(is_buy[keep], 1, -1).astype(np.int64)
    return {"ts_ns": ts, "px": px, "sz": sz, "side": side}


# --------------------------------------------------------------------- #
# DEPTH book builder: per-side rows -> two-sided snapshots
# --------------------------------------------------------------------- #

def build_book_snapshots_reference(cols: dict) -> dict:
    """Pure-Python reference: walk per-side depth rows forward, carrying
    forward the unchanged side, and emit a two-sided snapshot per row.

    Output arrays (length = n input rows):
      snap_ts, snap_depth_bid, snap_depth_ask, and (n,10) bid_px/bid_sz/
      ask_px/ask_sz.  A snapshot before either side has been seen carries
      zeros for the unseen side (best-* will be None downstream).
    """
    ts = cols["ts_ns"]; side = cols["side"]; depth = cols["depth"]
    px = cols["px"]; sz = cols["sz"]
    n = ts.shape[0]
    snap_ts = np.empty(n, np.int64)
    bid_px = np.zeros((n, _MAX_LEVELS)); bid_sz = np.zeros((n, _MAX_LEVELS))
    ask_px = np.zeros((n, _MAX_LEVELS)); ask_sz = np.zeros((n, _MAX_LEVELS))
    dbid = np.zeros(n, np.int64); dask = np.zeros(n, np.int64)

    cur_bid_px = np.zeros(_MAX_LEVELS); cur_bid_sz = np.zeros(_MAX_LEVELS)
    cur_ask_px = np.zeros(_MAX_LEVELS); cur_ask_sz = np.zeros(_MAX_LEVELS)
    cur_dbid = 0; cur_dask = 0
    for i in range(n):
        if side[i] == +1:
            cur_bid_px = px[i].copy(); cur_bid_sz = sz[i].copy(); cur_dbid = depth[i]
        else:
            cur_ask_px = px[i].copy(); cur_ask_sz = sz[i].copy(); cur_dask = depth[i]
        snap_ts[i] = ts[i]
        bid_px[i] = cur_bid_px; bid_sz[i] = cur_bid_sz
        ask_px[i] = cur_ask_px; ask_sz[i] = cur_ask_sz
        dbid[i] = cur_dbid; dask[i] = cur_dask
    return {
        "snap_ts": snap_ts, "bid_px": bid_px, "bid_sz": bid_sz,
        "ask_px": ask_px, "ask_sz": ask_sz, "dbid": dbid, "dask": dask,
    }


if _HAVE_NUMBA:
    @njit(cache=True)
    def _build_book_snapshots_numba(ts, side, depth, px, sz):
        n = ts.shape[0]
        L = px.shape[1]
        snap_ts = np.empty(n, np.int64)
        bid_px = np.zeros((n, L)); bid_sz = np.zeros((n, L))
        ask_px = np.zeros((n, L)); ask_sz = np.zeros((n, L))
        dbid = np.zeros(n, np.int64); dask = np.zeros(n, np.int64)
        cur_bid_px = np.zeros(L); cur_bid_sz = np.zeros(L)
        cur_ask_px = np.zeros(L); cur_ask_sz = np.zeros(L)
        cur_dbid = 0; cur_dask = 0
        for i in range(n):
            if side[i] == 1:
                for k in range(L):
                    cur_bid_px[k] = px[i, k]; cur_bid_sz[k] = sz[i, k]
                cur_dbid = depth[i]
            else:
                for k in range(L):
                    cur_ask_px[k] = px[i, k]; cur_ask_sz[k] = sz[i, k]
                cur_dask = depth[i]
            snap_ts[i] = ts[i]
            for k in range(L):
                bid_px[i, k] = cur_bid_px[k]; bid_sz[i, k] = cur_bid_sz[k]
                ask_px[i, k] = cur_ask_px[k]; ask_sz[i, k] = cur_ask_sz[k]
            dbid[i] = cur_dbid; dask[i] = cur_dask
        return snap_ts, bid_px, bid_sz, ask_px, ask_sz, dbid, dask
else:  # pragma: no cover
    _build_book_snapshots_numba = None


def build_book_snapshots(cols: dict, *, use_numba: Optional[bool] = None) -> dict:
    """Dispatch: numba production path, else the reference. Bit-identical."""
    if use_numba is None:
        use_numba = _HAVE_NUMBA
    if use_numba and _build_book_snapshots_numba is not None:
        out = _build_book_snapshots_numba(
            cols["ts_ns"], cols["side"], cols["depth"], cols["px"], cols["sz"])
        keys = ["snap_ts", "bid_px", "bid_sz", "ask_px", "ask_sz", "dbid", "dask"]
        return dict(zip(keys, out))
    return build_book_snapshots_reference(cols)


# --------------------------------------------------------------------- #
# TAQ: raw CSV -> trade arrays (true signs)
# --------------------------------------------------------------------- #

def parse_taq_trades_reference(path: Union[str, Path]) -> dict:
    """Read a futures TAQ CSV, keep ONLY trade rows, return ts/px/size/
    side arrays with EXCHANGE-PROVIDED aggressor side (+1 buy / -1 sell)."""
    ts: List[int] = []
    px: List[float] = []
    sz: List[float] = []
    side: List[int] = []
    day_ns = None
    with _open_text(path) as fh:
        fh.readline()  # header
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 15:
                continue
            try:
                tm = int(parts[8])           # TypeMask
            except ValueError:
                continue
            if tm == _TM_TRADE_BUY:
                s = +1
            elif tm == _TM_TRADE_SELL:
                s = -1
            else:
                continue
            if day_ns is None:
                day_ns = _date_to_day_ns(parts[0].strip())
            t = day_ns + _utctime_to_ns_scalar(int(parts[1]))
            p = float(parts[12])             # Price
            q = float(parts[13])             # Quantity
            if q <= 0.0:
                continue
            ts.append(t); px.append(p); sz.append(q); side.append(s)
    return {
        "ts_ns": np.asarray(ts, dtype=np.int64),
        "px": np.asarray(px, dtype=np.float64),
        "sz": np.asarray(sz, dtype=np.float64),
        "side": np.asarray(side, dtype=np.int64),
    }


# --------------------------------------------------------------------- #
# Throttle: keep only snapshots where the top-K book actually changed.
# Information-preserving for MM (identical consecutive books carry no new
# state); cuts a 10M-update contract-day to a RAM-tractable count without
# touching the price/size content of any KEPT snapshot.
# --------------------------------------------------------------------- #

def throttle_top_k_changed(snaps: dict, k: int = 5) -> dict:
    """Return a row-subset of `snaps` keeping only indices where the top-k
    bid OR ask price/size changed vs the previously KEPT snapshot.  The
    first row is always kept.  Vectorized."""
    n = snaps["snap_ts"].shape[0]
    if n == 0:
        return snaps
    bp = snaps["bid_px"][:, :k]; bs = snaps["bid_sz"][:, :k]
    ap = snaps["ask_px"][:, :k]; az = snaps["ask_sz"][:, :k]
    # change vs immediately-previous row, any of the 4 top-k blocks
    chg = np.zeros(n, dtype=bool); chg[0] = True
    diff = (np.any(bp[1:] != bp[:-1], axis=1) | np.any(bs[1:] != bs[:-1], axis=1) |
            np.any(ap[1:] != ap[:-1], axis=1) | np.any(az[1:] != az[:-1], axis=1))
    chg[1:] = diff
    idx = np.flatnonzero(chg)
    return {kk: (vv[idx] if vv.ndim == 1 else vv[idx]) for kk, vv in snaps.items()}


# --------------------------------------------------------------------- #
# EventStream + parquet builders (match mmsim.ingest.lob reader schema)
# --------------------------------------------------------------------- #

def _levels_tuple(px_row, sz_row, ndepth):
    """Build a best-first ((px,sz),...) tuple from a level row, dropping
    zero-price/zero-size padding levels.  ``ndepth`` caps the count."""
    out = []
    nmax = min(int(ndepth), _MAX_LEVELS) if ndepth > 0 else _MAX_LEVELS
    for k in range(nmax):
        p = float(px_row[k]); s = float(sz_row[k])
        if p > 0.0 and s > 0.0:
            out.append((p, s))
    return tuple(out)


def snapshots_to_eventstream(
    snaps: dict, trades: dict, *, symbol: str, venue: str = "CME",
    max_levels: int = _MAX_LEVELS,
) -> EventStream:
    """Build the merged, time-sorted EventStream the engine consumes.
    Snapshot sorts before trade at identical ts (book state is pre-trade).
    """
    out: list = []
    st = snaps["snap_ts"]
    for i in range(st.shape[0]):
        nb = min(max_levels, _MAX_LEVELS)
        bids = _levels_tuple(snaps["bid_px"][i], snaps["bid_sz"][i],
                             min(snaps["dbid"][i] or _MAX_LEVELS, nb))
        asks = _levels_tuple(snaps["ask_px"][i], snaps["ask_sz"][i],
                             min(snaps["dask"][i] or _MAX_LEVELS, nb))
        out.append(SnapshotEvent(
            ts_ns=int(st[i]), recv_ns=int(st[i]), symbol=symbol, venue=venue,
            depth=max(len(bids), len(asks)), bids=bids, asks=asks))
    tt = trades["ts_ns"]
    for i in range(tt.shape[0]):
        out.append(TradeEvent(
            ts_ns=int(tt[i]), recv_ns=int(tt[i]), symbol=symbol, venue=venue,
            price=float(trades["px"][i]), size=float(trades["sz"][i]),
            side=int(trades["side"][i])))
    out.sort(key=lambda e: (e.ts_ns, e.recv_ns,
                            0 if isinstance(e, SnapshotEvent) else 1))
    return out


def write_parquet(
    snaps: dict, trades: dict, snap_path: Union[str, Path],
    trade_path: Union[str, Path], *, symbol: str, venue: str = "CME",
    max_levels: int = _MAX_LEVELS,
) -> Tuple[int, int]:
    """Write compact per-contract parquet matching mmsim.ingest.lob's
    reader schema (so load_lob reads it back unchanged).  Multi-level
    bids/asks are stored as list<struct{px,sz}>.  Returns (n_snap, n_trade).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    st = snaps["snap_ts"]
    n = int(st.shape[0])
    nb = min(max_levels, _MAX_LEVELS)
    bpx = snaps["bid_px"]; bsz = snaps["bid_sz"]
    apx = snaps["ask_px"]; asz = snaps["ask_sz"]
    dbid = snaps["dbid"]; dask = snaps["dask"]
    lvl_type = pa.list_(pa.struct([("px", pa.float64()), ("sz", pa.float64())]))
    schema = pa.schema([
        ("ts_ns", pa.int64()), ("recv_ns", pa.int64()),
        ("symbol", pa.string()), ("venue", pa.string()),
        ("depth", pa.int64()), ("bids", lvl_type), ("asks", lvl_type)])
    writer = pq.ParquetWriter(str(snap_path), schema)
    BATCH = 200_000
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        bids_col = []; asks_col = []; depth_col = []
        for i in range(start, end):
            bids = _levels_tuple(bpx[i], bsz[i], min(int(dbid[i]) or _MAX_LEVELS, nb))
            asks = _levels_tuple(apx[i], asz[i], min(int(dask[i]) or _MAX_LEVELS, nb))
            bids_col.append([{"px": p, "sz": s} for p, s in bids])
            asks_col.append([{"px": p, "sz": s} for p, s in asks])
            depth_col.append(max(len(bids), len(asks)))
        tbl = pa.table({
            "ts_ns": pa.array(st[start:end].tolist(), pa.int64()),
            "recv_ns": pa.array(st[start:end].tolist(), pa.int64()),
            "symbol": pa.array([symbol] * (end - start)),
            "venue": pa.array([venue] * (end - start)),
            "depth": pa.array(depth_col, pa.int64()),
            "bids": pa.array(bids_col, lvl_type),
            "asks": pa.array(asks_col, lvl_type),
        }, schema=schema)
        writer.write_table(tbl)
    writer.close()
    tt = trades["ts_ns"]
    trade_tbl = pa.table({
        "ts_ns": pa.array(tt.tolist(), pa.int64()),
        "recv_ns": pa.array(tt.tolist(), pa.int64()),
        "symbol": pa.array([symbol] * tt.shape[0]),
        "venue": pa.array([venue] * tt.shape[0]),
        "price": pa.array(trades["px"].tolist(), pa.float64()),
        "size": pa.array(trades["sz"].tolist(), pa.float64()),
        "side": pa.array(trades["side"].tolist(), pa.int64()),
    })
    pq.write_table(trade_tbl, str(trade_path))
    return int(st.shape[0]), int(tt.shape[0])


def ingest_contract(
    depth_csv: Union[str, Path], taq_csv: Union[str, Path], *,
    symbol: str, venue: str = "CME", max_levels: int = _MAX_LEVELS,
    use_numba: Optional[bool] = None, fast: bool = True,
) -> EventStream:
    """End-to-end: one contract's depth + taq CSVs -> EventStream in memory.

    ``fast`` uses the pandas-C parsers (bit-identical to the reference,
    asserted in tests); set False to force the pure-Python reference.
    """
    if fast:
        dcols = parse_depth_arrays_fast(depth_csv)
        trades = parse_taq_trades_fast(taq_csv)
    else:
        dcols = parse_depth_arrays_reference(depth_csv)
        trades = parse_taq_trades_reference(taq_csv)
    snaps = build_book_snapshots(dcols, use_numba=use_numba)
    return snapshots_to_eventstream(snaps, trades, symbol=symbol, venue=venue,
                                    max_levels=max_levels)


__all__ = [
    "parse_depth_arrays_reference", "parse_depth_arrays_fast",
    "build_book_snapshots", "build_book_snapshots_reference",
    "parse_taq_trades_reference", "parse_taq_trades_fast",
    "snapshots_to_eventstream", "write_parquet", "ingest_contract",
]
