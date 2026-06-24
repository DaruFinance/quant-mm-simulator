"""Unit tests for the futures depth-10 + TAQ ingester (mmsim.ingest.fut_depth).

Covers: numba<->reference parity (book builder), exchange-true side
correctness, depth-10 book integrity (levels, ordering, carry-forward),
causality (no future row changes a past snapshot), and parquet round-trip
back through mmsim.ingest.lob.load_lob.

Fixtures are written as tiny synthetic CSVs in the exact column layout
so the test is hermetic; a separate (skipped-by-default) parity test runs
on the real ESZ3 slice when present.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np
import pytest

from mmsim.ingest import fut_depth as fd
from mmsim.ingest.lob import SnapshotEvent, TradeEvent, load_lob


_DEPTH_HEADER = (
    "UTCDate,UTCTime,LocalDate,LocalTime,SecurityID,Product,Group,Ticker,"
    "Side,Flags,PriceDecimals,MainFraction,SubFraction,PriceDisplayFormat,Depth,"
    + ",".join(f"L{n}Price,L{n}Size,L{n}Orders" for n in range(1, 11))
)
_TAQ_HEADER = (
    "UTCDate,UTCTime,LocalDate,LocalTime,SecurityID,Product,Group,Ticker,"
    "TypeMask,Info,Side,Level,Price,Quantity,Orders,Flags,PriceDecimals,"
    "MainFraction,SubFraction,PriceDisplayFormat"
)


def _depth_row(date, utctime, side, depth, levels):
    """levels = list of (px,sz,orders) best-first; padded to 10."""
    lv = list(levels) + [(0.0, 0, 0)] * (10 - len(levels))
    cells = [date, f"{utctime:015d}", date, f"{utctime:015d}", "1", "ES", "ES",
             "ESZ3", side, "0", "2", "0", "0", "0", str(depth)]
    for p, s, o in lv:
        cells += [str(p), str(s), str(o)]
    return ",".join(cells)


def _taq_row(date, utctime, typemask, info, side, price, qty):
    cells = [date, f"{utctime:015d}", date, f"{utctime:015d}", "1", "ES", "ES",
             "ESZ3", str(typemask), info, side, "0", str(price), str(qty),
             "0", "0", "2", "0", "0", "0"]
    return ",".join(cells)


@pytest.fixture
def synth_depth(tmp_path):
    p = tmp_path / "depth.csv"
    rows = [_DEPTH_HEADER]
    # t0: bid side 3 levels
    rows.append(_depth_row("20231006", 1000, "B", 3,
                           [(100.0, 5, 2), (99.75, 10, 3), (99.5, 7, 1)]))
    # t0: ask side 3 levels
    rows.append(_depth_row("20231006", 1000, "S", 3,
                           [(100.25, 4, 2), (100.5, 8, 3), (100.75, 6, 2)]))
    # t1: bid updates only (ask carries forward)
    rows.append(_depth_row("20231006", 2000, "B", 3,
                           [(100.0, 9, 4), (99.75, 10, 3), (99.5, 7, 1)]))
    # t2: full depth-10 bid
    rows.append(_depth_row("20231006", 3000, "B", 10,
                           [(100.0, 1, 1), (99.75, 2, 1), (99.5, 3, 1),
                            (99.25, 4, 1), (99.0, 5, 1), (98.75, 6, 1),
                            (98.5, 7, 1), (98.25, 8, 1), (98.0, 9, 1),
                            (97.75, 10, 1)]))
    p.write_text("\n".join(rows) + "\n")
    return p


@pytest.fixture
def synth_taq(tmp_path):
    p = tmp_path / "taq.csv"
    rows = [_TAQ_HEADER]
    rows.append(_taq_row("20231006", 1500, 161, "QUOTE BUY FINAL", "B", 100.0, 5))
    rows.append(_taq_row("20231006", 1600, 98, "TRADE AGGRESSOR ON SELL FINAL", "S", 100.0, 3))
    rows.append(_taq_row("20231006", 1700, 162, "TRADE AGGRESSOR ON BUY FINAL", "B", 100.25, 2))
    rows.append(_taq_row("20231006", 1800, 49, "ELECTRONIC VOLUME FINAL", "", 0, 99))
    p.write_text("\n".join(rows) + "\n")
    return p


def test_utctime_decode():
    # 00:00:01.007081052 UTC -> 1_007_081_052 ns within day
    assert fd._utctime_to_ns_scalar(1_007_081_052) == 1_007_081_052
    # 13:30:00.000000000 -> (13*3600+30*60)*1e9
    assert fd._utctime_to_ns_scalar(133000_000000000) == (13 * 3600 + 30 * 60) * 1_000_000_000
    # 14:29:59.500000000
    expect = (14 * 3600 + 29 * 60 + 59) * 1_000_000_000 + 500_000_000
    assert fd._utctime_to_ns_scalar(142959_500000000) == expect
    # vectorized agrees with scalar
    arr = np.array([1_007_081_052, 133000_000000000, 142959_500000000], np.int64)
    vec = fd._utctime_to_ns_vec(arr)
    for i, v in enumerate(arr):
        assert vec[i] == fd._utctime_to_ns_scalar(int(v))


def test_depth_parse_shapes(synth_depth):
    cols = fd.parse_depth_arrays_reference(synth_depth)
    assert cols["ts_ns"].shape[0] == 4
    assert cols["px"].shape == (4, 10)
    # ts monotone non-decreasing
    assert np.all(np.diff(cols["ts_ns"]) >= 0)
    # sides: B,S,B,B -> +1,-1,+1,+1
    assert list(cols["side"]) == [1, -1, 1, 1]


def test_book_builder_carry_forward(synth_depth):
    cols = fd.parse_depth_arrays_reference(synth_depth)
    snaps = fd.build_book_snapshots_reference(cols)
    # row 0 (B): ask side not yet seen -> ask all zero
    assert snaps["ask_px"][0, 0] == 0.0
    # row 1 (S): bid carried from row 0
    assert snaps["bid_px"][1, 0] == 100.0 and snaps["bid_sz"][1, 0] == 5.0
    # row 2 (B updates): ask carried from row 1
    assert snaps["ask_px"][2, 0] == 100.25 and snaps["ask_sz"][2, 0] == 4.0
    # row 2 best bid size updated to 9
    assert snaps["bid_sz"][2, 0] == 9.0


@pytest.mark.skipif(not fd._HAVE_NUMBA, reason="numba unavailable")
def test_book_builder_numba_parity(synth_depth):
    cols = fd.parse_depth_arrays_reference(synth_depth)
    ref = fd.build_book_snapshots(cols, use_numba=False)
    nb = fd.build_book_snapshots(cols, use_numba=True)
    for k in ref:
        assert np.array_equal(ref[k], nb[k]), f"mismatch in {k}"


def test_taq_true_side(synth_taq):
    tr = fd.parse_taq_trades_reference(synth_taq)
    # only 2 trade rows kept (quote + volume dropped)
    assert tr["ts_ns"].shape[0] == 2
    # first trade: sell-aggressor -1; second: buy-aggressor +1
    assert list(tr["side"]) == [-1, 1]
    assert tr["px"][0] == 100.0 and tr["sz"][0] == 3.0
    assert tr["px"][1] == 100.25 and tr["sz"][1] == 2.0


def test_depth10_book_integrity(synth_depth):
    cols = fd.parse_depth_arrays_reference(synth_depth)
    snaps = fd.build_book_snapshots(cols)
    trades = {"ts_ns": np.array([], np.int64), "px": np.array([], np.float64),
              "sz": np.array([], np.float64), "side": np.array([], np.int64)}
    stream = fd.snapshots_to_eventstream(snaps, trades, symbol="ESZ3")
    snap_events = [e for e in stream if isinstance(e, SnapshotEvent)]
    last = snap_events[-1]  # the depth-10 bid row (ts=3000)
    # 10 bid levels, strictly descending in price (best-first)
    assert len(last.bids) == 10
    bp = [p for p, _ in last.bids]
    assert all(bp[i] > bp[i + 1] for i in range(len(bp) - 1))
    # asks carried forward = 3 levels, strictly ascending
    ap = [p for p, _ in last.asks]
    assert len(ap) == 3 and all(ap[i] < ap[i + 1] for i in range(len(ap) - 1))
    # no zero-padding leaked into the level tuples
    assert all(p > 0 and s > 0 for p, s in last.bids + last.asks)


def test_causality_no_future_leak(synth_depth):
    """A snapshot's book at row i must depend only on rows <= i.
    Truncating the stream after row i must not change snapshot i."""
    cols = fd.parse_depth_arrays_reference(synth_depth)
    snaps_full = fd.build_book_snapshots(cols)
    # truncate to first 2 rows
    cut = {k: v[:2] for k, v in cols.items()}
    snaps_cut = fd.build_book_snapshots(cut)
    for k in ("snap_ts", "bid_px", "bid_sz", "ask_px", "ask_sz"):
        assert np.array_equal(snaps_full[k][:2], snaps_cut[k]), f"leak in {k}"


def test_parquet_round_trip(tmp_path, synth_depth, synth_taq):
    cols = fd.parse_depth_arrays_reference(synth_depth)
    snaps = fd.build_book_snapshots(cols)
    trades = fd.parse_taq_trades_reference(synth_taq)
    sp = tmp_path / "snap.parquet"; tp = tmp_path / "trade.parquet"
    n_snap, n_tr = fd.write_parquet(snaps, trades, sp, tp, symbol="ESZ3")
    assert n_snap == 4 and n_tr == 2
    stream = load_lob(sp, tp)
    snap_events = [e for e in stream if isinstance(e, SnapshotEvent)]
    trade_events = [e for e in stream if isinstance(e, TradeEvent)]
    assert len(snap_events) == 4 and len(trade_events) == 2
    # multi-level survived the round trip
    last_snap = max(snap_events, key=lambda e: e.ts_ns)
    assert len(last_snap.bids) == 10
    # true side survived
    assert sorted(t.side for t in trade_events) == [-1, 1]
    # merge order: at equal ts snapshot precedes trade (none here share ts,
    # but the stream must be globally ts-sorted)
    ts = [e.ts_ns for e in stream]
    assert ts == sorted(ts)


# --------------------------------------------------------------------- #
# Real-slice parity (runs only when the ESZ3 extract is present)
# --------------------------------------------------------------------- #

_FUT_STAGE = os.environ.get("FUT_STAGE_DIR", "data/fut_stage")
_REAL_DEPTH = os.environ.get("REAL_DEPTH_SLICE", os.path.join(_FUT_STAGE, "ESZ3_depth.csv.gz"))
_REAL_TAQ = os.environ.get("REAL_TAQ_SLICE", os.path.join(_FUT_STAGE, "ESZ3_taq.csv.gz"))


@pytest.mark.skipif(not os.path.exists(_REAL_DEPTH),
                    reason="real ESZ3 depth slice not present")
def test_real_slice_numba_parity():
    """First 200k depth rows: numba book builder bit-identical to reference."""
    cols = fd.parse_depth_arrays_reference(_REAL_DEPTH)
    cut = {k: (v[:200_000] if v.ndim == 1 else v[:200_000]) for k, v in cols.items()}
    ref = fd.build_book_snapshots(cut, use_numba=False)
    nb = fd.build_book_snapshots(cut, use_numba=True)
    for k in ref:
        assert np.array_equal(ref[k], nb[k]), f"real-slice mismatch in {k}"


def test_fast_equals_reference_depth(synth_depth):
    ref = fd.parse_depth_arrays_reference(synth_depth)
    fast = fd.parse_depth_arrays_fast(synth_depth)
    for k in ref:
        assert np.array_equal(ref[k], fast[k]), f"fast!=ref depth {k}"


def test_fast_equals_reference_taq(synth_taq):
    ref = fd.parse_taq_trades_reference(synth_taq)
    fast = fd.parse_taq_trades_fast(synth_taq)
    for k in ref:
        assert np.array_equal(ref[k], fast[k]), f"fast!=ref taq {k}"


@pytest.mark.skipif(not os.path.exists(_REAL_DEPTH),
                    reason="real ESZ3 depth slice not present")
def test_real_slice_fast_equals_reference():
    """Fast pandas parser bit-identical to the reference on a real slice.

    Reference parser is slow, so compare on a head-truncated plain CSV.
    """
    import gzip, tempfile
    tmp = os.path.join(tempfile.gettempdir(), "_es_parity.csv")
    with gzip.open(_REAL_DEPTH, "rt") as fh, open(tmp, "w") as out:
        out.write(fh.readline())
        for i, line in enumerate(fh):
            out.write(line)
            if i >= 100_000:
                break
    try:
        ref = fd.parse_depth_arrays_reference(tmp)
        fast = fd.parse_depth_arrays_fast(tmp)
        for k in ref:
            assert np.array_equal(ref[k], fast[k]), f"real fast!=ref {k}"
    finally:
        os.remove(tmp)
