"""Parity + correctness for the futures TAQ-only (L1) ingester.

fast path == pure-Python reference (bit-identical), and the L1 book / true-sign
tape are built correctly off a small synthetic TAQ fixture.  No licensed data.
"""
import numpy as np
import pytest
from pathlib import Path
from mmsim.ingest import fut_taq, fut_depth

FIX = Path(__file__).parent / "fixtures" / "fut_taq_synth.csv"


def test_quote_parser_parity():
    a = fut_taq.parse_taq_quotes_reference(FIX)
    b = fut_taq.parse_taq_quotes_fast(FIX)
    for k in ("ts_ns", "qside", "px", "sz", "empty"):
        assert np.array_equal(a[k], b[k]), k


def test_level_filter_and_codes():
    q = fut_taq.parse_taq_quotes_fast(FIX)
    # Level-0 quote/empty rows: 161 bid x2, 97 ask x3, 108 empty-sell x1 = 6 kept;
    # the trailing Level-1 161 row is dropped (Level != 0).
    assert q["ts_ns"].shape[0] == 6
    assert (q["empty"] == 1).sum() == 1            # the EMPTY BOOK SELL
    assert set(np.unique(q["qside"]).tolist()) <= {-1, 1}


def test_l1_book_carryforward_and_empty():
    q = fut_taq.parse_taq_quotes_fast(FIX)
    snaps = fut_taq.build_l1_book(q, use_numba=False)
    # first row is a bid 100.00; ask still unseen -> dask 0
    assert snaps["bid_px"][0, 0] == 100.00 and snaps["dask"][0] == 0
    # after the EMPTY BOOK SELL row, that snapshot's ask is cleared (dask 0)
    eb = np.flatnonzero(q["empty"] == 1)[0]
    assert snaps["ask_px"][eb, 0] == 0.0 and snaps["dask"][eb] == 0


def test_numba_matches_reference():
    q = fut_taq.parse_taq_quotes_fast(FIX)
    r = fut_taq.build_l1_book(q, use_numba=False)
    n = fut_taq.build_l1_book(q, use_numba=True)
    for k in r:
        assert np.array_equal(r[k], n[k]), k


def test_true_sign_tape_shared_with_depth():
    # tape comes from the shared fut_depth parser: +1 buy-aggr (162), -1 sell (98)
    t = fut_depth.parse_taq_trades_fast(FIX)
    assert t["side"].tolist() == [1, -1]
    assert t["px"].tolist() == [100.25, 100.00]
