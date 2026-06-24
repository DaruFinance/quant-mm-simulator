"""Tests for the equity-TAQ ingester (mmsim.ingest.eq_taq).

Covers: reference↔numba bit-identical event builder, quote-rule sign
inference, NBBO book carry, causality (sign uses only ≤t mid), and the
parquet round-trip through the frozen load_lob reader.
"""
from __future__ import annotations

import numpy as np
import pytest

from mmsim.ingest import eq_taq as E
from mmsim.ingest.lob import load_lob, SnapshotEvent, TradeEvent


# A tiny synthetic TAQ CSV in the exact source format (leading spaces,
# ms timestamps, NB quotes + trades). Regular-hours timestamps so the
# default regular_hours_only filter keeps them.
_CSV = (
    "Date, Timestamp, EventType, Ticker, Price, Quantity, Exchange, Conditions\n"
    "20150616, 10:00:00.000, QUOTE BID NB, IBM, 100.00, 200, ARCA, 00000001\n"
    "20150616, 10:00:00.000, QUOTE ASK NB, IBM, 100.10, 300, ARCA, 00000001\n"
    "20150616, 10:00:00.500, TRADE, IBM, 100.10, 100, NASDAQ, 00000001\n"   # at ask -> buy +1
    "20150616, 10:00:01.000, TRADE, IBM, 100.00, 150, NASDAQ, 00000001\n"   # at bid -> sell -1
    "20150616, 10:00:01.500, TRADE NB, IBM, 100.05, 50, NASDAQ, 00000001\n" # at mid -> 0
    "20150616, 10:00:02.000, QUOTE BID NB, IBM, 100.02, 100, BATS, 00000001\n"
    "20150616, 10:00:02.500, TRADE, IBM, 100.10, 75, NASDAQ, 00000001\n"    # mid=100.06 -> buy +1
)


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "tiny_taq.csv"
    p.write_text(_CSV)
    return str(p)


def test_parse_classifies_and_filters(csv_path):
    cols = E.parse_taq_arrays_reference(csv_path)
    # 3 NB-quote rows (codes 0/1) + 4 trades (code 2) = 7 kept rows
    assert cols["ts_ns"].shape[0] == 7
    codes = cols["code"].tolist()
    assert codes.count(2) == 4  # four trades


def test_quote_rule_signs(csv_path):
    cols = E.parse_taq_arrays_reference(csv_path)
    ev = E.build_event_columns_reference(cols)
    # trades in order: at-ask(+1), at-bid(-1), at-mid(0), above-mid(+1)
    assert ev["tr_side"].tolist() == [1, -1, 0, 1]


def test_builder_ref_eq_numba_bit_identical(csv_path):
    cols = E.parse_taq_arrays_reference(csv_path)
    r = E.build_event_columns(cols, use_numba=False)
    n = E.build_event_columns(cols, use_numba=True)
    for k in r:
        a, b = r[k], n[k]
        if a.dtype.kind == "f":
            assert np.array_equal(a, b, equal_nan=True), k
        else:
            assert np.array_equal(a, b), k


def test_nbbo_book_carry(csv_path):
    cols = E.parse_taq_arrays_reference(csv_path)
    ev = E.build_event_columns_reference(cols)
    # After the BATS bid update (100.02), the NBBO snapshot should carry
    # bid 100.02 with ask still 100.10.
    last_bid = ev["snap_bidpx"][-1]
    last_ask = ev["snap_askpx"][-1]
    assert abs(last_bid - 100.02) < 1e-9
    assert abs(last_ask - 100.10) < 1e-9


def test_causality_sign_uses_only_past_mid(csv_path):
    # The at-mid trade (third) must be signed against the mid known at its
    # OWN time (100.05 = (100.00+100.10)/2), not the later 100.06 mid.
    cols = E.parse_taq_arrays_reference(csv_path)
    ev = E.build_event_columns_reference(cols)
    assert ev["tr_side"][2] == 0  # at-mid given the contemporaneous NBBO


def test_parquet_roundtrip_via_load_lob(csv_path, tmp_path):
    cols = E.parse_taq_arrays_reference(csv_path)
    ev = E.build_event_columns(cols, use_numba=True)
    sp = str(tmp_path / "snap.parquet")
    tp = str(tmp_path / "trades.parquet")
    ns, nt = E.write_parquet(ev, sp, tp, symbol="IBM", venue="NBBO")
    assert ns == 3 and nt == 4
    stream = load_lob(sp, tp)
    snaps = [e for e in stream if isinstance(e, SnapshotEvent)]
    trades = [e for e in stream if isinstance(e, TradeEvent)]
    assert len(snaps) == 3 and len(trades) == 4
    # stream is time-sorted, snapshot-before-trade tiebreak
    assert all(stream[i].ts_ns <= stream[i + 1].ts_ns for i in range(len(stream) - 1))
    # depth-1 NBBO; the FIRST snapshot is emitted on the bid update before
    # the ask arrives (one-sided warmup), so the SECOND snapshot is the first
    # two-sided NBBO.
    assert snaps[0].depth == 1
    assert snaps[0].bids and not snaps[0].asks   # bid-only warmup
    assert snaps[1].bids and snaps[1].asks       # first complete NBBO


def test_in_memory_stream_matches_parquet(csv_path, tmp_path):
    cols = E.parse_taq_arrays_reference(csv_path)
    ev = E.build_event_columns(cols, use_numba=True)
    mem = E.columns_to_eventstream(ev, symbol="IBM", venue="NBBO")
    sp = str(tmp_path / "s.parquet"); tp = str(tmp_path / "t.parquet")
    E.write_parquet(ev, sp, tp, symbol="IBM", venue="NBBO")
    disk = load_lob(sp, tp)
    assert len(mem) == len(disk)
    for a, b in zip(mem, disk):
        assert a.ts_ns == b.ts_ns
        assert type(a) is type(b)
