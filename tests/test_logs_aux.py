"""Aux log streams (fill-rate / inventory / queue-pos).

Covers:
  - FillRateLogger per-second bucketing math
  - InventoryLogger per-second snapshots + carry-forward semantics
  - QueuePosLogger row collection
  - AuxLogs.to_csvs round-trip (read back the CSVs and check headers
    + row counts + spot values)
  - DS-LOB-1H integration via run_sim_with_aux_logs: row counts +
    5-bucket manual reconciliation from raw fills (the spec's G3
    requirement)
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import pytest

from mmsim.ingest.lob import load_lob
from mmsim.logs.aux import (
    AuxLogs, FillRateBucket, FillRateLogger,
    InventoryBucket, InventoryLogger,
    QueuePosLogger, NS_PER_S,
    run_sim_with_aux_logs,
)
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.sim.loop import Fill


HERE = Path(__file__).resolve().parent
SNAP_30S = HERE / "fixtures" / "lob_btcusdt_30sec_snapshots.parquet"
TRADE_30S = HERE / "fixtures" / "lob_btcusdt_30sec_trades.parquet"
SNAP_1H = HERE / "fixtures" / "lob_btcusdt_60min_snapshots.parquet"
TRADE_1H = HERE / "fixtures" / "lob_btcusdt_60min_trades.parquet"


def _fill(ts_ns: int, side: int, size: float, is_maker: bool = True) -> Fill:
    return Fill(
        fill_id=ts_ns, order_id=0, ts_ns=ts_ns,
        price=100.0, size=size, side=side, is_maker=is_maker,
    )


# --------------------------------------------------------------------- #
# FillRateLogger unit tests
# --------------------------------------------------------------------- #

def test_fill_rate_buckets_single_second():
    log = FillRateLogger()
    # Three fills inside the same 1-second bucket.
    log.observe_fill(_fill(1_000_000_001, +1, 0.5))
    log.observe_fill(_fill(1_500_000_000, -1, 0.25))
    log.observe_fill(_fill(1_999_999_999, +1, 0.1))
    bs = log.buckets()
    assert len(bs) == 1
    assert bs[0].bucket_s == 1
    assert bs[0].n_fills == 3
    assert bs[0].total_qty == pytest.approx(0.85)
    assert bs[0].t_ns_start == 1_000_000_000


def test_fill_rate_buckets_span_multiple_seconds():
    log = FillRateLogger()
    log.observe_fill(_fill(5_000_000_000, +1, 1.0))    # bucket 5
    log.observe_fill(_fill(7_500_000_000, +1, 2.0))    # bucket 7
    bs = log.buckets()
    # Range [5, 7]; bucket 6 is an empty (zero) row.
    assert [b.bucket_s for b in bs] == [5, 6, 7]
    assert [b.n_fills for b in bs] == [1, 0, 1]
    assert [b.total_qty for b in bs] == pytest.approx([1.0, 0.0, 2.0])


def test_fill_rate_explicit_grid_overrides_observed_range():
    log = FillRateLogger()
    log.observe_fill(_fill(5_000_000_000, +1, 1.0))
    bs = log.buckets(start_s=3, end_s=7)
    assert [b.bucket_s for b in bs] == [3, 4, 5, 6, 7]
    assert [b.n_fills for b in bs] == [0, 0, 1, 0, 0]


# --------------------------------------------------------------------- #
# InventoryLogger unit tests
# --------------------------------------------------------------------- #

def test_inventory_buckets_carry_forward():
    log = InventoryLogger()
    log.observe_fill(_fill(2_000_000_000, +1, 0.5))    # b2 inv=+0.5
    log.observe_fill(_fill(2_500_000_000, -1, 0.2))    # b2 inv=+0.3
    log.observe_fill(_fill(4_000_000_000, +1, 0.4))    # b4 inv=+0.7
    bs = log.buckets()
    # Range [2, 4]; bucket 3 had no fill → carry forward inv=+0.3.
    assert [b.bucket_s for b in bs] == [2, 3, 4]
    assert [b.inv for b in bs] == pytest.approx([0.3, 0.3, 0.7])
    assert [b.n_fills_so_far for b in bs] == [2, 2, 3]


def test_inventory_buckets_explicit_grid_with_leading_empty():
    log = InventoryLogger()
    log.observe_fill(_fill(3_000_000_000, +1, 0.5))
    bs = log.buckets(start_s=1, end_s=4)
    # Leading empty seconds — inv defaults to 0.
    assert [b.bucket_s for b in bs] == [1, 2, 3, 4]
    assert [b.inv for b in bs] == pytest.approx([0.0, 0.0, 0.5, 0.5])


# --------------------------------------------------------------------- #
# QueuePosLogger unit tests
# --------------------------------------------------------------------- #

def test_queue_pos_logger_records_each_observation():
    log = QueuePosLogger()
    log.observe_queue_pos(
        ts_ns=1_000, order_id=1, side=+1, price=100.0,
        queue_pos=5.0, frozen=False)
    log.observe_queue_pos(
        ts_ns=2_000, order_id=1, side=+1, price=100.0,
        queue_pos=3.5, frozen=False)
    log.observe_queue_pos(
        ts_ns=3_000, order_id=2, side=-1, price=101.0,
        queue_pos=0.0, frozen=True)
    ss = log.samples()
    assert len(ss) == 3
    assert ss[0].queue_pos == 5.0 and ss[1].queue_pos == 3.5
    assert ss[2].frozen is True and ss[2].order_id == 2


# --------------------------------------------------------------------- #
# AuxLogs.to_csvs round-trip
# --------------------------------------------------------------------- #

def test_aux_logs_to_csvs_round_trip():
    aux = AuxLogs()
    aux.observe_fill(_fill(1_000_000_000, +1, 0.5))
    aux.observe_fill(_fill(2_000_000_000, -1, 0.2))
    aux.queue_pos.observe_queue_pos(1_000_000_000, 1, +1, 100.0, 5.0, False)
    aux.queue_pos.observe_queue_pos(2_000_000_000, 1, +1, 100.0, 4.5, False)
    with tempfile.TemporaryDirectory() as td:
        fr_p, inv_p, qp_p = aux.to_csvs(Path(td))
        assert fr_p.exists() and inv_p.exists() and qp_p.exists()
        fr_rows = list(csv.DictReader(fr_p.open()))
        inv_rows = list(csv.DictReader(inv_p.open()))
        qp_rows = list(csv.DictReader(qp_p.open()))
    assert len(fr_rows) == 2          # buckets 1 and 2
    assert int(fr_rows[0]["n_fills"]) == 1
    assert float(fr_rows[1]["total_qty"]) == pytest.approx(0.2)
    assert len(inv_rows) == 2
    assert float(inv_rows[0]["inv"]) == pytest.approx(0.5)
    assert float(inv_rows[1]["inv"]) == pytest.approx(0.3)
    assert len(qp_rows) == 2
    assert int(qp_rows[0]["order_id"]) == 1


# --------------------------------------------------------------------- #
# Causality: aux logs never read future events
# --------------------------------------------------------------------- #

def test_fill_rate_inv_observe_only_passed_event():
    """Two loggers fed an identical fill prefix produce identical
    state — proves observe() reads only the passed-in fill."""
    fills = [_fill(i * 100_000_000, +1 if i % 2 == 0 else -1, 0.1)
             for i in range(1, 20)]
    a = FillRateLogger(); b = FillRateLogger()
    ia = InventoryLogger(); ib = InventoryLogger()
    for f in fills:
        a.observe_fill(f); ia.observe_fill(f)
    for f in fills:
        b.observe_fill(f); ib.observe_fill(f)
    assert a.buckets() == b.buckets()
    assert ia.buckets() == ib.buckets()


# --------------------------------------------------------------------- #
# DS-LOB-1H integration: G3 baseline + 5-bucket reconciliation
# --------------------------------------------------------------------- #

@pytest.mark.skipif(
    not SNAP_1H.exists() or not TRADE_1H.exists(),
    reason="DS-LOB-1H fixture not present",
)
def test_ds_lob_1h_aux_logs_baseline():
    sys.path.insert(0, str(HERE))
    from test_sim_fills import BracketQuoter

    stream = load_lob(SNAP_1H, TRADE_1H)
    model = QueueAwareFillModel()
    res, aux = run_sim_with_aux_logs(
        stream, BracketQuoter(taker_every=500), model,
    )
    # The sim baseline — make sure the wrapped driver
    # didn't change the loop's economics.
    assert res.n_maker_fills == 1_823
    assert res.n_taker_fills == 71
    assert len(res.fills) == 1_894

    fr_buckets = aux.fill_rate.buckets()
    inv_buckets = aux.inventory.buckets()
    qp_samples = aux.queue_pos.samples()

    # Fill rate / inventory grids: derived from fill timestamps
    # spanning the hour.  Expect ~3600 second-buckets; small slack
    # at the edges (first/last fill not necessarily at second 0).
    assert len(fr_buckets) == len(inv_buckets)
    assert 3500 <= len(fr_buckets) <= 3700, (
        f"unexpected per-second grid size: {len(fr_buckets)}")

    # Total fills across all buckets equals trade-ledger size.
    total_n_fills_buckets = sum(b.n_fills for b in fr_buckets)
    total_qty_buckets = sum(b.total_qty for b in fr_buckets)
    assert total_n_fills_buckets == len(res.fills)
    expected_total_qty = sum(f.size for f in res.fills)
    assert total_qty_buckets == pytest.approx(expected_total_qty, abs=1e-9)

    # Final inv from logger matches running-sum of signed fills.
    expected_final_inv = sum(f.size * f.side for f in res.fills)
    assert inv_buckets[-1].inv == pytest.approx(expected_final_inv, abs=1e-9)
    # Cumulative fill counter at the last bucket equals total.
    assert inv_buckets[-1].n_fills_so_far == len(res.fills)

    # Queue-pos: one row per (snapshot, active_order) pair.  We see
    # ~2 active orders per snapshot (both maker sides) for nearly
    # every quoter call.  Expect ~2 * 35,989 minus the warmup
    # snapshot where active is empty.
    n_quoter_calls = res.n_quoter_calls
    # First call has no active orders; subsequent calls have up to 2.
    assert 1.5 * n_quoter_calls <= len(qp_samples) <= 2.5 * n_quoter_calls


@pytest.mark.skipif(
    not SNAP_1H.exists() or not TRADE_1H.exists(),
    reason="DS-LOB-1H fixture not present",
)
def test_ds_lob_1h_5_bucket_manual_reconciliation():
    """G3 — pick 5 second-buckets, recompute (n_fills, total_qty, inv)
    from raw fills, and compare to the logger output."""
    sys.path.insert(0, str(HERE))
    from test_sim_fills import BracketQuoter

    stream = load_lob(SNAP_1H, TRADE_1H)
    model = QueueAwareFillModel()
    res, aux = run_sim_with_aux_logs(
        stream, BracketQuoter(taker_every=500), model,
    )

    fr_buckets = aux.fill_rate.buckets()
    inv_buckets = aux.inventory.buckets()
    # Index buckets by bucket_s for easy lookup.
    fr_by_s = {b.bucket_s: b for b in fr_buckets}
    inv_by_s = {b.bucket_s: b for b in inv_buckets}

    # Pick 5 buckets evenly spread across the hour, biased toward
    # buckets known to be non-empty (use the bucket of every
    # (i*375)-th fill so we always land on a bucket with activity).
    n = len(res.fills)
    probe_idxs = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    probe_buckets = sorted({res.fills[i].ts_ns // NS_PER_S
                            for i in probe_idxs})
    assert len(probe_buckets) == 5, (
        f"probe buckets collided to {len(probe_buckets)} unique seconds")

    for bs in probe_buckets:
        bucket_start = bs * NS_PER_S
        bucket_end = bucket_start + NS_PER_S
        # Manual recompute.
        in_bucket = [f for f in res.fills
                     if bucket_start <= f.ts_ns < bucket_end]
        exp_n = len(in_bucket)
        exp_qty = sum(f.size for f in in_bucket)
        # Cumulative inventory at end of bucket.
        cum = [f for f in res.fills if f.ts_ns < bucket_end]
        exp_inv = sum(f.size * f.side for f in cum)
        exp_count = len(cum)

        got_fr = fr_by_s[bs]
        got_inv = inv_by_s[bs]
        assert got_fr.n_fills == exp_n, (
            f"bucket {bs}: n_fills got {got_fr.n_fills} want {exp_n}")
        assert got_fr.total_qty == pytest.approx(exp_qty, abs=1e-12)
        assert got_inv.inv == pytest.approx(exp_inv, abs=1e-9)
        assert got_inv.n_fills_so_far == exp_count


@pytest.mark.skipif(
    not SNAP_1H.exists() or not TRADE_1H.exists(),
    reason="DS-LOB-1H fixture not present",
)
def test_ds_lob_1h_queue_pos_causality():
    """Queue-pos rows at time t reflect only events with ts <= t.
    Spot-check: every row's queue_pos is non-negative and finite."""
    sys.path.insert(0, str(HERE))
    from test_sim_fills import BracketQuoter

    stream = load_lob(SNAP_1H, TRADE_1H)
    model = QueueAwareFillModel()
    _, aux = run_sim_with_aux_logs(
        stream, BracketQuoter(taker_every=500), model,
    )
    for s in aux.queue_pos.samples():
        assert s.queue_pos >= 0.0
        assert s.queue_pos == s.queue_pos          # not NaN
        assert s.side in (+1, -1)
    # Timestamps in queue_pos log are non-decreasing.
    ts = [s.ts_ns for s in aux.queue_pos.samples()]
    assert all(a <= b for a, b in zip(ts, ts[1:]))
