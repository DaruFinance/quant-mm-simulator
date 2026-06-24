"""InventoryTracker tests + DS-LOB-1H baseline."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

from mmsim.ingest.lob import load_lob
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.sim.inventory import (
    InventorySample, InventoryTrace, InventoryTracker, inventory_path,
)
from mmsim.sim.loop import Fill, run_sim


HERE = Path(__file__).resolve().parent
SNAP_30S = HERE / "fixtures" / "lob_btcusdt_30sec_snapshots.parquet"
TRADE_30S = HERE / "fixtures" / "lob_btcusdt_30sec_trades.parquet"
SNAP_1H = HERE / "fixtures" / "lob_btcusdt_60min_snapshots.parquet"
TRADE_1H = HERE / "fixtures" / "lob_btcusdt_60min_trades.parquet"


def _fill(ts: int, side: int, size: float, is_maker: bool = True) -> Fill:
    return Fill(
        fill_id=ts, order_id=0, ts_ns=ts,
        price=100.0, size=size, side=side, is_maker=is_maker,
    )


def test_initial_inv_zero():
    t = InventoryTracker()
    assert t.inv == 0.0
    assert t.peak_long == 0.0
    assert t.peak_short == 0.0


def test_signed_accumulation():
    t = InventoryTracker()
    t.observe(_fill(1, +1, 0.5))   # bid fill: +0.5
    t.observe(_fill(2, -1, 0.3))   # ask fill: -0.3
    t.observe(_fill(3, +1, 0.2))   # +0.2
    assert t.inv == pytest.approx(0.4)


def test_peaks_track_extrema():
    t = InventoryTracker()
    t.observe(_fill(1, +1, 1.5))   # inv = +1.5
    t.observe(_fill(2, -1, 4.0))   # inv = -2.5
    t.observe(_fill(3, +1, 1.0))   # inv = -1.5
    assert t.peak_long == pytest.approx(1.5)
    assert t.peak_short == pytest.approx(-2.5)


def test_taker_fills_count_the_same_as_makers():
    t = InventoryTracker()
    t.observe(_fill(1, +1, 0.5, is_maker=True))
    t.observe(_fill(2, +1, 0.3, is_maker=False))  # taker buy: +0.3
    assert t.inv == pytest.approx(0.8)


def test_trace_records_every_fill():
    t = InventoryTracker()
    for ts, side, sz in [(1, +1, 0.1), (2, -1, 0.05), (3, +1, 0.02)]:
        t.observe(_fill(ts, side, sz))
    tr = t.trace()
    assert tr.n_fills == 3
    assert tr.final_inv == pytest.approx(0.07)
    assert tr.samples[0].inv == pytest.approx(0.1)
    assert tr.samples[1].inv == pytest.approx(0.05)
    assert tr.samples[2].inv == pytest.approx(0.07)


def test_inventory_path_one_shot():
    fills = [_fill(1, +1, 0.5), _fill(2, -1, 0.3)]
    tr = inventory_path(fills)
    assert tr.n_fills == 2
    assert tr.final_inv == pytest.approx(0.2)


# --------------------------------------------------------------------- #
# DS-LOB-1H baseline: bracket quoter run; inventory path bounded.
# --------------------------------------------------------------------- #

@pytest.mark.skipif(
    not SNAP_1H.exists() or not TRADE_1H.exists(),
    reason="DS-LOB-1H fixture not present",
)
def test_ds_lob_1h_inventory_baseline():
    sys.path.insert(0, str(HERE))
    from test_sim_fills import BracketQuoter

    stream = load_lob(SNAP_1H, TRADE_1H)
    model = QueueAwareFillModel()
    res = run_sim(stream, BracketQuoter(taker_every=500), model)
    tr = inventory_path(res.fills)

    # Pinned baselines — these track the bracket-quoter behaviour
    # over DS-LOB-1H.  The fixture skews short-maker (29,728 ask
    # fills vs 35,755 bid fills with the naive stub; queue-aware shifts
    # the ratio but final inv is still positive given the buy-side
    # dominance).
    expected_n_fills = 1_894
    expected_final_inv = 0.060150         # net +60 mg long after 1h
    expected_peak_long = 0.067340         # max long ever held
    expected_peak_short = -0.006540       # most negative ever (~6.5 mg short)
    assert tr.n_fills == expected_n_fills
    assert tr.final_inv == pytest.approx(expected_final_inv, abs=1e-6)
    assert tr.peak_long == pytest.approx(expected_peak_long, abs=1e-6)
    assert tr.peak_short == pytest.approx(expected_peak_short, abs=1e-6)
    # Bound: peak inv stays well under what an unhedged accumulating
    # quoter would generate over an hour without flat-and-go.
    # 0.001 BTC per maker × ~1800 maker fills ≈ 1.8 BTC if all one-side;
    # actual bound 0.23 means 87% of maker fills are getting offset by
    # opposite-side fills (which is the bracket quoter doing its job).
    assert abs(tr.peak_long) < 1.0
    assert abs(tr.peak_short) < 1.0
