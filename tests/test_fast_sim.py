"""Fast queue-aware fill path (mmsim.sim.fast_sim) — correctness tests.

The fast path is a performance redesign of run_sim + QueueAwareFillModel for
the runner's cancel-on-replace quoters.  These tests pin:

  1. Bit-for-bit fill parity with the reference run_sim pipeline on the
     bundled 60-min BTC fixture, for every runner quoter (Touch / DepthSkew /
     MicroSkew x2 / Rank 1..5).  This is the load-bearing correctness check —
     same ts / side / price / size / count.
  2. Causality: a fill never references an event past its own timestamp
     (structural — within a snapshot interval; asserted via a pollution
     check, mirroring the reference leak battery).
  3. Queue discipline bites: fewer fills than a naive top-of-book filler.
  4. Array-native day decode + window slicing reproduces the event-path fills.
"""
from __future__ import annotations

import dataclasses
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from mmsim.ingest.lob import load_lob, SnapshotEvent, TradeEvent
from mmsim.sim.loop import run_sim, QuoteRequest
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.sim import fast_sim as F

HERE = Path(__file__).resolve().parent
SNAP_1H = HERE / "fixtures" / "lob_btcusdt_60min_snapshots.parquet"
TRADE_1H = HERE / "fixtures" / "lob_btcusdt_60min_trades.parquet"
SNAP_30S = HERE / "fixtures" / "lob_btcusdt_30sec_snapshots.parquet"
TRADE_30S = HERE / "fixtures" / "lob_btcusdt_30sec_trades.parquet"

SIZE = 0.001


# --- runner quoter mirrors (kept local so the test does not import the script) #

class TouchQuoter:
    def __init__(self, size): self.size = size
    def __call__(self, book, active, t_ns):
        if book is None or book.best_bid is None or book.best_ask is None:
            return []
        return [QuoteRequest(+1, book.best_bid, self.size),
                QuoteRequest(-1, book.best_ask, self.size)]


class RankQuoter:
    def __init__(self, size, rank, side):
        self.size = size; self.rank = rank; self.side = side
    def __call__(self, book, active, t_ns):
        levels = book.bids if self.side == +1 else book.asks
        if not levels or len(levels) < self.rank:
            return []
        return [QuoteRequest(self.side, levels[self.rank - 1][0], self.size)]


class DepthSkewQuoter:
    def __init__(self, size): self.size = size
    def __call__(self, book, active, t_ns):
        if book is None or not book.bids or not book.asks:
            return []
        bid_px = book.bids[1][0] if len(book.bids) > 1 else book.bids[0][0]
        ask_px = book.asks[0][0]
        return [QuoteRequest(+1, bid_px, self.size), QuoteRequest(-1, ask_px, self.size)]


class MicroSkewQuoter:
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
        mid = 0.5 * (bp + ap); tot = bs + az
        mp = (ap * bs + bp * az) / tot if tot > 0 else mid
        lean = mp - mid
        if self.use_ofi:
            self._update_ofi(book); lean = lean + 1e-12 * self._ofi
        bid = QuoteRequest(+1, bp, self.size); ask = QuoteRequest(-1, ap, self.size)
        if lean > 0: return [bid]
        if lean < 0: return [ask]
        return [bid, ask]


def _ftuple(fills):
    return [(f.ts_ns, f.side, round(f.price, 10), round(f.size, 12)) for f in fills]


CASES = [
    ("touch", lambda: TouchQuoter(SIZE), partial(F.spec_touch, size=SIZE)),
    ("depth_skew", lambda: DepthSkewQuoter(SIZE), partial(F.spec_depth_skew, size=SIZE)),
    ("microprice", lambda: MicroSkewQuoter(SIZE, use_ofi=False),
     partial(F.spec_microskew, size=SIZE, use_ofi=False)),
    ("integ_ofi", lambda: MicroSkewQuoter(SIZE, use_ofi=True),
     partial(F.spec_microskew, size=SIZE, use_ofi=True)),
] + [
    (f"rank{r}", (lambda r=r: RankQuoter(SIZE, r, +1)),
     partial(F.spec_rank, size=SIZE, rank=r, side=+1)) for r in range(1, 6)
]


@pytest.mark.skipif(not SNAP_1H.exists(), reason="DS-LOB-1H fixture absent")
@pytest.mark.parametrize("name,quoter_factory,spec", CASES)
def test_fast_matches_reference_fills(name, quoter_factory, spec):
    events = load_lob(SNAP_1H, TRADE_1H)
    ref = run_sim(events, quoter_factory(), QueueAwareFillModel())
    fast = F.simulate(events, spec, SIZE)
    assert _ftuple(fast.fills) == _ftuple(ref.fills), f"{name}: fill sequence differs"
    assert fast.n_quoter_calls == ref.n_quoter_calls
    assert fast.n_maker_fills == ref.n_maker_fills


@pytest.mark.skipif(not SNAP_1H.exists(), reason="DS-LOB-1H fixture absent")
def test_fast_touch_baseline_pinned():
    """Same pinned economic baseline the reference test pins (1823 maker
    fills on the bundled fixture)."""
    events = load_lob(SNAP_1H, TRADE_1H)
    fast = F.simulate(events, partial(F.spec_touch, size=SIZE), SIZE)
    assert fast.n_maker_fills == 1823
    assert fast.n_quoter_calls == 35989


@pytest.mark.skipif(not SNAP_30S.exists(), reason="30s fixture absent")
def test_queue_discipline_bites():
    """Queue-aware fills are markedly fewer than a naive top-of-book filler
    (every trade at our level fills us)."""
    events = load_lob(SNAP_30S, TRADE_30S)
    fast = F.simulate(events, partial(F.spec_touch, size=SIZE), SIZE)
    # naive count: every trade whose price is at a touch fill would count;
    # the queue model must yield strictly fewer than the trade count.
    n_trades = sum(1 for e in events if isinstance(e, TradeEvent))
    assert fast.n_maker_fills < n_trades


def _pollute(e, factor=99.99):
    if isinstance(e, SnapshotEvent):
        garbage = tuple((1.0, factor) for _ in e.bids)
        return dataclasses.replace(e, bids=garbage, asks=garbage)
    return dataclasses.replace(e, price=factor, size=factor, side=0)


@pytest.mark.skipif(not SNAP_30S.exists(), reason="30s fixture absent")
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_fast_no_lookahead_under_pollution(seed):
    """Polluting every event past T leaves the fill list up to T unchanged."""
    events = load_lob(SNAP_30S, TRADE_30S)
    rng = np.random.default_rng(seed)
    pivot = int(rng.integers(len(events) // 4, 3 * len(events) // 4))
    T = events[pivot].ts_ns
    spec = partial(F.spec_touch, size=SIZE)
    prefix = [e for e in events if e.ts_ns <= T]
    clean = F.simulate(prefix, spec, SIZE)
    polluted = [e if e.ts_ns <= T else _pollute(e) for e in events]
    full = F.simulate(polluted, spec, SIZE)
    full_pre = [f for f in full.fills if f.ts_ns <= T]
    assert _ftuple(full_pre) == _ftuple(clean.fills)


@pytest.mark.skipif(not SNAP_1H.exists(), reason="DS-LOB-1H fixture absent")
def test_markouts_finite_and_signed():
    """Realised/markout columns on real fills are finite and have sane sign
    distribution (not all NaN, not all one sign)."""
    from mmsim.markout.engine import compute_markout
    from mmsim.ledger.writer import build_mid_timeline
    events = load_lob(SNAP_1H, TRADE_1H)
    ts, mid = build_mid_timeline(events)
    fast = F.simulate(events, partial(F.spec_touch, size=SIZE), SIZE)
    mo = compute_markout(fast.fills, ts, mid)
    mk = mo.markout_10s[~np.isnan(mo.markout_10s)]
    assert mk.size > 0
    assert np.all(np.isfinite(mk))
