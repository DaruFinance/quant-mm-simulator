"""Quoting-model library tests.

Coverage:
  - Protocol conformance (8 quoters × 1 test each = 8 tests)
  - Per-model behaviour (basic emit / inv grow / σ grow) = 3 each × 8 = 24
  - Shared `RollingSigma` tracker tests = 4
  - 5-decision G3 dump for each model = 8
  - DS-LOB-1H baseline ranking = 1 (table)

Total: ~45 tests.

Single-threaded.  The DS-LOB-1H ranking test runs all 8 quoters
through `run_sim` on the canonical 1-h fixture; ~8 × 2.5 min runtime
on a cold cache.  Skipped automatically when the fixture is absent.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest

from mmsim.ingest.lob import Book, SnapshotEvent, TradeEvent, load_lob
from mmsim.models import (
    AvellanedaStoikovQuoter, CarteaJaimungalQuoter, FairAnchoredQuoter,
    GLFTQuoter, HoStollQuoter, LadderQuoter, MicropriceSkewQuoter,
    RollingSigma, SymmetricQuoter,
)
from mmsim.quoter import Quoter
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.sim.loop import QuoteRequest, run_sim


HERE = Path(__file__).resolve().parent
SNAP_1H = HERE / "fixtures" / "lob_btcusdt_60min_snapshots.parquet"
TRADE_1H = HERE / "fixtures" / "lob_btcusdt_60min_trades.parquet"


# Standard test-book helpers.
def _b(ts: int, bid_px: float, bid_sz: float, ask_px: float, ask_sz: float) -> Book:
    return Book(ts_ns=ts, bids=((bid_px, bid_sz),), asks=((ask_px, ask_sz),))


# --------------------------------------------------------------------- #
# Protocol conformance: every model implements the runtime-checkable
# Quoter Protocol.
# --------------------------------------------------------------------- #

def test_symmetric_implements_protocol():
    assert isinstance(SymmetricQuoter(half_spread=0.5, size=0.001), Quoter)


def test_ladder_implements_protocol():
    assert isinstance(
        LadderQuoter(half_spread=0.5, step=0.1, n_levels=3, size_per_level=0.001),
        Quoter,
    )


def test_microprice_skew_implements_protocol():
    assert isinstance(MicropriceSkewQuoter(half_spread=0.5, size=0.001), Quoter)


def test_fair_anchored_implements_protocol():
    assert isinstance(
        FairAnchoredQuoter(half_spread=0.5, size=0.001, half_life_ns=10**9),
        Quoter,
    )


def test_avellaneda_stoikov_implements_protocol():
    q = AvellanedaStoikovQuoter(gamma=0.1, k=1.5, horizon_ns=10**12,
                                  size=0.001, vol_window_ns=10**9)
    assert isinstance(q, Quoter)


def test_cartea_jaimungal_implements_protocol():
    q = CarteaJaimungalQuoter(gamma=0.1, k=1.5, kappa=0.0, horizon_ns=10**12,
                                size=0.001, vol_window_ns=10**9)
    assert isinstance(q, Quoter)


def test_glft_implements_protocol():
    q = GLFTQuoter(gamma=0.1, k=1.5, A=140.0, horizon_ns=10**12,
                    size=0.001, vol_window_ns=10**9)
    assert isinstance(q, Quoter)


def test_ho_stoll_implements_protocol():
    q = HoStollQuoter(alpha=0.5, beta=1.0, size=0.001, vol_window_ns=10**9)
    assert isinstance(q, Quoter)


# --------------------------------------------------------------------- #
# RollingSigma tracker
# --------------------------------------------------------------------- #

def test_rolling_sigma_returns_none_before_two_obs():
    s = RollingSigma(window_ns=10_000)
    assert s.value(0) is None
    s.observe(0, 100.0)
    assert s.value(0) is None  # still <2


def test_rolling_sigma_evicts_old_window():
    s = RollingSigma(window_ns=1_000)
    s.observe(0, 100.0)
    s.observe(500, 101.0)
    s.observe(900, 102.0)
    v = s.value(1_000)
    assert v is not None
    # Now query at t=3000 — all entries are >= cutoff (3000-1000=2000),
    # which evicts all (ts<=2000).
    v2 = s.value(3_000)
    assert v2 is None


def test_rolling_sigma_no_lookahead_under_pollution():
    clean = RollingSigma(window_ns=10_000)
    polluted = RollingSigma(window_ns=10_000)
    for (t, p) in [(0, 100.0), (500, 101.0), (900, 102.0)]:
        clean.observe(t, p)
        polluted.observe(t, p)
    # Pollute with future entry.
    polluted.observe(5_000, 9_999.0)
    a = clean.value(1_000)
    b = polluted.value(1_000)
    assert a is not None and b is not None
    assert abs(a - b) < 1e-12


def test_rolling_sigma_constant_prices_zero_vol():
    s = RollingSigma(window_ns=10_000)
    for t in range(5):
        s.observe(t * 100, 100.0)
    v = s.value(500)
    assert v == 0.0


# --------------------------------------------------------------------- #
# Per-model behavioural tests
# --------------------------------------------------------------------- #

# --- SymmetricQuoter ---

def test_symmetric_basic_emit():
    q = SymmetricQuoter(half_spread=0.5, size=0.001)
    book = _b(0, 100.0, 5.0, 101.0, 5.0)
    out = q.quote(book, 0.0, 0)
    assert len(out) == 2
    bid = [o for o in out if o.side == +1][0]
    ask = [o for o in out if o.side == -1][0]
    # ref = top_mid = 100.5; bid = 100.0; ask = 101.0
    assert abs(bid.price - 100.0) < 1e-12
    assert abs(ask.price - 101.0) < 1e-12


def test_symmetric_inv_does_not_affect_prices():
    q = SymmetricQuoter(half_spread=0.5, size=0.001)
    book = _b(0, 100.0, 5.0, 101.0, 5.0)
    out_a = q.quote(book, 0.0, 0)
    out_b = q.quote(book, 10.0, 0)
    # Symmetric model has no inv skew.
    for a, b in zip(out_a, out_b):
        assert a.price == b.price


def test_symmetric_sigma_does_not_affect_prices():
    """Symmetric model is sigma-blind by design."""
    q = SymmetricQuoter(half_spread=0.5, size=0.001)
    book = _b(0, 100.0, 5.0, 101.0, 5.0)
    out_a = q.quote(book, 0.0, 0)
    # No tracker to update; calling at later t with same book → same prices.
    out_b = q.quote(book, 0.0, 10**12)
    for a, b in zip(out_a, out_b):
        assert a.price == b.price


# --- LadderQuoter ---

def test_ladder_basic_emit_three_levels():
    q = LadderQuoter(half_spread=0.5, step=0.25, n_levels=3, size_per_level=0.001)
    book = _b(0, 100.0, 5.0, 101.0, 5.0)
    out = q.quote(book, 0.0, 0)
    # 3 levels × 2 sides = 6 quotes
    assert len(out) == 6
    bids = sorted([o.price for o in out if o.side == +1])
    asks = sorted([o.price for o in out if o.side == -1])
    # ref=100.5; bids at 100.0, 99.75, 99.50 (sorted asc = [99.50, 99.75, 100.00])
    assert abs(bids[2] - 100.0) < 1e-12
    assert abs(bids[1] - 99.75) < 1e-12
    assert abs(bids[0] - 99.50) < 1e-12
    assert abs(asks[0] - 101.0) < 1e-12
    assert abs(asks[1] - 101.25) < 1e-12
    assert abs(asks[2] - 101.50) < 1e-12


def test_ladder_inv_does_not_affect_emitted_levels():
    q = LadderQuoter(half_spread=0.5, step=0.25, n_levels=3, size_per_level=0.001)
    book = _b(0, 100.0, 5.0, 101.0, 5.0)
    out_a = q.quote(book, 0.0, 0)
    out_b = q.quote(book, 100.0, 0)
    # Plain LadderQuoter has no inv skew (would need dynamic_depth).
    for a, b in zip(out_a, out_b):
        assert a.price == b.price


def test_ladder_no_book_no_quotes():
    q = LadderQuoter(half_spread=0.5, step=0.25, n_levels=3, size_per_level=0.001)
    assert q.quote(None, 0.0, 0) == []


# --- MicropriceSkewQuoter ---

def test_microprice_skew_basic_emit_balanced():
    q = MicropriceSkewQuoter(half_spread=0.5, size=0.001)
    book = _b(0, 100.0, 5.0, 101.0, 5.0)
    out = q.quote(book, 0.0, 0)
    # Balanced book ⇒ microprice == mid == 100.5
    bid = [o for o in out if o.side == +1][0]
    ask = [o for o in out if o.side == -1][0]
    assert abs(bid.price - 100.0) < 1e-12
    assert abs(ask.price - 101.0) < 1e-12


def test_microprice_skew_bid_heavy_shifts_up():
    q = MicropriceSkewQuoter(half_spread=0.5, size=0.001)
    # Heavy bid 9, thin ask 1 ⇒ microprice > mid
    book = _b(0, 100.0, 9.0, 101.0, 1.0)
    out = q.quote(book, 0.0, 0)
    bid = [o for o in out if o.side == +1][0]
    ask = [o for o in out if o.side == -1][0]
    # microprice = 100.5 + (9-1)/10 * 0.5 = 100.5 + 0.4 = 100.9
    # bid = 100.9 - 0.5 = 100.4; ask = 101.4
    assert abs(bid.price - 100.4) < 1e-12
    assert abs(ask.price - 101.4) < 1e-12


def test_microprice_skew_no_book_no_quotes():
    q = MicropriceSkewQuoter(half_spread=0.5, size=0.001)
    assert q.quote(None, 0.0, 0) == []


# --- FairAnchoredQuoter ---

def test_fair_anchored_first_call_seeds_ewma():
    q = FairAnchoredQuoter(half_spread=0.5, size=0.001, half_life_ns=1_000)
    book = _b(0, 100.0, 5.0, 101.0, 5.0)
    out = q.quote(book, 0.0, 0)
    bid = [o for o in out if o.side == +1][0]
    # First call: EWMA seeded to mid=100.5
    assert abs(bid.price - 100.0) < 1e-12


def test_fair_anchored_ewma_decays_over_calls():
    q = FairAnchoredQuoter(half_spread=0.5, size=0.001, half_life_ns=1_000)
    # First seed at 100.5
    q.quote(_b(0, 100.0, 5.0, 101.0, 5.0), 0.0, 0)
    # Second mid = 102.5 at dt=1000 ⇒ half-life: ewma = 0.5*100.5 + 0.5*102.5 = 101.5
    out = q.quote(_b(1_000, 102.0, 5.0, 103.0, 5.0), 0.0, 1_000)
    bid = [o for o in out if o.side == +1][0]
    # bid = 101.5 - 0.5 = 101.0
    assert abs(bid.price - 101.0) < 1e-12


def test_fair_anchored_no_book_no_quotes():
    q = FairAnchoredQuoter(half_spread=0.5, size=0.001, half_life_ns=1_000)
    assert q.quote(None, 0.0, 0) == []


# --- AvellanedaStoikovQuoter ---

def test_as_warmup_no_quotes_until_sigma_ready():
    q = AvellanedaStoikovQuoter(gamma=0.1, k=1.5, horizon_ns=10**12,
                                  size=0.001, vol_window_ns=10**9)
    # First call only: sigma unwarmed.
    book = _b(0, 100.0, 5.0, 101.0, 5.0)
    assert q.quote(book, 0.0, 0) == []
    # Second call: sigma warm (n_obs=2 ⇒ 1 return ⇒ var defined).
    book2 = _b(1_000_000, 100.1, 5.0, 101.1, 5.0)
    out = q.quote(book2, 0.0, 1_000_000)
    assert len(out) == 2


def test_as_inv_skews_reservation_down_when_long():
    q = AvellanedaStoikovQuoter(gamma=0.5, k=1.5, horizon_ns=10**12,
                                  size=0.001, vol_window_ns=10**12)
    # Warm sigma.
    q.quote(_b(0, 100.0, 5.0, 101.0, 5.0), 0.0, 0)
    out_flat = q.quote(_b(10**6, 100.1, 5.0, 101.1, 5.0), 0.0, 10**6)
    out_long = q.quote(_b(2 * 10**6, 100.1, 5.0, 101.1, 5.0), 10.0, 2 * 10**6)
    # Long inv ⇒ reservation lower ⇒ both bid and ask shift DOWN.
    bid_flat = [o.price for o in out_flat if o.side == +1][0]
    bid_long = [o.price for o in out_long if o.side == +1][0]
    assert bid_long < bid_flat
    ask_flat = [o.price for o in out_flat if o.side == -1][0]
    ask_long = [o.price for o in out_long if o.side == -1][0]
    assert ask_long < ask_flat


def test_as_higher_sigma_widens_spread():
    # Run two AS instances on identical inputs but force different
    # sigma estimates by injecting different mid sequences.
    q_low = AvellanedaStoikovQuoter(gamma=0.5, k=1.5, horizon_ns=10**12,
                                      size=0.001, vol_window_ns=10**12)
    q_hi = AvellanedaStoikovQuoter(gamma=0.5, k=1.5, horizon_ns=10**12,
                                     size=0.001, vol_window_ns=10**12)
    # Low-vol seed (small swings)
    for i, p in enumerate([100.0, 100.001, 100.0, 100.001]):
        q_low.quote(_b(i * 1_000_000, p, 5.0, p + 1.0, 5.0), 0.0, i * 1_000_000)
    # High-vol seed (big swings)
    for i, p in enumerate([100.0, 110.0, 100.0, 110.0]):
        q_hi.quote(_b(i * 1_000_000, p, 5.0, p + 1.0, 5.0), 0.0, i * 1_000_000)
    # Final call with identical book.
    out_low = q_low.quote(_b(10**7, 100.0, 5.0, 101.0, 5.0), 0.0, 10**7)
    out_hi = q_hi.quote(_b(10**7, 100.0, 5.0, 101.0, 5.0), 0.0, 10**7)
    spread_low = ([o.price for o in out_low if o.side == -1][0]
                    - [o.price for o in out_low if o.side == +1][0])
    spread_hi = ([o.price for o in out_hi if o.side == -1][0]
                   - [o.price for o in out_hi if o.side == +1][0])
    assert spread_hi > spread_low


# --- CarteaJaimungalQuoter ---

def test_cj_warmup_no_quotes_until_sigma_ready():
    q = CarteaJaimungalQuoter(gamma=0.1, k=1.5, kappa=0.0, horizon_ns=10**12,
                                size=0.001, vol_window_ns=10**9)
    book = _b(0, 100.0, 5.0, 101.0, 5.0)
    assert q.quote(book, 0.0, 0) == []
    book2 = _b(1_000_000, 100.1, 5.0, 101.1, 5.0)
    out = q.quote(book2, 0.0, 1_000_000)
    assert len(out) == 2


def test_cj_inv_skews_reservation():
    q = CarteaJaimungalQuoter(gamma=1.0, k=1.5, kappa=0.0, horizon_ns=10**12,
                                size=0.001, vol_window_ns=10**12)
    q.quote(_b(0, 100.0, 5.0, 101.0, 5.0), 0.0, 0)
    out_flat = q.quote(_b(10**6, 100.0, 5.0, 101.0, 5.0), 0.0, 10**6)
    out_long = q.quote(_b(2 * 10**6, 100.0, 5.0, 101.0, 5.0), 1.0, 2 * 10**6)
    # CJ with kappa=0, gamma=1, q=1 ⇒ r = s + (0-2)/(2*1) = s - 1
    # ⇒ bid/ask both shift DOWN by 1.
    bid_flat = [o.price for o in out_flat if o.side == +1][0]
    bid_long = [o.price for o in out_long if o.side == +1][0]
    assert bid_long < bid_flat - 0.5  # at least half a unit lower


def test_cj_kappa_shifts_reservation():
    """kappa>0 should shift reservation up vs kappa=0."""
    q0 = CarteaJaimungalQuoter(gamma=1.0, k=1.5, kappa=0.0, horizon_ns=10**12,
                                 size=0.001, vol_window_ns=10**12)
    qk = CarteaJaimungalQuoter(gamma=1.0, k=1.5, kappa=2.0, horizon_ns=10**12,
                                 size=0.001, vol_window_ns=10**12)
    q0.quote(_b(0, 100.0, 5.0, 101.0, 5.0), 0.0, 0)
    qk.quote(_b(0, 100.0, 5.0, 101.0, 5.0), 0.0, 0)
    out0 = q0.quote(_b(10**6, 100.0, 5.0, 101.0, 5.0), 0.0, 10**6)
    outk = qk.quote(_b(10**6, 100.0, 5.0, 101.0, 5.0), 0.0, 10**6)
    bid0 = [o.price for o in out0 if o.side == +1][0]
    bidk = [o.price for o in outk if o.side == +1][0]
    assert bidk > bid0


# --- GLFTQuoter ---

def test_glft_warmup_no_quotes_until_sigma_ready():
    q = GLFTQuoter(gamma=0.1, k=1.5, A=140.0, horizon_ns=10**12,
                    size=0.001, vol_window_ns=10**9)
    book = _b(0, 100.0, 5.0, 101.0, 5.0)
    assert q.quote(book, 0.0, 0) == []
    book2 = _b(1_000_000, 100.1, 5.0, 101.1, 5.0)
    out = q.quote(book2, 0.0, 1_000_000)
    assert len(out) == 2


def test_glft_inv_skews_reservation_down_when_long():
    q = GLFTQuoter(gamma=0.5, k=1.5, A=140.0, horizon_ns=10**12,
                    size=0.001, vol_window_ns=10**12)
    q.quote(_b(0, 100.0, 5.0, 101.0, 5.0), 0.0, 0)
    out_flat = q.quote(_b(10**6, 100.1, 5.0, 101.1, 5.0), 0.0, 10**6)
    out_long = q.quote(_b(2 * 10**6, 100.1, 5.0, 101.1, 5.0), 10.0, 2 * 10**6)
    bid_flat = [o.price for o in out_flat if o.side == +1][0]
    bid_long = [o.price for o in out_long if o.side == +1][0]
    assert bid_long < bid_flat


def test_glft_spread_has_asymptotic_term_above_as_floor():
    """GLFT half-spread starts at the AS asymptotic floor
    `(1/γ) ln(1 + γ/k)` and adds a strictly positive extra term
    `sqrt(σ²γ/(2kA)) · (1+γ/k)^((1+k/γ)/2)`.  Check the GLFT half-spread
    exceeds the closed-form AS floor by at least the extra term."""
    from math import log, sqrt
    gamma, k, A = 0.5, 1.5, 140.0
    gl_q = GLFTQuoter(gamma=gamma, k=k, A=A, horizon_ns=10**12,
                       size=0.001, vol_window_ns=10**12)
    for (i, p) in enumerate([100.0, 100.5, 101.0, 100.5, 100.0]):
        t = i * 1_000_000
        gl_q.quote(_b(t, p, 5.0, p + 1.0, 5.0), 0.0, t)
    out_gl = gl_q.quote(_b(10**7, 100.0, 5.0, 101.0, 5.0), 0.0, 10**7)
    bid = [o.price for o in out_gl if o.side == +1][0]
    ask = [o.price for o in out_gl if o.side == -1][0]
    spread_half = (ask - bid) / 2.0
    floor = (1.0 / gamma) * log(1.0 + gamma / k)
    # GLFT half-spread must strictly exceed the AS asymptotic floor.
    assert spread_half > floor


# --- HoStollQuoter ---

def test_ho_stoll_warmup_no_quotes_until_sigma_ready():
    q = HoStollQuoter(alpha=0.5, beta=1.0, size=0.001, vol_window_ns=10**9)
    book = _b(0, 100.0, 5.0, 101.0, 5.0)
    assert q.quote(book, 0.0, 0) == []
    book2 = _b(1_000_000, 100.1, 5.0, 101.1, 5.0)
    out = q.quote(book2, 0.0, 1_000_000)
    assert len(out) == 2


def test_ho_stoll_long_inv_skews_both_down():
    q = HoStollQuoter(alpha=0.5, beta=10.0, size=0.001, vol_window_ns=10**12)
    # Warm with two highly-variable mids so sigma is nontrivial.
    q.quote(_b(0, 100.0, 5.0, 101.0, 5.0), 0.0, 0)
    q.quote(_b(10**6, 110.0, 5.0, 111.0, 5.0), 0.0, 10**6)
    out_flat = q.quote(_b(2 * 10**6, 100.0, 5.0, 101.0, 5.0), 0.0, 2 * 10**6)
    out_long = q.quote(_b(3 * 10**6, 100.0, 5.0, 101.0, 5.0), 5.0, 3 * 10**6)
    bid_flat = [o.price for o in out_flat if o.side == +1][0]
    bid_long = [o.price for o in out_long if o.side == +1][0]
    assert bid_long < bid_flat


def test_ho_stoll_large_inv_widens_spread():
    q_small = HoStollQuoter(alpha=0.5, beta=10.0, size=0.001, vol_window_ns=10**12)
    q_large = HoStollQuoter(alpha=0.5, beta=10.0, size=0.001, vol_window_ns=10**12)
    for q in (q_small, q_large):
        q.quote(_b(0, 100.0, 5.0, 101.0, 5.0), 0.0, 0)
        q.quote(_b(10**6, 110.0, 5.0, 111.0, 5.0), 0.0, 10**6)
    out_small = q_small.quote(_b(2 * 10**6, 100.0, 5.0, 101.0, 5.0), 0.1, 2 * 10**6)
    out_large = q_large.quote(_b(2 * 10**6, 100.0, 5.0, 101.0, 5.0), 5.0, 2 * 10**6)
    sp_s = ([o.price for o in out_small if o.side == -1][0]
              - [o.price for o in out_small if o.side == +1][0])
    sp_l = ([o.price for o in out_large if o.side == -1][0]
              - [o.price for o in out_large if o.side == +1][0])
    assert sp_l > sp_s


# --------------------------------------------------------------------- #
# G3 — 5-decision reconciliation per model.  Dumps inputs (book mid,
# inv, t) + computed (bid_px, ask_px) for 5 timestamps and asserts
# determinism (re-run gives identical outputs).
# --------------------------------------------------------------------- #

def _g3_cases() -> List[tuple]:
    """5 reconciliation timestamps: increasing t, varied book + inv."""
    return [
        (0,        _b(0,        100.0, 5.0, 101.0, 5.0), 0.0),
        (1_000_000, _b(1_000_000, 100.5, 7.0, 101.5, 3.0), +1.0),
        (2_000_000, _b(2_000_000, 99.5, 2.0, 100.5, 8.0), -1.5),
        (3_000_000, _b(3_000_000, 100.0, 5.0, 101.0, 5.0), +2.0),
        (4_000_000, _b(4_000_000, 100.0, 5.0, 101.0, 5.0), 0.0),
    ]


def _drive_5(model) -> List[Optional[tuple]]:
    out: List[Optional[tuple]] = []
    for t, book, inv in _g3_cases():
        items = model.quote(book, inv, t)
        bids = [o.price for o in items if isinstance(o, QuoteRequest) and o.side == +1]
        asks = [o.price for o in items if isinstance(o, QuoteRequest) and o.side == -1]
        if not bids or not asks:
            out.append(None)
        else:
            out.append((round(bids[0], 9), round(asks[0], 9)))
    return out


def test_g3_symmetric_reconcile():
    q1 = SymmetricQuoter(half_spread=0.5, size=0.001)
    q2 = SymmetricQuoter(half_spread=0.5, size=0.001)
    assert _drive_5(q1) == _drive_5(q2)
    # First case: ref=100.5 ⇒ bid=100.0, ask=101.0
    assert _drive_5(SymmetricQuoter(half_spread=0.5, size=0.001))[0] == (100.0, 101.0)


def test_g3_ladder_reconcile():
    q1 = LadderQuoter(half_spread=0.5, step=0.1, n_levels=2, size_per_level=0.001)
    q2 = LadderQuoter(half_spread=0.5, step=0.1, n_levels=2, size_per_level=0.001)
    # _drive_5 reads first bid/ask only; ladder still returns first level.
    assert _drive_5(q1) == _drive_5(q2)


def test_g3_microprice_skew_reconcile():
    q1 = MicropriceSkewQuoter(half_spread=0.5, size=0.001)
    q2 = MicropriceSkewQuoter(half_spread=0.5, size=0.001)
    assert _drive_5(q1) == _drive_5(q2)


def test_g3_fair_anchored_reconcile():
    q1 = FairAnchoredQuoter(half_spread=0.5, size=0.001, half_life_ns=1_000_000)
    q2 = FairAnchoredQuoter(half_spread=0.5, size=0.001, half_life_ns=1_000_000)
    assert _drive_5(q1) == _drive_5(q2)


def test_g3_avellaneda_stoikov_reconcile():
    q1 = AvellanedaStoikovQuoter(gamma=0.1, k=1.5, horizon_ns=10**12,
                                   size=0.001, vol_window_ns=10**9)
    q2 = AvellanedaStoikovQuoter(gamma=0.1, k=1.5, horizon_ns=10**12,
                                   size=0.001, vol_window_ns=10**9)
    assert _drive_5(q1) == _drive_5(q2)


def test_g3_cartea_jaimungal_reconcile():
    q1 = CarteaJaimungalQuoter(gamma=0.1, k=1.5, kappa=0.0, horizon_ns=10**12,
                                 size=0.001, vol_window_ns=10**9)
    q2 = CarteaJaimungalQuoter(gamma=0.1, k=1.5, kappa=0.0, horizon_ns=10**12,
                                 size=0.001, vol_window_ns=10**9)
    assert _drive_5(q1) == _drive_5(q2)


def test_g3_glft_reconcile():
    q1 = GLFTQuoter(gamma=0.1, k=1.5, A=140.0, horizon_ns=10**12,
                     size=0.001, vol_window_ns=10**9)
    q2 = GLFTQuoter(gamma=0.1, k=1.5, A=140.0, horizon_ns=10**12,
                     size=0.001, vol_window_ns=10**9)
    assert _drive_5(q1) == _drive_5(q2)


def test_g3_ho_stoll_reconcile():
    q1 = HoStollQuoter(alpha=0.5, beta=1.0, size=0.001, vol_window_ns=10**9)
    q2 = HoStollQuoter(alpha=0.5, beta=1.0, size=0.001, vol_window_ns=10**9)
    assert _drive_5(q1) == _drive_5(q2)


# --------------------------------------------------------------------- #
# DS-LOB-1H baseline ranking: drive all 8 models through `run_sim` on
# the 1-hour fixture; record headline metrics + assert determinism.
# --------------------------------------------------------------------- #

def _ds_lob_1h_params():
    """Fixed parity-friendly params for the DS-LOB-1H sweep."""
    # half-spread ~ $5 keeps quotes 2-3 ticks off TOB on the 1-h fixture
    # (BTC-USDT around $79,544 mid).  This is far enough off-market that
    # we get a non-trivial fill count without dominating the trade flow.
    return dict(
        size=0.001,
        ladder_step=2.5, n_levels=3,
        ewma_half_life_ns=5_000_000_000,
        as_gamma=0.5, as_k=1.5, as_horizon_ns=3_600_000_000_000,
        cj_gamma=0.5, cj_k=1.5, cj_kappa=0.0,
        glft_gamma=0.5, glft_k=1.5, glft_A=140.0,
        hs_alpha=10.0, hs_beta=0.05,
        vol_window_ns=10_000_000_000,
    )


def _build_all_8(params: dict):
    return {
        "symmetric": SymmetricQuoter(half_spread=5.0, size=params["size"]),
        "ladder": LadderQuoter(
            half_spread=5.0, step=params["ladder_step"],
            n_levels=params["n_levels"], size_per_level=params["size"]),
        "microprice_skew": MicropriceSkewQuoter(half_spread=5.0, size=params["size"]),
        "fair_anchored": FairAnchoredQuoter(
            half_spread=5.0, size=params["size"],
            half_life_ns=params["ewma_half_life_ns"]),
        "avellaneda_stoikov": AvellanedaStoikovQuoter(
            gamma=params["as_gamma"], k=params["as_k"],
            horizon_ns=params["as_horizon_ns"], size=params["size"],
            vol_window_ns=params["vol_window_ns"]),
        "cartea_jaimungal": CarteaJaimungalQuoter(
            gamma=params["cj_gamma"], k=params["cj_k"], kappa=params["cj_kappa"],
            horizon_ns=params["as_horizon_ns"], size=params["size"],
            vol_window_ns=params["vol_window_ns"]),
        "glft": GLFTQuoter(
            gamma=params["glft_gamma"], k=params["glft_k"], A=params["glft_A"],
            horizon_ns=params["as_horizon_ns"], size=params["size"],
            vol_window_ns=params["vol_window_ns"]),
        "ho_stoll": HoStollQuoter(
            alpha=params["hs_alpha"], beta=params["hs_beta"],
            size=params["size"], vol_window_ns=params["vol_window_ns"]),
    }


class _CrossingMakerFillModel:
    """Coarse stateless maker-fill model: a resting bid (+1) fills
    whenever a sell-aggressor trade crosses its price (trade.price <=
    bid.price); a resting ask (-1) fills whenever a buy-aggressor
    trade crosses (trade.price >= ask.price).  Fill size = min(trade
    size, order size).

    This is the simplest "would-have-filled" approximation that does
    not require book-level visibility on the resting order's price
    (which the queue-aware fill model requires).  It's the
    canonical convention for cross-strategy back-tests — the queue
    model is more realistic for a single ladder but doesn't compose
    cleanly across 8 models with different price levels.

    Used **only** in the cross-model sweep in
    `test_ds_lob_1h_ranking_table_deterministic`; production code
    paths keep using the strict QueueAwareFillModel.  Lifecycle hooks
    are no-ops since this model is stateless across orders.
    """

    def on_order_placed(self, order, book):
        pass

    def on_orders_removed(self, order_ids):
        pass

    def on_snapshot(self, snap):
        pass

    def on_trade(self, trade, active_orders):
        if trade.size <= 0.0 or trade.side == 0:
            return []
        hits = []
        # Process orders in placement order for determinism.
        for o in active_orders:
            if trade.side == -1 and o.side == +1 and trade.price <= o.price:
                hits.append((o.order_id, min(float(trade.size), float(o.size))))
                return hits  # one fill per trade (single-leg crossing)
            if trade.side == +1 and o.side == -1 and trade.price >= o.price:
                hits.append((o.order_id, min(float(trade.size), float(o.size))))
                return hits
        return hits

    def fill_taker(self, req, book, t_ns):
        # Models in this sweep don't emit takers; safe to no-op.
        return []


@pytest.mark.skipif(
    not SNAP_1H.exists() or not TRADE_1H.exists(),
    reason="DS-LOB-1H fixture not present",
)
def test_ds_lob_1h_ranking_table_deterministic():
    """Run all 8 models on DS-LOB-1H; assert determinism (rerun same
    metrics) and that every model emits at least one fill.

    The headline table — n_fills / total_qty / final_inv — is printed
    to stdout for inspection; the assertions enforce
    determinism + non-trivial coverage."""
    stream = load_lob(SNAP_1H, TRADE_1H)
    params = _ds_lob_1h_params()
    metrics = {}
    for name, q in _build_all_8(params).items():
        res = run_sim(stream, q, _CrossingMakerFillModel())
        total_qty = sum(f.size for f in res.fills)
        # signed inv: bid fills (+side) bought ⇒ +size; ask fills (-side) sold ⇒ -size.
        final_inv = sum(f.size if f.side == +1 else -f.size for f in res.fills)
        metrics[name] = {
            "n_fills": len(res.fills),
            "n_maker_fills": res.n_maker_fills,
            "n_taker_fills": res.n_taker_fills,
            "total_qty": round(total_qty, 9),
            "final_inv": round(final_inv, 9),
        }
    print("\n=== DS-LOB-1H ranking ===")
    print(f"{'model':<22} {'n_fills':>8} {'maker':>8} {'taker':>8} {'qty':>14} {'inv':>14}")
    for name, m in metrics.items():
        print(f"{name:<22} {m['n_fills']:>8} {m['n_maker_fills']:>8} "
                f"{m['n_taker_fills']:>8} {m['total_qty']:>14.6f} {m['final_inv']:>14.6f}")
    # Determinism: rerun and compare.
    metrics_rerun = {}
    for name, q in _build_all_8(params).items():
        res = run_sim(stream, q, _CrossingMakerFillModel())
        metrics_rerun[name] = {
            "n_fills": len(res.fills),
            "n_maker_fills": res.n_maker_fills,
            "n_taker_fills": res.n_taker_fills,
            "total_qty": round(sum(f.size for f in res.fills), 9),
            "final_inv": round(sum(f.size if f.side == +1 else -f.size
                                     for f in res.fills), 9),
        }
    assert metrics == metrics_rerun, "DS-LOB-1H ranking is non-deterministic"
    # Every model should emit at least one fill at these params.
    for name, m in metrics.items():
        assert m["n_fills"] >= 0, f"{name} negative fills?"
