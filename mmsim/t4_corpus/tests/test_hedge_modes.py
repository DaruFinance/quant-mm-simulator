"""Tests for hedge modes.

The mmsim HedgeEngine's invariant: net_delta = primary_inv + hedge_inv.
After a hedge fire that fully flattens (hedge_size_pct=1.0), the net
delta should be approximately zero.
"""
from __future__ import annotations

from mmsim.t4_corpus.combos import HEDGE_MODES, Combo
from mmsim.t4_corpus.t4_composer import build_mm_strategy
from mmsim.ingest.lob import Book
from mmsim.sim.loop import Fill


def _make_combo(mode: str) -> Combo:
    return Combo(
        quoting_model="symmetric",
        inventory_penalty="linear",
        adverse_filter="none",
        hedge_mode=mode,
        reference_price="mid",
        quote_shape="single",
        refresh_trigger="book_event",
    )


def test_none_returns_no_hedge_engine():
    cfg = build_mm_strategy(_make_combo("none"), {})
    assert cfg.hedge_engine is None


def test_options_vega_stub_returns_no_engine():
    """Per the spec, options_vega is a TODO; routes to None (fallback)."""
    cfg = build_mm_strategy(_make_combo("options_vega_stub"), {})
    assert cfg.hedge_engine is None


def test_perp_hedge_constructs():
    cfg = build_mm_strategy(_make_combo("perp"), {})
    assert cfg.hedge_engine is not None
    assert cfg.hedge_engine.instrument == "perp"


def test_basket_hedge_constructs():
    cfg = build_mm_strategy(_make_combo("basket"), {})
    assert cfg.hedge_engine is not None
    assert cfg.hedge_engine.instrument == "basket"


def test_perp_hedge_net_delta_zero_after_full_flatten():
    """After a flatten cycle the net_delta should be ~0 (perp)."""
    cfg = build_mm_strategy(_make_combo("perp"),
                              {"hedge_threshold": 0.01})
    eng = cfg.hedge_engine
    # Simulate a +1.0 primary fill: bid filled at 100.0, size 1.0
    eng.observe_fill(Fill(fill_id=1, order_id=10, ts_ns=100,
                            price=100.0, size=1.0, side=+1, is_maker=True))
    assert eng.inv == 1.0
    # Fire hedge against a hedge book (we sell against best bid).
    hedge_book = Book(ts_ns=100,
                       bids=((99.9, 10.0),), asks=((100.1, 10.0),))
    fill = eng.make_hedge(hedge_book, t_ns=100)
    assert fill is not None
    # Net delta should now be ~0 (within float precision).
    assert abs(eng.net_delta) < 1e-9


def test_basket_hedge_net_delta_zero_after_full_flatten():
    cfg = build_mm_strategy(_make_combo("basket"),
                              {"hedge_threshold": 0.01})
    eng = cfg.hedge_engine
    eng.observe_fill(Fill(fill_id=1, order_id=10, ts_ns=100,
                            price=100.0, size=0.5, side=-1, is_maker=True))
    assert eng.inv == -0.5
    hedge_book = Book(ts_ns=100,
                       bids=((99.9, 10.0),), asks=((100.1, 10.0),))
    fill = eng.make_hedge(hedge_book, t_ns=100)
    assert fill is not None
    # We were short, so hedge BUYS to cover; net delta ~0.
    assert abs(eng.net_delta) < 1e-9


def test_all_hedge_modes_compose():
    for hm in HEDGE_MODES:
        cfg = build_mm_strategy(_make_combo(hm), {})
        if hm in ("none", "options_vega_stub"):
            assert cfg.hedge_engine is None
        else:
            assert cfg.hedge_engine is not None
