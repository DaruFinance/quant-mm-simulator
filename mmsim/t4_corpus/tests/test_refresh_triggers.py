"""Tests for refresh-trigger primitives."""
from __future__ import annotations

from mmsim.ingest.lob import Book
from mmsim.t4_corpus.combos import REFRESH_TRIGGERS, Combo
from mmsim.t4_corpus.t4_composer import build_mm_strategy


def _make_combo(trig: str) -> Combo:
    return Combo(
        quoting_model="symmetric",
        inventory_penalty="linear",
        adverse_filter="none",
        hedge_mode="none",
        reference_price="mid",
        quote_shape="single",
        refresh_trigger=trig,
    )


def _book(ts: int, mid: float) -> Book:
    return Book(ts_ns=ts,
                 bids=((mid - 0.5, 1.0),), asks=((mid + 0.5, 1.0),))


def test_time_trigger_fires_after_interval():
    cfg = build_mm_strategy(_make_combo("time"),
                              {"refresh_interval_ns": 1_000_000_000})  # 1s
    t = cfg.refresh_trigger
    # First call always fires.
    assert t.step(_book(100, 100.0), 0.0, 100) is True
    # Same instant -> no fire.
    assert t.step(_book(100, 100.0), 0.0, 100) is False
    # 0.5s later -> no fire.
    assert t.step(_book(100, 100.0), 0.0, 500_000_100) is False
    # 1.1s later -> fires.
    assert t.step(_book(100, 100.0), 0.0, 1_100_000_100) is True


def test_mid_move_trigger_fires_on_threshold():
    cfg = build_mm_strategy(_make_combo("mid_move"),
                              {"trigger_mid_move_bp": 10.0})  # 10 bp
    t = cfg.refresh_trigger
    # First call baseline-sets.
    assert t.step(_book(100, 100.0), 0.0, 100) is True
    # 5bp move -> no fire.
    assert t.step(_book(200, 100.05), 0.0, 200) is False
    # 20bp move -> fires.
    assert t.step(_book(300, 100.20), 0.0, 300) is True


def test_inv_change_trigger_fires_on_threshold():
    cfg = build_mm_strategy(_make_combo("inv_change"),
                              {"trigger_inv_change": 0.05})
    t = cfg.refresh_trigger
    # First call always fires.
    assert t.step(None, 0.0, 100) is True
    # |0.02 - 0.0| = 0.02 < 0.05 -> no fire.
    assert t.step(None, 0.02, 200) is False
    # |0.10 - 0.0| = 0.10 >= 0.05 -> fires.
    assert t.step(None, 0.10, 300) is True


def test_book_event_trigger_always_fires():
    cfg = build_mm_strategy(_make_combo("book_event"), {})
    t = cfg.refresh_trigger
    for i in range(20):
        assert t.step(_book(i, 100.0), 0.0, i) is True


def test_hybrid_trigger_fires_when_any_child_fires():
    cfg = build_mm_strategy(_make_combo("hybrid"),
                              {"refresh_interval_ns": 1_000_000_000,
                               "trigger_mid_move_bp": 10.0})
    t = cfg.refresh_trigger
    # First call: both children's first calls fire, so hybrid fires.
    assert t.step(_book(100, 100.0), 0.0, 100) is True
    # Soon after, neither child fires -> hybrid doesn't fire.
    assert t.step(_book(150, 100.0), 0.0, 150) is False
    # Mid moves a lot -> mid_move fires -> hybrid fires.
    assert t.step(_book(200, 110.0), 0.0, 200) is True


def test_all_refresh_triggers_compose():
    for tr in REFRESH_TRIGGERS:
        cfg = build_mm_strategy(_make_combo(tr), {})
        assert cfg.refresh_trigger is not None
        # Each trigger's step returns a bool.
        out = cfg.refresh_trigger.step(_book(100, 100.0), 0.0, 100)
        assert isinstance(out, bool)
