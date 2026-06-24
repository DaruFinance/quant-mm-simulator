"""Tests for reference-price strategies.

The math invariant: every pure book-based reference price lies in
[best_bid, best_ask] (the natural admissible range for any fair).
Stateful trackers may produce values outside this range before
warmup or under heavy fading.
"""
from __future__ import annotations

from mmsim.ingest.lob import Book
from mmsim.t4_corpus.combos import REFERENCE_PRICES, Combo
from mmsim.t4_corpus.t4_composer import _RefPriceFn, build_mm_strategy


def _make_combo(ref: str) -> Combo:
    return Combo(
        quoting_model="symmetric",
        inventory_penalty="linear",
        adverse_filter="none",
        hedge_mode="none",
        reference_price=ref,
        quote_shape="single",
        refresh_trigger="book_event",
    )


def _book() -> Book:
    return Book(ts_ns=100,
                 bids=((100.0, 1.0),), asks=((101.0, 1.0),))


def _asymmetric_book() -> Book:
    # heavy bid side: bid_sz=10, ask_sz=1 → microprice/weighted_mid lean up.
    return Book(ts_ns=100,
                 bids=((100.0, 10.0),), asks=((101.0, 1.0),))


def test_mid_in_bid_ask():
    cfg = build_mm_strategy(_make_combo("mid"), {})
    v = cfg.ref_price_fn(_book())
    assert v == 100.5
    assert 100.0 <= v <= 101.0


def test_microprice_in_bid_ask_range():
    cfg = build_mm_strategy(_make_combo("microprice"), {})
    book = _asymmetric_book()
    v = cfg.ref_price_fn(book)
    assert 100.0 <= v <= 101.0
    # Heavy bid => microprice > mid.
    assert v > 100.5


def test_weighted_mid_in_bid_ask_range():
    cfg = build_mm_strategy(_make_combo("weighted_mid"), {})
    book = _asymmetric_book()
    v = cfg.ref_price_fn(book)
    assert 100.0 <= v <= 101.0
    # Heavy bid (more buyers) => weighted_mid leans toward ask (price up)
    # per the convention in mmsim.quoter.refprice.weighted_mid.
    assert v > 100.5


def test_ewma_fair_seeds_at_first_observation():
    cfg = build_mm_strategy(_make_combo("ewma_fair"),
                              {"ewma_half_life_ns": 1_000_000_000})
    book = _book()
    v = cfg.ref_price_fn(book)
    assert v == 100.5  # First obs seeds at mid


def test_vwap_falls_back_to_mid_without_trades():
    """VWAP needs trade-tape input; without trades it falls back to mid."""
    cfg = build_mm_strategy(_make_combo("vwap"), {})
    v = cfg.ref_price_fn(_book())
    assert v == 100.5


def test_model_pred_stub_returns_mid():
    """The composer's model_pred routes to a no-op fallback returning mid."""
    cfg = build_mm_strategy(_make_combo("model_pred"), {})
    v = cfg.ref_price_fn(_book())
    assert v == 100.5


def test_all_reference_prices_compose():
    for rp in REFERENCE_PRICES:
        cfg = build_mm_strategy(_make_combo(rp), {})
        assert cfg.ref_price_fn is not None


def test_ref_price_no_book_returns_none():
    """Every ref-price strategy returns None when the book is missing."""
    for rp in REFERENCE_PRICES:
        cfg = build_mm_strategy(_make_combo(rp), {})
        v = cfg.ref_price_fn(None)
        # mid / microprice / weighted_mid / vwap fallback return None.
        # ewma_fair returns None pre-warmup. model_pred returns top_mid(None)=None.
        assert v is None


def test_ref_price_deterministic():
    """Same input book → same output across calls."""
    for rp in REFERENCE_PRICES:
        cfg = build_mm_strategy(_make_combo(rp), {})
        b = _book()
        v1 = cfg.ref_price_fn(b)
        v2 = cfg.ref_price_fn(b)
        # For stateful trackers (ewma_fair) v2 may differ from v1 only by
        # the deterministic update — but here we pass the same book with
        # the same timestamp twice; ewma's same-instant rule means v2 == v1.
        if rp == "vwap":
            # vwap state-independent until trades observed (fallback both times).
            assert v1 == v2
        else:
            assert v1 == v2
