"""Tests for inventory-penalty primitives.

Each penalty's ``∂skew/∂inventory`` has a known sign convention:
positive inventory should push the quote skew DOWN (encouraging
selling); negative inventory should push skew UP. We test that
convention per-penalty.
"""
from __future__ import annotations

from mmsim.t4_corpus.combos import INVENTORY_PENALTIES, Combo
from mmsim.t4_corpus.t4_composer import build_mm_strategy


def _make_combo(inv_pen: str) -> Combo:
    return Combo(
        quoting_model="symmetric",
        inventory_penalty=inv_pen,
        adverse_filter="none",
        hedge_mode="none",
        reference_price="mid",
        quote_shape="single",
        refresh_trigger="book_event",
    )


def test_linear_skew_sign_long_inventory():
    cfg = build_mm_strategy(_make_combo("linear"), {})
    s_pos = cfg.inv_penalty_fn(+1.0)
    s_neg = cfg.inv_penalty_fn(-1.0)
    # Long inventory => negative price_offset (shift quotes DOWN)
    assert s_pos.price_offset < 0
    assert s_neg.price_offset > 0
    assert s_pos.size_scale_bid == 1.0 == s_pos.size_scale_ask


def test_quadratic_skew_grows_with_magnitude():
    cfg = build_mm_strategy(_make_combo("quadratic"), {})
    s1 = cfg.inv_penalty_fn(+1.0)
    s2 = cfg.inv_penalty_fn(+2.0)
    # |offset(2)| > |offset(1)| because quadratic.
    assert abs(s2.price_offset) > abs(s1.price_offset)
    assert s1.price_offset < 0 and s2.price_offset < 0


def test_exponential_skew_explodes():
    cfg = build_mm_strategy(_make_combo("exponential"),
                              {"gamma": 1.0, "inventory_cap": 0.1})
    small = cfg.inv_penalty_fn(0.01)
    large = cfg.inv_penalty_fn(1.0)
    assert abs(large.price_offset) > 10 * abs(small.price_offset)


def test_asymmetric_long_vs_short_differs():
    cfg = build_mm_strategy(_make_combo("asymmetric"), {"gamma": 1.0})
    s_long = cfg.inv_penalty_fn(+1.0)
    s_short = cfg.inv_penalty_fn(-1.0)
    # gamma_long=1.0, gamma_short=0.5 (per composer convention)
    # |long offset| = 1.0, |short offset| = 0.5
    assert abs(s_long.price_offset) > abs(s_short.price_offset)


def test_soft_cap_extra_ramp_beyond_cap():
    cfg = build_mm_strategy(_make_combo("soft_cap"),
                              {"gamma": 1.0, "inventory_cap": 0.5})
    at_cap = cfg.inv_penalty_fn(0.5)
    over_cap = cfg.inv_penalty_fn(0.75)
    # soft_cap adds a quadratic ramp beyond |inv| >= cap. so |over| > |at|.
    assert abs(over_cap.price_offset) > abs(at_cap.price_offset)


def test_hard_cap_drops_side():
    cfg = build_mm_strategy(_make_combo("hard_cap"),
                              {"inventory_cap": 0.5})
    # Long >= cap => drop the bid (size_scale_bid=0)
    s = cfg.inv_penalty_fn(+1.0)
    assert s.size_scale_bid == 0.0
    assert s.size_scale_ask == 1.0
    # Short <= -cap => drop the ask
    s2 = cfg.inv_penalty_fn(-1.0)
    assert s2.size_scale_bid == 1.0
    assert s2.size_scale_ask == 0.0


def test_all_inventory_penalties_zero_at_zero():
    """At inv=0 every penalty produces a zero price offset (boundary)."""
    for ip in INVENTORY_PENALTIES:
        cfg = build_mm_strategy(_make_combo(ip),
                                  {"gamma": 1.0, "inventory_cap": 0.5})
        s = cfg.inv_penalty_fn(0.0)
        assert s.price_offset == 0.0


def test_no_lookahead_under_pollute():
    """Pure-function check: passing inv=x then inv=y then inv=x must give
    the same Skew the first and third times (the primitives have no
    hidden state)."""
    for ip in INVENTORY_PENALTIES:
        cfg = build_mm_strategy(_make_combo(ip),
                                  {"gamma": 1.0, "inventory_cap": 0.5})
        s_a1 = cfg.inv_penalty_fn(+0.3)
        _ = cfg.inv_penalty_fn(+0.9)
        s_a2 = cfg.inv_penalty_fn(+0.3)
        assert s_a1 == s_a2
