"""Tests for quote shapes."""
from __future__ import annotations

from mmsim.t4_corpus.combos import QUOTE_SHAPES, Combo
from mmsim.t4_corpus.t4_composer import build_mm_strategy
from mmsim.quoter.shapes import (
    single, paired, ladder, geometric, dynamic_depth,
    SingleSpec, PairedSpec, LadderSpec, GeometricSpec, DynamicDepthSpec,
)


def _make_combo(shape: str) -> Combo:
    return Combo(
        quoting_model="symmetric",
        inventory_penalty="linear",
        adverse_filter="none",
        hedge_mode="none",
        reference_price="mid",
        quote_shape=shape,
        refresh_trigger="book_event",
    )


def test_single_spec_constructs():
    cfg = build_mm_strategy(_make_combo("single"), {})
    assert isinstance(cfg.shape_spec, SingleSpec)


def test_paired_spec_constructs():
    cfg = build_mm_strategy(_make_combo("paired"), {})
    assert isinstance(cfg.shape_spec, PairedSpec)
    # 2 levels per side (per composer default)
    assert len(cfg.shape_spec.levels_bid) == 2
    assert len(cfg.shape_spec.levels_ask) == 2


def test_ladder_spec_constructs():
    cfg = build_mm_strategy(_make_combo("ladder"), {"ladder_n_levels": 4})
    assert isinstance(cfg.shape_spec, LadderSpec)
    assert cfg.shape_spec.n_levels == 4
    # Run the ladder primitive: should produce 2*n_levels orders.
    out = ladder(cfg.shape_spec, ref_price=100.0)
    assert len(out) == 2 * 4


def test_geometric_spec_constructs():
    cfg = build_mm_strategy(_make_combo("geometric"),
                              {"ladder_n_levels": 3, "geometric_ratio": 2.0})
    assert isinstance(cfg.shape_spec, GeometricSpec)
    # Geometric spacing: level offsets = h, h*r, h*r^2.
    out = geometric(cfg.shape_spec, ref_price=100.0)
    assert len(out) == 6
    # Verify ratio: bid_2 offset == bid_1 offset * ratio
    bids = [q for q in out if q.side == +1]
    asks = [q for q in out if q.side == -1]
    # offsets from ref (positive numbers)
    off_b0 = 100.0 - bids[0].price
    off_b1 = 100.0 - bids[1].price
    off_b2 = 100.0 - bids[2].price
    assert abs(off_b1 / off_b0 - 2.0) < 1e-9
    assert abs(off_b2 / off_b1 - 2.0) < 1e-9


def test_dynamic_depth_taper():
    cfg = build_mm_strategy(_make_combo("dynamic_depth"),
                              {"ladder_n_levels": 5, "inventory_cap": 1.0})
    assert isinstance(cfg.shape_spec, DynamicDepthSpec)
    # At inv=0, all 5 levels are active.
    out_zero = dynamic_depth(cfg.shape_spec, ref_price=100.0, inv=0.0)
    assert len(out_zero) == 2 * 5
    # At inv well past 2*threshold, taper to 1 level per side.
    out_max = dynamic_depth(cfg.shape_spec, ref_price=100.0, inv=10.0)
    assert len(out_max) == 2


def test_ladder_geometric_spacing_correct():
    """Linear vs geometric step semantics."""
    spec = LadderSpec(half_spread=1.0, step=2.0, n_levels=3,
                       size_per_level=0.1)
    out = ladder(spec, ref_price=100.0)
    bids = sorted([q.price for q in out if q.side == +1], reverse=True)
    # innermost at 100 - 1 = 99; step 2 ⇒ 97, 95
    assert abs(bids[0] - 99.0) < 1e-12
    assert abs(bids[1] - 97.0) < 1e-12
    assert abs(bids[2] - 95.0) < 1e-12


def test_all_quote_shapes_compose():
    for sh in QUOTE_SHAPES:
        cfg = build_mm_strategy(_make_combo(sh), {})
        assert cfg.shape_spec is not None
