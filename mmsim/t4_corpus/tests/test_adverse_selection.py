"""Tests for adverse-selection filters."""
from __future__ import annotations

from mmsim.t4_corpus.combos import ADVERSE_FILTERS, Combo
from mmsim.t4_corpus.t4_composer import build_mm_strategy


def _make_combo(filt: str) -> Combo:
    return Combo(
        quoting_model="symmetric",
        inventory_penalty="linear",
        adverse_filter=filt,
        hedge_mode="none",
        reference_price="mid",
        quote_shape="single",
        refresh_trigger="book_event",
    )


def test_none_returns_no_filter():
    cfg = build_mm_strategy(_make_combo("none"), {})
    assert cfg.adverse_filter is None


def test_ofi_filter_constructs():
    cfg = build_mm_strategy(_make_combo("ofi"), {"filter_threshold": 0.4})
    assert cfg.adverse_filter is not None
    assert hasattr(cfg.adverse_filter, "is_adverse")


def test_toxicity_filter_threshold_in_legal_range():
    cfg = build_mm_strategy(_make_combo("toxicity"), {"filter_threshold": 0.1})
    # composer remaps to [0.5, 0.95]
    assert 0.5 <= cfg.adverse_filter.threshold <= 1.0


def test_vol_surge_filter_constructs():
    cfg = build_mm_strategy(_make_combo("vol_surge"), {})
    assert cfg.adverse_filter is not None


def test_microprice_dev_filter_constructs():
    cfg = build_mm_strategy(_make_combo("microprice_dev"), {})
    assert cfg.adverse_filter is not None


def test_queue_imb_filter_constructs():
    cfg = build_mm_strategy(_make_combo("queue_imb"), {"filter_threshold": 0.4})
    assert cfg.adverse_filter is not None
    assert hasattr(cfg.adverse_filter, "threshold")


def test_hybrid_filter_constructs():
    cfg = build_mm_strategy(_make_combo("hybrid"), {})
    assert cfg.adverse_filter is not None
    # HybridAdverseFilter has a `children` list.
    assert hasattr(cfg.adverse_filter, "children")
    assert len(cfg.adverse_filter.children) == 2


def test_filter_is_adverse_returns_bool():
    """Every filter's is_adverse(t) returns a bool (sanity)."""
    for filt in ADVERSE_FILTERS:
        if filt == "none":
            continue
        cfg = build_mm_strategy(_make_combo(filt), {})
        out = cfg.adverse_filter.is_adverse(0)
        assert isinstance(out, bool)


def test_all_filters_construct():
    """Every filter axis option composes without error."""
    for filt in ADVERSE_FILTERS:
        cfg = build_mm_strategy(_make_combo(filt), {})
        if filt == "none":
            assert cfg.adverse_filter is None
        else:
            assert cfg.adverse_filter is not None
