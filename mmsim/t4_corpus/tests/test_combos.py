"""Tests for the T4 structural combo grid."""
from __future__ import annotations

from mmsim.t4_corpus import combos as C


def test_grid_size_is_201600():
    assert C.GRID_SIZE == 201_600
    # Sanity: 8 * 6 * 7 * 4 * 6 * 5 * 5 = 201_600
    assert C.GRID_SIZE == len(C.QUOTING_MODELS) * len(C.INVENTORY_PENALTIES) \
        * len(C.ADVERSE_FILTERS) * len(C.HEDGE_MODES) \
        * len(C.REFERENCE_PRICES) * len(C.QUOTE_SHAPES) * len(C.REFRESH_TRIGGERS)


def test_iter_all_combos_count():
    n = sum(1 for _ in C.iter_all_combos())
    assert n == C.GRID_SIZE


def test_iter_all_combos_unique():
    seen = set()
    for c in C.iter_all_combos():
        assert c not in seen
        seen.add(c)
    assert len(seen) == C.GRID_SIZE


def test_sample_combos_deterministic_same_seed():
    a = C.sample_combos(50, seed=2026)
    b = C.sample_combos(50, seed=2026)
    assert a == b


def test_sample_combos_different_seeds_differ():
    a = C.sample_combos(50, seed=2026)
    b = C.sample_combos(50, seed=2027)
    assert a != b


def test_sample_combos_unique_within_sample():
    s = C.sample_combos(100, seed=2026)
    assert len(set(s)) == 100


def test_sample_combos_n_zero():
    s = C.sample_combos(0, seed=2026)
    assert s == []


def test_sample_combos_rejects_negative_n():
    import pytest
    with pytest.raises(ValueError):
        C.sample_combos(-1)


def test_sample_combos_rejects_oversized_n():
    import pytest
    with pytest.raises(ValueError):
        C.sample_combos(C.GRID_SIZE + 1)


def test_combo_at_zero_first_token_per_axis():
    # iter_all_combos's first element should be tuple-of-firsts.
    first = next(iter(C.iter_all_combos()))
    assert first.quoting_model == C.QUOTING_MODELS[0]
    assert first.inventory_penalty == C.INVENTORY_PENALTIES[0]
    assert first.adverse_filter == C.ADVERSE_FILTERS[0]
    assert first.hedge_mode == C.HEDGE_MODES[0]
    assert first.reference_price == C.REFERENCE_PRICES[0]
    assert first.quote_shape == C.QUOTE_SHAPES[0]
    assert first.refresh_trigger == C.REFRESH_TRIGGERS[0]


def test_strategy_name_has_seven_parts():
    s = C.sample_combos(20, seed=2026)
    for c in s:
        name = C.combo_to_strategy_name(c)
        parts = name.split("_")
        assert len(parts) == 7, f"strategy name has wrong part count: {name}"


def test_combo_dict_round_trip():
    s = C.sample_combos(20, seed=2026)
    for c in s:
        d = C.combo_to_dict(c)
        back = C.combo_from_dict(d)
        assert back == c


def test_to_dataframe_has_columns():
    s = C.sample_combos(5, seed=2026)
    df = C.to_dataframe(s)
    assert len(df) == 5
    for col in ("quoting_model", "inventory_penalty", "adverse_filter",
                "hedge_mode", "reference_price", "quote_shape",
                "refresh_trigger", "strategy_name"):
        assert col in df.columns


def test_pre_prune_is_permissive():
    s = C.sample_combos(20, seed=2026)
    pruned = C.pre_prune_combos(s)
    assert len(pruned) == 20


# --- 50-T pollute leak tests (the combos public surface is pure;
# polluting the global RNG state should not change a seeded draw). ---

def test_sample_combos_no_lookahead_under_pollution():
    import numpy as np
    # Pollute the default RNG; sampler should still match.
    np.random.seed(99999)
    expected = C.sample_combos(50, seed=2026)
    np.random.seed(11111)
    got = C.sample_combos(50, seed=2026)
    assert expected == got
