"""Tests for the T4 IS-axis search space."""
from __future__ import annotations

import math

import pytest

from mmsim.t4_corpus.search_space import SEARCH_SPACE_T4, search_space_t4, Axis


def test_seven_axes():
    sp = search_space_t4()
    names = sp.names
    assert "gamma" in names
    assert "k" in names
    assert "horizon_ns" in names
    assert "inventory_cap" in names
    assert "refresh_interval_ns" in names
    assert "spread_floor" in names
    assert "filter_threshold" in names
    assert len(names) == 7


def test_sample_count_matches():
    sp = SEARCH_SPACE_T4
    s = sp.sample(64, seed=2026)
    assert len(s) == 64


def test_sample_keys_match_axes():
    sp = SEARCH_SPACE_T4
    s = sp.sample(8, seed=2026)
    expected = set(sp.names)
    for sample in s:
        assert set(sample.keys()) == expected


def test_sample_values_in_range():
    sp = SEARCH_SPACE_T4
    s = sp.sample(32, seed=2026)
    for sample in s:
        for ax in sp.axes:
            v = sample[ax.name]
            if ax.kind == "int":
                assert ax.low <= v <= ax.high
            elif ax.kind == "float":
                assert ax.low <= v < ax.high or math.isclose(v, ax.high)
            elif ax.kind == "log_float":
                assert ax.low <= v <= ax.high * (1.0 + 1e-9)
            elif ax.kind == "choice":
                assert v in ax.choices


def test_sample_deterministic_same_seed():
    sp = SEARCH_SPACE_T4
    a = sp.sample(16, seed=2026)
    b = sp.sample(16, seed=2026)
    assert a == b


def test_sample_different_seed_differs():
    sp = SEARCH_SPACE_T4
    a = sp.sample(16, seed=2026)
    b = sp.sample(16, seed=2027)
    assert a != b


def test_grid_count_matches():
    sp = SEARCH_SPACE_T4
    g = sp.grid(2)
    # 7 numeric axes × 2 levels = 128 combos
    assert len(g) == sp.n_grid_combinations(2)


def test_duplicate_axis_rejected():
    with pytest.raises(ValueError):
        from mmsim.t4_corpus.search_space import SearchSpace
        SearchSpace((Axis("x", "float", 0, 1), Axis("x", "float", 0, 1)))


def test_n_negative_rejected():
    with pytest.raises(ValueError):
        SEARCH_SPACE_T4.sample(-1)


def test_pollute_then_sample_unchanged():
    """50-T pollute leak rail: SearchSpace.sample is seed-deterministic."""
    import numpy as np
    np.random.seed(99999)
    expected = SEARCH_SPACE_T4.sample(32, seed=2026)
    np.random.seed(11111)
    got = SEARCH_SPACE_T4.sample(32, seed=2026)
    assert expected == got
