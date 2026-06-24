"""Inventory-penalty primitive tests."""
from __future__ import annotations

from math import exp, isclose

import pytest

from mmsim.quoter.inv_penalty import (
    Skew, asymmetric, exponential, hard_cap, linear, quadratic, soft_cap,
)


# --------------------------------------------------------------------- #
# linear
# --------------------------------------------------------------------- #

def test_linear_zero_inv_zero_skew():
    s = linear(inv=0.0, gamma=1.0)
    assert s.price_offset == 0.0
    assert s.size_scale_bid == 1.0
    assert s.size_scale_ask == 1.0


def test_linear_long_inv_shifts_down():
    s = linear(inv=2.0, gamma=0.5)
    # -gamma * inv = -1.0
    assert s.price_offset == -1.0


def test_linear_short_inv_shifts_up():
    s = linear(inv=-2.0, gamma=0.5)
    assert s.price_offset == 1.0


# --------------------------------------------------------------------- #
# quadratic
# --------------------------------------------------------------------- #

def test_quadratic_zero_inv():
    assert quadratic(0.0, 1.0).price_offset == 0.0


def test_quadratic_grows_nonlinearly():
    s1 = quadratic(1.0, 0.5)
    s2 = quadratic(2.0, 0.5)
    # |skew_2| / |skew_1| == (inv_2/inv_1)^2 == 4
    assert abs(s2.price_offset) / abs(s1.price_offset) == pytest.approx(4.0)


def test_quadratic_sign_opposes_inv():
    assert quadratic(2.0, 1.0).price_offset < 0  # long -> down
    assert quadratic(-2.0, 1.0).price_offset > 0  # short -> up


# --------------------------------------------------------------------- #
# exponential
# --------------------------------------------------------------------- #

def test_exponential_zero_inv_zero_skew():
    assert exponential(0.0, 1.0, 1.0).price_offset == 0.0


def test_exponential_matches_formula():
    s = exponential(2.0, 1.0, 1.0)
    expected = -1.0 * (exp(2.0) - 1.0)  # sign=-1 (long), gamma=1, |inv|/scale=2
    assert s.price_offset == pytest.approx(expected)


def test_exponential_short_inv_positive_skew():
    s = exponential(-1.0, 1.0, 1.0)
    expected = +1.0 * (exp(1.0) - 1.0)
    assert s.price_offset == pytest.approx(expected)


# --------------------------------------------------------------------- #
# asymmetric
# --------------------------------------------------------------------- #

def test_asymmetric_different_long_short_coeffs():
    s_long = asymmetric(2.0, gamma_long=0.5, gamma_short=2.0)
    s_short = asymmetric(-2.0, gamma_long=0.5, gamma_short=2.0)
    assert s_long.price_offset == -1.0  # -0.5 * 2
    assert s_short.price_offset == 4.0   # -2.0 * -2


def test_asymmetric_zero_inv():
    assert asymmetric(0.0, 1.0, 1.0).price_offset == 0.0


# --------------------------------------------------------------------- #
# soft_cap
# --------------------------------------------------------------------- #

def test_soft_cap_below_cap_linear():
    s = soft_cap(inv=1.0, gamma=0.5, cap=2.0)
    # |inv| < cap; only linear term: -0.5 * 1 = -0.5
    assert s.price_offset == pytest.approx(-0.5)


def test_soft_cap_above_cap_extra_penalty():
    s_below = soft_cap(inv=1.5, gamma=1.0, cap=2.0)
    s_above = soft_cap(inv=3.0, gamma=1.0, cap=2.0)
    # Above cap: extra penalty kicks in -> larger magnitude
    assert abs(s_above.price_offset) > abs(s_below.price_offset)


def test_soft_cap_rejects_bad_cap():
    with pytest.raises(ValueError):
        soft_cap(inv=1.0, gamma=1.0, cap=0.0)


# --------------------------------------------------------------------- #
# hard_cap
# --------------------------------------------------------------------- #

def test_hard_cap_below_cap_full_size():
    s = hard_cap(inv=1.0, cap=2.0)
    assert s.size_scale_bid == 1.0
    assert s.size_scale_ask == 1.0
    assert s.price_offset == 0.0


def test_hard_cap_long_at_cap_drops_bid():
    s = hard_cap(inv=2.0, cap=2.0)
    assert s.size_scale_bid == 0.0  # don't add more long
    assert s.size_scale_ask == 1.0


def test_hard_cap_short_at_cap_drops_ask():
    s = hard_cap(inv=-2.0, cap=2.0)
    assert s.size_scale_bid == 1.0
    assert s.size_scale_ask == 0.0  # don't add more short


def test_hard_cap_rejects_bad_cap():
    with pytest.raises(ValueError):
        hard_cap(inv=1.0, cap=0.0)


# --------------------------------------------------------------------- #
# Purity: all primitives are deterministic functions of inputs.
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("inv", [-3.0, -1.5, -0.001, 0.0, 0.001, 1.5, 3.0])
def test_all_primitives_deterministic(inv):
    a1 = linear(inv, 0.5)
    a2 = linear(inv, 0.5)
    assert a1 == a2
    b1 = quadratic(inv, 0.5)
    b2 = quadratic(inv, 0.5)
    assert b1 == b2
    c1 = exponential(inv, 0.5, 1.0)
    c2 = exponential(inv, 0.5, 1.0)
    assert c1 == c2
    d1 = asymmetric(inv, 0.5, 0.7)
    d2 = asymmetric(inv, 0.5, 0.7)
    assert d1 == d2
    e1 = soft_cap(inv, 0.5, 2.0)
    e2 = soft_cap(inv, 0.5, 2.0)
    assert e1 == e2
    f1 = hard_cap(inv, 2.0)
    f2 = hard_cap(inv, 2.0)
    assert f1 == f2


# --------------------------------------------------------------------- #
# 5-event reconciliation per spec (5 inv values × all 6 primitives;
# enumerate expected Skew outputs explicitly).
# --------------------------------------------------------------------- #

def test_g3_five_inv_values_reconciled():
    cases = [
        (-2.0, linear, (0.5,), Skew(price_offset=+1.0, size_scale_bid=1.0, size_scale_ask=1.0)),
        (-1.0, quadratic, (1.0,), Skew(price_offset=+1.0, size_scale_bid=1.0, size_scale_ask=1.0)),
        (0.0, exponential, (1.0, 1.0), Skew(price_offset=0.0, size_scale_bid=1.0, size_scale_ask=1.0)),
        (1.0, asymmetric, (0.5, 2.0), Skew(price_offset=-0.5, size_scale_bid=1.0, size_scale_ask=1.0)),
        (3.0, hard_cap, (2.0,), Skew(price_offset=0.0, size_scale_bid=0.0, size_scale_ask=1.0)),
    ]
    for inv, fn, params, expected in cases:
        actual = fn(inv, *params)
        assert actual == expected, f"reconciliation failed for {fn.__name__}(inv={inv}, *{params}): got {actual}"
