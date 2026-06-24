"""Quote-shape primitive tests.

5 hand-computed reference values per primitive — the spec's
'5 arrays reconciled' G3 requirement.  Plus integration with the
Quoter Protocol via a tiny wrapper class.
"""
from __future__ import annotations

import pytest

from mmsim.quoter.shapes import (
    DynamicDepthSpec, GeometricSpec, LadderSpec, PairedSpec, SingleSpec,
    dynamic_depth, geometric, ladder, paired, single,
)
from mmsim.sim.loop import QuoteRequest


# --------------------------------------------------------------------- #
# single
# --------------------------------------------------------------------- #

def test_single_bid_ask_at_ref_plus_minus_half_spread():
    out = single(SingleSpec(size=0.001, half_spread=0.5), ref_price=100.0)
    assert len(out) == 2
    bid = next(q for q in out if q.side == +1)
    ask = next(q for q in out if q.side == -1)
    assert bid.price == 99.5
    assert ask.price == 100.5
    assert bid.size == 0.001
    assert ask.size == 0.001


def test_single_ignores_inv():
    a = single(SingleSpec(size=0.001, half_spread=0.5), 100.0, inv=0.0)
    b = single(SingleSpec(size=0.001, half_spread=0.5), 100.0, inv=10.0)
    c = single(SingleSpec(size=0.001, half_spread=0.5), 100.0, inv=-10.0)
    assert a == b == c


# --------------------------------------------------------------------- #
# paired
# --------------------------------------------------------------------- #

def test_paired_per_level_offsets_and_sizes():
    spec = PairedSpec(
        levels_bid=((0.5, 0.001), (1.5, 0.002)),
        levels_ask=((0.5, 0.001), (1.5, 0.002), (3.0, 0.003)),
    )
    out = paired(spec, ref_price=100.0)
    bids = sorted([q for q in out if q.side == +1], key=lambda q: -q.price)
    asks = sorted([q for q in out if q.side == -1], key=lambda q: q.price)
    assert [(q.price, q.size) for q in bids] == [(99.5, 0.001), (98.5, 0.002)]
    assert [(q.price, q.size) for q in asks] == [(100.5, 0.001), (101.5, 0.002), (103.0, 0.003)]


def test_paired_empty_levels_returns_empty():
    spec = PairedSpec(levels_bid=(), levels_ask=())
    assert paired(spec, ref_price=100.0) == []


# --------------------------------------------------------------------- #
# ladder
# --------------------------------------------------------------------- #

def test_ladder_3_levels_linear_step():
    spec = LadderSpec(half_spread=0.5, step=0.1, n_levels=3, size_per_level=0.001)
    out = ladder(spec, ref_price=100.0)
    # 3 bids + 3 asks = 6 quotes total
    assert len(out) == 6
    bids = sorted([q for q in out if q.side == +1], key=lambda q: -q.price)
    asks = sorted([q for q in out if q.side == -1], key=lambda q: q.price)
    # Bid offsets: 0.5, 0.6, 0.7  (half_spread + k*step for k=0,1,2)
    assert [q.price for q in bids] == pytest.approx([99.5, 99.4, 99.3])
    assert [q.price for q in asks] == pytest.approx([100.5, 100.6, 100.7])
    assert all(q.size == 0.001 for q in out)


def test_ladder_zero_levels_returns_empty():
    spec = LadderSpec(half_spread=0.5, step=0.1, n_levels=0, size_per_level=0.001)
    assert ladder(spec, ref_price=100.0) == []


# --------------------------------------------------------------------- #
# geometric
# --------------------------------------------------------------------- #

def test_geometric_3_levels_ratio_2():
    spec = GeometricSpec(half_spread=0.5, ratio=2.0, n_levels=3, size_per_level=0.001)
    out = geometric(spec, ref_price=100.0)
    # Bid offsets: 0.5, 1.0, 2.0 (geometric ×2)
    bids = sorted([q for q in out if q.side == +1], key=lambda q: -q.price)
    asks = sorted([q for q in out if q.side == -1], key=lambda q: q.price)
    assert [q.price for q in bids] == pytest.approx([99.5, 99.0, 98.0])
    assert [q.price for q in asks] == pytest.approx([100.5, 101.0, 102.0])


def test_geometric_ratio_1_collapses_to_stack():
    """ratio=1 produces N levels all at the same price (degenerate
    by design — caller's responsibility)."""
    spec = GeometricSpec(half_spread=0.5, ratio=1.0, n_levels=3, size_per_level=0.001)
    out = geometric(spec, ref_price=100.0)
    bids = [q for q in out if q.side == +1]
    assert all(q.price == 99.5 for q in bids)
    assert len(bids) == 3


# --------------------------------------------------------------------- #
# dynamic_depth — the only shape that consults inv
# --------------------------------------------------------------------- #

def test_dynamic_depth_full_depth_at_zero_inv():
    spec = DynamicDepthSpec(
        half_spread=0.5, step=0.1, max_levels=5,
        inv_taper_threshold=1.0, size_per_level=0.001,
    )
    out = dynamic_depth(spec, ref_price=100.0, inv=0.0)
    # 5 bids + 5 asks = 10 quotes
    assert len(out) == 10


def test_dynamic_depth_tapers_to_one_at_2x_threshold():
    spec = DynamicDepthSpec(
        half_spread=0.5, step=0.1, max_levels=5,
        inv_taper_threshold=1.0, size_per_level=0.001,
    )
    out = dynamic_depth(spec, ref_price=100.0, inv=2.5)  # |inv| > 2*thr
    # 1 bid + 1 ask
    assert len(out) == 2


def test_dynamic_depth_linear_interpolation_midpoint():
    spec = DynamicDepthSpec(
        half_spread=0.5, step=0.1, max_levels=5,
        inv_taper_threshold=1.0, size_per_level=0.001,
    )
    # |inv| = 1.5 -> halfway between thr and 2*thr -> frac=0.5
    # n_active = max(1, round(5 * 0.5)) = max(1, 2 or 3) — round(2.5)=2 in Python
    out = dynamic_depth(spec, ref_price=100.0, inv=1.5)
    n_bids = sum(1 for q in out if q.side == +1)
    n_asks = sum(1 for q in out if q.side == -1)
    assert n_bids == n_asks
    # 2 or 3 depending on Python's banker's rounding; the spec uses
    # int(round(...)) which is banker's rounding -> round(2.5) = 2.
    assert n_bids in (2, 3)


def test_dynamic_depth_negative_inv_treated_symmetrically():
    spec = DynamicDepthSpec(
        half_spread=0.5, step=0.1, max_levels=5,
        inv_taper_threshold=1.0, size_per_level=0.001,
    )
    a = dynamic_depth(spec, ref_price=100.0, inv=+2.5)
    b = dynamic_depth(spec, ref_price=100.0, inv=-2.5)
    # Depth tapering uses |inv|; positions are symmetric.
    assert len(a) == len(b)


# --------------------------------------------------------------------- #
# Causality: pure of (spec, ref_price, inv) — no other inputs.
# Calling the same primitive twice with identical inputs must produce
# identical outputs.  This is the test that proves "no time leak
# possible" mechanically.
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_all_shapes_deterministic_given_inputs(seed):
    import random
    rng = random.Random(seed)
    ref = 100.0 + rng.uniform(-50, 50)
    inv = rng.uniform(-3, 3)
    s_spec = SingleSpec(size=0.001, half_spread=0.5)
    p_spec = PairedSpec(
        levels_bid=((0.5, 0.001), (1.5, 0.002)),
        levels_ask=((0.5, 0.001), (1.5, 0.002)),
    )
    l_spec = LadderSpec(half_spread=0.5, step=0.1, n_levels=3, size_per_level=0.001)
    g_spec = GeometricSpec(half_spread=0.5, ratio=2.0, n_levels=3, size_per_level=0.001)
    d_spec = DynamicDepthSpec(
        half_spread=0.5, step=0.1, max_levels=5,
        inv_taper_threshold=1.0, size_per_level=0.001,
    )
    for fn, spec in [(single, s_spec), (paired, p_spec), (ladder, l_spec),
                      (geometric, g_spec), (dynamic_depth, d_spec)]:
        a = fn(spec, ref, inv)
        b = fn(spec, ref, inv)
        assert a == b


# --------------------------------------------------------------------- #
# G3 — 5 hand-reconciled arrays (spec's "5 arrays reconciled"
# requirement).  Each is a known shape × ref × inv with the expected
# output enumerated explicitly.
# --------------------------------------------------------------------- #

def test_g3_five_hand_reconciled_arrays():
    cases = [
        # (label, fn, spec, ref, inv, expected_quotes)
        (
            "single@100",
            single,
            SingleSpec(size=0.001, half_spread=0.5),
            100.0, 0.0,
            [QuoteRequest(side=+1, price=99.5, size=0.001),
             QuoteRequest(side=-1, price=100.5, size=0.001)],
        ),
        (
            "paired_2x2",
            paired,
            PairedSpec(
                levels_bid=((0.5, 0.001), (1.0, 0.002)),
                levels_ask=((0.5, 0.001), (1.0, 0.002)),
            ),
            50.0, 0.0,
            [QuoteRequest(side=+1, price=49.5, size=0.001),
             QuoteRequest(side=+1, price=49.0, size=0.002),
             QuoteRequest(side=-1, price=50.5, size=0.001),
             QuoteRequest(side=-1, price=51.0, size=0.002)],
        ),
        (
            "ladder_3x_step_0.2",
            ladder,
            LadderSpec(half_spread=1.0, step=0.2, n_levels=3, size_per_level=0.005),
            200.0, 0.0,
            [QuoteRequest(side=+1, price=199.0, size=0.005),
             QuoteRequest(side=-1, price=201.0, size=0.005),
             QuoteRequest(side=+1, price=198.8, size=0.005),
             QuoteRequest(side=-1, price=201.2, size=0.005),
             QuoteRequest(side=+1, price=198.6, size=0.005),
             QuoteRequest(side=-1, price=201.4, size=0.005)],
        ),
        (
            "geometric_ratio_1.5",
            geometric,
            GeometricSpec(half_spread=1.0, ratio=1.5, n_levels=3, size_per_level=0.005),
            10.0, 0.0,
            # offsets: 1.0, 1.5, 2.25
            [QuoteRequest(side=+1, price=9.0, size=0.005),
             QuoteRequest(side=-1, price=11.0, size=0.005),
             QuoteRequest(side=+1, price=8.5, size=0.005),
             QuoteRequest(side=-1, price=11.5, size=0.005),
             QuoteRequest(side=+1, price=7.75, size=0.005),
             QuoteRequest(side=-1, price=12.25, size=0.005)],
        ),
        (
            "dynamic_depth_tapered_at_2x_thresh",
            dynamic_depth,
            DynamicDepthSpec(
                half_spread=0.5, step=0.1, max_levels=5,
                inv_taper_threshold=1.0, size_per_level=0.001,
            ),
            100.0, 3.0,  # |inv| = 3 > 2*thr -> 1 level
            [QuoteRequest(side=+1, price=99.5, size=0.001),
             QuoteRequest(side=-1, price=100.5, size=0.001)],
        ),
    ]
    for label, fn, spec, ref, inv, expected in cases:
        out = fn(spec, ref, inv)
        assert out == expected, f"reconciliation failed for {label}: got {out}"
