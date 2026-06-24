"""End-to-end smoke for the T4 corpus layer on real LOB data.

These tests drive 1 representative combo per [B] axis option through
the full engine pipeline on the DS-LOB-30sec fixture and verify the
cost-identity (every fill has fee + slip > 0) plus the loop
contract (n_quoter_calls == n_snapshots).
"""
from __future__ import annotations

from mmsim.t4_corpus.combos import (
    ADVERSE_FILTERS, HEDGE_MODES, INVENTORY_PENALTIES,
    QUOTE_SHAPES, QUOTING_MODELS, REFERENCE_PRICES, REFRESH_TRIGGERS,
    Combo,
)
from mmsim.t4_corpus.t4_runner import (
    MAKER_FEE_PCT, SLIP_PCT, TAKER_FEE_PCT, run_t4_combo,
)


def _combo_with(axis: str, value: str) -> Combo:
    """One combo with the given axis set to ``value`` and every other
    axis at its first option (defensive baseline)."""
    base = dict(
        quoting_model="symmetric",
        inventory_penalty="linear",
        adverse_filter="none",
        hedge_mode="none",
        reference_price="mid",
        quote_shape="single",
        refresh_trigger="book_event",
    )
    base[axis] = value
    return Combo(**base)


# --------------------------------------------------------------------- #
# Per-axis smoke: 1 combo per option, full engine pass.
# --------------------------------------------------------------------- #

def _assert_run_ok(rr):
    """Sanity invariants for a non-trivial T4 run."""
    # The 30sec fixture has many snapshots; quoter must be called every one.
    assert rr.metrics["n_snapshots"] > 0
    assert rr.metrics["n_quoter_calls"] == rr.metrics["n_snapshots"]
    # Cost identity: every realized fill contributes a non-negative
    # fee + slip. (Some axis combos may produce zero fills on the 30s
    # fixture; that's expected. When fills exist they must be costed.)
    if rr.n_fills > 0:
        assert rr.metrics["total_fees"] > 0
        assert rr.metrics["total_slippage"] > 0
        assert rr.metrics["total_cost"] > 0


def test_quoting_models_smoke(stream_30s):
    for qm in QUOTING_MODELS:
        rr = run_t4_combo(_combo_with("quoting_model", qm), {}, stream_30s)
        _assert_run_ok(rr)


def test_inventory_penalties_smoke(stream_30s):
    for ip in INVENTORY_PENALTIES:
        rr = run_t4_combo(_combo_with("inventory_penalty", ip), {}, stream_30s)
        _assert_run_ok(rr)


def test_adverse_filters_smoke(stream_30s):
    for filt in ADVERSE_FILTERS:
        rr = run_t4_combo(_combo_with("adverse_filter", filt), {}, stream_30s)
        _assert_run_ok(rr)


def test_hedge_modes_smoke(stream_30s):
    for hm in HEDGE_MODES:
        rr = run_t4_combo(_combo_with("hedge_mode", hm), {}, stream_30s)
        _assert_run_ok(rr)


def test_reference_prices_smoke(stream_30s):
    for rp in REFERENCE_PRICES:
        rr = run_t4_combo(_combo_with("reference_price", rp), {}, stream_30s)
        _assert_run_ok(rr)


def test_quote_shapes_smoke(stream_30s):
    for sh in QUOTE_SHAPES:
        rr = run_t4_combo(_combo_with("quote_shape", sh), {}, stream_30s)
        _assert_run_ok(rr)


def test_refresh_triggers_smoke(stream_30s):
    for tr in REFRESH_TRIGGERS:
        rr = run_t4_combo(_combo_with("refresh_trigger", tr), {}, stream_30s)
        _assert_run_ok(rr)


# --------------------------------------------------------------------- #
# Composability of a fully-non-default combo.
# --------------------------------------------------------------------- #

def test_complex_composition_runs(stream_30s):
    """Pick a non-trivial combo touching multiple non-default axes; verify
    the runner produces a coherent result."""
    combo = Combo(
        quoting_model="avellaneda_stoikov",
        inventory_penalty="soft_cap",
        adverse_filter="ofi",
        hedge_mode="perp",
        reference_price="microprice",
        quote_shape="ladder",
        refresh_trigger="time",
    )
    rr = run_t4_combo(combo, {"gamma": 0.3, "k": 1.0,
                                "inventory_cap": 0.1,
                                "refresh_interval_ns": 1_000_000_000},
                       stream_30s)
    _assert_run_ok(rr)


# --------------------------------------------------------------------- #
# Cost-stack constants match the project-wide convention.
# --------------------------------------------------------------------- #

def test_cost_constants_match_project_defaults():
    """The crypto cost defaults documented in feedback_no_costless_backtests.md.

    Maker 0.02%, Taker 0.05%, Slippage 0.02%.
    """
    assert MAKER_FEE_PCT == 2e-4
    assert TAKER_FEE_PCT == 5e-4
    assert SLIP_PCT == 2e-4


# --------------------------------------------------------------------- #
# 50-T pollute leak rail: re-running the same combo on the same stream
# produces identical results regardless of intermediate calls with other
# combos.  (T4 is leak-free by composition: every primitive is.)
# --------------------------------------------------------------------- #

def test_reproducibility_same_combo(stream_30s):
    combo = Combo(
        quoting_model="symmetric",
        inventory_penalty="linear",
        adverse_filter="none",
        hedge_mode="none",
        reference_price="mid",
        quote_shape="single",
        refresh_trigger="book_event",
    )
    rr_a = run_t4_combo(combo, {}, stream_30s)
    # Intervening call with a polluting combo.
    _ = run_t4_combo(
        Combo(quoting_model="glft", inventory_penalty="quadratic",
               adverse_filter="hybrid", hedge_mode="perp",
               reference_price="microprice", quote_shape="ladder",
               refresh_trigger="hybrid"), {}, stream_30s)
    rr_b = run_t4_combo(combo, {}, stream_30s)
    assert rr_a.n_fills == rr_b.n_fills
    assert rr_a.metrics["total_cost"] == rr_b.metrics["total_cost"]
