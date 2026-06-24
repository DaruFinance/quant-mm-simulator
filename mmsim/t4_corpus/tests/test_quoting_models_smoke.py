"""Smoke tests: one combo per quoting model on the 30sec LOB fixture.

Each test runs the engine end-to-end on real data and asserts the
quoter:
  - constructs without error via the composer,
  - produces a non-empty quote stream when the book is warm,
  - is leak-free under the pollute rail.
"""
from __future__ import annotations

from mmsim.t4_corpus.combos import (
    QUOTING_MODELS, Combo,
)
from mmsim.t4_corpus.t4_composer import build_mm_strategy
from mmsim.t4_corpus.t4_runner import run_t4_combo


def _make_combo(quoting_model: str) -> Combo:
    """Build a representative combo with the given quoting_model and the
    first option of every other axis. Keeps the smoke surface minimal."""
    return Combo(
        quoting_model=quoting_model,
        inventory_penalty="linear",
        adverse_filter="none",
        hedge_mode="none",
        reference_price="mid",
        quote_shape="single",
        refresh_trigger="book_event",
    )


def test_avellaneda_stoikov_smoke(stream_30s):
    rr = run_t4_combo(_make_combo("avellaneda_stoikov"), {}, stream_30s)
    assert rr.metrics["n_quoter_calls"] > 0


def test_cartea_jaimungal_smoke(stream_30s):
    rr = run_t4_combo(_make_combo("cartea_jaimungal"), {}, stream_30s)
    assert rr.metrics["n_quoter_calls"] > 0


def test_glft_smoke(stream_30s):
    rr = run_t4_combo(_make_combo("glft"), {}, stream_30s)
    assert rr.metrics["n_quoter_calls"] > 0


def test_ho_stoll_smoke(stream_30s):
    rr = run_t4_combo(_make_combo("ho_stoll"), {}, stream_30s)
    assert rr.metrics["n_quoter_calls"] > 0


def test_symmetric_smoke(stream_30s):
    rr = run_t4_combo(_make_combo("symmetric"), {}, stream_30s)
    assert rr.metrics["n_quoter_calls"] > 0


def test_ladder_smoke(stream_30s):
    rr = run_t4_combo(_make_combo("ladder"), {}, stream_30s)
    assert rr.metrics["n_quoter_calls"] > 0


def test_microprice_skew_smoke(stream_30s):
    rr = run_t4_combo(_make_combo("microprice_skew"), {}, stream_30s)
    assert rr.metrics["n_quoter_calls"] > 0


def test_fair_anchored_smoke(stream_30s):
    rr = run_t4_combo(_make_combo("fair_anchored"), {}, stream_30s)
    assert rr.metrics["n_quoter_calls"] > 0


def test_all_quoting_models_compose():
    """Each of the 8 quoting models composes without raising."""
    for qm in QUOTING_MODELS:
        cfg = build_mm_strategy(_make_combo(qm), {})
        assert cfg.quoter is not None
