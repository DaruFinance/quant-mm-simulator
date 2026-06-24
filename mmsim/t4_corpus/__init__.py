"""T4 Market-Making corpus generator layer.

The T4 corpus layer drives the existing mmsim engine primitives at scale by
enumerating the cross-product of 7 structural [B] axes:

    QUOTING_MODELS (8) x INVENTORY_PENALTIES (6) x ADVERSE_FILTERS (7) x
    HEDGE_MODES (4) x REFERENCE_PRICES (6) x QUOTE_SHAPES (5) x
    REFRESH_TRIGGERS (5) = 8 * 6 * 7 * 4 * 6 * 5 * 5 = 201,600 combos

(The plan summary lists `8*6*7*6*6*5*5 = 302,400`; in our enumeration the
HEDGE_MODES axis is `{none, perp, basket, options_vega_stub}` = 4 — the
plan's 6 included two duplicate variants which we collapse. The true
cross-product is 100,800; `GRID_SIZE` exposes the actual cardinality.)

IS-tune axes layered on top of each structural combo:
    gamma (risk aversion) - k (intensity decay) - T (horizon ns) -
    inventory_cap - refresh_interval_ns - spread_floor -
    filter threshold

This module is **additive** to the existing mmsim engine: every axis
choice routes to an existing primitive in `mmsim.quoter`, `mmsim.models`,
or `mmsim.hedge`. No engine code is duplicated.

Real LOB fixtures only: DS-LOB-1H or DS-LOB-30s. Fees + slippage flow
through the existing mmsim cost layer (per-fill maker/taker fee applied
in the runner).
"""
from __future__ import annotations

from .combos import (
    Combo,
    GRID_SIZE,
    QUOTING_MODELS, INVENTORY_PENALTIES, ADVERSE_FILTERS,
    HEDGE_MODES, REFERENCE_PRICES, QUOTE_SHAPES, REFRESH_TRIGGERS,
    iter_all_combos, sample_combos,
    combo_to_dict, combo_from_dict, combo_to_strategy_name,
)
from .search_space import (
    Axis, SearchSpace, SEARCH_SPACE_T4, search_space_t4,
)
from .t4_composer import build_mm_strategy, T4StrategyConfig
from .t4_runner import run_t4_combo, T4RunResult, MAKER_FEE_PCT, TAKER_FEE_PCT, SLIP_PCT
from .t4_corpus_gen import generate_t4_corpus, T4CorpusResult

__all__ = [
    # combos
    "Combo", "GRID_SIZE",
    "QUOTING_MODELS", "INVENTORY_PENALTIES", "ADVERSE_FILTERS",
    "HEDGE_MODES", "REFERENCE_PRICES", "QUOTE_SHAPES", "REFRESH_TRIGGERS",
    "iter_all_combos", "sample_combos",
    "combo_to_dict", "combo_from_dict", "combo_to_strategy_name",
    # search space
    "Axis", "SearchSpace", "SEARCH_SPACE_T4", "search_space_t4",
    # composer
    "build_mm_strategy", "T4StrategyConfig",
    # runner
    "run_t4_combo", "T4RunResult",
    "MAKER_FEE_PCT", "TAKER_FEE_PCT", "SLIP_PCT",
    # corpus gen
    "generate_t4_corpus", "T4CorpusResult",
]
