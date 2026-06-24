"""T4 Market-Making structural combo grid.

Encodes the 7 structural [B] axes from the T4 spec and provides
deterministic sampling for corpus generation.

Axes (in canonical order):
    1. quoting_model      (8 options) — AS / CJ / GLFT / Ho-Stoll / symmetric /
                                         ladder / microprice_skew / fair_anchored
    2. inventory_penalty  (6 options) — linear / quadratic / exponential /
                                         asymmetric / soft_cap / hard_cap
    3. adverse_filter     (7 options) — none / ofi / toxicity / vol_surge /
                                         microprice_dev / queue_imb / hybrid
    4. hedge_mode         (4 options) — none / perp / basket / options_vega_stub
    5. reference_price    (6 options) — mid / microprice / weighted_mid /
                                         ewma_fair / model_pred / vwap
    6. quote_shape        (5 options) — single / paired / ladder / geometric /
                                         dynamic_depth
    7. refresh_trigger    (5 options) — time / mid_move / inv_change /
                                         book_event / hybrid

Cartesian product: 8 * 6 * 7 * 4 * 6 * 5 * 5 = 201,600 combos.

Grid-size note
--------------
The T4 plan summary writes ``8*6*7*6*6*5*5 = 302,400`` for the structural
cardinality. Inspection of the spec text reveals the HEDGE_MODES axis is
``{none, perp, basket, options-vega(TODO)}`` (4 distinct primitives, not
6); the plan's "6" lumped together the structural variants of
``basket`` + ``perp+basket`` that we treat as a single ``basket`` mode
with the basket size resolved by IS-tune. The structural axis remains
4 distinct primitives. We surface the actual cardinality (201,600)
via ``GRID_SIZE`` and document the deviation here (not in the runner).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Axis enumerations
# ---------------------------------------------------------------------------

QUOTING_MODELS: tuple[str, ...] = (
    "avellaneda_stoikov",
    "cartea_jaimungal",
    "glft",
    "ho_stoll",
    "symmetric",
    "ladder",
    "microprice_skew",
    "fair_anchored",
)

INVENTORY_PENALTIES: tuple[str, ...] = (
    "linear",
    "quadratic",
    "exponential",
    "asymmetric",
    "soft_cap",
    "hard_cap",
)

ADVERSE_FILTERS: tuple[str, ...] = (
    "none",
    "ofi",
    "toxicity",
    "vol_surge",
    "microprice_dev",
    "queue_imb",
    "hybrid",
)

HEDGE_MODES: tuple[str, ...] = (
    "none",
    "perp",
    "basket",
    "options_vega_stub",  # TODO per the spec; routes to a no-op stub.
)

REFERENCE_PRICES: tuple[str, ...] = (
    "mid",
    "microprice",
    "weighted_mid",
    "ewma_fair",
    "model_pred",
    "vwap",
)

QUOTE_SHAPES: tuple[str, ...] = (
    "single",
    "paired",
    "ladder",
    "geometric",
    "dynamic_depth",
)

REFRESH_TRIGGERS: tuple[str, ...] = (
    "time",
    "mid_move",
    "inv_change",
    "book_event",
    "hybrid",
)


# Short codes for canonical strategy_name strings. Collision-free per-axis.
_SHORT_CODES: dict[str, str] = {
    # quoting_model
    "avellaneda_stoikov": "AS",
    "cartea_jaimungal": "CJ",
    "glft": "GLFT",
    "ho_stoll": "HS",
    "symmetric": "sym",
    "ladder": "lad",
    "microprice_skew": "mps",
    "fair_anchored": "fair",
    # inventory_penalty
    "linear": "lin",
    "quadratic": "quad",
    "exponential": "exp",
    "asymmetric": "asym",
    "soft_cap": "scap",
    "hard_cap": "hcap",
    # adverse_filter
    "none": "nofilt",  # disambiguated for adverse axis
    "ofi": "ofi",
    "toxicity": "tox",
    "vol_surge": "vsurge",
    "microprice_dev": "mpdev",
    "queue_imb": "qimb",
    "hybrid": "hybA",
    # hedge_mode (none re-mapped per axis position below)
    "perp": "perp",
    "basket": "bask",
    "options_vega_stub": "opvg",
    # reference_price
    "mid": "mid",
    "microprice": "micro",
    "weighted_mid": "wmid",
    "ewma_fair": "ewma",
    "model_pred": "mpred",
    "vwap": "vwap",
    # quote_shape
    "single": "sgl",
    "paired": "par",
    # "ladder" already mapped above as "lad" (quoting); same code reused.
    "geometric": "geo",
    "dynamic_depth": "dyn",
    # refresh_trigger
    "time": "time",
    "mid_move": "mvmid",
    "inv_change": "invch",
    "book_event": "bkev",
    # "hybrid" already mapped above as "hybA"
}


# Literal aliases for type-checker friendliness.
QuotingModel = Literal[
    "avellaneda_stoikov", "cartea_jaimungal", "glft", "ho_stoll",
    "symmetric", "ladder", "microprice_skew", "fair_anchored",
]
InventoryPenalty = Literal[
    "linear", "quadratic", "exponential",
    "asymmetric", "soft_cap", "hard_cap",
]
AdverseFilter = Literal[
    "none", "ofi", "toxicity", "vol_surge",
    "microprice_dev", "queue_imb", "hybrid",
]
HedgeMode = Literal["none", "perp", "basket", "options_vega_stub"]
ReferencePrice = Literal[
    "mid", "microprice", "weighted_mid", "ewma_fair", "model_pred", "vwap",
]
QuoteShape = Literal["single", "paired", "ladder", "geometric", "dynamic_depth"]
RefreshTrigger = Literal["time", "mid_move", "inv_change", "book_event", "hybrid"]


# ---------------------------------------------------------------------------
# Combo dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Combo:
    """One structural T4 combo.

    Frozen + slotted: hashable (so combos can be deduped via set()) and
    memory-cheap when materializing larger samples.
    """
    quoting_model: QuotingModel
    inventory_penalty: InventoryPenalty
    adverse_filter: AdverseFilter
    hedge_mode: HedgeMode
    reference_price: ReferencePrice
    quote_shape: QuoteShape
    refresh_trigger: RefreshTrigger


# Ordered tuple of (field_name, axis_values) used by the iterator + samplers.
_AXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("quoting_model", QUOTING_MODELS),
    ("inventory_penalty", INVENTORY_PENALTIES),
    ("adverse_filter", ADVERSE_FILTERS),
    ("hedge_mode", HEDGE_MODES),
    ("reference_price", REFERENCE_PRICES),
    ("quote_shape", QUOTE_SHAPES),
    ("refresh_trigger", REFRESH_TRIGGERS),
)

# Total grid cardinality: 8 * 6 * 7 * 4 * 6 * 5 * 5 = 201,600.
GRID_SIZE: int = 1
for _name, _vals in _AXES:
    GRID_SIZE *= len(_vals)
assert GRID_SIZE == 201_600, f"Grid size drift: expected 201,600, got {GRID_SIZE}"


# ---------------------------------------------------------------------------
# Enumeration / sampling
# ---------------------------------------------------------------------------

def iter_all_combos() -> Iterator[Combo]:
    """Yield every combo in the Cartesian product without materializing.

    Ordering is the natural ``itertools.product`` order over ``_AXES``: the
    last axis (``refresh_trigger``) varies fastest. This ordering is the
    canonical index used by ``sample_combos``.
    """
    axis_values = [vals for _name, vals in _AXES]
    for tup in itertools.product(*axis_values):
        yield Combo(*tup)


def _combo_at(idx: int) -> Combo:
    """Materialize the combo at a given linear index without enumerating the
    whole grid. Uses mixed-radix decomposition over the axis cardinalities."""
    if not 0 <= idx < GRID_SIZE:
        raise IndexError(f"combo index {idx} out of range [0, {GRID_SIZE})")
    parts: list[str] = []
    remaining = idx
    sizes = [len(vals) for _n, vals in _AXES]
    strides: list[int] = []
    acc = 1
    for s in reversed(sizes):
        strides.append(acc)
        acc *= s
    strides.reverse()
    for (name, vals), stride in zip(_AXES, strides):
        q, remaining = divmod(remaining, stride)
        parts.append(vals[q])
    return Combo(*parts)


def sample_combos(n: int, seed: int = 2026) -> list[Combo]:
    """Deterministic uniform-without-replacement sample of ``n`` combos.

    Uses ``numpy.random.default_rng(seed)`` for reproducibility; seed 2026
    matches the project-wide convention.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n > GRID_SIZE:
        raise ValueError(f"requested n={n} exceeds grid size {GRID_SIZE}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(GRID_SIZE, size=n, replace=False)
    return [_combo_at(int(i)) for i in indices]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def combo_to_dict(c: Combo) -> dict[str, str]:
    """Plain-string dict representation. Suitable for JSON / parquet rows."""
    return asdict(c)


def combo_from_dict(d: dict[str, str]) -> Combo:
    """Inverse of :func:`combo_to_dict`. Extra keys are ignored; missing keys
    raise ``KeyError``."""
    field_names = [f.name for f in fields(Combo)]
    return Combo(**{name: d[name] for name in field_names})


# Per-axis short-code resolver (handles tokens shared across axes like "ladder"
# and "hybrid" / "none").
def _axis_short_code(axis_name: str, token: str) -> str:
    """Return the canonical short code for ``token`` at axis ``axis_name``.

    Some tokens (``none``, ``ladder``, ``hybrid``) appear at multiple axes;
    we disambiguate by axis position so a combo's string name has 7
    underscore-separated parts uniquely keyed by axis.
    """
    if axis_name == "adverse_filter":
        if token == "none":
            return "nofilt"
        if token == "hybrid":
            return "hybA"
    if axis_name == "hedge_mode" and token == "none":
        return "nohed"
    if axis_name == "refresh_trigger":
        if token == "hybrid":
            return "hybR"
    if axis_name == "quote_shape" and token == "ladder":
        return "ladS"
    return _SHORT_CODES[token]


def combo_to_strategy_name(c: Combo) -> str:
    """Canonical short-code strategy name.

    Format: 7 underscore-separated short codes, one per axis, in the fixed
    ``_AXES`` order (quoting / inv_pen / adverse / hedge / refprice /
    shape / trigger). Example::

        "AS_lin_nofilt_perp_mid_sgl_time"
    """
    return "_".join(
        _axis_short_code(name, getattr(c, name)) for name, _vals in _AXES
    )


# ---------------------------------------------------------------------------
# Pruning hooks (post-prune to ~720K of 2.4M in the plan; here pre-prune is
# permissive — the smoke surface ships every combo; gates kick in at corpus
# generation time once cross-axis incompatibilities are discovered).
# ---------------------------------------------------------------------------

def is_combo_admissible(combo: Combo) -> bool:
    """Reject combos that are structurally incoherent.

    Permissive stub: every (quoting × inv × adverse × hedge × ref × shape ×
    trigger) tuple is admissible at the structural level. Composability is
    enforced by the composer (typing) rather than by axis exclusions.

    Note: ``options_vega_stub`` is admissible structurally but the composer
    routes it to a no-op fallback (per the spec: T3 options book is
    deferred; the axis is preserved so the cardinality matches the plan).
    """
    del combo
    return True


def pre_prune_combos(combos: list[Combo]) -> list[Combo]:
    """Apply combo-level static filters before any runner is invoked.

    Currently a permissive pass-through; the no-op composer fallback for
    options_vega_stub is the only "filter" applied at runtime.
    """
    return [c for c in combos if is_combo_admissible(c)]


# ---------------------------------------------------------------------------
# DataFrame inspection helper
# ---------------------------------------------------------------------------

def to_dataframe(combos: list[Combo]) -> pd.DataFrame:
    """Tidy DataFrame with one row per combo and a derived ``strategy_name``
    column. Column order matches ``_AXES`` order."""
    rows = [combo_to_dict(c) | {"strategy_name": combo_to_strategy_name(c)}
            for c in combos]
    column_order = [name for name, _vals in _AXES] + ["strategy_name"]
    return pd.DataFrame(rows, columns=column_order)


__all__ = [
    "Combo", "GRID_SIZE",
    "QUOTING_MODELS", "INVENTORY_PENALTIES", "ADVERSE_FILTERS",
    "HEDGE_MODES", "REFERENCE_PRICES", "QUOTE_SHAPES", "REFRESH_TRIGGERS",
    "iter_all_combos", "sample_combos",
    "combo_to_dict", "combo_from_dict", "combo_to_strategy_name",
    "is_combo_admissible", "pre_prune_combos",
    "to_dataframe",
]
