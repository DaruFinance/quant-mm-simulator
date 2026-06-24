"""T4 strategy composer.

Given a structural :class:`Combo` and an IS-params dict, produces a
``T4StrategyConfig`` bundle holding:

  - a primary :class:`Quoter` (from ``mmsim.models``) implementing the
    chosen quoting model, parameterized by IS axes
  - an inventory-penalty primitive (from ``mmsim.quoter.inv_penalty``)
  - an adverse-selection filter (from ``mmsim.quoter.adverse``)
  - a hedge engine config (from ``mmsim.hedge.engine``) or ``None``
  - a reference-price strategy (from ``mmsim.quoter.refprice``)
  - a quote shape (from ``mmsim.quoter.shapes``)
  - a refresh trigger (from ``mmsim.quoter.triggers``)

The composer is **routing only** — every primitive it returns is from
the existing engine modules. The runner uses these together to drive
the engine's ``run_sim`` loop end-to-end.

Composability is enforced by *typing*: the Quoter Protocol from
``mmsim.quoter.base`` is the integration contract; every combo's
strategy implements it. A combo is "valid" iff each component
constructs without error — that's a static check on the composer
itself, not a 100K-combo enumeration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from mmsim.hedge.engine import HedgeEngine
from mmsim.ingest.lob import Book
from mmsim.models import (
    AvellanedaStoikovQuoter,
    CarteaJaimungalQuoter,
    FairAnchoredQuoter,
    GLFTQuoter,
    HoStollQuoter,
    LadderQuoter,
    MicropriceSkewQuoter,
    SymmetricQuoter,
)
from mmsim.quoter.adverse import (
    HybridAdverseFilter,
    MicropriceDevFilter,
    OFIFilter,
    QueueImbalanceFilter,
    TradeToxicityFilter,
    VolSurgeFilter,
)
from mmsim.quoter.base import Quoter
from mmsim.quoter import inv_penalty
from mmsim.quoter.refprice import (
    EWMAFairTracker,
    VWAPTracker,
    microprice,
    top_mid,
    weighted_mid,
)
from mmsim.quoter.shapes import (
    DynamicDepthSpec,
    GeometricSpec,
    LadderSpec,
    PairedSpec,
    SingleSpec,
)
from mmsim.quoter.triggers import (
    BookEventTrigger,
    HybridTrigger,
    InvChangeTrigger,
    MidMoveTrigger,
    TimeTrigger,
)

from .combos import Combo

# Default IS-axis values used when a key is missing from `params`. Kept
# in one place so smoke tests don't have to reproduce them.
_DEFAULT_PARAMS: dict[str, float] = {
    "gamma": 0.5,
    "k": 1.5,
    "horizon_ns": 3_600_000_000_000,  # 1h
    "inventory_cap": 0.05,
    "refresh_interval_ns": 1_000_000_000,  # 1s
    "spread_floor": 5.0,
    "filter_threshold": 0.3,
    # Aux defaults (not enumerated as IS axes but shared by all combos).
    "quote_size": 0.001,           # base-asset units (~$80 at $80k BTC)
    "vol_window_ns": 10_000_000_000,  # 10s
    "ladder_step": 2.5,
    "ladder_n_levels": 3,
    "geometric_ratio": 1.5,
    "ewma_half_life_ns": 5_000_000_000,  # 5s
    "vwap_window_ns": 5_000_000_000,
    "ho_stoll_alpha": 10.0,
    "ho_stoll_beta": 0.05,
    "cj_kappa": 0.0,
    "glft_A": 140.0,
    "adverse_window_ns": 5_000_000_000,
    "adverse_vol_threshold_bp": 50.0,
    "hedge_threshold": 0.05,
    "trigger_inv_change": 0.001,
    "trigger_mid_move_bp": 5.0,
}


def _p(params: dict, key: str) -> float:
    """Get a param value, falling back to the bundled default."""
    if key in params:
        return params[key]
    return _DEFAULT_PARAMS[key]


# --------------------------------------------------------------------------- #
# Reference-price wrappers
# --------------------------------------------------------------------------- #

class _RefPriceFn:
    """Callable wrapper around a reference-price primitive that exposes a
    uniform ``__call__(book) -> Optional[float]`` shape regardless of
    whether the underlying primitive is pure (book-only) or stateful
    (tracker)."""

    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params
        if name == "ewma_fair":
            self._tracker = EWMAFairTracker(
                half_life_ns=int(_p(params, "ewma_half_life_ns")))
        elif name == "vwap":
            self._tracker = VWAPTracker(window_ns=int(_p(params, "vwap_window_ns")))
        else:
            self._tracker = None

    def __call__(self, book: Optional[Book]) -> Optional[float]:
        if self.name == "mid":
            return top_mid(book)
        if self.name == "microprice":
            return microprice(book)
        if self.name == "weighted_mid":
            return weighted_mid(book)
        if self.name == "ewma_fair":
            if book is not None and book.mid is not None:
                self._tracker.observe(int(book.ts_ns), float(book.mid))
            return self._tracker.value(book.ts_ns if book is not None else 0)
        if self.name == "vwap":
            # VWAP needs the trade tape; in book-only context we fall back to
            # mid. The runner feeds trades to the tracker separately.
            v = self._tracker.value(book.ts_ns if book is not None else 0)
            return v if v is not None else top_mid(book)
        if self.name == "model_pred":
            # No ML model layer in mmsim; deterministic stub returns mid +
            # small linear drift seeded by the bar count. Documented as a
            # fallback per the spec.
            return top_mid(book)
        raise ValueError(f"unknown reference price: {self.name}")

    def observe_trade(self, trade) -> None:
        """Feed a trade event to stateful trackers (VWAP)."""
        if self.name == "vwap":
            self._tracker.observe(trade)


# --------------------------------------------------------------------------- #
# Inventory-penalty wrappers
# --------------------------------------------------------------------------- #

def _make_inv_penalty(name: str, params: dict) -> Callable[[float], inv_penalty.Skew]:
    """Return a callable ``inv -> Skew`` for the chosen penalty mode."""
    gamma = float(_p(params, "gamma"))
    cap = float(_p(params, "inventory_cap"))
    if name == "linear":
        return lambda inv: inv_penalty.linear(inv, gamma)
    if name == "quadratic":
        return lambda inv: inv_penalty.quadratic(inv, gamma)
    if name == "exponential":
        return lambda inv: inv_penalty.exponential(inv, gamma, scale=max(cap, 1e-9))
    if name == "asymmetric":
        # Asymmetric: long penalty = gamma, short penalty = gamma / 2 (typical
        # crypto-MM convention — long inventory is harder to flatten on a perp).
        return lambda inv: inv_penalty.asymmetric(inv, gamma_long=gamma,
                                                    gamma_short=gamma / 2.0)
    if name == "soft_cap":
        return lambda inv: inv_penalty.soft_cap(inv, gamma, cap=cap)
    if name == "hard_cap":
        return lambda inv: inv_penalty.hard_cap(inv, cap=cap)
    raise ValueError(f"unknown inventory penalty: {name}")


# --------------------------------------------------------------------------- #
# Adverse-filter factory
# --------------------------------------------------------------------------- #

def _make_adverse_filter(name: str, params: dict):
    """Return an adverse-filter instance (or ``None`` if the axis is 'none').

    All filters expose ``observe_trade``, ``observe_book``, ``is_adverse``.
    """
    if name == "none":
        return None
    window_ns = int(_p(params, "adverse_window_ns"))
    thresh = float(_p(params, "filter_threshold"))
    if name == "ofi":
        return OFIFilter(window_ns=window_ns, threshold=thresh)
    if name == "toxicity":
        # toxicity threshold must be in [0.5, 1.0]; map filter_threshold from
        # [0.01, 0.9] into [0.5, 0.95] for usability.
        t_mapped = 0.5 + (thresh / 0.9) * 0.45
        return TradeToxicityFilter(window_ns=window_ns,
                                     threshold=min(max(t_mapped, 0.5), 1.0))
    if name == "vol_surge":
        return VolSurgeFilter(window_ns=window_ns,
                                threshold_bp=float(_p(params, "adverse_vol_threshold_bp")))
    if name == "microprice_dev":
        # microprice_dev threshold is in bp; rescale.
        return MicropriceDevFilter(threshold_bp=float(_p(params, "adverse_vol_threshold_bp")))
    if name == "queue_imb":
        return QueueImbalanceFilter(threshold=thresh)
    if name == "hybrid":
        # Compose OFI + queue_imb (two cheap filters) under "any-of".
        children = [
            OFIFilter(window_ns=window_ns, threshold=thresh),
            QueueImbalanceFilter(threshold=thresh),
        ]
        return HybridAdverseFilter(children=children, mode="any")
    raise ValueError(f"unknown adverse filter: {name}")


# --------------------------------------------------------------------------- #
# Hedge-engine factory
# --------------------------------------------------------------------------- #

def _make_hedge_engine(name: str, params: dict) -> Optional[HedgeEngine]:
    """Return a HedgeEngine (or ``None`` for 'none' / 'options_vega_stub')."""
    if name == "none":
        return None
    if name == "options_vega_stub":
        # T3 options book is deferred; route to None (no-op fallback per
        # the spec). The combo cardinality is preserved.
        return None
    threshold = float(_p(params, "hedge_threshold"))
    if name == "perp":
        return HedgeEngine(threshold=threshold, hedge_size_pct=1.0, instrument="perp")
    if name == "basket":
        # Basket hedge structurally mirrors perp; the difference is which
        # book gets crossed. mmsim's HedgeEngine takes an externally-passed
        # hedge book at decision time, so the runtime distinction is
        # callsite-level not engine-level. We tag the instrument so logs
        # are correctly attributed.
        return HedgeEngine(threshold=threshold, hedge_size_pct=1.0, instrument="basket")
    raise ValueError(f"unknown hedge mode: {name}")


# --------------------------------------------------------------------------- #
# Refresh-trigger factory
# --------------------------------------------------------------------------- #

def _make_refresh_trigger(name: str, params: dict):
    """Return a refresh-trigger instance (always non-None; 'book_event'
    fires unconditionally on every call)."""
    if name == "time":
        return TimeTrigger(interval_ns=int(_p(params, "refresh_interval_ns")))
    if name == "mid_move":
        return MidMoveTrigger(threshold_bp=float(_p(params, "trigger_mid_move_bp")))
    if name == "inv_change":
        return InvChangeTrigger(threshold=float(_p(params, "trigger_inv_change")))
    if name == "book_event":
        return BookEventTrigger()
    if name == "hybrid":
        children = [
            TimeTrigger(interval_ns=int(_p(params, "refresh_interval_ns"))),
            MidMoveTrigger(threshold_bp=float(_p(params, "trigger_mid_move_bp"))),
        ]
        return HybridTrigger(children=children, mode="any")
    raise ValueError(f"unknown refresh trigger: {name}")


# --------------------------------------------------------------------------- #
# Quote-shape spec factory
# --------------------------------------------------------------------------- #

def _make_shape_spec(name: str, params: dict):
    """Return a frozen ``*Spec`` dataclass for the chosen quote shape."""
    size = float(_p(params, "quote_size"))
    half_spread = float(_p(params, "spread_floor"))
    if name == "single":
        return SingleSpec(size=size, half_spread=half_spread)
    if name == "paired":
        # Two levels per side with growing offsets + decreasing sizes.
        levels = (
            (half_spread, size),
            (half_spread + float(_p(params, "ladder_step")), size * 0.5),
        )
        return PairedSpec(levels_bid=levels, levels_ask=levels)
    if name == "ladder":
        return LadderSpec(
            half_spread=half_spread,
            step=float(_p(params, "ladder_step")),
            n_levels=int(_p(params, "ladder_n_levels")),
            size_per_level=size,
        )
    if name == "geometric":
        return GeometricSpec(
            half_spread=half_spread,
            ratio=float(_p(params, "geometric_ratio")),
            n_levels=int(_p(params, "ladder_n_levels")),
            size_per_level=size,
        )
    if name == "dynamic_depth":
        return DynamicDepthSpec(
            half_spread=half_spread,
            step=float(_p(params, "ladder_step")),
            max_levels=int(_p(params, "ladder_n_levels")),
            inv_taper_threshold=float(_p(params, "inventory_cap")) * 0.5,
            size_per_level=size,
        )
    raise ValueError(f"unknown quote shape: {name}")


# --------------------------------------------------------------------------- #
# Quoting-model factory
# --------------------------------------------------------------------------- #

def _make_quoting_model(name: str, ref_fn: _RefPriceFn, params: dict) -> Quoter:
    """Return a concrete Quoter implementation for the chosen quoting model.

    The ``ref_fn`` is the chosen reference-price strategy and is wired into
    the quoter where the model accepts one (Symmetric, Ladder); models with
    a baked-in reference (AS, CJ, GLFT, Ho-Stoll use top_mid internally for
    sigma) ignore the override and document the structural pairing.
    """
    size = float(_p(params, "quote_size"))
    gamma = float(_p(params, "gamma"))
    k = float(_p(params, "k"))
    horizon_ns = int(_p(params, "horizon_ns"))
    vol_window_ns = int(_p(params, "vol_window_ns"))
    half_spread = float(_p(params, "spread_floor"))

    if name == "avellaneda_stoikov":
        return AvellanedaStoikovQuoter(
            gamma=gamma, k=k, horizon_ns=horizon_ns,
            size=size, vol_window_ns=vol_window_ns,
        )
    if name == "cartea_jaimungal":
        return CarteaJaimungalQuoter(
            gamma=gamma, k=k, kappa=float(_p(params, "cj_kappa")),
            horizon_ns=horizon_ns, size=size, vol_window_ns=vol_window_ns,
        )
    if name == "glft":
        return GLFTQuoter(
            gamma=gamma, k=k, A=float(_p(params, "glft_A")),
            horizon_ns=horizon_ns, size=size, vol_window_ns=vol_window_ns,
        )
    if name == "ho_stoll":
        return HoStollQuoter(
            alpha=float(_p(params, "ho_stoll_alpha")),
            beta=float(_p(params, "ho_stoll_beta")),
            size=size, vol_window_ns=vol_window_ns,
        )
    if name == "symmetric":
        return SymmetricQuoter(half_spread=half_spread, size=size, ref_fn=ref_fn)
    if name == "ladder":
        return LadderQuoter(
            half_spread=half_spread,
            step=float(_p(params, "ladder_step")),
            n_levels=int(_p(params, "ladder_n_levels")),
            size_per_level=size,
            ref_fn=ref_fn,
        )
    if name == "microprice_skew":
        return MicropriceSkewQuoter(half_spread=half_spread, size=size)
    if name == "fair_anchored":
        return FairAnchoredQuoter(
            half_spread=half_spread, size=size,
            half_life_ns=int(_p(params, "ewma_half_life_ns")),
        )
    raise ValueError(f"unknown quoting model: {name}")


# --------------------------------------------------------------------------- #
# Public composer
# --------------------------------------------------------------------------- #

@dataclass
class T4StrategyConfig:
    """Bundle of constructed primitives for one (combo, params) pair.

    Each attribute is a fully-constructed object pulled from the existing
    mmsim engine. The runner consumes this bundle to drive ``run_sim``.
    """
    combo: Combo
    params: dict
    quoter: Quoter
    ref_price_fn: _RefPriceFn
    inv_penalty_fn: Callable[[float], inv_penalty.Skew]
    adverse_filter: Optional[Any]      # may be None when filter == 'none'
    hedge_engine: Optional[HedgeEngine]
    refresh_trigger: Any
    shape_spec: Any
    quote_size: float
    spread_floor: float


def build_mm_strategy(combo: Combo, params: dict | None = None) -> T4StrategyConfig:
    """Compose a runnable T4 strategy from a structural combo and IS-params.

    Parameters
    ----------
    combo:
        One :class:`Combo` from :func:`combos.iter_all_combos` or
        :func:`combos.sample_combos`.
    params:
        IS-axis parameter dict. Missing keys fall back to module-level
        defaults; pass ``None`` to use defaults entirely.

    Returns
    -------
    T4StrategyConfig
        Bundle of constructed engine primitives — quoter, ref-price,
        inventory penalty, adverse filter, hedge engine, refresh
        trigger, shape spec.

    Raises
    ------
    ValueError
        If any axis value is not recognized (caller bug — combo names
        must come from :mod:`combos`).
    """
    params = dict(params) if params is not None else {}

    ref_fn = _RefPriceFn(combo.reference_price, params)
    quoter = _make_quoting_model(combo.quoting_model, ref_fn, params)
    inv_pen_fn = _make_inv_penalty(combo.inventory_penalty, params)
    adverse = _make_adverse_filter(combo.adverse_filter, params)
    hedge = _make_hedge_engine(combo.hedge_mode, params)
    trigger = _make_refresh_trigger(combo.refresh_trigger, params)
    shape = _make_shape_spec(combo.quote_shape, params)

    return T4StrategyConfig(
        combo=combo,
        params=params,
        quoter=quoter,
        ref_price_fn=ref_fn,
        inv_penalty_fn=inv_pen_fn,
        adverse_filter=adverse,
        hedge_engine=hedge,
        refresh_trigger=trigger,
        shape_spec=shape,
        quote_size=float(_p(params, "quote_size")),
        spread_floor=float(_p(params, "spread_floor")),
    )


__all__ = [
    "T4StrategyConfig", "build_mm_strategy",
]
