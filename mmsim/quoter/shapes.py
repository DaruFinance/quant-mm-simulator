"""Quote-shape primitives.

Five pure-function shape primitives.  Every primitive is a function of
``(spec, ref_price, inv)`` only — they never consult time, never
consult future state, never close over mutable book state.  This
makes leak-freedom obvious by inspection (the function signature
itself is the proof).

The five shapes:
  - ``single``        — one bid + one ask at ref ± half_spread
  - ``paired``        — N bid levels + N ask levels with per-level
                         offsets and sizes
  - ``ladder``        — N evenly-spaced (linear step) levels per side
  - ``geometric``     — N geometrically-spaced levels per side
  - ``dynamic_depth`` — N tapers down as |inv| grows past threshold

Each is callable with the same trailing-inv convention so the
inventory-penalty primitives can compose with any shape
uniformly.

Composition with the Quoter Protocol: a Quoter implementation
stores a shape spec + a reference-price strategy + an inv-skew
and emits ``shape(spec, ref(book), inv)`` from its ``quote()`` method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from mmsim.sim.loop import QuoteRequest


# --------------------------------------------------------------------- #
# Spec types (frozen dataclasses; hash-stable so unit tests can compare
# spec-to-spec equality without surprises).
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class SingleSpec:
    """One bid + one ask of equal size at ref ± half_spread."""
    size: float
    half_spread: float


@dataclass(frozen=True)
class PairedSpec:
    """N levels per side with per-level (offset_from_ref, size).

    ``levels_bid`` and ``levels_ask`` are each tuples of (offset, size)
    where offset is the absolute distance from ref (positive number;
    bids sit at ``ref - offset``, asks at ``ref + offset``).  Sizes
    per level can differ across sides; the two arms are independent.
    """
    levels_bid: Tuple[Tuple[float, float], ...]
    levels_ask: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class LadderSpec:
    """N linear-spaced levels per side.  Level k sits at
    ``ref ± (half_spread + k * step)`` with size ``size_per_level``."""
    half_spread: float
    step: float
    n_levels: int
    size_per_level: float


@dataclass(frozen=True)
class GeometricSpec:
    """N geometrically-spaced levels per side.  Level k sits at
    ``ref ± half_spread * ratio**k`` with size ``size_per_level``.
    ``ratio > 1`` means expanding spacing; ``ratio == 1`` collapses
    to a degenerate stack at the same price (caller's choice)."""
    half_spread: float
    ratio: float
    n_levels: int
    size_per_level: float


@dataclass(frozen=True)
class DynamicDepthSpec:
    """Like ``LadderSpec`` but the active level count tapers when
    ``|inv|`` exceeds ``inv_taper_threshold``.  At ``|inv| <= thresh``
    the shape posts ``max_levels`` per side; at ``|inv| >= 2 * thresh``
    it tapers down to 1 level; linear in between."""
    half_spread: float
    step: float
    max_levels: int
    inv_taper_threshold: float
    size_per_level: float


# --------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------- #

def single(spec: SingleSpec, ref_price: float, inv: float = 0.0) -> List[QuoteRequest]:
    """One bid + one ask centred on ``ref_price``."""
    return [
        QuoteRequest(side=+1, price=ref_price - spec.half_spread, size=spec.size),
        QuoteRequest(side=-1, price=ref_price + spec.half_spread, size=spec.size),
    ]


def paired(spec: PairedSpec, ref_price: float, inv: float = 0.0) -> List[QuoteRequest]:
    """Per-level (offset, size) on each side, independent arms."""
    out: List[QuoteRequest] = []
    for offset, size in spec.levels_bid:
        out.append(QuoteRequest(side=+1, price=ref_price - float(offset), size=float(size)))
    for offset, size in spec.levels_ask:
        out.append(QuoteRequest(side=-1, price=ref_price + float(offset), size=float(size)))
    return out


def ladder(spec: LadderSpec, ref_price: float, inv: float = 0.0) -> List[QuoteRequest]:
    """N linear-spaced levels per side, bid + ask interleaved by level."""
    if spec.n_levels < 1:
        return []
    out: List[QuoteRequest] = []
    for k in range(spec.n_levels):
        offset = spec.half_spread + k * spec.step
        out.append(QuoteRequest(side=+1, price=ref_price - offset, size=spec.size_per_level))
        out.append(QuoteRequest(side=-1, price=ref_price + offset, size=spec.size_per_level))
    return out


def geometric(spec: GeometricSpec, ref_price: float, inv: float = 0.0) -> List[QuoteRequest]:
    """N geometrically-spaced levels per side."""
    if spec.n_levels < 1:
        return []
    out: List[QuoteRequest] = []
    offset = spec.half_spread
    for _ in range(spec.n_levels):
        out.append(QuoteRequest(side=+1, price=ref_price - offset, size=spec.size_per_level))
        out.append(QuoteRequest(side=-1, price=ref_price + offset, size=spec.size_per_level))
        offset *= spec.ratio
    return out


def dynamic_depth(spec: DynamicDepthSpec, ref_price: float, inv: float = 0.0) -> List[QuoteRequest]:
    """Ladder whose active depth tapers with ``|inv|``."""
    abs_inv = abs(inv)
    thr = spec.inv_taper_threshold
    if abs_inv <= thr:
        n_active = spec.max_levels
    elif abs_inv >= 2.0 * thr:
        n_active = 1
    else:
        # Linear taper from max_levels down to 1 across [thr, 2*thr].
        frac = (abs_inv - thr) / thr  # in [0, 1]
        n_active = max(1, int(round(spec.max_levels * (1.0 - frac))))
    out: List[QuoteRequest] = []
    for k in range(n_active):
        offset = spec.half_spread + k * spec.step
        out.append(QuoteRequest(side=+1, price=ref_price - offset, size=spec.size_per_level))
        out.append(QuoteRequest(side=-1, price=ref_price + offset, size=spec.size_per_level))
    return out


__all__ = [
    "SingleSpec", "PairedSpec", "LadderSpec", "GeometricSpec", "DynamicDepthSpec",
    "single", "paired", "ladder", "geometric", "dynamic_depth",
]
