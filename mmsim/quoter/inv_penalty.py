"""Inventory-penalty primitives.

Six pure functions of (inv, params) that compute a price skew (in
the same currency unit as `ref_price`) and an optional size scale
applied to bid vs ask quotes.  The output is a `Skew` record with
two scalar fields:

  - ``price_offset``: signed currency offset; **positive shifts both
    quotes UP** (skewing in favour of selling — discourages buying).
    Bid price becomes ``ref - half_spread + price_offset``;
    ask price becomes ``ref + half_spread + price_offset``.
  - ``size_scale_bid`` / ``size_scale_ask``: multiplicative factors
    in `[0.0, 1.0]` applied to each side's quote sizes.  1.0 = no
    change; 0.0 = refuse to quote that side (hard-cap path).

All primitives are pure functions of ``inv`` and the spec — no time,
no book, no future state.  Leak-freedom is the signature itself.

Composition with shapes: a Quoter takes a shape's output
QuoteRequest list and applies the Skew's price_offset to every
quote's price + the size_scale per side.  The composition is at
the strategy level; these primitives ship the math.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class Skew:
    """Output of every inventory-penalty primitive."""
    price_offset: float          # signed; +ve shifts quotes UP
    size_scale_bid: float        # in [0.0, 1.0]
    size_scale_ask: float


# --------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------- #

def linear(inv: float, gamma: float) -> Skew:
    """Price skew = -gamma * inv (long inv -> shift quotes DOWN to
    encourage selling).  No size skew."""
    return Skew(price_offset=-gamma * float(inv),
                 size_scale_bid=1.0, size_scale_ask=1.0)


def quadratic(inv: float, gamma: float) -> Skew:
    """Quadratic price skew: sign(inv) * gamma * inv**2.  Penalty
    grows non-linearly with position size."""
    inv = float(inv)
    sign = -1.0 if inv > 0 else (1.0 if inv < 0 else 0.0)
    return Skew(price_offset=sign * gamma * inv * inv,
                 size_scale_bid=1.0, size_scale_ask=1.0)


def exponential(inv: float, gamma: float, scale: float) -> Skew:
    """Exponential price skew: sign(-inv) * gamma * (exp(|inv|/scale) - 1).
    Skew explodes as |inv| grows; useful for hard-skew at large
    positions while staying small near zero."""
    inv = float(inv)
    abs_inv = abs(inv)
    base = exp(abs_inv / scale) - 1.0
    sign = -1.0 if inv > 0 else (1.0 if inv < 0 else 0.0)
    return Skew(price_offset=sign * gamma * base,
                 size_scale_bid=1.0, size_scale_ask=1.0)


def asymmetric(inv: float, gamma_long: float, gamma_short: float) -> Skew:
    """Different linear coefficient for long vs short inventory.
    Useful when one side is structurally harder to hedge."""
    inv = float(inv)
    if inv > 0:
        offset = -gamma_long * inv
    elif inv < 0:
        offset = -gamma_short * inv  # inv<0 -> offset>0 -> shifts UP
    else:
        offset = 0.0
    return Skew(price_offset=offset,
                 size_scale_bid=1.0, size_scale_ask=1.0)


def soft_cap(inv: float, gamma: float, cap: float) -> Skew:
    """Linear skew plus an additional quadratic ramp as |inv|
    approaches ``cap``.  Smoothly slows accumulation without an
    abrupt cliff."""
    if cap <= 0:
        raise ValueError("cap must be > 0")
    inv = float(inv)
    base = -gamma * inv
    if abs(inv) >= cap:
        # Quadratic ramp: extra penalty proportional to (|inv|/cap - 1)^2
        extra = gamma * cap * ((abs(inv) / cap - 1.0) ** 2)
        sign_extra = -1.0 if inv > 0 else 1.0
        offset = base + sign_extra * extra
    else:
        offset = base
    return Skew(price_offset=offset,
                 size_scale_bid=1.0, size_scale_ask=1.0)


def hard_cap(inv: float, cap: float) -> Skew:
    """No price skew; instead, refuses to quote on the side that
    would worsen position.  Long >= cap: drop the bid (size_bid=0);
    short <= -cap: drop the ask (size_ask=0)."""
    if cap <= 0:
        raise ValueError("cap must be > 0")
    inv = float(inv)
    bid_scale = 0.0 if inv >= cap else 1.0
    ask_scale = 0.0 if inv <= -cap else 1.0
    return Skew(price_offset=0.0,
                 size_scale_bid=bid_scale,
                 size_scale_ask=ask_scale)


__all__ = ["Skew", "linear", "quadratic", "exponential",
            "asymmetric", "soft_cap", "hard_cap"]
