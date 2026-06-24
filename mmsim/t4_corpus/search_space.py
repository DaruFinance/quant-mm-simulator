"""T4 IS-optimization search space.

Per the T4 spec [IS] axes:
    gamma (risk aversion) - k (intensity decay) - T (horizon ns) -
    inventory_cap - refresh_interval_ns - spread_floor -
    filter threshold

Each axis is named so combos and runners can pass `params` dicts through
unchanged. Bounds are defensible defaults — wide enough to find good
params, tight enough to avoid burning the IS-opt budget on pathological
parameter values.

All sampling is deterministic given the seed (NumPy PCG64). This is part
of the leak-free discipline: the IS optimizer cannot accidentally use
OOS data to bias the sample because the samples are fixed by seed alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Iterator, Literal, Sequence

import numpy as np


AxisKind = Literal["int", "float", "log_float", "choice"]


@dataclass(frozen=True)
class Axis:
    """One IS-tune axis. Use ``low``/``high`` for numeric kinds; use
    ``choices`` for ``choice``."""
    name: str
    kind: AxisKind
    low: float = 0.0
    high: float = 1.0
    choices: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class SearchSpace:
    """Joint search space across multiple axes."""
    axes: tuple[Axis, ...]

    def __post_init__(self):
        names = [a.name for a in self.axes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate axis names: {names}")

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.axes]

    def sample(self, n: int, seed: int = 2026) -> list[dict]:
        """Draw ``n`` random samples from the joint distribution.

        Determinism: ``np.random.default_rng(seed)`` plus per-axis sequential
        draws → identical results across calls.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        rng = np.random.default_rng(seed)
        out: list[dict] = []
        for _ in range(n):
            params: dict = {}
            for ax in self.axes:
                params[ax.name] = _draw_axis(ax, rng)
            out.append(params)
        return out

    def grid(self, levels_per_axis: int) -> list[dict]:
        """Cartesian grid of ``levels_per_axis`` per numeric axis."""
        per_axis_values: list[list] = []
        for ax in self.axes:
            if ax.kind == "choice":
                per_axis_values.append(list(ax.choices))
            elif ax.kind == "int":
                vals = np.linspace(ax.low, ax.high, levels_per_axis)
                per_axis_values.append([int(round(v)) for v in vals])
            elif ax.kind == "log_float":
                lo, hi = math.log10(ax.low), math.log10(ax.high)
                per_axis_values.append([float(10 ** v)
                                         for v in np.linspace(lo, hi, levels_per_axis)])
            else:  # float
                per_axis_values.append([float(v) for v in
                                         np.linspace(ax.low, ax.high, levels_per_axis)])
        names = self.names
        return [dict(zip(names, combo))
                for combo in product(*per_axis_values)]

    def n_grid_combinations(self, levels_per_axis: int) -> int:
        total = 1
        for ax in self.axes:
            if ax.kind == "choice":
                total *= max(1, len(ax.choices))
            else:
                total *= levels_per_axis
        return total


def _draw_axis(ax: Axis, rng: np.random.Generator) -> float | int | str:
    if ax.kind == "int":
        return int(rng.integers(int(ax.low), int(ax.high) + 1))
    if ax.kind == "float":
        return float(rng.uniform(ax.low, ax.high))
    if ax.kind == "log_float":
        lo, hi = math.log10(ax.low), math.log10(ax.high)
        return float(10 ** rng.uniform(lo, hi))
    if ax.kind == "choice":
        return ax.choices[int(rng.integers(0, len(ax.choices)))]
    raise ValueError(f"unknown axis kind: {ax.kind}")


# --------------------- T4 IS axes --------------------------- #

# Bounds tuned for crypto-LOB market making at BTCUSDT-scale prices.
# - gamma in (0.05, 5.0) covers from "very-risk-tolerant" to "ultra-conservative".
# - k in (0.1, 10) covers reasonable intensity-decay calibrations for Poisson MO arrivals.
# - horizon_ns spans 10s (~1e10) to 1h (~3.6e12).
# - inventory_cap is in base-asset units; 0.001 to 1.0 BTC.
# - refresh_interval_ns from 100us (~1e5) to 10s (~1e10).
# - spread_floor in price units; 0.5 to 50.0 (BTCUSDT tick is 0.01-0.1).
# - filter_threshold spans 0.01 to 0.9 (OFI / queue imbalance unit ratio).
SEARCH_SPACE_T4 = SearchSpace((
    Axis("gamma", "log_float", 0.05, 5.0),
    Axis("k", "log_float", 0.1, 10.0),
    Axis("horizon_ns", "log_float", 1e10, 3.6e12),
    Axis("inventory_cap", "log_float", 0.001, 1.0),
    Axis("refresh_interval_ns", "log_float", 1e5, 1e10),
    Axis("spread_floor", "log_float", 0.5, 50.0),
    Axis("filter_threshold", "float", 0.01, 0.9),
))


def search_space_t4() -> SearchSpace:
    """Return the T4 SearchSpace (singleton convenience accessor)."""
    return SEARCH_SPACE_T4


__all__ = [
    "Axis", "SearchSpace",
    "SEARCH_SPACE_T4", "search_space_t4",
]
