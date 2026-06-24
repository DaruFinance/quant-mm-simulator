"""Post-fill markout / realised-spread decomposition (research layer, H3).

ABSENT from the engine today (grep markout/realised_spread => nothing).
This module is the genuinely new research substrate.

For each fill it computes, at horizons 1s / 10s / 60s:
  - mid(t+tau): most-recent snapshot mid at-or-before fill_ts + tau
  - markout(tau)            = side * (mid(tau) - price) / mid0
  - realised_spread(tau)    = side * (price - mid(tau)) / mid0
  - adverse_selection(tau)  = side * (mid(tau) - mid0) / mid0
with mid0 = mid at-or-before fill_ts. Definitions are locked in
the pre-registration. Identity (checked): quoted_half_spread =
realised_spread(tau) + adverse_selection(tau).

The per-fill mid lookup over N snapshots is the hot loop; it is
implemented in pure Python (reference) and numba (production), verified
bit-identical. See mmsim.markout.engine.
"""
from __future__ import annotations

from mmsim.markout.engine import (
    MarkoutResult,
    compute_markout,
    compute_markout_reference,
    HORIZONS_NS,
)

__all__ = [
    "MarkoutResult",
    "compute_markout",
    "compute_markout_reference",
    "HORIZONS_NS",
]
