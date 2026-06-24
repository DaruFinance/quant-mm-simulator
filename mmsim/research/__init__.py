"""Research / validation layer for the MM simulator (H1..H8).

The simulation engine (mmsim.sim, mmsim.models, mmsim.quoter, mmsim.hedge)
is frozen and parity-validated. This package is the NEW research substrate
built on top of it, per the pre-registration:

  - prereg     : the locked hypotheses + thresholds as code constants.
  - h1_replay  : held-out slice carve + queue-aware-vs-naive differentiation.
  - wfo        : MM-objective adapter over the qrf walk-forward harness (H2).
  - gating     : markout-gated quote-pull policy (H3).
  - perm_null  : strategy-level permutation null + DSR effective-trials (H7).

The modules are import-clean and unit-smoke-able on the existing fixtures.
"""
from __future__ import annotations

__all__ = []
