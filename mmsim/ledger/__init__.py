"""Costed per-fill ledger (research layer, H1).

The simulation engine (mmsim.sim) is frozen and parity-validated; it
emits ``Fill`` records but no costs and no reference mid. This module
adds a **non-invasive post-processing pass** over a ``SimResult`` that:

  - attaches the reference mid at fill time (from the snapshot stream),
  - applies the locked cost model (maker/taker fee, slippage, funding),
  - computes per-fill gross/net PnL against a running mark,
  - threads inventory after each fill,
  - writes the frozen ledger schema to parquet via a streaming writer
    (RAM-bounded; never materializes the whole ledger as Python objects).

Markout columns (mid_{1s,10s,60s}, markout_*, realised_spread,
adverse_selection_cost) are filled by ``mmsim.markout`` and merged in;
this module leaves them null when markout is not requested.

The cost model is the locked one from the pre-registration:
    taker fee 5 bp, maker fee 2 bp, slippage 2 bp, funding 1 bp/8h/leg.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from mmsim.ledger.schema import LEDGER_COLUMNS, LedgerRow
from mmsim.ledger.costs import CostModel, DEFAULT_COST_MODEL
from mmsim.ledger.writer import LedgerWriter, build_ledger

__all__ = [
    "LEDGER_COLUMNS",
    "LedgerRow",
    "CostModel",
    "DEFAULT_COST_MODEL",
    "LedgerWriter",
    "build_ledger",
]
