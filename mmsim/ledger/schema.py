"""Frozen per-fill ledger schema (locked).

The column order here IS the contract. Downstream readers index by
name, but the parquet is written in this order for stable diffs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

# Locked column order. Any change requires a v2 schema, not an edit.
LEDGER_COLUMNS = [
    "fill_id", "ts_ns", "venue", "symbol", "side", "is_maker", "price", "size",
    "ref_mid_at_fill", "queue_pos_at_fill", "fee", "slippage", "funding_accrued",
    "gas", "swap_slippage", "mid_1s", "mid_10s", "mid_60s",
    "markout_1s", "markout_10s", "markout_60s", "realised_spread",
    "adverse_selection_cost", "inv_after", "gross_pnl", "net_pnl",
    "regime_posterior", "run_id", "commit_hash",
]


@dataclass
class LedgerRow:
    """One costed fill. Markout/regime fields default to NaN/None until
    the markout pass (mmsim.markout) or regime pass (H4) fills them."""
    fill_id: int
    ts_ns: int
    venue: str
    symbol: str
    side: int
    is_maker: bool
    price: float
    size: float
    ref_mid_at_fill: float
    queue_pos_at_fill: float
    fee: float
    slippage: float
    funding_accrued: float
    gas: float
    swap_slippage: float
    inv_after: float
    gross_pnl: float
    net_pnl: float
    run_id: str
    commit_hash: str
    mid_1s: float = float("nan")
    mid_10s: float = float("nan")
    mid_60s: float = float("nan")
    markout_1s: float = float("nan")
    markout_10s: float = float("nan")
    markout_60s: float = float("nan")
    realised_spread: float = float("nan")
    adverse_selection_cost: float = float("nan")
    regime_posterior: Optional[float] = None

    def as_ordered(self) -> dict:
        return {c: getattr(self, c) for c in LEDGER_COLUMNS}


def _assert_schema_complete() -> None:
    have = {f.name for f in fields(LedgerRow)}
    want = set(LEDGER_COLUMNS)
    missing = want - have
    extra = have - want
    if missing or extra:
        raise AssertionError(
            f"LedgerRow schema drift: missing={missing} extra={extra}")


_assert_schema_complete()

__all__ = ["LEDGER_COLUMNS", "LedgerRow"]
