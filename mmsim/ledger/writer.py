"""Streaming ledger writer + the non-invasive costing pass.

``build_ledger`` walks a SimResult's fills in fill order, attaches the
reference mid (most-recent snapshot mid at-or-before each fill), applies
the locked cost model, threads inventory + running PnL, and streams rows
to parquet through ``LedgerWriter`` (RAM-bounded ParquetWriter).

PnL convention (mark-to-mid):
  - signed_size = side * size  (bid fill +, ask fill -)
  - cash flow of the fill = -signed_size * price  (we pay to buy, receive to sell)
  - gross_pnl per fill = mark-to-mid PnL increment using the running
    inventory revalued at the new reference mid, minus the trade's cash
    impact. We use the standard incremental MTM:
        d(equity) = inv_before * (mid_now - mid_prev) + signed_size * (mid_now - price)
    The first term is the carry on existing inventory as the mid moves;
    the second is the immediate edge of the new fill vs mid.
  - net_pnl = gross_pnl - fee - slippage - funding_accrued.

This is a research-layer accounting pass; it does NOT modify the engine.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from mmsim.ledger.schema import LEDGER_COLUMNS, LedgerRow
from mmsim.ledger.costs import CostModel, DEFAULT_COST_MODEL


def build_mid_timeline(stream) -> tuple:
    """Build (ts_ns array, mid array) from the snapshot events in a
    loaded EventStream. Sorted ascending by ts. Snapshots with no
    two-sided book are skipped."""
    from mmsim.ingest.lob import SnapshotEvent
    ts: List[int] = []
    mids: List[float] = []
    for ev in stream:
        if isinstance(ev, SnapshotEvent):
            if ev.bids and ev.asks:
                m = (ev.bids[0][0] + ev.asks[0][0]) / 2.0
                ts.append(int(ev.ts_ns))
                mids.append(float(m))
    return np.asarray(ts, dtype=np.int64), np.asarray(mids, dtype=np.float64)


def _mid_at_or_before(ts_arr: np.ndarray, mid_arr: np.ndarray, t: int) -> float:
    """Most-recent mid at-or-before t. NaN if t precedes the first snap."""
    idx = int(np.searchsorted(ts_arr, t, side="right")) - 1
    if idx < 0:
        return float("nan")
    return float(mid_arr[idx])


class LedgerWriter:
    """Streaming parquet writer for the frozen ledger schema. RAM-bounded:
    rows are buffered and flushed in batches; never holds the full ledger.
    Falls back to an in-memory list when no path is given (tests)."""

    def __init__(self, path: Optional[str] = None, batch_size: int = 50_000):
        self.path = path
        self.batch_size = batch_size
        self._buf: List[dict] = []
        self._writer = None
        self._schema = None
        self.rows_written = 0
        self.collected: List[dict] = []  # only used when path is None

    def _flush(self) -> None:
        if not self._buf:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        cols = {c: [r[c] for r in self._buf] for c in LEDGER_COLUMNS}
        table = pa.table(cols)
        if self.path is not None:
            if self._writer is None:
                self._schema = table.schema
                self._writer = pq.ParquetWriter(self.path, self._schema)
            self._writer.write_table(table.cast(self._schema))
        self.rows_written += len(self._buf)
        self._buf.clear()

    def write(self, row: LedgerRow) -> None:
        d = row.as_ordered()
        if self.path is None:
            self.collected.append(d)
        else:
            self._buf.append(d)
            if len(self._buf) >= self.batch_size:
                self._flush()

    def close(self) -> None:
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def build_ledger(
    sim_result,
    stream,
    *,
    cost_model: CostModel = DEFAULT_COST_MODEL,
    venue: str = "binance",
    symbol: str = "BTCUSDT",
    run_id: str = "run",
    commit_hash: str = "",
    out_path: Optional[str] = None,
    queue_pos_lookup: Optional[dict] = None,
    mid_timeline: Optional[tuple] = None,
) -> LedgerWriter:
    """Cost and write the per-fill ledger for one SimResult.

    ``queue_pos_lookup`` optionally maps fill_id -> queue_pos_at_fill
    (the engine does not expose it on the Fill record; left NaN if absent).
    ``mid_timeline`` optionally supplies a precomputed ``(ts_arr, mid_arr)``
    sorted snapshot mid timeline so callers running many ledgers over the same
    (or a superset) snapshot stream avoid rebuilding it each call; the
    most-recent-mid-at-or-before lookup is identical whether the timeline is
    the window's snapshots or a superset day timeline (fills fall inside the
    window).  When None, it is built from ``stream``.
    Returns the (closed) LedgerWriter; read rows via ``.collected`` when
    out_path is None, else they are on disk.
    """
    if mid_timeline is not None:
        ts_arr, mid_arr = mid_timeline
    else:
        ts_arr, mid_arr = build_mid_timeline(stream)
    writer = LedgerWriter(out_path)

    inv = 0.0
    prev_mid: Optional[float] = None
    prev_ts: Optional[int] = None

    for f in sim_result.fills:
        m0 = _mid_at_or_before(ts_arr, mid_arr, int(f.ts_ns))
        signed = float(f.side) * float(f.size)
        notional = float(f.price) * float(f.size)
        fee = cost_model.fee_for(notional, bool(f.is_maker))
        # Passive (maker) fills incur NO slippage — a resting limit order fills
        # at its limit price and never crosses the book.  slippage_for() returns
        # 0 for makers unless the model explicitly sets maker_pays_slippage.
        slip = cost_model.slippage_for(notional, bool(f.is_maker))
        # Funding on inventory carried since the previous fill.
        funding = 0.0
        if prev_ts is not None and prev_mid is not None:
            funding = cost_model.funding_for(inv, prev_mid, int(f.ts_ns) - prev_ts)

        # Incremental mark-to-mid PnL.
        if prev_mid is None or m0 != m0:  # NaN check
            carry = 0.0
            edge = 0.0 if m0 != m0 else signed * (m0 - float(f.price))
        else:
            carry = inv * (m0 - prev_mid)
            edge = signed * (m0 - float(f.price))
        gross = carry + edge

        inv += signed
        net = gross - fee - slip - funding

        qp = float("nan")
        if queue_pos_lookup is not None:
            qp = float(queue_pos_lookup.get(int(f.fill_id), float("nan")))

        row = LedgerRow(
            fill_id=int(f.fill_id), ts_ns=int(f.ts_ns), venue=venue,
            symbol=symbol, side=int(f.side), is_maker=bool(f.is_maker),
            price=float(f.price), size=float(f.size),
            ref_mid_at_fill=m0, queue_pos_at_fill=qp,
            fee=fee, slippage=slip, funding_accrued=funding,
            gas=0.0, swap_slippage=0.0,
            inv_after=inv, gross_pnl=gross, net_pnl=net,
            run_id=run_id, commit_hash=commit_hash,
        )
        writer.write(row)

        if m0 == m0:  # only advance the mark when we have a real mid
            prev_mid = m0
            prev_ts = int(f.ts_ns)

    writer.close()
    return writer


__all__ = ["LedgerWriter", "build_ledger", "build_mid_timeline"]
