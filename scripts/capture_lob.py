"""Focused capture: BTCUSDT L2 snapshots + trade tape from Binance.

Drives `cryptodata.sources.binance.BinanceSpot` for a fixed duration
and writes two parquet files:
  - lob_btcusdt_{n}min_snapshots.parquet  (L2 snapshots, 5s cadence)
  - lob_btcusdt_{n}min_trades.parquet     (tick-by-tick trades)

These two are the inputs `load_lob` expects.  We capture
them as separate parquets rather than a single combined file so that
downstream tests can pin them independently.

Usage:
    python scripts/capture_lob.py --minutes 60 --out tests/fixtures/

Output schema mirrors the `crypto-data-aggregator` data dictionary
(book_l2_snapshot and trades tables) so `load_lob` can read either
this script's output or a future query against the aggregator's
DuckDB-managed Parquet partitions interchangeably.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from cryptodata.sources.binance import BinanceSpot


def _flatten_book_snapshot(s: Any) -> dict:
    """`BookSnapshot` -> a flat row.  Matches the data-dictionary
    schema for `book_l2_snapshot` (ts_ns, recv_ns, symbol, venue,
    depth, bids, asks)."""
    return {
        "ts_ns": int(s.ts_ns),
        "recv_ns": int(s.recv_ns),
        "symbol": str(s.symbol),
        "venue": str(s.venue),
        "depth": int(s.depth),
        "bids": [{"px": float(b["px"]), "sz": float(b["sz"])} for b in s.bids],
        "asks": [{"px": float(a["px"]), "sz": float(a["sz"])} for a in s.asks],
    }


def _flatten_trade(t: Any) -> dict:
    """`Trade` -> a flat row matching `trades` schema."""
    return {
        "ts_ns": int(t.ts_ns),
        "recv_ns": int(t.recv_ns),
        "symbol": str(t.symbol),
        "venue": str(t.venue),
        "price": float(t.price),
        "size": float(t.size),
        "side": int(t.side),
        "trade_id": str(t.trade_id) if t.trade_id is not None else None,
    }


async def _capture(seconds: int, depth: int) -> tuple[list[dict], list[dict]]:
    src = BinanceSpot()
    snapshots: list[dict] = []
    trades: list[dict] = []
    stop_at = time.monotonic() + seconds

    async def collect_snapshots():
        async for s in src.stream_book(["BTC-USDT"], depth=depth):
            snapshots.append(_flatten_book_snapshot(s))
            if time.monotonic() >= stop_at:
                return

    async def collect_trades():
        async for t in src.stream_trades(["BTC-USDT"]):
            trades.append(_flatten_trade(t))
            if time.monotonic() >= stop_at:
                return

    snap_task = asyncio.create_task(collect_snapshots())
    trade_task = asyncio.create_task(collect_trades())
    # Bound by wall-clock; whichever stream is slower defines the cap.
    await asyncio.sleep(seconds + 1)
    for tk in (snap_task, trade_task):
        if not tk.done():
            tk.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await tk
    return snapshots, trades


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=int, default=60,
                    help="capture duration in minutes (default 60)")
    p.add_argument("--depth", type=int, default=20,
                    help="L2 depth per side (5/10/20 — Binance valid depths)")
    p.add_argument("--out", type=Path, default=Path("tests/fixtures"),
                    help="output directory for the two parquet files")
    args = p.parse_args()

    seconds = args.minutes * 60
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"capture_lob: starting BTCUSDT/binance L2+trades capture "
          f"for {args.minutes} min (depth {args.depth})")
    t0 = time.time()
    snaps, trades = asyncio.run(_capture(seconds, args.depth))
    elapsed = time.time() - t0
    print(f"capture_lob: ingested {len(snaps)} snapshots and "
          f"{len(trades)} trades in {elapsed:.1f}s")

    if snaps:
        snap_path = args.out / f"lob_btcusdt_{args.minutes}min_snapshots.parquet"
        pq.write_table(pa.Table.from_pylist(snaps), snap_path)
        print(f"  wrote {snap_path} ({snap_path.stat().st_size:,} bytes)")
    else:
        print("  WARNING: no snapshots captured")
    if trades:
        trade_path = args.out / f"lob_btcusdt_{args.minutes}min_trades.parquet"
        pq.write_table(pa.Table.from_pylist(trades), trade_path)
        print(f"  wrote {trade_path} ({trade_path.stat().st_size:,} bytes)")
    else:
        print("  WARNING: no trades captured")

    return 0 if (snaps and trades) else 1


if __name__ == "__main__":
    sys.exit(main())
