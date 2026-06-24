"""Bit-identical parity: streamed engine vs validated reference engine.

Asserts, on an ES smoke slice:
  1. iter_events_streaming(parquet) yields the SAME event sequence (field by
     field) as load_lob(parquet) -- same order, same SnapshotEvent/TradeEvent.
  2. run_sim_streaming(generator) == run_sim(materialised list): identical
     fills (price/size/side/ts/order_id/fill_id/is_maker), identical counts,
     identical queue/maker/taker state -- for both a maker quoter and a
     deeper-rank quoter (so the queue-aware fill path is exercised).
  3. The streamed mid timeline == build_mid_timeline(load_lob(...)).

Exit code 0 = PASS (bit-identical). Non-zero = parity broken.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from mmsim.ingest.lob import load_lob, SnapshotEvent, TradeEvent
from mmsim.ingest.stream import iter_events_streaming, iter_mid_timeline_streaming
from mmsim.ledger.writer import build_mid_timeline
from mmsim.sim.loop import run_sim, QuoteRequest
from mmsim.sim.loop_stream import run_sim_streaming
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.quoter.builtin import TopOfBookQuoter


class RankQuoter:
    def __init__(self, size, rank, side):
        self.size = size; self.rank = rank; self.side = side
    def __call__(self, book, active, t_ns):
        levels = book.bids if self.side == +1 else book.asks
        if not levels or len(levels) < self.rank:
            return []
        px = levels[self.rank - 1][0]
        return [QuoteRequest(side=self.side, price=px, size=self.size)]


def _event_tuple(e):
    if isinstance(e, SnapshotEvent):
        return ("S", e.ts_ns, e.recv_ns, e.symbol, e.venue, e.depth,
                tuple(e.bids), tuple(e.asks))
    return ("T", e.ts_ns, e.recv_ns, e.symbol, e.venue, e.price, e.size, e.side)


def _fills_equal(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (x.fill_id, x.order_id, x.ts_ns, x.price, x.size, x.side, x.is_maker) != \
           (y.fill_id, y.order_id, y.ts_ns, y.price, y.size, y.side, y.is_maker):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--size", type=float, default=1.0)
    args = ap.parse_args()

    ok = True

    # ---- 1. event-sequence parity ----------------------------------------
    ref_stream = load_lob(args.snap, args.trades)
    ref_tuples = [_event_tuple(e) for e in ref_stream]
    stream_tuples = [_event_tuple(e)
                     for e in iter_events_streaming(args.snap, args.trades)]
    seq_ok = ref_tuples == stream_tuples
    print(f"[1] event-sequence bit-identical: {seq_ok} "
          f"(ref={len(ref_tuples):,} stream={len(stream_tuples):,})")
    ok &= seq_ok

    # ---- 2. engine parity (touch quoter + deep-rank quoter) --------------
    for label, qfac in [
        ("TopOfBook", lambda: TopOfBookQuoter(size=args.size)),
        ("Rank-3-bid", lambda: RankQuoter(args.size, 3, +1)),
    ]:
        r_ref = run_sim(ref_stream, qfac(), QueueAwareFillModel())
        r_str = run_sim_streaming(
            iter_events_streaming(args.snap, args.trades), qfac(),
            QueueAwareFillModel())
        eng_ok = (
            _fills_equal(r_ref.fills, r_str.fills)
            and r_ref.n_events_processed == r_str.n_events_processed
            and r_ref.n_snapshot_events == r_str.n_snapshot_events
            and r_ref.n_trade_events == r_str.n_trade_events
            and r_ref.n_quoter_calls == r_str.n_quoter_calls
            and r_ref.n_maker_fills == r_str.n_maker_fills
            and r_ref.n_taker_fills == r_str.n_taker_fills
        )
        print(f"[2:{label}] engine bit-identical: {eng_ok} "
              f"(fills ref={len(r_ref.fills):,} stream={len(r_str.fills):,}; "
              f"events={r_ref.n_events_processed:,})")
        ok &= eng_ok

    # ---- 3. mid-timeline parity ------------------------------------------
    ts_ref, mid_ref = build_mid_timeline(ref_stream)
    pairs = list(iter_mid_timeline_streaming(args.snap))
    ts_str = np.array([p[0] for p in pairs], dtype=np.int64)
    mid_str = np.array([p[1] for p in pairs], dtype=np.float64)
    mid_ok = (np.array_equal(ts_ref, ts_str)
              and np.array_equal(mid_ref, mid_str, equal_nan=True))
    print(f"[3] mid-timeline bit-identical: {mid_ok} "
          f"(ref={ts_ref.size:,} stream={ts_str.size:,})")
    ok &= mid_ok

    print(f"\nPARITY {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
