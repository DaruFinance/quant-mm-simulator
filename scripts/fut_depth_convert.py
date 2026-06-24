"""Convert one futures contract's depth + taq CSV.gz -> compact parquet.

Per-contract: parse depth (per-side rows) -> two-sided depth-10 snapshots,
parse taq -> true-signed trades, write both parquet files matching the
engine's lob reader schema.  numba book-builder (bit-identical to the
reference, checked in tests).  cProfile smoke is gated by --profile.

Usage:
  fut_depth_convert.py --depth ESZ3_depth.csv.gz --taq ESZ3_taq.csv.gz \
      --symbol ESZ3 --out-dir data/fut_smoke [--max-levels 10] [--profile]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from mmsim.ingest import fut_depth as fd


def _window_mask(ts, day_ns, start_hhmm, end_hhmm):
    """Boolean mask for [start,end) in UTC HH:MM within the day."""
    import numpy as np
    def _hhmm_ns(hhmm):
        hh = int(hhmm[:2]); mm = int(hhmm[2:])
        return day_ns + (hh * 3600 + mm * 60) * 1_000_000_000
    lo = _hhmm_ns(start_hhmm); hi = _hhmm_ns(end_hhmm)
    return (ts >= lo) & (ts < hi)


def convert(depth, taq, symbol, out_dir, max_levels, throttle_k,
            start_hhmm=None, end_hhmm=None):
    import numpy as np
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dcols = fd.parse_depth_arrays_fast(depth)
    t1 = time.time()
    snaps = fd.build_book_snapshots(dcols, use_numba=True)
    n_raw = snaps["snap_ts"].shape[0]
    del dcols
    if start_hhmm and end_hhmm and n_raw:
        day_ns = (snaps["snap_ts"][0] // (86_400 * 1_000_000_000)) * (86_400 * 1_000_000_000)
        m = _window_mask(snaps["snap_ts"], day_ns, start_hhmm, end_hhmm)
        snaps = {k: v[m] for k, v in snaps.items()}
    if throttle_k > 0:
        snaps = fd.throttle_top_k_changed(snaps, k=throttle_k)
    t2 = time.time()
    trades = fd.parse_taq_trades_fast(taq)
    if start_hhmm and end_hhmm and trades["ts_ns"].shape[0]:
        day_ns = (trades["ts_ns"][0] // (86_400 * 1_000_000_000)) * (86_400 * 1_000_000_000)
        mt = _window_mask(trades["ts_ns"], day_ns, start_hhmm, end_hhmm)
        trades = {k: v[mt] for k, v in trades.items()}
    t3 = time.time()
    sp = out / f"{symbol}_snap.parquet"
    tp = out / f"{symbol}_trades.parquet"
    n_snap, n_tr = fd.write_parquet(snaps, trades, sp, tp, symbol=symbol,
                                    max_levels=max_levels)
    t4 = time.time()
    print(f"[{symbol}] depth_parse={t1-t0:.1f}s book_build+throttle={t2-t1:.1f}s "
          f"taq_parse={t3-t2:.1f}s write={t4-t3:.1f}s total={t4-t0:.1f}s")
    print(f"[{symbol}] raw_updates={n_raw:,} -> kept(top-{throttle_k}-changed)="
          f"{n_snap:,} ({n_snap/n_raw*100:.1f}%), trades={n_tr:,}")
    return n_snap, n_tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", required=True)
    ap.add_argument("--taq", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-levels", type=int, default=10)
    ap.add_argument("--throttle-k", type=int, default=5)
    ap.add_argument("--start-hhmm", default=None, help="UTC HHMM window start")
    ap.add_argument("--end-hhmm", default=None, help="UTC HHMM window end")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    if args.profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
        convert(args.depth, args.taq, args.symbol, args.out_dir, args.max_levels, args.throttle_k, args.start_hhmm, args.end_hhmm)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
        print(s.getvalue())
    else:
        convert(args.depth, args.taq, args.symbol, args.out_dir, args.max_levels, args.throttle_k, args.start_hhmm, args.end_hhmm)


if __name__ == "__main__":
    main()
