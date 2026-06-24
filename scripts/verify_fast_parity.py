"""Parity check: fast numba fill path vs the reference run_sim pipeline.

For each runner quoter (TouchQuoter, DepthSkewQuoter, MicroSkewQuoter x2,
RankQuoter ranks 1..5) we run BOTH:
  - the reference: mmsim.sim.loop.run_sim(events, quoter, QueueAwareFillModel())
  - the fast path: mmsim.sim.fast_sim.simulate(events, spec, size)
and assert the emitted fills are economically identical (same ts/side/price/
size sequence) and the scalar counts match.

Run:
  PYTHONPATH=. python3 scripts/verify_fast_parity.py [snap.parquet trades.parquet [symbol]]
Default: the bundled 60-min BTC fixture.
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from mmsim.ingest.lob import load_lob
from mmsim.sim.loop import run_sim
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.sim import fast_sim as F
import scripts.run_mm_full as R


def _fills_tuple(fills):
    return [(f.ts_ns, f.side, round(f.price, 10), round(f.size, 12), f.is_maker)
            for f in fills]


def _cmp(name, ref_res, fast_res):
    rt = _fills_tuple(ref_res.fills)
    ft = _fills_tuple(fast_res.fills)
    ok = True
    if len(rt) != len(ft):
        print(f"  [{name}] FAIL n_fills ref={len(rt)} fast={len(ft)}")
        ok = False
    n = min(len(rt), len(ft))
    mism = 0
    for i in range(n):
        if rt[i] != ft[i]:
            mism += 1
            if mism <= 5:
                print(f"  [{name}] mism[{i}] ref={rt[i]} fast={ft[i]}")
    if mism:
        print(f"  [{name}] FAIL {mism}/{n} fill mismatches")
        ok = False
    if ref_res.n_quoter_calls != fast_res.n_quoter_calls:
        print(f"  [{name}] FAIL n_quoter_calls ref={ref_res.n_quoter_calls} "
              f"fast={fast_res.n_quoter_calls}")
        ok = False
    if ref_res.n_maker_fills != fast_res.n_maker_fills:
        print(f"  [{name}] FAIL n_maker_fills ref={ref_res.n_maker_fills} "
              f"fast={fast_res.n_maker_fills}")
        ok = False
    if ok:
        print(f"  [{name}] OK  fills={len(rt)} quoter_calls={ref_res.n_quoter_calls}")
    return ok


def main():
    if len(sys.argv) >= 3:
        snap, trades = sys.argv[1], sys.argv[2]
        symbol = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        here = Path(__file__).resolve().parent.parent
        snap = str(here / "tests/fixtures/lob_btcusdt_60min_snapshots.parquet")
        trades = str(here / "tests/fixtures/lob_btcusdt_60min_trades.parquet")
        symbol = None

    events = load_lob(snap, trades, symbol=symbol)
    print(f"events={len(events)}  snap={snap}")
    size = 0.001

    cases = [
        ("touch", R.TouchQuoter(size), partial(F.spec_touch, size=size)),
        ("depth_skew", R.DepthSkewQuoter(size), partial(F.spec_depth_skew, size=size)),
        ("microprice", R.MicroSkewQuoter(size, use_ofi=False),
         partial(F.spec_microskew, size=size, use_ofi=False)),
        ("integ_ofi", R.MicroSkewQuoter(size, use_ofi=True),
         partial(F.spec_microskew, size=size, use_ofi=True)),
    ]
    for rank in range(1, 6):
        cases.append((f"rank{rank}", R.RankQuoter(size, rank, +1),
                      partial(F.spec_rank, size=size, rank=rank, side=+1)))

    all_ok = True
    for name, quoter, spec in cases:
        ref = run_sim(events, quoter, QueueAwareFillModel())
        fast = F.simulate(events, spec, size)
        all_ok &= _cmp(name, ref, fast)

    print("ALL PARITY OK" if all_ok else "PARITY FAILURES")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
