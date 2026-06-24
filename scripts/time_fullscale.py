"""Full-scale wall-clock timing: numba fast path vs pure-Python reference on a
real CME root-day (millions of events), where the per-event Python overhead the
kernel removes actually dominates. Writes runs/engine_timing_fullscale.json.

Usage: PYTHONPATH=. python3 scripts/time_fullscale.py <snap.parquet> <trades.parquet>
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from functools import partial
from pathlib import Path

sys.path.insert(0, ".")
from mmsim.ingest.lob import load_lob
from mmsim.sim.loop import run_sim
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.sim import fast_sim as F
import scripts.run_mm_full as R


def main():
    snap, trades = sys.argv[1], sys.argv[2]
    events = load_lob(snap, trades, symbol=None)
    size = 1.0
    quoter = R.TouchQuoter(size)
    spec = partial(F.spec_touch, size=size)

    F.simulate(events, spec, size)  # JIT warmup
    best_fast = float("inf")
    for _ in range(3):
        t0 = time.perf_counter(); fr = F.simulate(events, spec, size)
        best_fast = min(best_fast, time.perf_counter() - t0)

    t0 = time.perf_counter(); rr = run_sim(events, quoter, QueueAwareFillModel())
    ref = time.perf_counter() - t0

    art = {
        "fixture": f"real CME root-day full scale ({Path(snap).name})",
        "n_events": int(len(events)),
        "n_fills": int(len(fr.fills)),
        "fast_numba_sec": round(best_fast, 3),
        "reference_python_sec": round(ref, 3),
        "speedup_x": round(ref / best_fast, 1) if best_fast > 0 else None,
        "parity_fills_ok": bool(len(rr.fills) == len(fr.fills)),
        "note": ("fast = min of 3 (numba JIT warmed up); reference = single run of "
                 "mmsim.sim.loop.run_sim (deterministic; concurrent load only inflates it). "
                 "Single-thread, OPENBLAS_NUM_THREADS=1."),
    }
    out = Path("runs/engine_timing_fullscale.json")
    out.write_text(json.dumps(art, indent=2))
    print(json.dumps(art, indent=2)); print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
