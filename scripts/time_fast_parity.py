"""Wall-clock timing artifact for the numba fast path vs the pure-Python reference.

Mirrors scripts/verify_fast_parity.py but times the touch quoter (the headline
engine) on the bundled 60-min BTC fixture: warm up the numba JIT, then take the
min of N repeats for each engine (min is robust to CPU contention). Writes the
absolute wall-clock split + speedup factor to runs/engine_timing.json so the
117x speedup claim rests on a saved artifact rather than an unrecorded run.

Run:  PYTHONPATH=. python3 scripts/time_fast_parity.py
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from functools import partial
from pathlib import Path

sys.path.insert(0, ".")
from mmsim.ingest.lob import load_lob
from mmsim.sim.loop import run_sim
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.sim import fast_sim as F
import scripts.run_mm_full as R

REPEATS = int(os.environ.get("TIMING_REPEATS", "3"))


def _best(fn, n):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def main():
    here = Path(__file__).resolve().parent.parent
    snap = str(here / "tests/fixtures/lob_btcusdt_60min_snapshots.parquet")
    trades = str(here / "tests/fixtures/lob_btcusdt_60min_trades.parquet")
    events = load_lob(snap, trades, symbol=None)
    size = 0.001
    quoter = R.TouchQuoter(size)
    spec = partial(F.spec_touch, size=size)

    # parity + numba warmup (JIT compile excluded from timing)
    ref0 = run_sim(events, quoter, QueueAwareFillModel())
    fast0 = F.simulate(events, spec, size)
    parity = (len(ref0.fills) == len(fast0.fills) and
              ref0.n_maker_fills == fast0.n_maker_fills)

    ref_s = _best(lambda: run_sim(events, quoter, QueueAwareFillModel()), REPEATS)
    fast_s = _best(lambda: F.simulate(events, spec, size), REPEATS)
    speedup = ref_s / fast_s if fast_s > 0 else float("nan")

    art = {
        "fixture": "lob_btcusdt_60min (bundled BTC depth book + tape)",
        "n_events": int(len(events)),
        "n_fills": int(len(fast0.fills)),
        "quoter": "touch",
        "repeats": REPEATS,
        "reference_python_sec": round(ref_s, 4),
        "fast_numba_sec": round(fast_s, 4),
        "speedup_x": round(speedup, 1),
        "parity_ok": bool(parity),
        "note": ("min-of-N wall clock, numba JIT warmed up before timing; "
                 "single-thread (OPENBLAS_NUM_THREADS=1). Reference = "
                 "mmsim.sim.loop.run_sim; fast = mmsim.sim.fast_sim.simulate."),
    }
    out = here / "runs" / "engine_timing.json"
    out.write_text(json.dumps(art, indent=2))
    print(json.dumps(art, indent=2))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
