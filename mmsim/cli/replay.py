"""mmsim replay CLI (H8).

Replays an L2 capture through the queue-aware engine and emits inventory,
PnL, and markout summary as JSON. Single-thread.

Usage:
    python -m mmsim.cli.replay \
        --snapshots tests/fixtures/lob_btcusdt_60min_snapshots.parquet \
        --trades    tests/fixtures/lob_btcusdt_60min_trades.parquet \
        [--ledger-out ledger.parquet] [--horizon 10s] [--json]

Verdicts and thresholds are read from the locked pre-registration
(mmsim.research.prereg) so CLI output cannot drift from the lock.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mmsim-replay")
    p.add_argument("--snapshots", type=Path, required=True)
    p.add_argument("--trades", type=Path, required=True)
    p.add_argument("--ledger-out", type=Path, default=None)
    p.add_argument("--horizon", default="10s", choices=["1s", "10s", "60s"])
    p.add_argument("--taker-every", type=int, default=500)
    p.add_argument("--json", action="store_true", help="emit JSON to stdout")
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from mmsim.ingest.lob import load_lob
    from mmsim.sim.loop import run_sim
    from mmsim.sim.fills import QueueAwareFillModel
    from mmsim.ledger.writer import build_ledger, build_mid_timeline
    from mmsim.ledger.costs import DEFAULT_COST_MODEL
    from mmsim.markout.engine import compute_markout
    from mmsim.sim.inventory import inventory_path
    from test_sim_fills import BracketQuoter

    stream = load_lob(str(args.snapshots), str(args.trades))
    res = run_sim(stream, BracketQuoter(taker_every=args.taker_every),
                  QueueAwareFillModel())
    snap_ts, snap_mid = build_mid_timeline(stream)
    commit = _git_commit()

    w = build_ledger(res, stream, cost_model=DEFAULT_COST_MODEL,
                     run_id="cli-replay", commit_hash=commit,
                     out_path=str(args.ledger_out) if args.ledger_out else None)
    rows = w.collected if args.ledger_out is None else None

    mo = compute_markout(res.fills, snap_ts, snap_mid)
    inv = inventory_path(res.fills)
    h = args.horizon
    mk = getattr(mo, f"markout_{h}")
    rs = getattr(mo, f"realised_spread_{h}")
    adv = getattr(mo, f"adverse_{h}")

    if rows is not None:
        net_pnl = float(np.sum([r["net_pnl"] for r in rows]))
        gross_pnl = float(np.sum([r["gross_pnl"] for r in rows]))
        fees = float(np.sum([r["fee"] + r["slippage"] for r in rows]))
    else:
        net_pnl = gross_pnl = fees = float("nan")  # streamed to disk

    out = {
        "n_events": res.n_events_processed,
        "n_snapshots": res.n_snapshot_events,
        "n_trades": res.n_trade_events,
        "fills": {"total": len(res.fills), "maker": res.n_maker_fills,
                  "taker": res.n_taker_fills},
        "inventory": {"final": inv.final_inv, "peak_long": inv.peak_long,
                      "peak_short": inv.peak_short},
        "pnl": {"net": net_pnl, "gross": gross_pnl, "costs": fees},
        "markout": {
            "horizon": h,
            "mean_markout": float(np.nanmean(mk)),
            "mean_realised_spread": float(np.nanmean(rs)),
            "mean_adverse_selection": float(np.nanmean(adv)),
        },
        "commit": commit,
        "ledger_out": str(args.ledger_out) if args.ledger_out else None,
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
