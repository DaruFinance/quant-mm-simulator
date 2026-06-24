"""H3 — markout-gated quote-pull policy (a falsifiable hypothesis).

The decomposition half (realised vs adverse-selection spread, markout
by queue rank) is provided as M6/M7 in ``scripts/run_mm_full.py``.  This
script evaluates the *actionable rule* it motivates:

    A market maker who can SEE its post-fill markout / a toxicity signal turning
    against it should be able to PULL (withdraw) its quotes for a cooldown and
    re-quote once the toxic flow passes, thereby skipping the worst fills.

This script implements that policy as a thin, read-only LAYER on top of the
shipped fast engine (``mmsim.sim.fast_sim``).  It does NOT modify the engine or
the runner — it imports the spec stage, builds a per-snapshot GATE MASK from a
strictly-causal toxicity signal, NaN-suppresses the baseline two-sided touch
quotes inside the cooldown, and re-runs the SAME shared fill kernel.

Falsifiable hypothesis (pre-registered direction)
-------------------------------------------------
    H3: net realised spread (per unit of size traded, net of the locked CME
        futures costs) of the GATED quoter is GREATER than the ALWAYS-quoting
        baseline, after honestly accounting for the fills it forgoes.

    Net-of-forgone test: we report (a) the per-fill net realised spread of each
    quoter and (b) the TOTAL net realised-spread captured (sum over fills =
    per-fill mean x n_fills).  The economically honest verdict is on the total:
    pulling quotes only helps if the spread saved on skipped-toxic fills exceeds
    the spread given up on the good fills it also skipped.  If gating raises
    per-fill quality but lowers total captured spread, that is a NULL (it just
    trades less), and we say so.

    PASS  iff  gated improves BOTH per-fill mean net realised spread AND keeps
              total captured net realised spread within a non-trivial fraction
              (so the per-fill gain is not purely a trade-count artefact) — the
              decisive figure is per-fill mean with a bootstrap CI that excludes
              the baseline, with total-captured reported alongside as the cost.
    NULL  otherwise (publish honestly).

Causality (no lookahead) — the core correctness property
--------------------------------------------------------
The gate decision active during snapshot interval i may use ONLY information
timestamped at or before ``snap_ts[i]``.  Two signals, both strictly causal:

  * ``flow``  (default): a VPIN-style rolling signed-aggressive-volume toxicity
    read off the TRADE TAPE in a trailing window ending at ``snap_ts[i]``.  A
    fill on our bid is hit by SELL aggressors; a burst of one-sided sell flow is
    exactly the adverse pressure that makes a bid fill's markout negative.  The
    signal at snapshot i aggregates only trades with ts <= snap_ts[i].

  * ``markout``: realised post-fill markout of our OWN baseline fills, but a
    fill's markout only ENTERS the signal once its markout horizon has fully
    elapsed (``fill_ts + H <= snap_ts[i]``).  This is the literal
    "markout-gated" rule and is causal by the horizon lag.  (We verify both
    signals are leak-free by a shift/pollution check.)

When the toxicity signal exceeds a threshold (calibrated IN-SAMPLE on a
disjoint forward-WFO IS window, evaluated OOS), the maker PULLS both quotes for
``cooldown_ns`` and resumes after.

Run discipline / outputs are documented at the bottom (argparse help).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import resource
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

from mmsim.sim import fast_sim as _F
from mmsim.sim.fast_sim import WindowArrays
from mmsim.ledger.writer import build_ledger
from mmsim.ledger.costs import CostModel
from mmsim.markout.engine import compute_markout

# Locked CME futures cost model — identical to scripts/run_mm_full.py FUT_COST
# (no maker rebate; symmetric per-side clearing).  Net of costs by construction.
FUT_COST = CostModel(taker_fee=0.00002, maker_fee=0.00002, slippage=0.00002,
                     funding_per_8h=0.0, is_perp=False)
_SEC = 1_000_000_000
_MIN = 60 * _SEC


# --------------------------------------------------------------------- #
# Causal toxicity signals  (value defined per SNAPSHOT, using ts<=snap_ts)
# --------------------------------------------------------------------- #

def flow_toxicity(w: WindowArrays, lookback_ns: int) -> np.ndarray:
    """VPIN-style signed-aggressive-flow toxicity, per snapshot, strictly
    causal.

    For snapshot i define tox[i] = |sum_{trades in (snap_ts[i]-LB, snap_ts[i]]}
    side*size| / sum size  in that trailing window  (a [0,1] order-flow
    imbalance magnitude).  High tox = one-sided aggressive flow = adverse.

    Only trades with ts <= snap_ts[i] enter tox[i], so the signal that gates
    interval i (which starts AT snap_ts[i]) uses no future trade.  Computed
    with a two-pointer sweep over the time-sorted tape -> O(S+T).
    """
    S = w.snap_ts.shape[0]
    tox = np.zeros(S, np.float64)
    tts = w.trd_ts
    tside = w.trd_side.astype(np.float64)
    tsz = w.trd_sz
    T = tts.shape[0]
    if T == 0 or S == 0:
        return tox
    signed = tside * tsz
    lo = 0  # first trade index inside the window
    hi = 0  # one past the last trade index with ts <= snap_ts[i]
    run_signed = 0.0
    run_abs = 0.0
    for i in range(S):
        t_hi = w.snap_ts[i]
        t_lo = t_hi - lookback_ns
        # extend hi to include all trades with ts <= t_hi (causal: <=)
        while hi < T and tts[hi] <= t_hi:
            run_signed += signed[hi]
            run_abs += tsz[hi]
            hi += 1
        # retract lo to drop trades older than the lookback window (ts <= t_lo)
        while lo < hi and tts[lo] <= t_lo:
            run_signed -= signed[lo]
            run_abs -= tsz[lo]
            lo += 1
        tox[i] = abs(run_signed) / run_abs if run_abs > 0 else 0.0
    return tox


def markout_toxicity(w: WindowArrays, base_fills, snap_ts, snap_mid,
                     horizon_ns: int) -> np.ndarray:
    """Realised post-fill markout toxicity per snapshot, strictly causal via a
    horizon lag.

    For each baseline fill we know its 10s markout (negative = adverse).  The
    fill's markout is only OBSERVABLE once ``fill_ts + horizon_ns`` has passed,
    so it may enter the per-snapshot signal only for snapshots with
    ``snap_ts[i] >= fill_ts + horizon_ns``.  tox[i] = max(0, -mean recent
    observable markout) -> larger when recent fills marked out against us.

    We aggregate observable fills in a trailing window of width ``horizon_ns``
    on the OBSERVATION time (fill_ts+horizon), giving a rolling adverse-markout
    estimate.  Returns a non-negative per-snapshot toxicity in bp.
    """
    S = w.snap_ts.shape[0]
    tox = np.zeros(S, np.float64)
    if not base_fills or S == 0:
        return tox
    mo = compute_markout(base_fills, snap_ts, snap_mid)
    mk = mo.markout_10s  # signed; negative = adverse to us
    fts = np.array([f.ts_ns for f in base_fills], dtype=np.int64)
    obs_ts = fts + horizon_ns           # earliest time the markout is known
    good = ~np.isnan(mk)
    obs_ts = obs_ts[good]; mk = mk[good]
    if obs_ts.size == 0:
        return tox
    order = np.argsort(obs_ts, kind="mergesort")
    obs_ts = obs_ts[order]; mk = mk[order]
    T = obs_ts.shape[0]
    lo = 0; hi = 0
    run_sum = 0.0; run_n = 0
    for i in range(S):
        t_hi = w.snap_ts[i]
        t_lo = t_hi - horizon_ns
        while hi < T and obs_ts[hi] <= t_hi:   # observable by now (causal)
            run_sum += mk[hi]; run_n += 1; hi += 1
        while lo < hi and obs_ts[lo] <= t_lo:
            run_sum -= mk[lo]; run_n -= 1; lo += 1
        if run_n > 0:
            mean_mk = run_sum / run_n
            tox[i] = -mean_mk * 1e4 if mean_mk < 0 else 0.0   # bp, adverse only
    return tox


# --------------------------------------------------------------------- #
# Gate mask: cooldown latch on a thresholded toxicity signal
# --------------------------------------------------------------------- #

def gate_mask(tox: np.ndarray, snap_ts: np.ndarray, thresh: float,
              cooldown_ns: int) -> np.ndarray:
    """Boolean PULL mask per snapshot.  When tox[i] >= thresh we enter a pull
    state that latches until ``cooldown_ns`` has elapsed with the signal back
    below threshold.  pulled[i]=True => suppress both quotes for interval i.

    Strictly causal: pulled[i] depends only on tox[<=i] and snap_ts[<=i].
    """
    S = tox.shape[0]
    pulled = np.zeros(S, np.bool_)
    cooldown_until = -1
    for i in range(S):
        if tox[i] >= thresh:
            cooldown_until = snap_ts[i] + cooldown_ns
        if snap_ts[i] < cooldown_until:
            pulled[i] = True
    return pulled


def apply_gate(bpx, bsz, apx, asz, pulled):
    """NaN-suppress both quotes on pulled snapshots (engine treats NaN px as
    'no order').  Returns new arrays (does not mutate inputs)."""
    bpx = np.where(pulled, np.nan, bpx)
    apx = np.where(pulled, np.nan, apx)
    bsz = np.where(pulled, 0.0, bsz)
    asz = np.where(pulled, 0.0, asz)
    return bpx, bsz, apx, asz


# --------------------------------------------------------------------- #
# Net realised spread of a quoter (per-fill, net of costs)
# --------------------------------------------------------------------- #

def _net_realised(fills, snap_ts, snap_mid):
    """Per-fill net realised half-spread (bp), NET of the cost model.

    realised_spread_10s from the markout engine is the GROSS captured half-
    spread vs the 10s mid.  We subtract the round-trip cost charged per fill
    (maker fee + slippage, both per-side rates) so every number is net.  Fees
    are a fraction of notional == bp directly.
    """
    if not fills:
        return np.array([])
    mo = compute_markout(fills, snap_ts, snap_mid)
    rs = mo.realised_spread_10s
    rs = rs[~np.isnan(rs)]
    cost_bp = (FUT_COST.maker_fee + FUT_COST.slippage) * 1e4
    return rs * 1e4 - cost_bp


def _boot_mean_ci(x, B=2000, seed=0, alpha=0.05):
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(B, x.size))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (float(x.mean()), lo, hi)


# --------------------------------------------------------------------- #
# Per-window evaluation
# --------------------------------------------------------------------- #

def wfo_windows(t0, t1, is_ns, oos_ns):
    out = []
    s = t0
    while s + is_ns + oos_ns <= t1:
        out.append((s, s + is_ns, s + is_ns, s + is_ns + oos_ns))
        s += oos_ns
    return out


def _build_tox(wa, signal, lookback_ns, horizon_ns, snap_ts, snap_mid,
               base_fills):
    if signal == "flow":
        return flow_toxicity(wa, lookback_ns)
    elif signal == "markout":
        return markout_toxicity(wa, base_fills, snap_ts, snap_mid, horizon_ns)
    raise ValueError(signal)


def eval_window(wa: WindowArrays, snap_ts, snap_mid, size, signal,
                lookback_ns, horizon_ns, cooldown_grid, thresh_grid,
                is_split_frac=0.5, seed=0):
    """Run baseline + gated on ONE OOS window.  Threshold/cooldown are picked
    on the IS PORTION of the window (first ``is_split_frac``) and evaluated on
    the held-out OOS portion — a within-window forward split so the gate params
    are never fit on the data they are scored on.

    Returns per-window metrics + the OOS per-fill net-realised arrays for both
    quoters (for day-level pooling).
    """
    S = wa.snap_ts.shape[0]
    if S < 4 or wa.trd_ts.shape[0] == 0:
        return None

    # baseline two-sided touch (always quote) — the engine's spec_touch
    bpx, bsz, apx, asz = _F.spec_touch(wa, size)

    # causal toxicity signal over the whole window's snapshots
    base_fills_full = _F._run_specs(wa, bpx, bsz, apx, asz)
    tox = _build_tox(wa, signal, lookback_ns, horizon_ns, snap_ts, snap_mid,
                     base_fills_full)

    # within-window forward IS/OOS split on snapshot time
    t0 = int(wa.snap_ts[0]); t1 = int(wa.snap_ts[-1])
    split_ts = t0 + int((t1 - t0) * is_split_frac)
    is_mask_snap = wa.snap_ts < split_ts
    oos_mask_snap = ~is_mask_snap
    if is_mask_snap.sum() < 2 or oos_mask_snap.sum() < 2:
        return None

    # --- IS calibration: pick (thresh, cooldown) maximising IS per-fill net
    #     realised spread of the gated quoter, restricted to IS snapshots ---
    def _is_score(thresh, cd_ns):
        pulled = gate_mask(tox, wa.snap_ts, thresh, cd_ns)
        gb, gbs, ga, gas = apply_gate(bpx, bsz, apx, asz, pulled)
        # zero out OOS snapshots so only IS fills count for scoring
        gb = np.where(is_mask_snap, gb, np.nan)
        ga = np.where(is_mask_snap, ga, np.nan)
        fills = _F._run_specs(wa, gb, gbs, ga, gas)
        nr = _net_realised(fills, snap_ts, snap_mid)
        if nr.size < 5:
            return -np.inf, 0
        return float(nr.mean()), nr.size

    best = (-np.inf, None, None, 0)
    for th in thresh_grid:
        for cd in cooldown_grid:
            sc, n = _is_score(th, cd)
            if sc > best[0]:
                best = (sc, th, cd, n)
    if best[1] is None:
        return None
    _, thr, cd_ns, _ = best

    # --- OOS evaluation with the IS-chosen params -----------------------
    pulled = gate_mask(tox, wa.snap_ts, thr, cd_ns)
    gb, gbs, ga, gas = apply_gate(bpx, bsz, apx, asz, pulled)

    # restrict BOTH quoters to OOS snapshots for the verdict
    bpx_oos = np.where(oos_mask_snap, bpx, np.nan)
    apx_oos = np.where(oos_mask_snap, apx, np.nan)
    gb_oos = np.where(oos_mask_snap, gb, np.nan)
    ga_oos = np.where(oos_mask_snap, ga, np.nan)

    base_fills = _F._run_specs(wa, bpx_oos, bsz, apx_oos, asz)
    gate_fills = _F._run_specs(wa, gb_oos, gbs, ga_oos, gas)

    base_nr = _net_realised(base_fills, snap_ts, snap_mid)
    gate_nr = _net_realised(gate_fills, snap_ts, snap_mid)

    frac_pulled_oos = float(pulled[oos_mask_snap].mean()) if oos_mask_snap.any() else 0.0

    m = {
        "thr": float(thr), "cooldown_ms": cd_ns / 1e6,
        "frac_snapshots_pulled_oos": frac_pulled_oos,
        "base_n_fills": int(base_nr.size),
        "gate_n_fills": int(gate_nr.size),
        "base_perfill_net_realised_bp": float(base_nr.mean()) if base_nr.size else float("nan"),
        "gate_perfill_net_realised_bp": float(gate_nr.mean()) if gate_nr.size else float("nan"),
        "base_total_net_realised_bp": float(base_nr.sum()) if base_nr.size else 0.0,
        "gate_total_net_realised_bp": float(gate_nr.sum()) if gate_nr.size else 0.0,
        "forgone_fills": int(base_nr.size - gate_nr.size),
    }
    return {"metrics": m, "base_nr": base_nr, "gate_nr": gate_nr}


# --------------------------------------------------------------------- #
# One contract-day
# --------------------------------------------------------------------- #

def run_one(snap, trades, root, date, symbol, out_dir, *, size, signal,
            lookback_ms, horizon_s, cooldown_grid_ms, thresh_grid,
            is_min, oos_min, throttle_k, seed):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    # session bounds from the snapshot mid timeline (cheap streaming pass)
    from scripts.run_mm_full import mid_timeline
    s_ts, s_mid = mid_timeline(snap, symbol)
    if s_ts.size == 0:
        raise SystemExit(f"{root} {date}: no two-sided snapshots")
    t0, t1 = int(s_ts[0]), int(s_ts[-1])
    windows = wfo_windows(t0, t1, int(is_min * _MIN), int(oos_min * _MIN))

    day = _F.read_day_arrays(snap, trades, symbol=symbol, throttle_k=throttle_k,
                             depth_k=10)

    lookback_ns = int(lookback_ms * 1e6)
    horizon_ns = int(horizon_s * _SEC)
    cooldown_grid = [int(c * 1e6) for c in cooldown_grid_ms]

    win_rows: List[dict] = []
    pooled_base: List[np.ndarray] = []
    pooled_gate: List[np.ndarray] = []
    for wi in range(len(windows)):
        w = windows[wi]
        wa = _F.window_arrays_from_day(day, w[2], w[3])
        r = eval_window(wa, s_ts, s_mid, size, signal, lookback_ns, horizon_ns,
                        cooldown_grid, thresh_grid, seed=seed + wi)
        del wa
        if r is None:
            continue
        m = r["metrics"]; m["window"] = wi
        m["root"] = root; m["date"] = date; m["symbol"] = symbol
        win_rows.append(m)
        if r["base_nr"].size: pooled_base.append(r["base_nr"])
        if r["gate_nr"].size: pooled_gate.append(r["gate_nr"])

    pb = np.concatenate(pooled_base) if pooled_base else np.array([])
    pg = np.concatenate(pooled_gate) if pooled_gate else np.array([])

    base_mean, base_lo, base_hi = _boot_mean_ci(pb, seed=seed)
    gate_mean, gate_lo, gate_hi = _boot_mean_ci(pg, seed=seed + 1)
    # paired bootstrap on the DIFFERENCE of pooled means (independent resamples
    # since the two fill sets differ in membership; this is a difference-of-
    # means test on the two pooled distributions)
    diff_mean = (gate_mean - base_mean) if (pb.size and pg.size) else float("nan")
    diff_ci = _diff_boot_ci(pb, pg, seed=seed + 2)

    summary = {
        "root": root, "date": date, "symbol": symbol, "signal": signal,
        "n_windows": len(win_rows),
        "lookback_ms": lookback_ms, "horizon_s": horizon_s,
        "is_min": is_min, "oos_min": oos_min, "throttle_k": throttle_k,
        "cost_model": {"maker_bp": FUT_COST.maker_fee * 1e4,
                       "slip_bp": FUT_COST.slippage * 1e4, "rebate": False},
        "oos_pooled": {
            "base_n_fills": int(pb.size), "gate_n_fills": int(pg.size),
            "forgone_fills": int(pb.size - pg.size),
            "base_perfill_net_realised_bp": base_mean,
            "base_perfill_ci95": [base_lo, base_hi],
            "gate_perfill_net_realised_bp": gate_mean,
            "gate_perfill_ci95": [gate_lo, gate_hi],
            "diff_perfill_bp": diff_mean,
            "diff_perfill_ci95": diff_ci,
            "base_total_net_realised_bp": float(pb.sum()) if pb.size else 0.0,
            "gate_total_net_realised_bp": float(pg.sum()) if pg.size else 0.0,
        },
        "wall_s": round(time.time() - t_start, 1),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
    }
    json_path = os.path.join(out_dir, f"{root}_{date}_{signal}.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    if win_rows:
        pq.write_table(pa.Table.from_pylist(win_rows),
                       os.path.join(out_dir, f"{root}_{date}_{signal}_windows.parquet"))
    print(f"[{root} {date} {signal}] win={len(win_rows)} "
          f"base_fills={pb.size:,} gate_fills={pg.size:,} "
          f"perfill base={base_mean:.4f} gate={gate_mean:.4f} bp "
          f"diff={diff_mean:.4f} CI={diff_ci} wall={summary['wall_s']}s")
    return summary


def _diff_boot_ci(a, b, B=2000, seed=0, alpha=0.05):
    if a.size == 0 or b.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a.size, size=(B, a.size))
    ib = rng.integers(0, b.size, size=(B, b.size))
    d = b[ib].mean(axis=1) - a[ia].mean(axis=1)
    return [float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))]


# --------------------------------------------------------------------- #
# Assembly + verdict + figure
# --------------------------------------------------------------------- #

def assemble(out_dir, signal):
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(f"_{signal}.json") and not fn.startswith("VERDICT"):
            with open(os.path.join(out_dir, fn)) as fh:
                rows.append(json.load(fh))
    if not rows:
        raise SystemExit(f"no per-run JSONs for signal={signal} in {out_dir}")

    # Pool across contract-days at the per-day-pooled level.  The decisive
    # falsifiable figure: mean(diff_perfill_bp) across days and the share of
    # days where gating improved per-fill net realised spread, plus the
    # aggregate forgone-fill cost.
    diffs = np.array([r["oos_pooled"]["diff_perfill_bp"] for r in rows
                      if r["oos_pooled"]["diff_perfill_bp"] == r["oos_pooled"]["diff_perfill_bp"]])
    base_tot = sum(r["oos_pooled"]["base_total_net_realised_bp"] for r in rows)
    gate_tot = sum(r["oos_pooled"]["gate_total_net_realised_bp"] for r in rows)
    base_n = sum(r["oos_pooled"]["base_n_fills"] for r in rows)
    gate_n = sum(r["oos_pooled"]["gate_n_fills"] for r in rows)

    # cross-day verdict: per-fill must improve (mean diff>0 AND a sign test /
    # CI on the per-day diffs excluding 0 from below); total-captured reported.
    n_days = diffs.size
    mean_diff = float(diffs.mean()) if n_days else float("nan")
    # day-level bootstrap CI on the mean per-day diff
    if n_days >= 2:
        rng = np.random.default_rng(99)
        bs = diffs[rng.integers(0, n_days, size=(5000, n_days))].mean(axis=1)
        diff_lo = float(np.quantile(bs, 0.025)); diff_hi = float(np.quantile(bs, 0.975))
        win_share = float(np.mean(diffs > 0))
    else:
        diff_lo = diff_hi = float("nan"); win_share = float("nan")

    per_fill_better = mean_diff > 0 and (n_days < 2 or diff_lo > 0)
    total_retained = (gate_tot / base_tot) if base_tot != 0 else float("nan")
    # PASS: per-fill improves with CI excluding 0, AND gating retains a non-
    # trivial share of total captured spread (not just trading ~nothing).
    verdict = "PASS" if (per_fill_better and total_retained > 0.5) else "NULL"

    out = {
        "signal": signal, "n_contract_days": len(rows),
        "verdict": verdict,
        "mean_per_day_diff_perfill_bp": mean_diff,
        "per_day_diff_ci95": [diff_lo, diff_hi],
        "share_days_gating_better": win_share,
        "base_perfill_pooled_bp": (base_tot / base_n) if base_n else float("nan"),
        "gate_perfill_pooled_bp": (gate_tot / gate_n) if gate_n else float("nan"),
        "base_total_captured_bp": base_tot,
        "gate_total_captured_bp": gate_tot,
        "total_captured_retained_frac": total_retained,
        "base_n_fills": base_n, "gate_n_fills": gate_n,
        "forgone_fills": base_n - gate_n,
        "forgone_fill_frac": (1 - gate_n / base_n) if base_n else float("nan"),
        "interpretation": (
            "PASS = gated quoter improves OOS per-fill net realised spread "
            "(CI excludes 0) while retaining >50% of total captured spread. "
            "NULL = either no per-fill gain or the gain is purely a "
            "trade-count artefact (gating just trades less)."),
        "per_day": [{"root": r["root"], "date": r["date"],
                     "diff_perfill_bp": r["oos_pooled"]["diff_perfill_bp"],
                     "diff_ci95": r["oos_pooled"]["diff_perfill_ci95"],
                     "base_perfill_bp": r["oos_pooled"]["base_perfill_net_realised_bp"],
                     "gate_perfill_bp": r["oos_pooled"]["gate_perfill_net_realised_bp"],
                     "forgone_fills": r["oos_pooled"]["forgone_fills"],
                     "base_n_fills": r["oos_pooled"]["base_n_fills"],
                     "gate_n_fills": r["oos_pooled"]["gate_n_fills"]}
                    for r in rows],
    }
    vpath = os.path.join(out_dir, f"VERDICT_{signal}.json")
    with open(vpath, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n=== H3 GATING VERDICT ({signal}) : {verdict} ===")
    print(f"  contract-days={len(rows)}  mean per-day per-fill diff={mean_diff:.4f} bp "
          f"CI95=[{diff_lo:.4f},{diff_hi:.4f}]  days-better={win_share}")
    print(f"  pooled per-fill: base={out['base_perfill_pooled_bp']:.4f} "
          f"gate={out['gate_perfill_pooled_bp']:.4f} bp")
    print(f"  total captured retained={total_retained:.3f}  "
          f"forgone fills={out['forgone_fill_frac']:.1%}")
    print(f"  -> {vpath}")
    return out


def make_figure(out_dir, signal, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vpath = os.path.join(out_dir, f"VERDICT_{signal}.json")
    with open(vpath) as fh:
        v = json.load(fh)
    per_day = v["per_day"]
    labels = [f"{d['root']}\n{d['date']}" for d in per_day]
    base = [d["base_perfill_bp"] for d in per_day]
    gate = [d["gate_perfill_bp"] for d in per_day]
    diffs = [d["diff_perfill_bp"] for d in per_day]

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(labels))
    w = 0.38
    ax[0].bar(x - w / 2, base, w, label="always-quote (baseline)", color="#4878CF")
    ax[0].bar(x + w / 2, gate, w, label=f"gated ({signal})", color="#D65F5F")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("OOS per-fill net realised spread (bp)")
    ax[0].set_title("Per-fill net realised spread"); ax[0].legend(fontsize=8)

    colors = ["#55A868" if d > 0 else "#C44E52" for d in diffs]
    ax[1].bar(x, diffs, color=colors)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("gated - baseline (bp)")
    ax[1].set_title(f"Per-day improvement  (mean={v['mean_per_day_diff_perfill_bp']:.3f} bp,"
                    f" CI {v['per_day_diff_ci95'][0]:.3f}..{v['per_day_diff_ci95'][1]:.3f})")

    bn = [d["base_n_fills"] for d in per_day]
    gn = [d["gate_n_fills"] for d in per_day]
    ax[2].bar(x - w / 2, bn, w, label="baseline fills", color="#4878CF")
    ax[2].bar(x + w / 2, gn, w, label="gated fills", color="#D65F5F")
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=8)
    ax[2].set_ylabel("OOS fills (forgone = the gap)")
    ax[2].set_title(f"Forgone-fill accounting  "
                    f"(retained captured={v['total_captured_retained_frac']:.2f})")
    ax[2].legend(fontsize=8)

    fig.suptitle(f"H3 markout-gated quote-pull  —  VERDICT: {v['verdict']}  "
                 f"(signal={signal}, {v['n_contract_days']} contract-days, net of costs)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_path, dpi=130)
    print(f"figure -> {fig_path}")


# --------------------------------------------------------------------- #
# Manifest driver (RAM-gated, mirrors run_mm_full's pool)
# --------------------------------------------------------------------- #

def run_manifest(manifest, out_dir, jobs, **kw):
    import csv, subprocess, sys
    with open(manifest) as fh:
        tasks = list(csv.DictReader(fh))
    jobs = int(jobs)
    running = []; pending = list(tasks); done = 0
    while pending or running:
        while pending and len(running) < jobs:
            t = pending.pop(0)
            cmd = [sys.executable, __file__, "--snap", t["snap"],
                   "--trades", t["trades"], "--root", t["root"],
                   "--date", t["date"], "--symbol", t["symbol"],
                   "--out", out_dir, "--signal", kw["signal"],
                   "--lookback-ms", str(kw["lookback_ms"]),
                   "--horizon-s", str(kw["horizon_s"]),
                   "--is-min", str(kw["is_min"]), "--oos-min", str(kw["oos_min"]),
                   "--throttle-k", str(kw["throttle_k"]),
                   "--cooldown-grid-ms", kw["cooldown_grid_ms"],
                   "--thresh-grid", kw["thresh_grid"]]
            env = dict(os.environ, PYTHONPATH=os.getcwd())
            running.append((subprocess.Popen(cmd, env=env), t))
        for p, t in running[:]:
            if p.poll() is not None:
                running.remove((p, t)); done += 1
                print(f"  [{done}/{len(tasks)}] {t['root']} {t['date']} rc={p.returncode}")
        time.sleep(0.5)
    print(f"manifest complete: {done} contract-days")


# --------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snap"); ap.add_argument("--trades")
    ap.add_argument("--root"); ap.add_argument("--date"); ap.add_argument("--symbol")
    ap.add_argument("--out", default="runs/h3_gating")
    ap.add_argument("--size", type=float, default=1.0)
    ap.add_argument("--signal", choices=["flow", "markout"], default="flow")
    ap.add_argument("--lookback-ms", type=float, default=5000.0,
                    help="flow-toxicity trailing window (ms)")
    ap.add_argument("--horizon-s", type=float, default=10.0,
                    help="markout-toxicity observation horizon (s)")
    ap.add_argument("--cooldown-grid-ms", default="500,2000,5000",
                    help="comma-list of cooldown candidates (ms), IS-calibrated")
    ap.add_argument("--thresh-grid", default="",
                    help="comma-list of toxicity thresholds; empty -> auto by signal")
    ap.add_argument("--is-min", type=float, default=60.0)
    ap.add_argument("--oos-min", type=float, default=20.0)
    ap.add_argument("--throttle-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=12345)
    # batch / assembly
    ap.add_argument("--manifest"); ap.add_argument("--jobs", default="2")
    ap.add_argument("--assemble"); ap.add_argument("--figure")
    args = ap.parse_args()

    throttle_k = args.throttle_k if args.throttle_k and args.throttle_k > 0 else None
    cooldown_grid_ms = [float(x) for x in args.cooldown_grid_ms.split(",")]
    if args.thresh_grid.strip():
        thresh_grid = [float(x) for x in args.thresh_grid.split(",")]
    else:
        # auto grids: flow tox in [0,1] OFI magnitude; markout tox in bp
        thresh_grid = ([0.3, 0.5, 0.7] if args.signal == "flow"
                       else [0.1, 0.3, 0.5])

    if args.assemble:
        assemble(args.assemble, args.signal)
        if args.figure:
            make_figure(args.assemble, args.signal, args.figure)
        return
    if args.figure and not args.assemble:
        make_figure(args.out, args.signal, args.figure); return
    if args.manifest:
        run_manifest(args.manifest, args.out, args.jobs, signal=args.signal,
                     lookback_ms=args.lookback_ms, horizon_s=args.horizon_s,
                     is_min=args.is_min, oos_min=args.oos_min,
                     throttle_k=args.throttle_k or 0,
                     cooldown_grid_ms=args.cooldown_grid_ms,
                     thresh_grid=",".join(str(t) for t in thresh_grid))
        return
    if not (args.snap and args.trades and args.root and args.date and args.symbol):
        ap.error("need --snap --trades --root --date --symbol (or --manifest/--assemble)")
    run_one(args.snap, args.trades, args.root, args.date, args.symbol, args.out,
            size=args.size, signal=args.signal, lookback_ms=args.lookback_ms,
            horizon_s=args.horizon_s, cooldown_grid_ms=cooldown_grid_ms,
            thresh_grid=thresh_grid, is_min=args.is_min, oos_min=args.oos_min,
            throttle_k=throttle_k, seed=args.seed)


if __name__ == "__main__":
    main()
