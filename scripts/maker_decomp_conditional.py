#!/usr/bin/env python3
"""Conditional maker-P&L decomposition: root x (session-open / mid-session) x
(high-vol / calm regime), on true-signed CME futures.

Forks scripts/maker_pnl_decomp.py.  Same engine path (fast_sim.spec_touch +
markout.compute_markout, MAKER_FEE_BP = 0.015).  Per root-day, after the
markout, each fill is tagged with:

  SESSION  - 'open' if the fill lands in the first OPEN_MINUTES of that
             root-day's fill timeline (the open auction + early RTH), else
             'mid'.  A purely-causal intraday bucket; no calendar lookahead.
  REGIME   - the ONLINE CAUSAL vol-regime label (h4_regime pipeline): 1-min
             mid bars -> ohlcv features -> K=4 Gaussian HMM fit on the FIRST
             TRAIN_MINUTES of bars (the IS warmup), vol-ordered so state K-1 =
             high-vol, then FORWARD-FILTERED (online) labels using only bars
             <= t.  The label at the fill is the online label of the snapshot
             bar containing it.  NO look-ahead: filtering consults no future
             bar; the HMM params are frozen from the in-day warmup.

We aggregate fill-weighted spread / adverse / net-after-fee per
(root x session x regime) cell and per cell pooled across root-days.

HONESTY (locked): the headline rigour is PBO 0.543 (resolution-independent)
and the raw effect sizes (the M1 inventory-skew family has 0 raw-significant
cells before any correction); 1126 nominal cells over the grid.  BH/BHY add
0/1126 only as a consistency note -- with M=1000 permutations the smallest
attainable p (1/1001) sits above the BH/BHY rank-1 thresholds, so 0/1126 is
mechanically guaranteed by the permutation resolution rather than evidence.
Any surviving (root x high-vol-open) cell with net-after-fee > 0 is reported
as an EXPLORATORY SENSITIVITY with its fill count, NOT a multiplicity-corrected
pass.  A single conditional positive is a slice of an already-null family, not
a discovery.  If nothing survives, the clean conditional null is documented.

Single-thread, one root-day in RAM at a time, RLIMIT_AS 12 GB.  NO parallel.
"""
import os, sys, json, time, resource
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
except Exception:
    pass

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
from mmsim.sim import fast_sim as _F
from mmsim.markout.engine import compute_markout
from run_mm_full import mid_timeline
import h4_regime as H4

MANIFEST = os.environ.get("FUT_UNIVERSE_MANIFEST",
                          "data/fut_universe/manifest_all.csv")
OUT = os.environ.get("MAKER_DECOMP_OUT", "runs/maker_decomp")
MAKER_FEE_BP = 0.015        # realistic CME maker fee, no rebate (matches decomp)
SIZE = 1.0
_MIN = 60 * 1_000_000_000   # 1 minute in ns
OPEN_MINUTES = 60           # first 60 min of the day's fills = 'session open'
TRAIN_MINUTES = 60          # HMM in-day warmup span (causal IS)
K = 4
SESSIONS = ("open", "mid")
REGIMES = ("calm", "hivol")  # calm = states 0..K-2 pooled; hivol = state K-1


def _online_regime_per_snapshot(snap_path, symbol):
    """Online causal high-vol-regime flag per snapshot ts for one root-day.
    Returns (snap_ts[int64], is_hivol[bool]) or (None, None) if too few bars.
    Mirrors h4_regime: bars -> features -> HMM fit on the first TRAIN_MINUTES
    of valid bars -> vol-order -> forward-filtered online labels."""
    ts, mid = H4.day_mid_timeline(snap_path, symbol)
    if ts.size == 0:
        return None, None
    bars = H4.build_bars(ts, mid)
    if bars is None:
        return None, None
    feat, valid = H4.ohlcv_features(bars)
    if valid.sum() < (2 * K + 5):
        return None, None
    bar_start = bars["bar_start_ns"]
    # in-day training span = bars whose start is within the first TRAIN_MINUTES
    t0 = int(bar_start[0])
    train_bar = (bar_start <= t0 + TRAIN_MINUTES * _MIN) & valid
    if train_bar.sum() < (2 * K + 2):
        # widen to the first 1/3 of valid bars if the 60-min warmup is too thin
        vidx = np.where(valid)[0]
        cut = vidx[max(1, len(vidx) // 3)]
        train_bar = valid.copy()
        train_bar[cut:] = False
        if train_bar.sum() < (2 * K + 2):
            return None, None
    mu = feat[train_bar].mean(axis=0)
    sd = feat[train_bar].std(axis=0) + 1e-9
    Xstd = (feat - mu) / sd
    try:
        model = H4.fit_hmm(Xstd[train_bar], K=K)
    except Exception:
        return None, None
    remap = H4.vol_state_order(model, vol_idx=1)
    hi_state = K - 1
    labels_full = np.full(feat.shape[0], -1, dtype=int)
    Xv = Xstd[valid]
    lab_v = H4.online_labels(model, Xv, remap)
    labels_full[valid] = lab_v
    # forward-fill warmup (-1) bars with the first known label (calm by
    # construction; never uses future info)
    cur = int(lab_v[0]) if lab_v.size else 0
    for i in range(labels_full.size):
        if labels_full[i] < 0:
            labels_full[i] = cur
        else:
            cur = labels_full[i]
    # map each snapshot ts -> its bar -> hi-vol flag
    bidx = np.searchsorted(bar_start, ts, side="right") - 1
    bidx = np.clip(bidx, 0, labels_full.size - 1)
    is_hivol = (labels_full[bidx] == hi_state)
    return ts.astype(np.int64), is_hivol.astype(bool)


def _cell_accum():
    return {"spread_sum": 0.0, "adverse_sum": 0.0, "markout_sum": 0.0,
            "n": 0}


def decomp_root_day(snap, trades, symbol):
    """Per-fill spread/adverse/markout (10s) + session + online regime tags,
    aggregated into the four (session x regime) cells.  Returns a dict of
    cell -> accum, or None."""
    s_ts, s_mid = mid_timeline(snap, symbol)
    if s_ts.size == 0:
        return None
    day = _F.read_day_arrays(snap, trades, symbol=symbol, depth_k=10)
    w = _F.window_arrays_from_day(day, int(s_ts[0]), int(s_ts[-1]))
    res = _F.simulate_from_arrays(w, lambda ww: _F.spec_touch(ww, SIZE), SIZE)
    if not res.fills:
        return None
    mk = compute_markout(res.fills, s_ts, s_mid)
    markout = mk.markout_10s
    adverse = mk.adverse_10s
    spread = markout - adverse
    fill_ts = np.array([f.ts_ns for f in res.fills], dtype=np.int64)
    good = np.isfinite(markout) & np.isfinite(adverse)

    # session tag: first OPEN_MINUTES of the day's fills = 'open'
    if fill_ts.size == 0:
        return None
    t_open_end = int(fill_ts.min()) + OPEN_MINUTES * _MIN
    is_open = fill_ts <= t_open_end

    # online causal regime per snapshot, mapped to each fill
    r_ts, r_hi = _online_regime_per_snapshot(snap, symbol)
    if r_ts is None:
        # regime undefined for this root-day (too few bars) -> skip cleanly;
        # such days are reported as "regime-undefined" not silently folded in.
        return {"_regime_undefined": True}
    idx = np.searchsorted(r_ts, fill_ts, side="right") - 1
    idx = np.clip(idx, 0, r_hi.size - 1)
    fill_hivol = r_hi[idx]

    cells = {(se, rg): _cell_accum() for se in SESSIONS for rg in REGIMES}
    for i in range(fill_ts.size):
        if not good[i]:
            continue
        se = "open" if is_open[i] else "mid"
        rg = "hivol" if fill_hivol[i] else "calm"
        c = cells[(se, rg)]
        c["spread_sum"] += float(spread[i])
        c["adverse_sum"] += float(adverse[i])
        c["markout_sum"] += float(markout[i])
        c["n"] += 1
    return cells


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_csv(MANIFEST, dtype={"date": str})
    only = os.environ.get("ONLY_ROOT")
    if only:
        m = m[m["root"] == only]

    # accumulate per (root, session, regime) across root-days
    from collections import defaultdict
    acc = defaultdict(_cell_accum)          # key (root, session, regime)
    n_rd = 0
    n_regime_undef = 0
    t0 = time.time()
    for _, r in m.iterrows():
        try:
            cells = decomp_root_day(r["snap"], r["trades"], r["symbol"])
        except Exception as e:
            print(f"  ! {r['root']} {r['date']}: {str(e)[:70]}", flush=True)
            continue
        if cells is None:
            continue
        if cells.get("_regime_undefined"):
            n_regime_undef += 1
            continue
        n_rd += 1
        for (se, rg), c in cells.items():
            a = acc[(r["root"], se, rg)]
            a["spread_sum"] += c["spread_sum"]
            a["adverse_sum"] += c["adverse_sum"]
            a["markout_sum"] += c["markout_sum"]
            a["n"] += c["n"]
        if n_rd % 25 == 0:
            print(f"  .. {n_rd} root-days, {time.time()-t0:.0f}s", flush=True)

    # build the per-cell table (fill-weighted means in bp)
    rows = []
    for (root, se, rg), a in acc.items():
        if a["n"] == 0:
            continue
        spread_bp = a["spread_sum"] / a["n"] * 1e4
        adverse_bp = a["adverse_sum"] / a["n"] * 1e4
        net_bp = a["markout_sum"] / a["n"] * 1e4
        net_fee_bp = net_bp - MAKER_FEE_BP
        rows.append({
            "root": root, "session": se, "regime": rg,
            "n_fills": a["n"],
            "spread_capture_bp": round(spread_bp, 4),
            "adverse_select_bp": round(adverse_bp, 4),
            "net_markout_bp": round(net_bp, 4),
            "net_after_fee_bp": round(net_fee_bp, 4),
        })
    df = pd.DataFrame(rows).sort_values(
        ["regime", "session", "net_after_fee_bp"], ascending=[True, True, False])
    df.to_csv(f"{OUT}/decomp_by_regime.csv", index=False)

    # the locked exploratory axis: root x high-vol x session-open
    hvo = df[(df["regime"] == "hivol") & (df["session"] == "open")].copy()
    survivors = hvo[hvo["net_after_fee_bp"] > 0].sort_values(
        "net_after_fee_bp", ascending=False)

    # pooled cell summaries (fill-weighted across roots) for each session x regime
    pooled = {}
    for se in SESSIONS:
        for rg in REGIMES:
            keys = [(root, se, rg) for root in df["root"].unique()]
            tot_n = sum(acc[k]["n"] for k in keys if k in acc)
            if tot_n == 0:
                continue
            sp = sum(acc[k]["spread_sum"] for k in keys if k in acc) / tot_n * 1e4
            ad = sum(acc[k]["adverse_sum"] for k in keys if k in acc) / tot_n * 1e4
            mk = sum(acc[k]["markout_sum"] for k in keys if k in acc) / tot_n * 1e4
            pooled[f"{se}_{rg}"] = {
                "n_fills": int(tot_n),
                "spread_capture_bp": round(sp, 4),
                "adverse_select_bp": round(ad, 4),
                "net_markout_bp": round(mk, 4),
                "net_after_fee_bp": round(mk - MAKER_FEE_BP, 4),
            }

    summary = {
        "maker_fee_bp": MAKER_FEE_BP,
        "axes": {"session": list(SESSIONS), "regime": list(REGIMES),
                 "open_minutes": OPEN_MINUTES, "hmm_K": K,
                 "hmm_train_minutes": TRAIN_MINUTES,
                 "regime_label": "online causal forward-filtered (no lookahead)"},
        "n_rootdays_used": n_rd,
        "n_rootdays_regime_undefined": n_regime_undef,
        "n_cells": len(rows),
        "pooled_cells": pooled,
        "hivol_open_cells": hvo.to_dict("records"),
        "hivol_open_survivors_net_after_fee_pos": survivors.to_dict("records"),
        "honesty": ("Survivors are an EXPLORATORY SENSITIVITY (a slice of the "
                    "already-null family), reported with fill counts; NOT a "
                    "multiplicity-corrected pass. Null rests on PBO 0.543 and "
                    "raw effect sizes; BH/BHY 0/1126 is a consistency note only "
                    "(mechanically guaranteed by the M=1000 permutation floor "
                    "sitting above the BH/BHY rank-1 thresholds)."),
    }
    with open(f"{OUT}/decomp_by_regime_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("\n=== Conditional decomposition: root x session x regime "
          "(10s, fill-weighted, net-after-fee at 0.015 bp) ===", flush=True)
    print(f"root-days used: {n_rd} (regime-undefined skipped: {n_regime_undef})",
          flush=True)
    print("\nPOOLED cells (fill-weighted across roots):", flush=True)
    print(f"{'cell':>12} {'fills':>9} {'spread':>8} {'adverse':>8} "
          f"{'net_mk':>8} {'net-fee':>8}", flush=True)
    for k, v in pooled.items():
        print(f"{k:>12} {v['n_fills']:>9} {v['spread_capture_bp']:>8.3f} "
              f"{v['adverse_select_bp']:>8.3f} {v['net_markout_bp']:>8.3f} "
              f"{v['net_after_fee_bp']:>8.3f}", flush=True)
    print(f"\nHIGH-VOL x SESSION-OPEN cells (the locked exploratory axis): "
          f"{len(hvo)} roots", flush=True)
    print(f"  survivors (net-after-fee > 0): {len(survivors)}", flush=True)
    for _, r in survivors.iterrows():
        print(f"    {r['root']:>4} net-after-fee {r['net_after_fee_bp']:+.3f} bp "
              f"(n_fills={int(r['n_fills'])})  [EXPLORATORY, not a pass]",
              flush=True)
    if survivors.empty:
        print("    NONE -- clean conditional null on root x high-vol-open.",
              flush=True)

    # the publication figure is generated by paper/figgen/fig_conditional_sweep.py
    # (vector PDF into paper/figs) from the CSV + summary this script writes.
    print(f"\n[done] {n_rd} root-days in {time.time()-t0:.0f}s -> {OUT}/"
          f"decomp_by_regime.csv + decomp_by_regime_summary.json", flush=True)


if __name__ == "__main__":
    main()
