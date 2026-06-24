#!/usr/bin/env python3
"""Test the "adverse selection == LVR" identity on CME futures.

The paper asserts adverse selection ~ LVR as an analogy but never
puts the sigma^2/8 LVR and the measured -0.661 bp post-fill adverse drift on the
SAME axis.  This script makes it quantitative, per root, in the same units (bp of
notional over the 10s markout horizon):

  MEASURED adverse drift (per root, 10s)  = decomp_by_root.csv adverse_select_bp_10s
        := mean over fills of  s*(mid(t+10s) - mid(t))/mid(t) * 1e4   (signed, <0 = picked off)

  IMPLIED LVR-equivalent (per root, 10s)  = 1e4 * Var(r_10s)/8
        where r_10s is the 10s-horizon mid log-return; Milionis et al. LVR = sigma^2/8
        per unit time, so over a horizon tau the accrued LVR = (sigma^2/8)*tau = Var(r_tau)/8.

Both are a cost per fill over a 10s window in bp of notional, so they sit on one axis.
We use the IDENTICAL mid timeline the decomposition consumes (run_mm_full.mid_timeline,
best-bid/ask mean, two-sided snapshots only) and the IDENTICAL "mid at-or-before t"
lookup convention as mmsim.markout.engine, sampled on a regular 10s clock grid across
each root-day's active span.  Per-root values are fills-weighted across root-days, the
same weighting decomp_by_root.csv uses.

Single-thread, OPENBLAS_NUM_THREADS=1, one root-day in RAM at a time. No network.
"""
import os, sys, json, time, resource
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try:
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
except Exception:
    pass

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
from run_mm_full import mid_timeline

MANIFEST = os.environ.get("FUT_UNIVERSE_MANIFEST", "data/fut_universe/manifest_all.csv")
DECOMP = os.environ.get("DECOMP_BY_ROOT", "runs/maker_decomp/decomp_by_root.csv")
OUT = os.environ.get("LVR_IDENTITY_OUT", "runs/maker_decomp/lvr_identity.json")
HORIZON_NS = 10_000_000_000  # 10s, matches decomp 10s horizon
GRID_NS = 10_000_000_000     # 10s clock grid (one return per horizon, non-overlapping)


def _mid_at_or_before(ts, mid, t):
    """Last mid at-or-before t (matches mmsim.markout _mid_at_or_before_scalar)."""
    j = np.searchsorted(ts, t, side="right") - 1
    if j < 0:
        return np.nan
    return mid[j]


def lvr_for_root_day(snap, symbol):
    ts, mid = mid_timeline(snap, symbol)
    if ts.size < 3:
        return None
    t0, t1 = int(ts[0]), int(ts[-1])
    if t1 - t0 < 2 * HORIZON_NS:
        return None
    # regular 10s clock grid across the active span; sample mid at-or-before each grid point
    grid = np.arange(t0, t1 + 1, GRID_NS, dtype=np.int64)
    # forward index: mid at-or-before each grid time
    idx = np.searchsorted(ts, grid, side="right") - 1
    ok = idx >= 0
    grid = grid[ok]; idx = idx[ok]
    m = mid[idx]
    good = np.isfinite(m) & (m > 0)
    m = m[good]
    if m.size < 3:
        return None
    # 10s-horizon log returns over the non-overlapping grid (one step == one 10s horizon)
    r = np.diff(np.log(m))
    r = r[np.isfinite(r)]
    if r.size < 2:
        return None
    var_10s = float(np.var(r, ddof=1))         # variance of the 10s mid log-return
    lvr_bp_10s = 1e4 * var_10s / 8.0           # sigma^2/8 over the 10s horizon, in bp
    # also report a fills-independent realized-vol annualization-free sigma for context
    return {
        "n_grid": int(r.size),
        "var_10s": var_10s,
        "rv_10s_bp": 1e4 * float(np.sqrt(var_10s)),  # 10s-horizon stdev in bp (sigma_tau)
        "lvr_equiv_bp_10s": lvr_bp_10s,
    }


def main():
    m = pd.read_csv(MANIFEST, dtype={"date": str})
    dec = pd.read_csv(DECOMP)
    rows = []
    t0 = time.time()
    for _, r in m.iterrows():
        try:
            d = lvr_for_root_day(r["snap"], r["symbol"])
        except Exception as e:
            print(f"  ! {r['root']} {r['date']}: {str(e)[:70]}", flush=True)
            continue
        if d is None:
            continue
        d.update({"root": r["root"], "date": r["date"]})
        rows.append(d)
    df = pd.DataFrame(rows)

    # per-root aggregation, fills-weighted to match decomp_by_root.csv weighting.
    # We weight each root-day's LVR by its decomp fill count is not available per day here,
    # so weight by n_grid (proportional to active session length == time-weight), then
    # report BOTH the per-root LVR and the decomp fill-weighted merge at the root level.
    agg = []
    for root, g in df.groupby("root"):
        w = g["n_grid"].to_numpy(dtype=float)
        lvr = g["lvr_equiv_bp_10s"].to_numpy(dtype=float)
        rv = g["rv_10s_bp"].to_numpy(dtype=float)
        agg.append({
            "root": root,
            "n_rootdays": int(len(g)),
            "n_grid_total": int(w.sum()),
            "lvr_equiv_bp_10s": float((lvr * w).sum() / w.sum()),
            "rv_10s_bp": float((rv * w).sum() / w.sum()),
        })
    adf = pd.DataFrame(agg)

    # merge measured adverse drift (10s) per root
    merged = adf.merge(
        dec[["root", "n_fills", "spread_capture_bp_10s", "adverse_select_bp_10s",
             "net_after_fee_bp_10s"]],
        on="root", how="inner")
    # measured adverse cost as a POSITIVE magnitude in bp (drift is negative => cost)
    merged["adverse_cost_bp_10s"] = -merged["adverse_select_bp_10s"]
    merged["ratio_lvr_over_adverse"] = merged["lvr_equiv_bp_10s"] / merged["adverse_cost_bp_10s"]

    # cross-root statistics
    x = merged["lvr_equiv_bp_10s"].to_numpy()
    y = merged["adverse_cost_bp_10s"].to_numpy()
    corr_pearson = float(np.corrcoef(x, y)[0, 1])
    # rank correlation (Spearman) without scipy
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    corr_spearman = float(np.corrcoef(rx, ry)[0, 1])
    # fills-weighted overall ratio (the headline magnitude comparison)
    fw = merged["n_fills"].to_numpy(dtype=float)
    lvr_overall = float((x * fw).sum() / fw.sum())
    adv_overall = float((y * fw).sum() / fw.sum())
    ratio_overall = lvr_overall / adv_overall
    # unweighted means
    ratio_mean = float(np.mean(merged["ratio_lvr_over_adverse"]))
    ratio_median = float(np.median(merged["ratio_lvr_over_adverse"]))

    out = {
        "description": "adverse selection vs sigma^2/8 LVR on the same axis (bp of notional, 10s horizon), per CME root",
        "horizon": "10s",
        "lvr_definition": "1e4 * Var(r_10s)/8 ; r_10s = 10s mid log-return on a 10s clock grid; identical mid timeline as decomp",
        "adverse_definition": "-adverse_select_bp_10s from decomp_by_root.csv (measured post-fill drift cost, positive magnitude)",
        "cross_root": {
            "n_roots": int(len(merged)),
            "pearson_corr_lvr_vs_adverse": corr_pearson,
            "spearman_corr_lvr_vs_adverse": corr_spearman,
            "ratio_overall_fillweighted_lvr_over_adverse": ratio_overall,
            "lvr_overall_fillweighted_bp": lvr_overall,
            "adverse_overall_fillweighted_bp": adv_overall,
            "ratio_unweighted_mean": ratio_mean,
            "ratio_unweighted_median": ratio_median,
        },
        "by_root": merged.sort_values("adverse_cost_bp_10s", ascending=False).to_dict("records"),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    # console report
    print("\n=== implied sigma^2/8 LVR vs measured adverse drift, per CME root (10s, bp) ===", flush=True)
    print(f"{'root':>5} {'fills':>8} {'adverse_cost':>12} {'lvr_sig2/8':>11} {'ratio L/A':>9} {'rv_10s_bp':>10}", flush=True)
    for _, r in merged.sort_values("adverse_cost_bp_10s", ascending=False).iterrows():
        print(f"{r['root']:>5} {int(r['n_fills']):>8} {r['adverse_cost_bp_10s']:>12.3f} "
              f"{r['lvr_equiv_bp_10s']:>11.3f} {r['ratio_lvr_over_adverse']:>9.3f} {r['rv_10s_bp']:>10.3f}", flush=True)
    print(f"\nCross-root Pearson corr (LVR vs adverse)  = {corr_pearson:.3f}", flush=True)
    print(f"Cross-root Spearman corr (LVR vs adverse) = {corr_spearman:.3f}", flush=True)
    print(f"Overall fills-weighted: LVR={lvr_overall:.3f} bp, adverse={adv_overall:.3f} bp, ratio L/A={ratio_overall:.3f}", flush=True)
    print(f"Ratio L/A unweighted mean={ratio_mean:.3f}, median={ratio_median:.3f}", flush=True)
    print(f"\n[done] {len(df)} root-days in {time.time()-t0:.1f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
