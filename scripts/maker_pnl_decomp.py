#!/usr/bin/env python3
"""Ground-truth maker-P&L decomposition on tick-bound CME futures.

Moallemi-Yuan-style split of the passive (queue-only touch) maker's per-fill economics into
SPREAD CAPTURE vs ADVERSE SELECTION, on REAL true-aggressor-signed depth-10 data. The literature's
queue/adverse-selection theory is equities / large-tick (Moallemi-Yuan 2016); futures (tick-bound,
deep) is under-modeled. We have the true signs to do it at scale.

Per fill, marked at horizon tau (10s, 60s), with s = maker side (+1 bid, -1 ask):
  spread_capture = s*(mid0 - fill_px)/mid0       (half-spread earned at fill; >=0 at the touch)
  adverse_select = s*(mid_tau - mid0)/mid0       (post-fill drift; <0 = picked off)
  net_markout    = spread_capture + adverse_select = s*(mid_tau - fill_px)/mid0
  net_after_fee  = net_markout - taker_fee_bp/1e4  (CME maker pays fee, no rebate)
Derived from mmsim.markout.engine.compute_markout (markout_tau, adverse_tau): spread = markout - adverse.

Reuses the fast_sim engine. Single-thread, one root-day in RAM at a time.
"""
import os, sys, json, time, resource
from pathlib import Path
import numpy as np
import pandas as pd

try:
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))  # VAS headroom for 9M-trade days; RSS stays ~2 GB
except Exception:
    pass

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
from mmsim.sim import fast_sim as _F
from mmsim.markout.engine import compute_markout
from run_mm_full import mid_timeline

MANIFEST = os.environ.get("FUT_UNIVERSE_MANIFEST", "data/fut_universe/manifest_all.csv")
OUT = os.environ.get("MAKER_DECOMP_OUT", "runs/maker_decomp")
MAKER_FEE_BP = 0.015   # realistic CME maker fee, no rebate
SIZE = 1.0


_GRID_NS = 10_000_000_000  # 10s clock grid (matches lvr_identity sigma_tau)


def realized_vol_10s_bp(s_ts, s_mid):
    """10s-horizon mid log-return stdev in bp (sigma_tau), matching lvr_identity binning."""
    ts = np.asarray(s_ts, dtype=np.int64)
    if ts.size < 3:
        return float('nan')
    t0, t1 = int(ts[0]), int(ts[-1])
    if t1 - t0 < 2 * _GRID_NS:
        return float('nan')
    grid = np.arange(t0, t1 + 1, _GRID_NS, dtype=np.int64)
    idx = np.searchsorted(ts, grid, side="right") - 1
    ok = idx >= 0
    m = np.asarray(s_mid)[idx[ok]]
    m = m[np.isfinite(m) & (m > 0)]
    if m.size < 3:
        return float('nan')
    r = np.diff(np.log(m)); r = r[np.isfinite(r)]
    if r.size < 2:
        return float('nan')
    return 1e4 * float(np.sqrt(np.var(r, ddof=1)))


def decomp_root_day(snap, trades, symbol):
    s_ts, s_mid = mid_timeline(snap, symbol)
    if s_ts.size == 0:
        return None
    day = _F.read_day_arrays(snap, trades, symbol=symbol, depth_k=10)
    w = _F.window_arrays_from_day(day, int(s_ts[0]), int(s_ts[-1]))
    res = _F.simulate_from_arrays(w, lambda ww: _F.spec_touch(ww, SIZE), SIZE)
    if not res.fills:
        return None
    mk = compute_markout(res.fills, s_ts, s_mid)
    out = {'n_fills': len(res.fills)}
    for tau in ('10s', '60s'):
        markout = getattr(mk, f'markout_{tau}')
        adverse = getattr(mk, f'adverse_{tau}')
        spread = markout - adverse                      # = s*(mid0 - px)/mid0
        good = np.isfinite(markout) & np.isfinite(adverse)
        out[f'spread_capture_bp_{tau}'] = float(np.nanmean(spread[good]) * 1e4)
        out[f'adverse_select_bp_{tau}'] = float(np.nanmean(adverse[good]) * 1e4)
        out[f'net_markout_bp_{tau}'] = float(np.nanmean(markout[good]) * 1e4)
        out[f'net_after_fee_bp_{tau}'] = float(np.nanmean(markout[good]) * 1e4 - MAKER_FEE_BP)
        out[f'n_good_{tau}'] = int(good.sum())
    out['rv_10s_bp'] = realized_vol_10s_bp(s_ts, s_mid)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_csv(MANIFEST, dtype={'date': str})
    only = os.environ.get('ONLY_ROOT')
    if only:
        m = m[m['root'] == only]
    rows = []
    t0 = time.time()
    for _, r in m.iterrows():
        try:
            d = decomp_root_day(r['snap'], r['trades'], r['symbol'])
        except Exception as e:
            print(f"  ! {r['root']} {r['date']}: {str(e)[:60]}", flush=True)
            continue
        if d is None:
            continue
        d.update({'root': r['root'], 'date': r['date']})
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUT}/decomp_by_rootday.parquet", index=False)

    # per-root fill-weighted aggregation
    def wavg(g, col, wcol):
        w = g[wcol].to_numpy(); v = g[col].to_numpy(); s = w.sum()
        return float((v * w).sum() / s) if s > 0 else float('nan')

    def wavg_nan(g, col, wcol):
        w = g[wcol].to_numpy(dtype=float); v = g[col].to_numpy(dtype=float)
        good = np.isfinite(v) & np.isfinite(w) & (w > 0)
        s = w[good].sum()
        return float((v[good] * w[good]).sum() / s) if s > 0 else float('nan')
    agg = []
    for root, g in df.groupby('root'):
        rowd = {'root': root, 'n_rootdays': len(g), 'n_fills': int(g['n_fills'].sum())}
        for tau in ('10s', '60s'):
            for comp in ('spread_capture', 'adverse_select', 'net_markout', 'net_after_fee'):
                rowd[f'{comp}_bp_{tau}'] = wavg(g, f'{comp}_bp_{tau}', f'n_good_{tau}')
        rowd['rv_10s_bp'] = wavg_nan(g, 'rv_10s_bp', 'n_good_10s')
        agg.append(rowd)
    adf = pd.DataFrame(agg).sort_values('net_after_fee_bp_10s', ascending=False)
    adf.to_csv(f"{OUT}/decomp_by_root.csv", index=False)

    # overall (fill-weighted across all roots)
    overall = {}
    for tau in ('10s', '60s'):
        for comp in ('spread_capture', 'adverse_select', 'net_markout', 'net_after_fee'):
            overall[f'{comp}_bp_{tau}'] = wavg(df, f'{comp}_bp_{tau}', f'n_good_{tau}')
    overall['rv_10s_bp'] = wavg_nan(df, 'rv_10s_bp', 'n_good_10s')
    overall['n_fills'] = int(df['n_fills'].sum())
    overall['n_rootdays'] = int(len(df))

    print("\n=== Maker-P&L decomposition (touch quoter, true-signed CME, per-fill, 10s horizon) ===", flush=True)
    print(f"{'root':>5} {'fills':>9} {'spread':>8} {'adverse':>8} {'net_mk':>8} {'net-fee':>8}", flush=True)
    for _, r in adf.iterrows():
        print(f"{r['root']:>5} {int(r['n_fills']):>9} {r['spread_capture_bp_10s']:>8.3f} "
              f"{r['adverse_select_bp_10s']:>8.3f} {r['net_markout_bp_10s']:>8.3f} {r['net_after_fee_bp_10s']:>8.3f}", flush=True)
    print(f"{'ALL':>5} {overall['n_fills']:>9} {overall['spread_capture_bp_10s']:>8.3f} "
          f"{overall['adverse_select_bp_10s']:>8.3f} {overall['net_markout_bp_10s']:>8.3f} "
          f"{overall['net_after_fee_bp_10s']:>8.3f}", flush=True)

    with open(f"{OUT}/decomp_summary.json", 'w') as fh:
        json.dump({'maker_fee_bp': MAKER_FEE_BP, 'overall': overall,
                   'by_root': adf.to_dict('records')}, fh, indent=2, default=float)
    print(f"\n[done] {len(df)} root-days, {overall['n_fills']} fills in {time.time()-t0:.1f}s -> {OUT}/", flush=True)


if __name__ == '__main__':
    main()
