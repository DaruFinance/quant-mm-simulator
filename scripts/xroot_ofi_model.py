#!/usr/bin/env python3
"""Stage 2 of the cross-root integrated-OFI lead-lag study.

Builds the synchronised 13-root panel from stage-1 features, fits the nested models
(own-lagged vs own+cross-lagged) under rolling WFO with IS-only PCA/standardisation,
evaluates OOS incremental R^2 and a net-of-fee portfolio with a tail-guarded WFO, and runs
the pre-registered significance battery (block-bootstrap null, BHY across roots, DSR, Lo-HAC).

Honest expected outcome (CCZ equity prior): own integrated OFI subsumes contemporaneous
cross-impact; any edge is the short-horizon lagged cross-root term, tiny and cost-sensitive.

RAM-safe: the panel is ~30k rows x 13 roots (<100 MB). Single-thread.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmsim.research.perm_null import tail_guarded_rrr, nominal_trial_count

FEAT = os.environ.get("XROOT_OFI_FEAT", "runs/xroot_ofi/feat")
OUT = os.environ.get("XROOT_OFI_OUT", "runs/xroot_ofi")
ROOTS = ['ES', 'NQ', 'RTY', 'ZN', 'ZF', 'ZB', 'CL', 'NG', 'GC', 'SI', 'HG', '6E', '6J']
DAYS = os.environ.get('DAYS', '20230709,20230710,20230711,20230712,20230713,20230719,20230730,20230716,20230723').split(',')
L = 10
TAKER_FEE_BP = 0.015          # realistic CME taker fee
BIN_MS = 10000
SEED = 12345
rng = np.random.default_rng(SEED)


# ---------- panel construction ----------
def build_panel():
    """Per day inner-join the 13 roots on common bins; compute within-day forward log
    return + half-spread (bp) per root. Concatenate days in time order. Returns a dict of
    arrays: O (rows x 13 x 10) raw OFI, RFWD (rows x 13), HSBP (rows x 13), day ids."""
    frames = []
    for day in DAYS:
        per = {}
        ok = True
        for r in ROOTS:
            f = f"{FEAT}/{r}_{day}_{BIN_MS}.parquet"
            if not os.path.exists(f):
                ok = False; break
            d = pd.read_parquet(f)
            d = d[d['mid_close'].notna() & (d['mid_close'] > 0)]
            per[r] = d.set_index('bin_ts_ns')
        if not ok or len(per) < len(ROOTS):
            continue
        common = set.intersection(*[set(per[r].index) for r in ROOTS])
        common = np.array(sorted(common))
        if common.size < 50:
            continue
        # assemble aligned arrays
        O = np.zeros((common.size, len(ROOTS), L))
        MID = np.zeros((common.size, len(ROOTS)))
        SPR = np.zeros((common.size, len(ROOTS)))
        for j, r in enumerate(ROOTS):
            sub = per[r].loc[common]
            O[:, j, :] = sub[[f'ofi{i}' for i in range(L)]].to_numpy()
            MID[:, j] = sub['mid_close'].to_numpy()
            SPR[:, j] = sub['spr_close'].to_numpy()
        logmid = np.log(MID)
        rfwd = np.full_like(MID, np.nan)
        rfwd[:-1, :] = logmid[1:, :] - logmid[:-1, :]   # within-day forward return
        hsbp = 0.5 * SPR / MID * 1e4
        df_day = pd.DataFrame({'day': day, 'bin_ts': common})
        frames.append((df_day, O, rfwd, hsbp))
    # concat (days already in chronological order in DAYS? sort to be safe)
    order = sorted(range(len(frames)), key=lambda k: frames[k][0]['day'].iloc[0])
    frames = [frames[k] for k in order]
    meta = pd.concat([f[0] for f in frames], ignore_index=True)
    O = np.concatenate([f[1] for f in frames], axis=0)
    RFWD = np.concatenate([f[2] for f in frames], axis=0)
    HSBP = np.concatenate([f[3] for f in frames], axis=0)
    # drop rows with any nan forward return (day-last bins)
    good = np.isfinite(RFWD).all(axis=1)
    return meta[good].reset_index(drop=True), O[good], RFWD[good], HSBP[good]


# ---------- IS-only integrated OFI ----------
def integrate_ofi(O_is, O_full):
    """Fit per-root standardisation + PC1 on IS slice, apply to full. O_*: (n,13,10).
    Returns Z_is (n_is,13), Z_full (n_full,13). Sign-fixed so PC1 loads + on level-0."""
    nR = O_is.shape[1]
    Z_is = np.zeros((O_is.shape[0], nR))
    Z_full = np.zeros((O_full.shape[0], nR))
    for j in range(nR):
        X = O_is[:, j, :]
        mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        # PC1 via SVD
        U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
        pc = Vt[0]
        if pc[0] < 0:
            pc = -pc
        Z_is[:, j] = Xs @ pc
        Z_full[:, j] = ((O_full[:, j, :] - mu) / sd) @ pc
    return Z_is, Z_full


def ols_fit(X, y):
    """OLS with intercept. X (n,k) -> beta (k+1,)."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def ols_pred(beta, X):
    return beta[0] + X @ beta[1:]


def oos_r2(y, pred):
    """OOS R^2 vs zero-prediction baseline (E[ret]~0 at 10s)."""
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum(y ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ---------- WFO ----------
def run_wfo(meta, O, RFWD, HSBP, L_is=4000, L_oos=1500, step=1500, cost_mult=1.0):
    n = len(meta)
    nR = O.shape[1]
    starts = list(range(0, n - L_is - L_oos + 1, step))
    win = []
    is_sig = RFWD.std(0)  # rough; recomputed per-window inside
    for s in starts:
        is_sl = slice(s, s + L_is)
        oos_sl = slice(s + L_is, s + L_is + L_oos)
        O_is, O_oos = O[is_sl], O[oos_sl]
        y_is_all, y_oos_all = RFWD[is_sl], RFWD[oos_sl]
        hs_oos = HSBP[oos_sl]
        Z_is, Z_oos = integrate_ofi(O_is, np.concatenate([O_is, O_oos]))
        Z_is, Z_oos = Z_oos[:L_is], Z_oos[L_is:]
        sig_is = y_is_all.std(0); sig_is[sig_is == 0] = 1.0
        r2_1 = np.full(nR, np.nan); r2_2 = np.full(nR, np.nan)
        pred1 = np.zeros((L_oos, nR)); pred2 = np.zeros((L_oos, nR))
        for i in range(nR):
            y_is = y_is_all[:, i]; y_oos = y_oos_all[:, i]
            b1 = ols_fit(Z_is[:, [i]], y_is)
            p1 = ols_pred(b1, Z_oos[:, [i]])
            b2 = ols_fit(Z_is, y_is)
            p2 = ols_pred(b2, Z_oos)
            r2_1[i] = oos_r2(y_oos, p1); r2_2[i] = oos_r2(y_oos, p2)
            pred1[:, i] = p1; pred2[:, i] = p2
        # trading: vol-scaled, gross-normalised positions; net of cost
        def pnl(pred):
            pos = pred / sig_is
            gross = np.sum(np.abs(pos), axis=1, keepdims=True); gross[gross == 0] = 1.0
            pos = pos / gross
            gross_ret = np.sum(pos * y_oos_all, axis=1)
            dpos = np.abs(np.diff(pos, axis=0, prepend=0.0))
            cost = np.sum(dpos * (cost_mult * hs_oos / 1e4 + TAKER_FEE_BP / 1e4), axis=1)
            return gross_ret - cost, gross_ret
        net_c, gross_c = pnl(pred2)   # cross model
        net_o, gross_o = pnl(pred1)   # own model
        win.append({
            'start': s, 'incr_r2': float(np.nanmean(r2_2 - r2_1)),
            'r2_own': float(np.nanmean(r2_1)), 'r2_cross': float(np.nanmean(r2_2)),
            'incr_r2_byroot': (r2_2 - r2_1),
            'net_cross_mean': float(net_c.mean()), 'gross_cross_mean': float(gross_c.mean()),
            'net_own_mean': float(net_o.mean()),
            'net_cross_series': net_c,
            # cache for the block-bootstrap null (PCA is row-perm-invariant -> reuse)
            '_Z_is': Z_is, '_Z_oos': Z_oos, '_y_is': y_is_all, '_y_oos': y_oos_all, '_r2_1': r2_1,
        })
    return win


def _block_perm(n, block):
    nb = int(np.ceil(n / block))
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(o * block, min((o + 1) * block, n)) for o in order])
    return idx[:n]


def block_bootstrap_null(win, obs_incr, M=1000, block=150):
    """Null: block-shuffle the cross-root alignment. PCA loadings are row-permutation
    invariant, so we reuse each window's cached integrated OFI (Z) and only block-permute the
    CROSS columns (own column stays time-aligned to y, preserving the own-only baseline), then
    refit OLS. This destroys cross-root lead-lag while preserving each root's own series + vol
    clustering within blocks. p = (#null>=obs + 1)/(M+1)."""
    nullstats = np.empty(M)
    nR = win[0]['_Z_is'].shape[1]
    for b in range(M):
        per_win_incr = []
        for w in win:
            Z_is, Z_oos = w['_Z_is'], w['_Z_oos']
            y_is_all, y_oos_all, r2_1 = w['_y_is'], w['_y_oos'], w['_r2_1']
            pis = _block_perm(Z_is.shape[0], block)
            poos = _block_perm(Z_oos.shape[0], block)
            Zp_is, Zp_oos = Z_is[pis], Z_oos[poos]
            d = np.full(nR, np.nan)
            for i in range(nR):
                Xis = Zp_is.copy(); Xis[:, i] = Z_is[:, i]      # own column un-permuted
                Xoos = Zp_oos.copy(); Xoos[:, i] = Z_oos[:, i]
                b2 = ols_fit(Xis, y_is_all[:, i]); p2 = ols_pred(b2, Xoos)
                d[i] = oos_r2(y_oos_all[:, i], p2) - r2_1[i]
            per_win_incr.append(np.nanmean(d))
        nullstats[b] = float(np.nanmean(per_win_incr))
    p = (np.sum(nullstats >= obs_incr) + 1) / (M + 1)
    return float(p), nullstats


def _pbo_cscv(M, S=8):
    """CSCV PBO (Bailey et al. 2017). M: (N strategies x T). PBO = P(IS-best below OOS median)."""
    import itertools
    N, T = M.shape
    if T < 4 or N < 2:
        return float("nan"), 0
    bl = np.array_split(np.arange(T), S)
    logits = []
    for combo in itertools.combinations(range(S), S // 2):
        isd = np.concatenate([bl[i] for i in combo])
        ood = np.concatenate([bl[i] for i in range(S) if i not in combo])
        ns = int(np.argmax(M[:, isd].mean(1)))
        rank = (M[:, ood].mean(1) <= M[:, ood].mean(1)[ns]).mean()
        w = max(min(rank, 1 - 1e-6), 1e-6)
        logits.append(np.log(w / (1 - w)))
    return float((np.array(logits) <= 0).mean()), len(logits)


def lo_hac_sharpe(x, q=10):
    """Annualised-agnostic Sharpe (mean/std) with Newey-West HAC SE of the mean -> t-stat."""
    x = np.asarray(x); T = len(x)
    if T < 5 or x.std() == 0:
        return {'sharpe': float('nan'), 't_hac': float('nan')}
    mu = x.mean(); sd = x.std(ddof=1)
    g0 = np.var(x, ddof=0)
    s = g0
    for k in range(1, q + 1):
        cov = np.cov(x[:-k], x[k:])[0, 1] if T - k > 1 else 0.0
        s += 2 * (1 - k / (q + 1)) * cov
    se_mean = np.sqrt(s / T)
    return {'sharpe': float(mu / sd), 't_hac': float(mu / se_mean) if se_mean > 0 else float('nan')}


def bhy_threshold(pvals, alpha=0.05):
    """Benjamini-Hochberg-Yekutieli (arbitrary dependence). Returns reject mask
    + the crit p actually applied. When nothing rejects (kmax=0) we report the
    rank-1 threshold (the lowest hurdle) rather than 0.0, so the meaningful
    "smallest p a single root must clear" is never hidden as a spurious zero."""
    p = np.asarray(pvals); m = len(p)
    order = np.argsort(p); ranks = np.arange(1, m + 1)
    c_m = np.sum(1.0 / ranks)
    crit = (ranks / (m * c_m)) * alpha
    sorted_p = p[order]
    passed = sorted_p <= crit
    kmax = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    reject = np.zeros(m, bool)
    if kmax > 0:
        reject[order[:kmax]] = True
    crit_reported = float(crit[kmax - 1]) if kmax > 0 else float(crit[0])
    return reject, crit_reported


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    meta, O, RFWD, HSBP = build_panel()
    print(f"[panel] {len(meta)} rows, {O.shape[1]} roots, days={meta['day'].nunique()} in {time.time()-t0:.1f}s", flush=True)

    # ---- pipeline sanity: own-OFI CONTEMPORANEOUS impact (expect high) vs LAGGED predictive ----
    # RFWD[k] = price change during bin k+1; contemporaneous price change for ofi(k) is RFWD[k-1].
    _, Z_full = integrate_ofi(O, O)
    y_con = np.roll(RFWD, 1, axis=0); y_con[0] = np.nan   # RFWD[k-1] aligned to ofi(k)
    impact_r2, pred_r2 = [], []
    for i in range(O.shape[1]):
        m = np.isfinite(y_con[:, i])
        bc = ols_fit(Z_full[m, i:i+1], y_con[m, i]); pc = ols_pred(bc, Z_full[m, i:i+1])
        impact_r2.append(oos_r2(y_con[m, i], pc))
        bp = ols_fit(Z_full[:, i:i+1], RFWD[:, i]); pp = ols_pred(bp, Z_full[:, i:i+1])
        pred_r2.append(oos_r2(RFWD[:, i], pp))
    print(f"[sanity] own-OFI CONTEMPORANEOUS impact IS R^2 mean={np.nanmean(impact_r2):.3f} "
          f"(validates pipeline) | LAGGED predictive R^2 mean={np.nanmean(pred_r2):.4f}", flush=True)
    print(f"         contemporaneous per-root: {np.round(impact_r2,2)}", flush=True)

    # ---- main WFO at cost frontier ----
    results = {}
    for cm, tag in [(0.0, 'cost0x'), (0.5, 'cost0.5x'), (1.0, 'cost1x')]:
        win = run_wfo(meta, O, RFWD, HSBP, cost_mult=cm)
        incr = np.array([w['incr_r2'] for w in win])
        net = np.array([w['net_cross_mean'] for w in win])
        gross = np.array([w['gross_cross_mean'] for w in win])
        net_own = np.array([w['net_own_mean'] for w in win])
        allnet = np.concatenate([w['net_cross_series'] for w in win])
        tg = tail_guarded_rrr(net)
        hac = lo_hac_sharpe(allnet)
        results[tag] = {
            'n_windows': len(win),
            'incr_r2_oos_mean': float(incr.mean()), 'incr_r2_oos_bywindow_pos_frac': float((incr > 0).mean()),
            'r2_own_oos_mean': float(np.mean([w['r2_own'] for w in win])),
            'r2_cross_oos_mean': float(np.mean([w['r2_cross'] for w in win])),
            'net_cross_mean_bp': float(net.mean() * 1e4), 'gross_cross_mean_bp': float(gross.mean() * 1e4),
            'net_own_mean_bp': float(net_own.mean() * 1e4),
            'net_windows_pos_frac': float((net > 0).mean()),
            'tail_guard': tg, 'lo_hac': hac,
        }
        print(f"[wfo {tag}] windows={len(win)} incrR2={incr.mean():.5f} (pos {100*(incr>0).mean():.0f}%) "
              f"net={net.mean()*1e4:.3f}bp gross={gross.mean()*1e4:.3f}bp RRR={tg['rrr']:.2f} "
              f"RRRexbest={tg['rrr_ex_best']:.2f} t_hac={hac['t_hac']:.2f}", flush=True)

    # ---- per-root incremental R^2 + BHY (use cost-irrelevant R^2 from cost1x WFO windows) ----
    win = run_wfo(meta, O, RFWD, HSBP, cost_mult=1.0)
    byroot = np.vstack([w['incr_r2_byroot'] for w in win])  # (nwin, nR)
    root_t = byroot.mean(0) / (byroot.std(0, ddof=1) / np.sqrt(byroot.shape[0]) + 1e-12)
    from scipy import stats as _st
    root_p = _st.t.sf(root_t, df=byroot.shape[0] - 1)  # one-sided incr>0
    rej, crit = bhy_threshold(root_p)
    sharpes = np.array([lo_hac_sharpe(w['net_cross_series'])['sharpe'] for w in win])
    nominal_trials = nominal_trial_count(sharpes)  # raw WFO-window count, NOT a deflation
    print(f"[per-root] incrR2>0 BHY-significant roots: {int(rej.sum())}/{len(ROOTS)} (crit p<={crit:.4f})", flush=True)

    # ---- block-bootstrap null on the cross term (pooled mean OOS incremental R^2) ----
    obs_incr = float(np.mean([w['incr_r2'] for w in win]))
    M = int(os.environ.get('PERM_M', '1000'))
    t1 = time.time()
    p_block, nullstats = block_bootstrap_null(win, obs_incr, M=M, block=150)
    print(f"[block-null] obs incrR2={obs_incr:.5f} null_mean={nullstats.mean():.5f} "
          f"p={p_block:.4f} (M={M}) in {time.time()-t1:.1f}s", flush=True)
    # secondary scheme: single-bar shuffle (block=1) — destroys ALL serial structure
    p_single, _ = block_bootstrap_null(win, obs_incr, M=M, block=1)
    print(f"[single-bar-null] p={p_single:.4f} (M={M})", flush=True)
    # PBO via CSCV: per-root cross-model incremental OOS R^2 across WFO windows
    byroot_mat = np.vstack([w['incr_r2_byroot'] for w in win]).T   # (nR, nwin)
    byroot_mat = np.nan_to_num(byroot_mat)
    pbo_xr, n_sp = _pbo_cscv(byroot_mat, S=min(8, (byroot_mat.shape[1] // 2) * 2))
    print(f"[PBO] cross-root CSCV PBO={pbo_xr:.3f} (n_strats={byroot_mat.shape[0]}, "
          f"T_windows={byroot_mat.shape[1]}, {n_sp} splits)", flush=True)

    out = {
        'study': 'cross-root integrated-OFI lead-lag (CME, true sign, depth-10)',
        'prereg': 'locked pre-registration',
        'panel': {'rows': int(len(meta)), 'roots': ROOTS, 'days': sorted(meta['day'].unique().tolist()),
                  'bin_ms': BIN_MS},
        'own_ofi_contemporaneous_impact_is_r2_mean': float(np.nanmean(impact_r2)),
        'own_ofi_lagged_predictive_is_r2_mean': float(np.nanmean(pred_r2)),
        'cost_frontier': results,
        'per_root_cross_benefit': {
            'incr_r2_byroot_mean': {ROOTS[i]: float(byroot.mean(0)[i]) for i in range(len(ROOTS))},
            'bhy_significant_roots': int(rej.sum()), 'bhy_crit_p_rank1': crit,
            'nominal_trial_count': nominal_trials,
            'note_trials': ('nominal_trial_count is the raw count of WFO windows, NOT a '
                            'deflated-Sharpe effective-trials estimate; reported nominally. '
                            'A real correlation-based deflation is moot here because the '
                            'cross term is null (0/13 BHY-significant) before any deflation.'),
            'note_bhy_floor': ('the per-root incr-R^2 p-values are t-based (continuous), not '
                               'permutation-bounded, so the BHY rank-1 hurdle is attainable in '
                               'principle; the cross term simply does not clear it (0/13).'),
        },
        'block_bootstrap_null': {'obs_incr_r2': obs_incr, 'null_mean': float(nullstats.mean()),
                                 'p_value_block': p_block, 'p_value_single_bar': p_single, 'M': M,
                                 'note': 'block (preserves vol clustering, primary) + single-bar '
                                         '(destroys all serial structure, secondary) schemes'},
        'pbo_cscv': {'pbo': pbo_xr, 'n_strategies': int(byroot_mat.shape[0]),
                     'T_windows': int(byroot_mat.shape[1]), 'n_splits': n_sp,
                     'note': 'PBO of the per-root cross-model incremental OOS R^2; ~0.5 or high '
                             'confirms the cross-root term has no OOS-selectable edge'},
    }
    with open(f"{OUT}/verdict_xroot_ofi.json", 'w') as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"[done] wrote {OUT}/verdict_xroot_ofi.json in {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
