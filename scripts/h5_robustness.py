"""H5 — Quoting-parameter robustness (surface-stability protocol for MM).

Standalone branch script.  Imports the MM engine READ-ONLY
(``mmsim.sim.fast_sim`` + the ledger/markout/cost modules); does NOT modify
``fast_sim.py`` or ``run_mm_full.py``.

PROTOCOL (adapted from the surface-stability project)
----------------------------------------------------------
The surface-stability claim is: *low in-sample perturbation-sensitivity
σ_micro predicts better out-of-sample performance*.  In the original equities
work the perturbations are {base, ENT(entry-confirm), FEE, SLI, ENT+IND} and
the OOS metric is forward Sharpe.  Here we transplant it to the MM quoting
grid the fast engine exposes.

Quoting configurations (the "strategies" — the structural quoting knobs the
engine exposes):
  * family = quoter shape:  touch | depth_skew | microskew | microskew_ofi
                            | rank2 | rank3   (the inventory-skew / requote /
                            base-spread analogues fast_sim supports)
  * size   = quote size in {1, 2}  (a quoting-size knob)
Each (family, size) is one configuration; configurations are the rows we rank.

Perturbations (the σ_micro suite — MM analogues of ENT/FEE/SLI/ENT+IND):
  * base : standard cost (maker 0.2 bp, slip 0.2 bp), standard size, no
           latency offset.
  * FEE  : maker/taker fee +50%  (cost-robustness — directly the FEE
           perturbation).
  * SLI  : slippage +50%        (the SLI perturbation).
  * SIZE : quote size x2         (MM analogue of the entry-confirm/ENT knob —
           changes queue position / fill set).
  * LAT  : markout reference shifted +5 ms (edge-decay; the ENT+IND combined
           stress, an MM-native latency perturbation).

IS metric per (config, window, perturbation):
    mean NET realised half-spread (bp) over the IS window's baseline-quoter
    fills, where NET = gross realised_spread_10s (bp) - round-trip cost (bp).
σ_micro per (config, window) = std of that IS metric across the 5
perturbations.

OOS metric R per (config, window):
    OOS realised half-spread (bp) of the IS-best configs (forward window),
    sign-flipped so that *larger R is better* (we negate realised-spread cost
    so the falsifiable direction matches the original β_smooth<0 elevation
    criterion: lower σ_micro -> higher R).  We report BOTH the raw realised
    spread and R for transparency.

CELL aggregation (family-as-family, window):
    For each (family, window) cell we take the IS-top-K configs by IS net
    realised spread, compute σ_micro = mean σ_micro over the top-K, and
    R_K = mean OOS R of those top-K in window w+1.  This mirrors
    compute_metrics.cell_metrics.

FIT:
    OLS  R_K ~ sigma_micro_std + C(family) + C(window)  with family-clustered
    robust SEs (mirrors fit.fit_mixed), the within-transform cross-check, and
    a within-(family,window) permutation null on β_smooth.  PASS iff
    β_smooth < 0 at permutation p < 0.05 (one-sided).  The SOL-pilot NULL is a
    pre-stated legitimate outcome.

No lookahead: σ_micro is computed entirely on IS window w; R_K is window w+1
OOS.  All markout uses only data at ts <= horizon, inherited from the engine's
structural causality.

USAGE
-----
  PYTHONPATH=. python3 scripts/h5_robustness.py \
      --manifest data/fut_universe/manifest.csv \
      --roots 6E,CL --max-dates 3 --jobs 2 --out runs/h5_robustness

The pilot establishes the σ_micro->OOS relationship on 1-2 roots x a few
dates; the full universe is NOT run here.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- engine, imported READ-ONLY -------------------------------------------
from mmsim.sim import fast_sim as _F
from mmsim.ledger.costs import CostModel
from mmsim.markout.engine import compute_markout


_SEC = 1_000_000_000
_MIN = 60 * _SEC

# Locked futures cost model (same numbers as run_mm_full.FUT_COST).
BASE_MAKER = 0.00002
BASE_TAKER = 0.00002
BASE_SLIP = 0.00002

# Perturbation suite (name -> dict of overrides).
PERTURBATIONS = ["base", "FEE", "SLI", "SIZE", "LAT"]


# --------------------------------------------------------------------- #
# Quoting configurations (family x size).  spec_fn produces the fast_sim
# spec for the family; the SIZE perturbation multiplies the size.
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class QConfig:
    family: str
    size: float
    def spec(self, size: float):
        f = self.family
        if f == "touch":
            return partial(_F.spec_touch, size=size)
        if f == "depth_skew":
            return partial(_F.spec_depth_skew, size=size)
        if f == "microskew":
            return partial(_F.spec_microskew, size=size, use_ofi=False)
        if f == "microskew_ofi":
            return partial(_F.spec_microskew, size=size, use_ofi=True)
        if f == "rank2":
            # two-sided depth-2 quoting: bid+ask each one level deeper.
            return partial(_spec_rank_both, size=size, rank=2)
        if f == "rank3":
            return partial(_spec_rank_both, size=size, rank=3)
        raise ValueError(f)


def _spec_rank_both(w, size: float, rank: int):
    """Two-sided resting quote at depth-rank `rank` on both sides (a wider
    base-spread quoting knob).  Mirrors spec_touch's shape but at a deeper
    level; requires that rank exists on both sides."""
    S = w.snap_ts.shape[0]
    has_both = (w.n_bid >= rank) & (w.n_ask >= rank)
    bpx = np.where(has_both, w.bid_px[:, rank - 1], np.nan)
    apx = np.where(has_both, w.ask_px[:, rank - 1], np.nan)
    bsz = np.where(has_both, size, 0.0)
    asz = np.where(has_both, size, 0.0)
    return bpx, bsz, apx, asz


# Only the families that actually FILL under the engine's queue-aware model
# are usable: the skew/rank-deep families post away from where aggression
# lands and almost never fill (verified empirically: <=5 fills/session vs
# 70-180 for touch/depth_skew), so a per-window realised-spread mean is
# undefined for them.  We keep the two filling quoter shapes (the base-spread
# / inventory-skew structural knobs) and vary quote size, which genuinely
# changes the spill-over fill set and the realised-spread metric (monotone in
# size).  This is the honest usable quoting-parameter grid for this universe.
FAMILIES = ["touch", "depth_skew"]
# Quote-size knobs per family so each (family, window) cell has a rankable
# pool of configurations whose IS-best size we rank within the family.
SIZES = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
CONFIGS = [QConfig(f, s) for f in FAMILIES for s in SIZES]

K_PRIMARY = 3            # IS top-K per cell (pool is len(SIZES) per family)
MIN_CONFIGS_PER_CELL = 3 # need a rankable pool


# --------------------------------------------------------------------- #
# Net realised half-spread (bp) per fill, under a cost model + latency.
# --------------------------------------------------------------------- #

def _net_realised_bp(fills, snap_ts, snap_mid, *, maker, slip,
                     lat_offset_ns=0) -> np.ndarray:
    """Per-fill NET realised half-spread in bp.

    gross = realised_spread_10s (already a per-fill bp-able fraction); NET
    subtracts the round-trip maker fee + slippage (both legs of the captured
    spread).  ``lat_offset_ns`` shifts each fill's timestamp forward (the
    latency / edge-decay perturbation) before the markout reference is taken.
    Returns an array of finite per-fill net realised spreads (in bp).
    """
    if not fills:
        return np.array([])
    if lat_offset_ns:
        # shift fill ts forward -> markout reference is taken later (edge decays)
        class _Fk:
            __slots__ = ("ts_ns", "price", "side", "size", "fill_id",
                         "order_id", "is_maker")
            def __init__(s, f):
                s.ts_ns = int(f.ts_ns) + lat_offset_ns
                s.price = float(f.price); s.side = int(f.side)
                s.size = float(getattr(f, "size", 1.0))
                s.fill_id = int(getattr(f, "fill_id", 0))
                s.order_id = int(getattr(f, "order_id", 0))
                s.is_maker = bool(getattr(f, "is_maker", True))
        use_fills = [_Fk(f) for f in fills]
    else:
        use_fills = fills
    mo = compute_markout(use_fills, snap_ts, snap_mid)
    rs = mo.realised_spread_10s
    rs = rs[~np.isnan(rs)]
    if rs.size == 0:
        return rs
    # cost in fraction-of-price terms; round-trip = entry + (assumed) exit leg
    cost_frac = 2.0 * (maker + slip)
    net = rs - cost_frac
    return net * 1e4  # -> bp


def _perturb_params(pert: str) -> dict:
    """Return the override params for a perturbation name."""
    if pert == "base":
        return dict(maker=BASE_MAKER, slip=BASE_SLIP, size_mult=1.0, lat_ns=0)
    if pert == "FEE":
        return dict(maker=BASE_MAKER * 1.5, slip=BASE_SLIP, size_mult=1.0, lat_ns=0)
    if pert == "SLI":
        return dict(maker=BASE_MAKER, slip=BASE_SLIP * 1.5, size_mult=1.0, lat_ns=0)
    if pert == "SIZE":
        return dict(maker=BASE_MAKER, slip=BASE_SLIP, size_mult=2.0, lat_ns=0)
    if pert == "LAT":
        return dict(maker=BASE_MAKER, slip=BASE_SLIP, size_mult=1.0,
                    lat_ns=5_000_000)  # +5 ms
    raise ValueError(pert)


# --------------------------------------------------------------------- #
# WFO windows (same scheme as run_mm_full.wfo_windows)
# --------------------------------------------------------------------- #

def wfo_windows(t0, t1, is_ns, oos_ns):
    out = []
    s = t0
    while s + is_ns + oos_ns <= t1:
        out.append((s, s + is_ns, s + is_ns, s + is_ns + oos_ns))
        s += oos_ns
    return out


# --------------------------------------------------------------------- #
# Per-(config, window, perturbation) IS metric + per-(config,window) OOS R
# --------------------------------------------------------------------- #

def _config_window_net(day, lo, hi, cfg: QConfig, snap_ts, snap_mid,
                       pert_params: dict) -> np.ndarray:
    """Per-fill net realised half-spread (bp) array over [lo,hi) for one config
    under one perturbation's params.  Empty array if no fills."""
    wa = _F.window_arrays_from_day(day, lo, hi)
    if wa.snap_ts.shape[0] == 0:
        return np.array([])
    size = cfg.size * pert_params["size_mult"]
    res = _F.simulate_from_arrays(wa, cfg.spec(size), size)
    return _net_realised_bp(res.fills, snap_ts, snap_mid,
                            maker=pert_params["maker"], slip=pert_params["slip"],
                            lat_offset_ns=pert_params["lat_ns"])


def _config_window_metric(day, lo, hi, cfg: QConfig, snap_ts, snap_mid,
                          pert_params: dict) -> float:
    """Mean net realised half-spread (bp), NaN if no fills."""
    net = _config_window_net(day, lo, hi, cfg, snap_ts, snap_mid, pert_params)
    return float(net.mean()) if net.size else float("nan")


def run_one_day(snap, trades, root, date, symbol, *, is_min, oos_min,
                throttle_k) -> List[dict]:
    """Produce the per-(family,window) cell rows for one contract-day.

    Returns a list of dicts with the cell schema expected by the fit
    (asset, family, window, sigma_micro, R_K, realised_spread_oos_bp, ...).
    """
    # session bounds from a cheap mid-timeline pass
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    day = _F.read_day_arrays(snap, trades, symbol=symbol,
                             throttle_k=throttle_k, depth_k=10)
    snap_ts = day["snap_ts"]
    if snap_ts.size == 0:
        return []
    # snapshot mid timeline (best bid/ask midpoint) for markout
    bb = day["bid_px"][:, 0]
    ba = day["ask_px"][:, 0]
    both = (~np.isnan(bb)) & (~np.isnan(ba))
    s_ts = snap_ts[both]
    s_mid = 0.5 * (bb[both] + ba[both])
    if s_ts.size == 0:
        return []
    # IMPORTANT: in this universe the depth feed spans the full ~12h session
    # but the TRADE tape is only captured for the active sub-period (often the
    # last ~2h).  Maker fills can only occur where trades exist, so we anchor
    # the WFO windows to the trade-active span intersected with snapshot
    # availability — otherwise every window but the last is fill-empty.
    tt = day["trd_ts"]
    if tt.size == 0:
        return []
    t0 = max(int(s_ts[0]), int(tt[0]))
    t1 = min(int(s_ts[-1]), int(tt[-1]))
    if t1 <= t0:
        return []
    windows = wfo_windows(t0, t1, int(is_min * _MIN), int(oos_min * _MIN))
    if len(windows) < 2:
        return []

    # For each config: IS metric per perturbation per window (for sigma_micro +
    # ranking) and per-fill OOS net-spread arrays per window (for POOLED R_K).
    # We need window w (IS) -> window w+1 (OOS) transitions.
    nW = len(windows)
    is_metric = np.full((len(CONFIGS), nW, len(PERTURBATIONS)), np.nan)
    # per (config, window) OOS base-pert per-fill net-bp arrays (pooled later)
    oos_fills: List[List[np.ndarray]] = [
        [np.array([]) for _ in range(nW)] for _ in range(len(CONFIGS))]
    oos_mean = np.full((len(CONFIGS), nW), np.nan)

    base_pp = _perturb_params("base")
    for ci, cfg in enumerate(CONFIGS):
        for wi, w in enumerate(windows):
            is_lo, is_hi = w[0], w[1]
            oos_lo, oos_hi = w[2], w[3]
            for pi, pert in enumerate(PERTURBATIONS):
                pp = _perturb_params(pert)
                is_metric[ci, wi, pi] = _config_window_metric(
                    day, is_lo, is_hi, cfg, s_ts, s_mid, pp)
            net = _config_window_net(day, oos_lo, oos_hi, cfg, s_ts, s_mid,
                                     base_pp)
            oos_fills[ci][wi] = net
            if net.size:
                oos_mean[ci, wi] = float(net.mean())

    # sigma_micro per (config, window): std across perturbations (IS only).
    # require >=2 finite perturbation metrics else NaN.
    n_finite = np.sum(np.isfinite(is_metric), axis=2)
    sigma = np.nanstd(is_metric, axis=2)
    sigma[n_finite < 2] = np.nan
    is_base = is_metric[:, :, 0]                       # IS base metric for ranking

    # Build (family, window-transition) cells: rank configs by IS base metric in
    # window w, take top-K, average sigma_micro, and POOL their OOS fills in w+1
    # to form R_K (pooling makes R_K finite even when each config's OOS window
    # holds few fills — the honest fix for the sparse maker-fill regime).
    rows: List[dict] = []
    fam_to_cidx: Dict[str, List[int]] = {}
    for ci, cfg in enumerate(CONFIGS):
        fam_to_cidx.setdefault(cfg.family, []).append(ci)

    for fam, cidxs in fam_to_cidx.items():
        cidxs = np.array(cidxs)
        for w in range(nW - 1):  # transition w -> w+1
            isb = is_base[cidxs, w]
            sig = sigma[cidxs, w]
            valid = np.isfinite(isb) & np.isfinite(sig)
            if valid.sum() < MIN_CONFIGS_PER_CELL:
                continue
            loc = np.flatnonzero(valid)
            isb_v = isb[loc]; sig_v = sig[loc]
            order = np.argsort(-isb_v)            # descending IS metric
            kk = min(K_PRIMARY, isb_v.size)
            top_loc = loc[order[:kk]]             # indices into cidxs
            top_cidx = cidxs[top_loc]
            # POOL OOS fills of the top-K configs in the forward window
            pooled = [oos_fills[ci][w + 1] for ci in top_cidx]
            pooled = [p for p in pooled if p.size]
            if not pooled:
                continue
            pooled_net = np.concatenate(pooled)
            realised_oos_bp = float(pooled_net.mean())
            sigma_micro_cell = float(np.mean(sig_v[order[:kk]]))
            # R_K: larger-is-better (realised half-spread is captured edge).
            R_K = realised_oos_bp
            rows.append({
                "asset": f"{root}",
                "root": root, "date": date, "symbol": symbol,
                "family": fam, "window": int(w + 1),  # 1-based transition id
                "n_configs": int(valid.sum()),
                "n_oos_fills": int(pooled_net.size),
                "sigma_micro": sigma_micro_cell,
                "R_K": R_K,
                "realised_spread_oos_bp": realised_oos_bp,
                "is_top_k_mean_bp": float(np.mean(isb_v[order[:kk]])),
            })
    return rows


# --------------------------------------------------------------------- #
# Fit: OLS + clustered SE + within-check + permutation null
# (mirrors quant-surface-stability/scripts/fit.py)
# --------------------------------------------------------------------- #

def fit_ols(df, sigma="sigma_micro", R="R_K"):
    import pandas as pd
    import statsmodels.formula.api as smf
    from scipy.stats import norm

    work = df.dropna(subset=[sigma, R, "family", "window"]).copy()
    if work.shape[0] < 8:
        return {"error": f"too few rows ({work.shape[0]})"}
    work["window_str"] = "w" + work["window"].astype(str)
    sd = work[sigma].std() or 1.0
    work["_sigma_std"] = (work[sigma] - work[sigma].mean()) / sd

    formula = f"{R} ~ _sigma_std + C(family) + C(window_str)"
    if work["asset"].nunique() > 1:
        formula += " + C(asset)"
    # need >1 cluster for clustered SE; else fall back to HC1
    n_clusters = work["family"].nunique()
    try:
        if n_clusters > 1:
            res = smf.ols(formula, data=work).fit(
                cov_type="cluster", cov_kwds={"groups": work["family"]})
        else:
            res = smf.ols(formula, data=work).fit(cov_type="HC1")
    except Exception as e:
        return {"error": f"fit failed: {e}"}
    if "_sigma_std" not in res.params.index:
        return {"error": "_sigma_std not in params"}
    beta = float(res.params["_sigma_std"])
    se = float(res.bse["_sigma_std"])
    tval = beta / se if se else np.nan
    p_one = float(norm.cdf(tval))         # one-sided H1: beta < 0
    p_two = float(2 * (1 - norm.cdf(abs(tval))))
    formula_null = formula.replace("_sigma_std + ", "")
    r2_full = float(res.rsquared)
    try:
        r2_null = float(smf.ols(formula_null, data=work).fit().rsquared)
    except Exception:
        r2_null = float("nan")
    f2 = (r2_full - r2_null) / (1 - r2_full) if r2_full < 1 else float("nan")
    return {"n": int(work.shape[0]), "beta_smooth": beta, "se": se,
            "tval": float(tval), "p_one_sided": p_one, "p_two_sided": p_two,
            "r2_full": r2_full, "r2_null": r2_null,
            "f2": (float(f2) if np.isfinite(f2) else None),
            "n_clusters": int(n_clusters)}


def fit_within(df, sigma="sigma_micro", R="R_K"):
    work = df.dropna(subset=[sigma, R, "family", "window"]).copy()
    sd = work[sigma].std() or 1.0
    work["_sigma_std"] = (work[sigma] - work[sigma].mean()) / sd
    cols = ["family", "window"]
    if work["asset"].nunique() > 1:
        cols.append("asset")
    y = work[R].astype(float).copy()
    x = work["_sigma_std"].astype(float).copy()
    for col in cols:
        y = y - work.groupby(col)[R].transform("mean")
        x = x - work.groupby(col)["_sigma_std"].transform("mean")
    if x.var() < 1e-12:
        return {"error": "no variance after demeaning"}
    beta = float((x * y).sum() / (x * x).sum())
    return {"beta_smooth_within": beta, "n": int(work.shape[0])}


def permutation_null(df, sigma="sigma_micro", R="R_K", M=1000, seed=7):
    rng = np.random.default_rng(seed)
    base = fit_ols(df, sigma, R)
    if "error" in base:
        return base
    obs = base["beta_smooth"]
    # within-(asset, window) shuffle of R (the OOS-side metric assignment)
    grouped = df.groupby(["asset", "window"]).groups
    null_betas = []
    base_R = df[R].values.copy()
    for _ in range(M):
        shuffled = df.copy()
        newR = base_R.copy()
        for key, idx in grouped.items():
            ipos = df.index.get_indexer(idx)
            perm = rng.permutation(ipos)
            newR[ipos] = base_R[perm]
        shuffled[R] = newR
        res = fit_ols(shuffled, sigma, R)
        if "error" not in res:
            null_betas.append(res["beta_smooth"])
    null_betas = np.array(null_betas)
    if null_betas.size == 0:
        return {"error": "all permutation fits failed"}
    p_one = (np.sum(null_betas <= obs) + 1) / (len(null_betas) + 1)
    return {"M": int(null_betas.size), "beta_obs": float(obs),
            "p_perm_one_sided": float(p_one),
            "null_mean": float(null_betas.mean()),
            "null_std": float(null_betas.std()),
            "null_q05": float(np.quantile(null_betas, 0.05)),
            "null_q95": float(np.quantile(null_betas, 0.95))}


def per_family_breakdown(df, sigma="sigma_micro", R="R_K"):
    import statsmodels.api as sm
    from scipy.stats import norm
    out = []
    for fam, sub in df.groupby("family"):
        sub = sub.dropna(subset=[sigma, R])
        if sub.shape[0] < 4 or sub[sigma].std() < 1e-12:
            continue
        x = (sub[sigma] - sub[sigma].mean()) / (sub[sigma].std() or 1.0)
        y = sub[R].values
        try:
            res = sm.OLS(y, sm.add_constant(x.values), missing="drop").fit()
            beta = float(res.params[1]); se = float(res.bse[1])
            tval = beta / se if se else np.nan
            out.append({"family": fam, "n": int(sub.shape[0]),
                        "beta": beta, "se": se,
                        "p_one_sided": float(norm.cdf(tval))})
        except Exception:
            pass
    return out


# --------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------- #

def make_figure(df, fit_res, perm_res, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    work = df.dropna(subset=["sigma_micro", "R_K"]).copy()
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    fams = sorted(work["family"].unique())
    cmap = plt.get_cmap("tab10")
    for i, fam in enumerate(fams):
        sub = work[work["family"] == fam]
        ax.scatter(sub["sigma_micro"], sub["R_K"], s=34, alpha=0.7,
                   color=cmap(i % 10), label=fam, edgecolors="none")
    # OLS fit line on raw (sigma, R) for visual
    if work.shape[0] >= 2 and work["sigma_micro"].std() > 0:
        b1, b0 = np.polyfit(work["sigma_micro"], work["R_K"], 1)
        xs = np.linspace(work["sigma_micro"].min(), work["sigma_micro"].max(), 50)
        ax.plot(xs, b0 + b1 * xs, "k--", lw=1.8,
                label=f"raw OLS slope={b1:+.2f} (uncontrolled)")
    ax.set_xlabel("in-sample σ_micro  (perturbation-sensitivity of net realised spread, bp)")
    ax.set_ylabel("OOS realised half-spread  R_K  (bp, larger = better)")
    beta = fit_res.get("beta_smooth", float("nan"))
    pperm = perm_res.get("p_perm_one_sided", float("nan"))
    verdict = ("PASS" if (np.isfinite(beta) and beta < 0
                          and np.isfinite(pperm) and pperm < 0.05) else "NULL")
    ax.set_title(f"H5 quoting-parameter robustness — β_smooth={beta:+.3f}  "
                 f"perm p={pperm:.3f}  [{verdict}]")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #

def load_manifest(path, roots: Optional[List[str]], max_dates: Optional[int],
                  data_root=os.environ.get("FUT_UNIVERSE_ROOT", "data/fut_universe")):
    """Accept either the enmetfx manifest schema (root,date,symbol,snap,trades)
    or the eqrates schema (root,date,contract,...).  For the eqrates schema we
    construct the snap/trades paths: snap=<root>/<date>_snap.parquet,
    trades=<root>/<date>_trades.parquet, symbol=contract."""
    with open(path) as fh:
        raw = list(csv.DictReader(fh))
    tasks = []
    for t in raw:
        if "snap" in t and "trades" in t and t.get("snap"):
            tasks.append({"root": t["root"], "date": t["date"],
                          "symbol": t.get("symbol") or t.get("contract"),
                          "snap": t["snap"], "trades": t["trades"]})
        else:
            root, date = t["root"], t["date"]
            sym = t.get("contract") or t.get("symbol")
            snap = os.path.join(data_root, root, f"{date}_snap.parquet")
            if not os.path.exists(snap):
                snap = os.path.join(data_root, root, f"{date}.parquet")
            trades = os.path.join(data_root, root, f"{date}_trades.parquet")
            tasks.append({"root": root, "date": date, "symbol": sym,
                          "snap": snap, "trades": trades})
    if roots:
        rs = set(roots)
        tasks = [t for t in tasks if t["root"] in rs]
    if max_dates:
        by_root: Dict[str, List[dict]] = {}
        for t in tasks:
            by_root.setdefault(t["root"], []).append(t)
        tasks = []
        for r, ts in by_root.items():
            tasks.extend(sorted(ts, key=lambda x: x["date"])[:max_dates])
    return tasks


def _worker(t, job_kw):
    """Module-level worker (picklable for ProcessPoolExecutor)."""
    ts0 = time.time()
    rows = run_one_day(t["snap"], t["trades"], t["root"], t["date"],
                       t["symbol"], **job_kw)
    return rows, time.time() - ts0, t


def free_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.environ.get("FUT_UNIVERSE_MANIFEST",
                                                          "data/fut_universe/manifest_eq0.csv"),
                    help="eqrates manifest (liquid roots fill; thin FX roots "
                         "almost never fill under the queue model)")
    ap.add_argument("--roots", default="ES,NQ",
                    help="comma list; TINY pilot default (liquid)")
    ap.add_argument("--max-dates", type=int, default=3)
    ap.add_argument("--out", default="runs/h5_robustness")
    # The trade tape is captured only over an active sub-period (~120 min in
    # this universe), so windows are anchored to that span; IS=40/OOS=20 yields
    # several transitions there while leaving enough fills per cell (pooled).
    ap.add_argument("--is-min", type=float, default=40.0)
    ap.add_argument("--oos-min", type=float, default=20.0)
    ap.add_argument("--throttle-k", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--M", type=int, default=1000, help="permutation count")
    ap.add_argument("--min-free-gb", type=float, default=22.0)
    args = ap.parse_args()

    roots = [r for r in args.roots.split(",") if r] if args.roots else None
    throttle_k = args.throttle_k if args.throttle_k > 0 else None
    os.makedirs(args.out, exist_ok=True)

    fg = free_gb()
    print(f"[h5] free RAM = {fg:.1f} GB (min required {args.min_free_gb})")
    if fg < args.min_free_gb:
        raise SystemExit(f"[h5] ABORT: free RAM {fg:.1f} GB < {args.min_free_gb} GB; "
                         f"shared box — wait/poll and retry.")

    tasks = load_manifest(args.manifest, roots, args.max_dates)
    print(f"[h5] pilot tasks: {len(tasks)} contract-days "
          f"({sorted(set(t['root'] for t in tasks))})")
    print(f"[h5] grid: {len(CONFIGS)} configs ({len(FAMILIES)} families x "
          f"{len(SIZES)} sizes) x {len(PERTURBATIONS)} perturbations")

    # Process tasks with a bounded thread pool (the heavy work is the numba
    # kernel which releases the GIL; --jobs caps concurrency for RAM).
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t_start = time.time()
    all_rows: List[dict] = []
    per_day_wall: List[float] = []

    job_kw = dict(is_min=args.is_min, oos_min=args.oos_min, throttle_k=throttle_k)

    if args.jobs <= 1:
        for t in tasks:
            rows, wall, tt = _worker(t, job_kw)
            all_rows.extend(rows)
            per_day_wall.append(wall)
            print(f"  [{tt['root']} {tt['date']}] cells={len(rows)} wall={wall:.1f}s")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_worker, t, job_kw): t for t in tasks}
            for fu in as_completed(futs):
                rows, wall, tt = fu.result()
                all_rows.extend(rows)
                per_day_wall.append(wall)
                print(f"  [{tt['root']} {tt['date']}] cells={len(rows)} wall={wall:.1f}s")

    if not all_rows:
        raise SystemExit("[h5] no cells produced — check windows / data.")

    import pandas as pd
    df = pd.DataFrame(all_rows)
    cells_path = os.path.join(args.out, "cells_h5.parquet")
    df.to_parquet(cells_path)
    print(f"[h5] cells: {df.shape[0]} -> {cells_path}")

    # ---- fit ----
    print("[h5] OLS fit  R_K ~ sigma_micro + C(family) + C(window) ...")
    fit_res = fit_ols(df)
    print(json.dumps(fit_res, indent=2))
    within = fit_within(df)
    if "beta_smooth" in fit_res and "beta_smooth_within" in within:
        within["xcheck_diff"] = abs(fit_res["beta_smooth"]
                                    - within["beta_smooth_within"])
    print("[h5] within-check:", json.dumps(within))
    print(f"[h5] permutation null M={args.M} ...")
    perm = permutation_null(df, M=args.M)
    print(json.dumps(perm, indent=2))
    pf = per_family_breakdown(df)

    beta = fit_res.get("beta_smooth", float("nan"))
    pperm = perm.get("p_perm_one_sided", float("nan"))
    passed = bool(np.isfinite(beta) and beta < 0
                  and np.isfinite(pperm) and pperm < 0.05)

    # ---- full-run cost estimate ----
    mean_wall = float(np.mean(per_day_wall)) if per_day_wall else float("nan")
    # full universe size from the manifest (all roots, all dates)
    with open(args.manifest) as fh:
        full_n = sum(1 for _ in csv.DictReader(fh))
    est_full_s = mean_wall * full_n / max(1, args.jobs)

    verdict = {
        "branch": "H5_quoting_parameter_robustness",
        "hypothesis": "low IS sigma_micro -> better OOS realised spread "
                      "(beta_smooth < 0 at perm p < 0.05, one-sided)",
        "verdict": "PASS" if passed else "NULL",
        "pre_stated_null_ok": True,
        "beta_smooth": beta,
        "beta_smooth_within": within.get("beta_smooth_within"),
        "within_xcheck_diff": within.get("xcheck_diff"),
        "perm_p_one_sided": pperm,
        "se": fit_res.get("se"),
        "p_one_sided_analytic": fit_res.get("p_one_sided"),
        "f2": fit_res.get("f2"),
        "n_cells": int(df.shape[0]),
        "n_clusters_family": fit_res.get("n_clusters"),
        "per_family": pf,
        "grid": {"families": FAMILIES, "sizes": SIZES,
                 "perturbations": PERTURBATIONS, "K_primary": K_PRIMARY},
        "pilot": {"roots": sorted(set(t["root"] for t in tasks)),
                  "n_contract_days": len(tasks),
                  "mean_wall_s_per_day": round(mean_wall, 2),
                  "total_wall_s": round(time.time() - t_start, 1)},
        "full_run_cost_estimate": {
            "universe_contract_days": full_n,
            "jobs": args.jobs,
            "est_wall_hours": round(est_full_s / 3600.0, 2),
            "note": "scales linearly in contract-days / jobs; perm-null "
                    "fit cost is fixed (runs once on assembled cells)."},
        "fit_full": fit_res,
        "perm_full": perm,
    }
    vpath = os.path.join(args.out, "verdict_h5.json")
    with open(vpath, "w") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    print(f"[h5] verdict -> {vpath}")

    fig_path = "runs/branch_figs/h5_robustness.png"
    make_figure(df, fit_res, perm, fig_path)
    print(f"[h5] figure -> {fig_path}")

    print("\n" + "=" * 64)
    print(f"H5 VERDICT: {verdict['verdict']}   "
          f"beta_smooth={beta:+.4f}  perm_p={pperm:.4f}  n_cells={df.shape[0]}")
    print(f"  per-family betas: " + ", ".join(
        f"{r['family']}={r['beta']:+.3f}(p1={r['p_one_sided']:.2f})" for r in pf))
    print(f"  full-run est: {verdict['full_run_cost_estimate']['est_wall_hours']} h "
          f"for {full_n} contract-days at jobs={args.jobs}")
    print("=" * 64)


if __name__ == "__main__":
    main()
