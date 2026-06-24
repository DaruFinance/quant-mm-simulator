#!/usr/bin/env python3
"""Frozen3 (n=33,956) robustness recompute: R1 horizon-invariance, R2 per-driver day-level
bootstrap significance, R3 per-driver time-stability. Deterministic (seed 7). Emits JSON.
Usage: python3 robustness_frozen3.py continuous/frozen3 > continuous/frozen3_robustness.json
"""
import json, sys, glob
import numpy as np, pandas as pd

DRV = ["NG", "HG", "CL", "6E", "6J"]
SRC = sys.argv[1] if len(sys.argv) > 1 else "continuous/frozen3_by_rootday.parquet"

import pandas as pd
rows = pd.read_parquet(SRC).to_dict("records") if str(SRC).endswith(".parquet") else [json.load(open(p)) for p in glob.glob(f"{SRC}/*.json")]
df = pd.DataFrame(rows)
df = df[df["n_fills"] >= 20].copy()
df["rv"] = df["rv_10s_bp"]
df["net10"] = df["net_after_fee_bp_10s"]
df["net60"] = df["net_after_fee_bp_60s"]
df = df.dropna(subset=["rv", "net10", "net60"])

def wslope(d, y, x="rv"):
    xd = d[x] - d[x].mean(); yd = d[y] - d[y].mean()
    v = float((xd * xd).sum())
    return float((xd * yd).sum() / v) if v > 0 else None

def fe_slope(d, y, x="rv"):
    """within-root FE slope (demean within each root, pooled)."""
    g = d.groupby("root")
    xd = d[x] - g[x].transform("mean"); yd = d[y] - g[y].transform("mean")
    v = float((xd * xd).sum())
    return float((xd * yd).sum() / v) if v > 0 else None

out = {}
# R1 — horizon invariance (10s vs 60s net), within-root FE slope
nd = df[~df.root.isin(DRV)]; dv = df[df.root.isin(DRV)]
out["R1_horizon"] = {
    "full_10s": round(fe_slope(df, "net10"), 4), "full_60s": round(fe_slope(df, "net60"), 4),
    "drivers_10s": round(fe_slope(dv, "net10"), 4), "drivers_60s": round(fe_slope(dv, "net60"), 4),
    "nondrivers_10s": round(fe_slope(nd, "net10"), 4), "nondrivers_60s": round(fe_slope(nd, "net60"), 4),
}

# R2 — per-driver day-level bootstrap CI of within-root net slope (significant if CI excludes 0)
rng = np.random.default_rng(7)
B = 10000
r2 = {}
for r in DRV:
    d = df[df.root == r]
    x = d["rv"].to_numpy(); y = d["net10"].to_numpy()
    n = len(d)
    pt = wslope(d, "net10")
    idx = rng.integers(0, n, size=(B, n))
    xb = x[idx]; yb = y[idx]
    # standard pairs bootstrap: recenter each resample by its own means (conservative)
    xc = xb - xb.mean(1, keepdims=True); yc = yb - yb.mean(1, keepdims=True)
    sl = (xc * yc).sum(1) / (xc * xc).sum(1)
    lo, hi = float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))
    r2[r] = {"slope": round(pt, 4), "ci": [round(lo, 4), round(hi, 4)], "n": n,
             "excludes_0": bool(hi < 0 or lo > 0)}
out["R2_per_driver_sig"] = r2
out["R2_n_significant"] = int(sum(v["excludes_0"] for v in r2.values()))

# R3 — per-driver time-stability: within-root slope first vs second chronological half
di = df["date"].astype(int)
md = int(np.median(di))
h1 = df[di <= md]; h2 = df[di > md]
r3 = {}
for r in DRV:
    s1 = wslope(h1[h1.root == r], "net10"); s2 = wslope(h2[h2.root == r], "net10")
    r3[r] = {"h1": round(s1, 4) if s1 is not None else None,
             "h2": round(s2, 4) if s2 is not None else None,
             "both_neg": bool(s1 is not None and s2 is not None and s1 < 0 and s2 < 0)}
out["R3_time_stability"] = r3
out["R3_split_date"] = md
out["R3_all_drivers_both_neg"] = bool(all(v["both_neg"] for v in r3.values()))

print(json.dumps(out, indent=1))
