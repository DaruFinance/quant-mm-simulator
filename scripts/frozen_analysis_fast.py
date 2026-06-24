"""FAST deterministic cross-regime analysis on the frozen snapshot.
Uses per-root SUFFICIENT STATISTICS (Sxx, Sxy) so the cluster-bootstrap is a vectorized
sum over sampled roots — milliseconds, not minutes. Exact same math as the OLS FE slope."""
import json, os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd
from pathlib import Path
from math import comb

SRC = sys.argv[1] if len(sys.argv) > 1 else "continuous/frozen3_by_rootday.parquet"

def load():
    df = pd.read_parquet(SRC) if str(SRC).endswith(".parquet") else pd.DataFrame([json.load(open(p)) for p in Path(SRC).glob("*.json")])
    df = df[df["n_fills"] >= 20].copy()
    df["net"] = df["net_after_fee_bp_10s"]; df["rv"] = df["rv_10s_bp"]
    df["ratio"] = -df["adverse_select_bp_10s"] / df["spread_capture_bp_10s"] * 100.0
    return df.dropna(subset=["rv", "net", "ratio"])

def suff_stats(df):
    """per-root sufficient stats for the within-root (FE) slope of net~rv and ratio~rv."""
    g = df.groupby("root")
    xdm = df["rv"] - g["rv"].transform("mean")
    ndm = df["net"] - g["net"].transform("mean")
    rdm = df["ratio"] - g["ratio"].transform("mean")
    t = pd.DataFrame({"root": df["root"], "sxx": xdm*xdm, "sxn": xdm*ndm, "sxr": xdm*rdm})
    s = t.groupby("root").sum()
    return s["sxx"].to_numpy(), s["sxn"].to_numpy(), s["sxr"].to_numpy(), list(s.index)

def fe(df, B=10000, seed=7):
    sxx, sxn, sxr, roots = suff_stats(df); R = len(roots)
    net = float(sxn.sum()/sxx.sum()); rat = float(sxr.sum()/sxx.sum())
    rng = np.random.default_rng(seed); idx = rng.integers(0, R, size=(B, R))
    sx = sxx[idx].sum(1); bn = sxn[idx].sum(1)/sx; br = sxr[idx].sum(1)/sx
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return dict(net=round(net, 4), net_ci=q(bn), ratio=round(rat, 3), ratio_ci=q(br), n=int(len(df)))

def main():
    df = load(); out = {"n_rootdays": int(len(df)), "n_roots": int(df.root.nunique()),
                        "rv_range": [round(float(df.rv.min()), 2), round(float(df.rv.max()), 2)]}
    out["fe_full"] = fe(df)
    # validator-requested scope cuts: the effect's concentration in energy/metals
    EM = ["CL", "NG", "GC", "SI", "HG"]; HI3 = ["CL", "SI", "NG"]
    BLOC = {"energy": ["CL","NG"], "metals": ["GC","SI","HG"], "equity": ["ES","NQ","RTY"],
            "rates": ["ZN","ZF","ZB"], "FX": ["6E","6J"]}
    out["fe_drop_hi3_CL_SI_NG"] = fe(df[~df.root.isin(HI3)])
    out["fe_drop_energy_metals"] = fe(df[~df.root.isin(EM)])
    out["fe_energy_metals_only"] = fe(df[df.root.isin(EM)])
    out["bloc_sigma_range"] = {b: dict(min=round(float(df[df.root.isin(rs)].rv.min()),2),
                                       med=round(float(df[df.root.isin(rs)].rv.median()),2),
                                       max=round(float(df[df.root.isin(rs)].rv.max()),2),
                                       within_root_span_med=round(float(df[df.root.isin(rs)].groupby("root").rv.agg(lambda s:s.max()-s.min()).median()),2))
                               for b, rs in BLOC.items()}
    out["bloc_fe"] = {b: fe(df[df.root.isin(rs)]) for b, rs in BLOC.items()}
    out["fe_drop_slope_drivers"] = fe(df[~df.root.isin(["CL","NG","HG","6E","6J"])])  # drop the 5 driver roots
    tail = df[df.rv >= 2.0]; tc = tail.groupby("root").size().sort_values(ascending=False)
    out["tail_ge2_top3_pct"] = round(float(tc.head(3).sum()/len(tail)*100), 1); out["tail_ge2_top3_roots"] = list(tc.head(3).index)
    # rank-robust (insensitive to extreme-σ magnitude) + capture/adverse channel decomposition
    def wslope(d, y, x):
        xd = d[x] - d.groupby("root")[x].transform("mean"); yd = d[y] - d.groupby("root")[y].transform("mean")
        v = float((xd*xd).sum()); return round(float((xd*yd).sum()/v), 4) if v > 0 else None
    dd = df.copy(); dd["rvrank"] = dd.groupby("root")["rv"].rank(pct=True)
    dd["cap"] = dd["spread_capture_bp_10s"]; dd["adv"] = dd["adverse_select_bp_10s"]
    DRV = ["NG","HG","CL","6E","6J"]
    out["rank_robust_net_slope"] = dict(full=wslope(dd,"net","rvrank"), drivers=wslope(dd[dd.root.isin(DRV)],"net","rvrank"),
                                        nondrivers=wslope(dd[~dd.root.isin(DRV)],"net","rvrank"))
    out["channel_full"] = dict(d_capture=wslope(dd,"cap","rv"), d_adverse=wslope(dd,"adv","rv"))
    out["channel_drivers"] = dict(d_capture=wslope(dd[dd.root.isin(DRV)],"cap","rv"), d_adverse=wslope(dd[dd.root.isin(DRV)],"adv","rv"))
    out["channel_per_root"] = {r: dict(d_capture=wslope(dd[dd.root==r],"cap","rv"), d_adverse=wslope(dd[dd.root==r],"adv","rv"),
                                       d_net=wslope(dd[dd.root==r],"net","rv")) for r in sorted(dd.root.unique())}
    out["fe_tail_drop_lt2"] = fe(df[df.rv < 2.0])
    di = df.date.astype(int); md = int(np.median(di))
    out["fe_first_half"] = fe(df[di <= md]); out["fe_second_half"] = fe(df[di > md])
    # LORO via sufficient stats (exact, instant)
    sxx, sxn, sxr, roots = suff_stats(df); Txx, Txn, Txr = sxx.sum(), sxn.sum(), sxr.sum()
    loro = {roots[i]: dict(net=round(float((Txn-sxn[i])/(Txx-sxx[i])), 4),
                           ratio=round(float((Txr-sxr[i])/(Txx-sxx[i])), 3)) for i in range(len(roots))}
    out["loro"] = loro
    nets = [v["net"] for v in loro.values()]; rats = [v["ratio"] for v in loro.values()]
    out["loro_net_range"] = [min(nets), max(nets)]; out["loro_net_closest_to_zero"] = max(nets)
    out["loro_ratio_range"] = [min(rats), max(rats)]
    # leverage share + per-root individual slope
    out["leverage_pct"] = {roots[i]: round(float(sxx[i]/Txx*100), 1) for i in range(len(roots))}
    out["leverage_top3_pct"] = round(sum(sorted(out["leverage_pct"].values(), reverse=True)[:3]), 1)
    persl = {roots[i]: dict(net=round(float(sxn[i]/sxx[i]), 4), ratio=round(float(sxr[i]/sxx[i]), 3)) for i in range(len(roots))}
    out["per_root_slope"] = persl
    out["roots_net_neg_slope"] = int(sum(1 for v in persl.values() if v["net"] < 0))
    out["roots_ratio_pos_slope"] = int(sum(1 for v in persl.values() if v["ratio"] > 0))
    # within-root tercile Delta map
    def fw(s, c): return float(np.average(s[c], weights=s["n_fills"].to_numpy().astype(float)))
    terc = []
    for r, g in df.groupby("root"):
        if len(g) < 9: continue
        lo = g[g.rv <= g.rv.quantile(1/3)]; hi = g[g.rv >= g.rv.quantile(2/3)]
        if len(lo) < 2 or len(hi) < 2: continue
        terc.append(dict(root=r, d_ratio=round(fw(hi,"ratio")-fw(lo,"ratio"),1), d_net=round(fw(hi,"net")-fw(lo,"net"),4)))
    tm = pd.DataFrame(terc); out["tercile_map"] = tm.to_dict("records")
    k = int((tm.d_net < 0).sum()); m = len(tm)
    one = sum(comb(m, i) for i in range(k, m+1))/2**m
    out["sign_test"] = dict(net_deeper=k, m=m, one_sided_p=round(one, 4), two_sided_p=round(min(1, 2*one), 4))
    calm = df[df.rv < 1.0]; mix = (calm.groupby("root").size()/len(calm)*100).round(1)
    out["calm_root_mix_pct"] = dict(mix.sort_values(ascending=False)); out["calm_n"] = int(len(calm))
    def spear(x, y):
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean(); return round(float((rx*ry).sum()/np.sqrt((rx*rx).sum()*(ry*ry).sum())), 3)
    out["pooled_spearman_ratio_sigma"] = spear(df.ratio.values, df.rv.values)
    json.dump(out, open("continuous/frozen3_results.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("loro","per_root_slope","tercile_map","leverage_pct","calm_root_mix_pct")}, indent=1, default=float))

if __name__ == "__main__":
    main()
