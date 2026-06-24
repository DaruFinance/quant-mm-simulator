#!/usr/bin/env python3
"""Same-dates reconstruction control: high-resolution vs coarser-resolution book
reconstruction on the IDENTICAL Jul-2023 calm root-days. Restricts both decompositions
to the intersection of (root,date) pairs present in both, fill-weights each, and reports
the adverse/capture ratio gap attributable to the reconstruction alone (dates, roots,
regime all held fixed)."""
import pandas as pd, numpy as np, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CTRL_DATES = ["20230710", "20230711", "20230712", "20230713", "20230714"]


def pooled(df):
    w = df["n_good_10s"].to_numpy(float)
    sp = float((df["spread_capture_bp_10s"] * w).sum() / w.sum())
    ad = float((df["adverse_select_bp_10s"] * w).sum() / w.sum())
    nf = float((df["net_after_fee_bp_10s"] * w).sum() / w.sum())
    return sp, ad, nf, -100 * ad / sp, int(df["n_fills"].sum()), len(df)


def main():
    hires = pd.read_parquet(REPO / "runs/maker_decomp/decomp_by_rootday.parquet")
    coarse = pd.read_parquet(REPO / "runs/maker_decomp_2023coarsectrl/decomp_by_rootday.parquet")
    for d in (hires, coarse):
        d["date"] = d["date"].astype(str)
    hires = hires[hires["date"].isin(CTRL_DATES)]
    coarse = coarse[coarse["date"].isin(CTRL_DATES)]
    # intersection of (root,date) present in BOTH reconstructions -> matched comparison
    key = lambda d: set(zip(d["root"], d["date"]))
    common = key(hires) & key(coarse)
    hm = hires[[ (r, dt) in common for r, dt in zip(hires["root"], hires["date"]) ]]
    cm = coarse[[ (r, dt) in common for r, dt in zip(coarse["root"], coarse["date"]) ]]
    sp_v, ad_v, nf_v, rat_v, nf_v_fills, nrd_v = pooled(hm)
    sp_l, ad_l, nf_l, rat_l, nf_l_fills, nrd_l = pooled(cm)
    out = {
        "control_dates": CTRL_DATES,
        "n_matched_rootdays": len(common),
        "hires_feed": {"spread": round(sp_v, 4), "adverse": round(ad_v, 4),
                       "net_after_fee": round(nf_v, 4), "adverse_over_capture_pct": round(rat_v, 2),
                       "n_fills": nf_v_fills},
        "coarse_feed": {"spread": round(sp_l, 4), "adverse": round(ad_l, 4),
                           "net_after_fee": round(nf_l, 4), "adverse_over_capture_pct": round(rat_l, 2),
                           "n_fills": nf_l_fills},
        "ratio_gap_pp": round(rat_v - rat_l, 2),
        "note": ("Same Jul-2023 calm dates/roots/regime; only the time resolution of the book "
                 "reconstruction differs (high- vs coarser-resolution). Any adverse/capture "
                 "gap is attributable to the reconstruction, not regime or volatility."),
    }
    (REPO / "runs/maker_decomp/feed_control_compare.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n  hires:  ratio {rat_v:.1f}%  net {nf_v:+.3f}  ({nrd_v} matched root-days)")
    print(f"  coarse: ratio {rat_l:.1f}%  net {nf_l:+.3f}  ({nrd_l} matched root-days)")
    print(f"  GAP:    {rat_v - rat_l:+.1f} percentage points (same dates/regime)")


if __name__ == "__main__":
    main()
