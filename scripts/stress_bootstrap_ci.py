"""Root-day cluster bootstrap CIs for the cross-regime stress decomposition.

The calm headline (runs/maker_decomp/decomp_bootstrap_ci.json) carries a root-day
cluster bootstrap CI on the adverse-to-captured ratio and net-after-fee. The stress
episodes (SVB / FOMC / COVID, on the nanosecond TAQ feed) reported only point estimates plus
median-root and roots-negative breadth checks. This script attaches the *same* CI to
each stress episode so the 105-111% over-consumption can be read against its own
within-feed sampling noise.

Procedure (identical to the calm CI): the resampling unit is the root-day; we draw
B root-day samples with replacement, recompute the fill-weighted pooled ratio and
net each draw, and take the 2.5/97.5 percentiles. The script first *reproduces* the
published calm CI as a methodological-parity check, then runs the three stress episodes.

Single-threaded, deterministic (fixed seed). Tiny data (<=232 root-days), trivial RAM.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CALM_PARQUET = REPO / "runs/maker_decomp/decomp_by_rootday.parquet"
STRESS_PARQUET = REPO / "runs/maker_decomp_xregime/stress_by_rootday.parquet"
OUT = REPO / "runs/maker_decomp_xregime/stress_bootstrap_ci.json"

B = 10000
SEED = 20260618


def pooled(n, cap, adv, net):
    """Fill-weighted pooled capture, adverse, adv/cap ratio (%), net-after-fee."""
    w = n.sum()
    cap_w = (n * cap).sum() / w
    adv_w = (n * adv).sum() / w
    net_w = (n * net).sum() / w
    ratio = -100.0 * adv_w / cap_w
    return cap_w, adv_w, ratio, net_w


def bootstrap(df, seed):
    n = df["n_fills"].to_numpy(float)
    cap = df["spread_capture_bp_10s"].to_numpy(float)
    adv = df["adverse_select_bp_10s"].to_numpy(float)
    net = df["net_after_fee_bp_10s"].to_numpy(float)
    k = len(df)

    cap0, adv0, ratio0, net0 = pooled(n, cap, adv, net)

    rng = np.random.default_rng(seed)
    ratios = np.empty(B)
    nets = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, k, k)  # resample root-days with replacement
        _, _, ratios[b], nets[b] = pooled(n[idx], cap[idx], adv[idx], net[idx])

    def ci(a):
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]

    r_ci = ci(ratios)
    n_ci = ci(nets)
    return {
        "n_rootdays": int(k),
        "n_fills": int(n.sum()),
        "capture_bp": round(float(cap0), 4),
        "adverse_bp": round(float(adv0), 4),
        "adverse_over_capture_pct": {
            "point": round(float(ratio0), 2),
            "ci95": r_ci,
            "excludes_100": bool(r_ci[0] > 100.0 or r_ci[1] < 100.0),
        },
        "net_after_fee_bp": {
            "point": round(float(net0), 4),
            "ci95": n_ci,
            "excludes_0": bool(n_ci[0] > 0.0 or n_ci[1] < 0.0),
            "p_negative": round(float((nets < 0).mean()), 3),
        },
    }


def main():
    out = {
        "method": f"root-day cluster bootstrap, B={B}, seed={SEED}, fill-weighted, 10s",
        "note": "Same procedure as runs/maker_decomp/decomp_bootstrap_ci.json (calm headline). "
                "Resampling unit is the root-day; ratio = fill-weighted adverse/captured.",
    }

    # (a) methodological-parity check: reproduce the published calm CI
    calm = pd.read_parquet(CALM_PARQUET)
    out["calm_parity_check"] = bootstrap(calm, SEED)
    out["calm_published"] = {
        "adverse_over_capture_pct": {"point": 101.1, "ci95": [97.7, 104.7]},
        "net_after_fee_bp": {"point": -0.0223, "ci95": [-0.0458, 0.0008]},
    }

    # (b) the three stress episodes, identical procedure
    stress = pd.read_parquet(STRESS_PARQUET)
    label = {"2023svb": "2023 SVB (Mar)", "2022fomc": "2022 FOMC (Jun)", "2020covid": "2020 COVID (Mar)"}
    out["stress"] = {}
    for i, key in enumerate(["2023svb", "2022fomc", "2020covid"]):
        ep = stress[stress["episode"] == key]
        res = bootstrap(ep, SEED + 1 + i)
        res["episode"] = label[key]
        out["stress"][key] = res

    OUT.write_text(json.dumps(out, indent=2))

    # console summary
    c = out["calm_parity_check"]
    print(f"CALM parity: ratio {c['adverse_over_capture_pct']['point']}% "
          f"CI {c['adverse_over_capture_pct']['ci95']}  "
          f"net {c['net_after_fee_bp']['point']} CI {c['net_after_fee_bp']['ci95']}")
    print("  (published: ratio 101.1% [97.7, 104.7]  net -0.0223 [-0.0458, 0.0008])")
    for key in ["2023svb", "2022fomc", "2020covid"]:
        s = out["stress"][key]
        rr = s["adverse_over_capture_pct"]
        nn = s["net_after_fee_bp"]
        print(f"{s['episode']:>18}: ratio {rr['point']:>6.1f}% CI {rr['ci95']} excl100={rr['excludes_100']}  "
              f"net {nn['point']:>7.4f} CI {nn['ci95']} excl0={nn['excludes_0']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
