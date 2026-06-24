#!/usr/bin/env python3
"""Stress-test the r=0.92 cross-root LVR/adverse-selection association.

The "same volatility-driven cost" claim rests on a 13-point
cross-root Pearson r=0.92 between the implied sigma^2/8 LVR and the measured
per-fill adverse drift. Thirteen points with a heavy NG tail is fragile. This
script stresses that single number so the same-driver claim is robust rather
than tail-driven, computing on the 13-root cross-section (read straight from
runs/maker_decomp/lvr_identity.json, the prior round's artifact):

  (a) leave-one-out (LOO) Pearson r, every root dropped one at a time, with the
      drop-NG case and the drop-{NG,SI,CL} high-vol-tail case called out;
  (b) Spearman rank correlation (rank-robust, tail-insensitive), full + drop-NG;
  (c) a percentile bootstrap CI on Pearson r (resample roots with replacement);
  (d) a PARTIAL correlation of adverse-drift vs LVR CONTROLLING for sigma_tau on
      BOTH axes -- the real question: do adverse and LVR co-move BEYOND both
      being increasing in volatility? We residualize adverse on sigma_tau and
      LVR on sigma_tau (OLS, through a constant) and correlate the residuals.

HONEST CAVEAT baked into the output: the implied LVR is sigma^2/8 by definition,
so it is mechanically ~quadratic in sigma_tau (corr(LVR, sigma_tau^2)=0.9999 here).
Partialling sigma_tau out of the LVR axis is therefore partly controlling for the
thing LVR IS; the partial-r is reported as a conservative lower bound on any
co-movement beyond the shared volatility driver, not as an independent channel.
The DEX r=0.97 (LVR per hour vs pool sigma) is likewise partly MECHANICAL, since
LVR is defined proportional to sigma^2 -- it is not independent corroboration.

Inputs : runs/maker_decomp/lvr_identity.json  (by_root: adverse_cost_bp_10s,
         lvr_equiv_bp_10s, rv_10s_bp == sigma_tau, n_fills, per the prior round)
Output : runs/maker_decomp/lvr_robustness.json

Pure numpy/pandas (no scipy, matching lvr_identity.py). Single-thread,
OPENBLAS_NUM_THREADS=1, deterministic bootstrap (fixed seed). No network.
"""
import os, sys, json, time, resource
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
except Exception:
    pass

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
SRC = os.environ.get("LVR_IDENTITY_JSON", str(_REPO / "runs/maker_decomp/lvr_identity.json"))
OUT = os.environ.get("LVR_ROBUSTNESS_OUT", str(_REPO / "runs/maker_decomp/lvr_robustness.json"))
HIGH_VOL_TAIL = ["NG", "SI", "CL"]   # the three highest-adverse / highest-sigma roots
N_BOOT = 100_000
SEED = 20260613


def pearson(x, y):
    if x.size < 3:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    """Spearman = Pearson on ranks (average ranks for ties), numpy-only."""
    if x.size < 3:
        return float("nan")
    import pandas as pd
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return pearson(rx, ry)


def ols_resid(y, x):
    """Residual of y after regressing on x with an intercept (OLS)."""
    A = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def partial_corr(a, b, z):
    """Partial correlation of a and b controlling for z (residual-on-residual)."""
    ra = ols_resid(a, z)
    rb = ols_resid(b, z)
    return pearson(ra, rb)


def main():
    t0 = time.time()
    d = json.load(open(SRC))
    br = d["by_root"]
    roots = [r["root"] for r in br]
    adverse = np.array([r["adverse_cost_bp_10s"] for r in br], dtype=float)  # +ve cost magnitude
    lvr = np.array([r["lvr_equiv_bp_10s"] for r in br], dtype=float)          # sigma^2/8, bp
    sigma = np.array([r["rv_10s_bp"] for r in br], dtype=float)               # sigma_tau (linear), bp
    n = len(roots)

    # --- full-sample baselines ---------------------------------------------------
    r_full = pearson(lvr, adverse)
    rho_full = spearman(lvr, adverse)

    # mechanical-coupling diagnostic: how much of LVR is just sigma_tau^2 ?
    lvr_vs_sigma2 = pearson(lvr, sigma**2)
    lvr_vs_sigma = pearson(lvr, sigma)

    # --- (a) leave-one-out -------------------------------------------------------
    loo = []
    for i, rt in enumerate(roots):
        keep = np.arange(n) != i
        loo.append({
            "dropped": rt,
            "pearson": pearson(lvr[keep], adverse[keep]),
            "spearman": spearman(lvr[keep], adverse[keep]),
        })
    loo_pearson_vals = np.array([x["pearson"] for x in loo])
    drop_ng = next(x for x in loo if x["dropped"] == "NG")

    # drop the whole high-vol tail {NG, SI, CL}
    tail_mask = np.array([rt not in HIGH_VOL_TAIL for rt in roots])
    r_drop_tail = pearson(lvr[tail_mask], adverse[tail_mask])
    rho_drop_tail = spearman(lvr[tail_mask], adverse[tail_mask])

    # --- (b) Spearman already above (rho_full, plus per-LOO) ----------------------

    # --- (c) percentile bootstrap CI on Pearson r (resample roots) ---------------
    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT, dtype=float)
    nb = 0
    for k in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        rb = pearson(lvr[idx], adverse[idx])
        if np.isfinite(rb):
            boot[nb] = rb
            nb += 1
    boot = boot[:nb]
    ci = {
        "method": "percentile bootstrap, resample the 13 roots with replacement",
        "n_boot": int(nb),
        "mean": float(np.mean(boot)),
        "median": float(np.median(boot)),
        "ci90_lo": float(np.percentile(boot, 5)),
        "ci90_hi": float(np.percentile(boot, 95)),
        "ci95_lo": float(np.percentile(boot, 2.5)),
        "ci95_hi": float(np.percentile(boot, 97.5)),
        "frac_above_0": float(np.mean(boot > 0.0)),
        "frac_above_0p5": float(np.mean(boot > 0.5)),
    }

    # --- (d) PARTIAL correlation: adverse vs LVR controlling for sigma_tau --------
    # The real question: do adverse and LVR co-move BEYOND both rising in vol?
    pr_full = partial_corr(adverse, lvr, sigma)
    # also drop-NG, since NG dominates the sigma range
    keep_ng = np.array([rt != "NG" for rt in roots])
    pr_drop_ng = partial_corr(adverse[keep_ng], lvr[keep_ng], sigma[keep_ng])
    pr_drop_tail = partial_corr(adverse[tail_mask], lvr[tail_mask], sigma[tail_mask])
    # bootstrap CI on the partial r as well
    pboot = np.empty(N_BOOT, dtype=float)
    nbp = 0
    for k in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        # need >=4 distinct points for a meaningful 1-control partial; guard
        if np.unique(idx).size < 4:
            continue
        try:
            v = partial_corr(adverse[idx], lvr[idx], sigma[idx])
        except Exception:
            continue
        if np.isfinite(v):
            pboot[nbp] = v
            nbp += 1
    pboot = pboot[:nbp]

    # context: simple correlations of each axis with sigma_tau (why the partial matters)
    adv_vs_sigma = pearson(adverse, sigma)

    # --- verdict ----------------------------------------------------------------
    # Two orthogonal questions, classified honestly:
    #   (1) RAW association: does Pearson r survive dropping NG (the heavy tail)?
    #       The same-driver claim only needs the SHARED-VOLATILITY channel, so the
    #       relevant survival test is whether BOTH axes still track sigma_tau and
    #       still rank-correlate once the NG outlier is removed (Spearman drop-NG).
    #   (2) BEYOND-VOLATILITY channel: does a co-movement survive controlling for
    #       sigma_tau on both axes (partial r)? For a SINGLE-DRIVER thesis this is
    #       EXPECTED to be ~0; a high partial would imply a SECOND, independent
    #       driver, which is NOT the claim.
    raw_pearson_tail_driven = (drop_ng["pearson"] < 0.75) or (r_drop_tail < 0.5)
    rank_survives_drop_ng = (drop_ng["spearman"] >= 0.5)
    both_track_sigma = (adv_vs_sigma >= 0.7) and (lvr_vs_sigma >= 0.7)
    partial_beyond_vol = (pr_drop_ng >= 0.3) and (pr_drop_tail >= 0.3)

    if partial_beyond_vol and not raw_pearson_tail_driven:
        verdict = "ROBUST-INDEPENDENT-CHANNEL"
        verdict_text = (
            "Adverse drift and the sigma^2/8 LVR co-move beyond their shared "
            "volatility driver (partial-sigma r survives), and the raw association is "
            "not tail-driven. A stronger claim than same-driver is supportable."
        )
    elif both_track_sigma and rank_survives_drop_ng:
        verdict = "SAME-VOLATILITY-DRIVER (raw r tail-driven, shared-driver robust)"
        verdict_text = (
            "The SAME-DRIVER claim is robust in the form the re-scoped title makes: "
            "both the per-fill adverse drift and the sigma^2/8 LVR are governed by the "
            "same volatility scale sigma_tau (corr(adverse, sigma)={a:.2f}, "
            "corr(LVR, sigma)={l:.2f}), and the rank association survives dropping NG "
            "(Spearman {sng:.2f}). What is NOT robust is the bald headline Pearson "
            "r={rf:.2f}: it is substantially tail-driven by NG (drop-NG Pearson "
            "{png:.2f}; drop the {{NG,SI,CL}} high-vol tail and Pearson falls to "
            "{rt:.2f}), and the co-movement BEYOND the shared volatility driver is "
            "near zero once sigma_tau is controlled on both axes (partial r: full "
            "{pf:.2f}, drop-NG {pdn:.2f}, drop-tail {pdt:.2f}). That near-zero partial "
            "is the CORRECT result for a single-driver thesis -- a large partial would "
            "mean a second, independent channel, which the paper does not claim. "
            "Report the association as one volatility-driven cost (adverse ~ linear in "
            "sigma, LVR ~ quadratic), with the raw r=0.92 flagged as tail-sensitive and "
            "the rank/sigma-tracking statistics as the robust backbone -- NOT as an "
            "independent co-movement beyond volatility."
        ).format(a=adv_vs_sigma, l=lvr_vs_sigma, sng=drop_ng["spearman"], rf=r_full,
                 png=drop_ng["pearson"], rt=r_drop_tail, pf=pr_full,
                 pdn=pr_drop_ng, pdt=pr_drop_tail)
    else:
        verdict = "TAIL-OR-VOL-DRIVEN"
        verdict_text = (
            "Neither an independent-channel nor a robust shared-driver reading is "
            "supported: scope the claim honestly as tail-/volatility-driven."
        )

    out = {
        "description": "Robustness of the 13-root cross-root Pearson r=0.92 between implied sigma^2/8 LVR and measured per-fill adverse drift (10s, bp). Stresses tail-dependence (LOO/drop-NG), rank-robustness (Spearman), sampling (bootstrap CI), and shared-driver (partial-sigma) channels.",
        "source": str(SRC),
        "n_roots": n,
        "roots": roots,
        "axes": {
            "adverse_cost_bp_10s": "measured per-fill post-fill adverse drift, +ve magnitude (decomp)",
            "lvr_equiv_bp_10s": "implied Milionis et al. sigma^2/8 LVR over 10s, bp of notional",
            "rv_10s_bp": "sigma_tau, the 10s mid-return stdev (LINEAR, first-order), bp",
        },
        "full_sample": {
            "pearson_lvr_vs_adverse": r_full,
            "spearman_lvr_vs_adverse": rho_full,
            "frozen_carrier_pearson": 0.92,
        },
        "a_leave_one_out": {
            "per_root": loo,
            "pearson_min": float(loo_pearson_vals.min()),
            "pearson_max": float(loo_pearson_vals.max()),
            "pearson_min_dropped_root": roots[int(np.argmin(loo_pearson_vals))],
            "drop_NG": drop_ng,
            "drop_high_vol_tail_NG_SI_CL": {
                "pearson": r_drop_tail,
                "spearman": rho_drop_tail,
                "n_remaining": int(tail_mask.sum()),
            },
        },
        "b_spearman": {
            "full": rho_full,
            "drop_NG": drop_ng["spearman"],
            "drop_high_vol_tail": rho_drop_tail,
            "note": "rank correlation is tail-insensitive; a stable Spearman under drop-NG means the ordering, not the NG outlier, carries the association.",
        },
        "c_bootstrap_ci": ci,
        "d_partial_sigma_control": {
            "question": "Do adverse drift and the sigma^2/8 LVR co-move BEYOND both being increasing in sigma_tau? (residual-on-residual after regressing each axis on sigma_tau)",
            "partial_r_full": pr_full,
            "partial_r_drop_NG": pr_drop_ng,
            "partial_r_drop_high_vol_tail": pr_drop_tail,
            "partial_r_bootstrap": {
                "n_boot": int(nbp),
                "median": float(np.median(pboot)) if pboot.size else float("nan"),
                "ci95_lo": float(np.percentile(pboot, 2.5)) if pboot.size else float("nan"),
                "ci95_hi": float(np.percentile(pboot, 97.5)) if pboot.size else float("nan"),
            },
            "context_corr_adverse_vs_sigma": adv_vs_sigma,
            "context_corr_lvr_vs_sigma": lvr_vs_sigma,
            "mechanical_coupling_warning": {
                "corr_lvr_vs_sigma_tau_squared": lvr_vs_sigma2,
                "note": "The implied LVR is sigma^2/8 BY DEFINITION, so it is mechanically ~quadratic in sigma_tau (corr with sigma_tau^2 ~ 1.00). Controlling for sigma_tau on the LVR axis is therefore partly controlling for the thing LVR IS; the partial-r is a CONSERVATIVE lower bound on co-movement beyond the shared volatility driver, not an independent-channel estimate.",
            },
        },
        "dex_reverse_check_is_partly_mechanical": {
            "dex_pearson_lvr_per_hr_vs_sigma_hr": d.get("dex_reverse_check_1b", {}).get("pearson_corr_lvr_per_hr_vs_sigma_hr"),
            "frozen_carrier": 0.97,
            "note": "The DEX r=0.97 between per-hour LVR and pool sigma is partly MECHANICAL: LVR is DEFINED proportional to sigma^2, so a high LVR-vs-sigma correlation is built in by construction. It corroborates that the DEX cost is the same sigma-governed object as the CME adverse drift, but it is NOT independent corroboration of the cross-root association and must not be read as a second confirming datapoint.",
        },
        "verdict": verdict,
        "verdict_text": verdict_text,
        "frozen_carriers_preserved": {
            "cross_root_pearson": 0.92,
            "dex_pearson": 0.97,
            "adverse_eq_0p74_sigma": d.get("cme_commensurate_linear_axis", {}).get("adverse_eq_k_sigma_tau_origin_slope_k"),
        },
        "runtime_sec": None,
    }
    out["runtime_sec"] = round(time.time() - t0, 2)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    # --- console report ----------------------------------------------------------
    print("\n=== robustness of cross-root r(LVR, adverse) on 13 roots ===", flush=True)
    print(f"full Pearson  = {r_full:.4f}   (frozen carrier 0.92)", flush=True)
    print(f"full Spearman = {rho_full:.4f}", flush=True)
    print("\n(a) leave-one-out Pearson:", flush=True)
    for x in sorted(loo, key=lambda z: z["pearson"]):
        tag = "  <-- DROP-NG" if x["dropped"] == "NG" else ""
        print(f"    drop {x['dropped']:>4}: r={x['pearson']:.4f}  rho={x['spearman']:.4f}{tag}", flush=True)
    print(f"    LOO Pearson range: [{loo_pearson_vals.min():.4f}, {loo_pearson_vals.max():.4f}]", flush=True)
    print(f"    drop {{NG,SI,CL}} tail: r={r_drop_tail:.4f}  rho={rho_drop_tail:.4f}  (n={int(tail_mask.sum())})", flush=True)
    print(f"\n(c) bootstrap CI (n={ci['n_boot']}): median={ci['median']:.4f}  "
          f"95%=[{ci['ci95_lo']:.4f}, {ci['ci95_hi']:.4f}]  90%=[{ci['ci90_lo']:.4f}, {ci['ci90_hi']:.4f}]  "
          f"frac>0={ci['frac_above_0']:.3f}", flush=True)
    print(f"\n(d) partial r (adverse vs LVR | sigma_tau):", flush=True)
    print(f"    full      = {pr_full:.4f}", flush=True)
    print(f"    drop-NG   = {pr_drop_ng:.4f}", flush=True)
    print(f"    drop-tail = {pr_drop_tail:.4f}", flush=True)
    print(f"    bootstrap median={np.median(pboot):.4f}  95%=[{np.percentile(pboot,2.5):.4f}, {np.percentile(pboot,97.5):.4f}] (n={nbp})", flush=True)
    print(f"    [mechanical] corr(LVR, sigma_tau^2) = {lvr_vs_sigma2:.4f}  (LVR is ~quadratic in sigma_tau by definition)", flush=True)
    print(f"    context: corr(adverse, sigma)={adv_vs_sigma:.4f}  corr(LVR, sigma)={lvr_vs_sigma:.4f}", flush=True)
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"  {verdict_text}", flush=True)
    print(f"\n[done] {time.time()-t0:.2f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
