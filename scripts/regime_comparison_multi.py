#!/usr/bin/env python3
"""Multi-regime maker-P&L decomposition + the reconstruction control.

The headline calm cancellation (high-resolution reconstruction, ~100%, net~0) was
compared in an early draft against a coarser-resolution stress week and called
'regime-dependent'. Adding four more episodes and a SAME-DATES control shows that
comparison is confounded by the time resolution of the book reconstruction: on identical
Jul-2023 dates the high-resolution reconstruction gives ~99.5% and a coarser-resolution
one ~82% (a ~17pp gap from the reconstruction alone). WITHIN the coarser reconstruction,
the adverse/capture ratio is flat (~79-88%) across realised vol from sigma_tau 1.0 to 7.3.

Writes runs/maker_decomp/regime_comparison_multi.json and the reconstruction-control figure.
"""
import os
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIGDIR = Path(os.environ.get("XREGIME_FIGDIR", "figs"))

# (key, label, feed, decomp_summary.json, rv_source)
EPISODES = [
    ("calm2023_hires",  "2023 calm (hires)", "hires",  REPO/"runs/maker_decomp/decomp_summary.json",           "lvr"),
    ("2021calm",     "2021 calm",        "coarse", REPO/"runs/maker_decomp_2021calm/decomp_summary.json",  None),
    ("coarsectrl2023",  "2023 calm (coarse ctrl)", "coarse", REPO/"runs/maker_decomp_2023coarsectrl/decomp_summary.json", None),
    ("svb2023",      "2023 SVB",         "coarse", REPO/"runs/maker_decomp_2023svb/decomp_summary.json",   None),
    ("fomc2022",     "2022 FOMC",        "coarse", REPO/"runs/maker_decomp_jun2022/decomp_summary.json",   None),
    ("covid2020",    "2020 COVID",       "coarse", REPO/"runs/maker_decomp_2020covid/decomp_summary.json", None),
]


def calm_rv():
    d = json.load(open(REPO/"runs/maker_decomp/lvr_identity.json"))["by_root"]
    rv = np.array([r["rv_10s_bp"] for r in d]); w = np.array([r["n_fills"] for r in d])
    return float((rv*w).sum()/w.sum())


def load_ep(key, label, feed, path, rv_source):
    o = json.load(open(path))["overall"]
    sp, ad = o["spread_capture_bp_10s"], o["adverse_select_bp_10s"]
    rv = calm_rv() if rv_source == "lvr" else o.get("rv_10s_bp", float("nan"))
    return {"key": key, "label": label, "feed": feed, "n_fills": int(o["n_fills"]),
            "spread_capture_bp": round(sp, 4), "adverse_select_bp": round(ad, 4),
            "net_after_fee_bp": round(o["net_after_fee_bp_10s"], 4),
            "adverse_over_capture_pct": round(-100*ad/sp, 2), "rv_10s_bp": round(rv, 4)}


def main():
    recs = [load_ep(*e) for e in EPISODES if e[3].exists()]
    recs.sort(key=lambda r: r["rv_10s_bp"])
    fc = json.load(open(REPO/"runs/maker_decomp/feed_control_compare.json"))
    out = {"episodes": recs, "feed_control": fc,
           "note": ("Adverse/capture ratio (|adverse|/capture, 10s, fill-weighted). The high- and "
                    "coarse-resolution reconstructions are not comparable (control: ~17pp gap on identical dates). "
                    "Within the coarse reconstruction the ratio is flat ~79-88% across sigma_tau 1.0-7.3.")}
    (REPO/"runs/maker_decomp/regime_comparison_multi.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'episode':<24}{'feed':<8}{'rv':>7}{'capture':>9}{'adverse':>9}{'net-fee':>9}{'adv/cap':>9}{'fills':>10}")
    for r in recs:
        print(f"{r['label']:<24}{r['feed']:<8}{r['rv_10s_bp']:>7.2f}{r['spread_capture_bp']:>9.3f}"
              f"{r['adverse_select_bp']:>9.3f}{r['net_after_fee_bp']:>9.3f}"
              f"{r['adverse_over_capture_pct']:>8.1f}%{r['n_fills']:>10,}")
    print(f"\nCONTROL (same dates {fc['control_dates'][0]}-{fc['control_dates'][-1][-2:]}): "
          f"hires={fc['hires_feed']['adverse_over_capture_pct']}% vs coarse="
          f"{fc['coarse_feed']['adverse_over_capture_pct']}%  gap={fc['ratio_gap_pp']}pp")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        leg = [r for r in recs if r["feed"] == "coarse"]
        hires = [r for r in recs if r["feed"] == "hires"]
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        ax.axhline(100, color="0.6", lw=0.8, ls="--", zorder=1)
        # coarse band
        lr = [r["adverse_over_capture_pct"] for r in leg]
        ax.axhspan(min(lr), max(lr), color="#a6611a", alpha=0.08, zorder=0)
        ax.plot([r["rv_10s_bp"] for r in leg], lr, "o", color="#a6611a", ms=8, zorder=3,
                label="coarser-resolution reconstruction")
        for r in leg:
            ax.annotate(f"{r['label']}\n{r['adverse_over_capture_pct']:.0f}%",
                        (r["rv_10s_bp"], r["adverse_over_capture_pct"]),
                        textcoords="offset points", xytext=(0, -22), ha="center", fontsize=6.8)
        ax.plot([r["rv_10s_bp"] for r in hires], [r["adverse_over_capture_pct"] for r in hires],
                "D", color="#1f4e79", ms=9, zorder=4, label="nanosecond feed")
        for r in hires:
            ax.annotate(f"{r['label']}\n{r['adverse_over_capture_pct']:.0f}%",
                        (r["rv_10s_bp"], r["adverse_over_capture_pct"]),
                        textcoords="offset points", xytext=(0, 10), ha="center", fontsize=6.8)
        # feed gap on identical control dates
        xg = 1.08
        yv = fc["hires_feed"]["adverse_over_capture_pct"]; yl = fc["coarse_feed"]["adverse_over_capture_pct"]
        ax.annotate("", (xg, yv), (xg, yl), arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
        ax.text(xg*1.04, (yv+yl)/2, f"{fc['ratio_gap_pp']:.0f}pp\nreconstruction gap\n(same dates)",
                fontsize=7.2, va="center")
        ax.set_xscale("log")
        ax.set_xlabel(r"realised volatility $\sigma_\tau$ (10s mid-return stdev, bp)  $\rightarrow$ more stress")
        ax.set_ylabel("adverse selection / captured spread (%)")
        ax.set_title("The ratio is flat across regimes within a reconstruction; the gap between reconstructions is an artifact",
                     fontsize=9.5)
        ax.legend(loc="center right", fontsize=8, frameon=False)
        ax.set_ylim(70, 108)
        fig.tight_layout()
        for p in (FIGDIR/"fig_feed_confound.pdf", REPO/"runs/maker_decomp/fig_feed_confound.pdf"):
            fig.savefig(p)
        print(f"[fig] -> {FIGDIR/'fig_feed_confound.pdf'}")
    except Exception as e:
        print(f"  ! figure skipped: {e}")


if __name__ == "__main__":
    main()
