#!/usr/bin/env python3
"""Continuous cross-regime sweep figure (frozen3, n=33,956). Two panels:
(a) composition-confound: asset-block mix shifts across sigma-regimes (the Simpson driver);
(b) capture-vs-adverse balance mechanism per root (drivers: adverse outpaces capture -> net deepens).
All numbers from continuous/frozen3 + frozen3_results.json. Deterministic. Output: paper figs/.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json, glob
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

def r3(x):  # round-half-up to 3dp so -0.0355 -> -0.036 (match caption/body convention)
    return float(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SRC = "continuous/frozen3_by_rootday.parquet"
OUT = os.environ.get("XREGIME_FIG_OUT", "continuous/fig_xregime_gradient.pdf")

BLOCK = {"ES": "equity", "NQ": "equity", "RTY": "equity",
         "ZF": "rates", "ZN": "rates", "ZB": "rates",
         "CL": "energy", "NG": "energy",
         "GC": "metals", "SI": "metals", "HG": "metals",
         "6E": "fx", "6J": "fx"}
BLOCK_COLOR = {"equity": "#4C72B0", "rates": "#55A868", "energy": "#C44E52",
               "metals": "#8172B3", "fx": "#CCB974"}
BLOCK_ORDER = ["equity", "rates", "energy", "metals", "fx"]
DRV = {"NG", "HG", "CL", "6E", "6J"}

res = json.load(open("continuous/frozen3_results.json"))

# ---- panel (a) data: asset-block share within each sigma-regime ----
import pandas as pd
rows = pd.read_parquet(SRC).to_dict("records")
import pandas as pd
df = pd.DataFrame(rows)
df = df[df["n_fills"] >= 20].copy()
df["rv"] = df["rv_10s_bp"]; df["block"] = df["root"].map(BLOCK)
df = df.dropna(subset=["rv", "block"])
bins = [("calm\n$\\sigma<1$", df[df.rv < 1.0]),
        ("moderate\n$1\\leq\\sigma<2$", df[(df.rv >= 1.0) & (df.rv < 2.0)]),
        ("stress\n$\\sigma\\geq2$", df[df.rv >= 2.0])]
# share matrix: rows=regimes, cols=blocks
shares = np.zeros((len(bins), len(BLOCK_ORDER)))
ns = []
for i, (_, sub) in enumerate(bins):
    ns.append(len(sub))
    vc = sub["block"].value_counts(normalize=True)
    for j, b in enumerate(BLOCK_ORDER):
        shares[i, j] = vc.get(b, 0.0) * 100.0

# ---- panel (b) data: per-root capture/adverse channels, sorted by net slope ----
cpr = res["channel_per_root"]; prs = res["per_root_slope"]
roots = sorted(cpr.keys(), key=lambda r: prs[r]["net"])  # most negative net first
cap = np.array([cpr[r]["d_capture"] for r in roots])
adv = np.array([cpr[r]["d_adverse"] for r in roots])
net = np.array([cpr[r]["d_net"] for r in roots])

# ================= plot =================
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.3), gridspec_kw={"width_ratios": [1.0, 1.35]})

# panel (a): stacked horizontal bars
yreg = np.arange(len(bins))[::-1]  # calm at top
left = np.zeros(len(bins))
for j, b in enumerate(BLOCK_ORDER):
    axA.barh(yreg, shares[:, j], left=left, color=BLOCK_COLOR[b], edgecolor="white",
             linewidth=0.6, label=b)
    left += shares[:, j]
axA.set_yticks(yreg)
axA.set_yticklabels([f"{lbl}\n(n={n:,})" for (lbl, _), n in zip(bins, ns)], fontsize=8)
axA.set_xlim(0, 100); axA.set_xlabel("share of root-days by asset block (%)", fontsize=8.5)
axA.set_title("(a) the pooled $\\sigma$-curve is composition-confounded", fontsize=9.5, loc="left")
axA.legend(handles=[Patch(facecolor=BLOCK_COLOR[b], label=b) for b in BLOCK_ORDER],
           ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16),
           frameon=False, columnspacing=1.0, handlelength=1.2)
axA.text(0.5, -0.34, f"high-$\\sigma$ days are a different root set than calm days, so the\n"
         f"pooled adverse-to-captured rank correlation with $\\sigma$ is only "
         f"{res['pooled_spearman_ratio_sigma']:.3f} (a Simpson artifact)",
         transform=axA.transAxes, ha="center", va="top", fontsize=7.2, color="#333333")
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)

# panel (b): diverging capture(up)/adverse(down) bars + net marker
x = np.arange(len(roots))
axB.bar(x, cap, color="#2c7fb8", width=0.66, label="$d$ capture$/d\\sigma$ (spread widens)")
axB.bar(x, adv, color="#d95f0e", width=0.66, label="$d$ adverse$/d\\sigma$ (pickoff deepens)")
axB.plot(x, net, "o", color="black", ms=4.2, label="$d$ net$/d\\sigma$ (= sum)", zorder=5)
axB.axhline(0, color="0.4", lw=0.8)
axB.set_xticks(x)
axB.set_xticklabels([("$\\mathbf{%s}$" % r if r in DRV else r) for r in roots], fontsize=7.8)
# bracket the drivers
drv_idx = [i for i, r in enumerate(roots) if r in DRV]
for i in drv_idx:
    axB.axvspan(i - 0.45, i + 0.45, color="#fdae6b", alpha=0.12, zorder=0)
axB.set_ylabel("$\\Delta$ per bp of $\\sigma$ (bp)", fontsize=8.5)
axB.set_title("(b) same two channels in every root; only the balance differs", fontsize=9.5, loc="left")
# legend below the x-axis (mirrors panel (a)), so it no longer overlaps the bars
axB.legend(fontsize=7.2, loc="upper center", frameon=False, ncol=3,
           bbox_to_anchor=(0.5, -0.13), columnspacing=1.3, handlelength=1.4)
# aggregate-slope annotation moves into the freed upper-right corner
axB.text(0.985, 0.975,
         f"aggregate net slope ${r3(res['fe_full']['net']):.3f}\\,$bp "
         f"(CI ${res['fe_full']['net_ci'][0]:.3f},{res['fe_full']['net_ci'][1]:.3f}$)\n"
         f"$\\to {res['fe_drop_slope_drivers']['net']:.3f}$ dropping the 5 drivers (CI incl. 0)",
         transform=axB.transAxes, ha="right", va="top", fontsize=7.0, color="#333333")
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
axB.text(np.mean(drv_idx), axB.get_ylim()[1]*0.92, "drivers", ha="center", fontsize=7.5,
         color="#a63603", style="italic")

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.subplots_adjust(wspace=0.22)
plt.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
print("panel a regime ns:", ns)
print("panel b roots (sorted by net):", roots)
print("net slopes:", [round(float(v), 4) for v in net])
