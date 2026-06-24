"""Locked pre-registration thresholds as code.

These constants are the machine-readable copy of the locked thresholds
that the CLI and the scorecard read, so verdicts cannot silently drift
from them.
"""
from __future__ import annotations

HEADLINE_HORIZON = "10s"

# Cost model (locked; crypto defaults; spot fixture -> funding 0)
COST = {
    "taker_fee": 0.0005, "maker_fee": 0.0002, "slippage": 0.0002,
    "funding_per_8h": 0.0001, "spot_funding": 0.0,
}

HOLDOUT_MINUTES = 10.0

# Per-hypothesis locked thresholds.
HYPOTHESES = {
    "H1": {"claim": "Replay parity + queue-aware != naive",
           "parity_tol": 1e-9, "fill_count_ratio_min": 2.0,
           "verdicts": ["PASS", "FALSIFIED"]},
    "H2": {"claim": "A-S inventory skew reduces inventory CVaR-95 at equal mean PnL",
           "perm_p_max": 0.05, "verdicts": ["PASS", "FALSIFIED"]},
    "H3": {"claim": "Markout-gated quote-pull improves OOS net realised spread",
           "perm_p_max": 0.05, "verdicts": ["PASS", "FALSIFIED"]},
    "H4": {"claim": "Regime-aware spread beats best static spread (OOS)",
           "verdicts": ["PASS", "NULL"]},
    "H5": {"claim": "Param-surface smoothness predicts OOS realised spread",
           "perm_p_max": 0.05, "verdicts": ["PASS", "NULL"]},
    "H7": {"claim": "Net realised spread outside zero-skill perm null + tail-guarded RRR",
           "perm_p_max": 0.05, "dsr": "effective-trials at assembly only",
           "verdicts": ["PASS", "FALSIFIED"]},
    "H8": {"claim": "Shippable artifact (reproducible repo + CLI)",
           "verdicts": ["PASS", "FALSIFIED"]},
}

DATA_SCOPE = (
    "One 60-min single-venue Binance-spot BTCUSDT L2 capture (35,989 snapshots "
    "+ 499,887 trades) + a 30-s smoke fixture. No ETH/SOL, no 2nd CEX, no DEX. "
    "WFO scope shrinks to the captured history; H2/H4/H5 are screening-grade."
)

__all__ = ["HEADLINE_HORIZON", "COST", "HOLDOUT_MINUTES", "HYPOTHESES", "DATA_SCOPE"]
