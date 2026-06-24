"""H7 — strategy-level permutation null + DSR effective-trials.

The MM-specific zero-skill null: shuffle the SIGN of each fill's mid-drift
(equivalently, randomize whether each fill was favorably or adversely
selected), keeping the magnitude distribution intact. Under the null, a
quoter has no skill at avoiding adverse selection, so the expected net
realised spread is centered at zero. The observed statistic is compared
to the permutation distribution.

  p = (#{null >= obs} + 1) / (M + 1)   (one-sided, pre-stated direction)

The permutation resampler is the hot loop (M >= 1000 shuffles over the
per-fill markout vector). Implemented in pure Python (reference) and numba
(production) with a SHARED explicit PCG/LCG so both are bit-identical for
the same seed. DSR effective-trials deflation is applied ONLY at assembly
(reused conceptually from quant-research-framework-rs-v2/src/dsr.rs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


# A tiny, explicit splitmix64-style PRNG so the Python reference and the
# numba kernel produce BIT-IDENTICAL streams for the same seed. We only
# need uniform sign flips, so we use the top bit of each 64-bit draw.

_MASK64 = (1 << 64) - 1


def _splitmix_next(state: int):
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = z ^ (z >> 31)
    return state, z


def _perm_null_reference(markout: np.ndarray, m: int, seed: int) -> np.ndarray:
    """Pure-Python reference. For each of m permutations, flip each fill's
    sign by a fresh coin, compute the mean of the sign-flipped markout.
    Returns the m null statistics."""
    n = markout.shape[0]
    out = np.empty(m, dtype=np.float64)
    state = seed & _MASK64
    for j in range(m):
        s = 0.0
        for i in range(n):
            state, z = _splitmix_next(state)
            sign = 1.0 if (z >> 63) & 1 else -1.0
            s += sign * markout[i]
        out[j] = s / n if n > 0 else 0.0
    return out


if _HAVE_NUMBA:
    @njit(cache=True)
    def _splitmix_next_nb(state):
        state = (state + np.uint64(0x9E3779B97F4A7C15))
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return state, z

    @njit(cache=True)
    def _perm_null_numba(markout, m, seed):
        n = markout.shape[0]
        out = np.empty(m, dtype=np.float64)
        state = np.uint64(seed)
        one = np.uint64(1)
        s63 = np.uint64(63)
        for j in range(m):
            s = 0.0
            for i in range(n):
                state, z = _splitmix_next_nb(state)
                bit = (z >> s63) & one
                sign = 1.0 if bit == one else -1.0
                s += sign * markout[i]
            out[j] = s / n if n > 0 else 0.0
        return out
else:  # pragma: no cover
    _perm_null_numba = None


@dataclass
class PermNullResult:
    observed: float
    null_mean: float
    null_q05: float
    null_q95: float
    p_value: float              # one-sided P(null >= obs)
    m: int
    n_fills: int


def permutation_null(markout: np.ndarray, *, m: int = 1000, seed: int = 12345,
                     use_numba: Optional[bool] = None) -> PermNullResult:
    """Run the sign-flip permutation null on a per-fill markout vector.

    The observed statistic is the mean markout (skill = avoiding adverse
    selection => positive markout). p = one-sided P(null >= obs)."""
    markout = np.asarray(markout, dtype=np.float64)
    markout = markout[~np.isnan(markout)]
    n = markout.shape[0]
    obs = float(np.mean(markout)) if n else 0.0
    if use_numba is None:
        use_numba = _HAVE_NUMBA
    if use_numba and _perm_null_numba is not None:
        null = _perm_null_numba(markout, m, np.uint64(seed))
    else:
        null = _perm_null_reference(markout, m, seed)
    p = (np.sum(null >= obs) + 1) / (m + 1)
    return PermNullResult(
        observed=obs, null_mean=float(np.mean(null)),
        null_q05=float(np.quantile(null, 0.05)),
        null_q95=float(np.quantile(null, 0.95)),
        p_value=float(p), m=m, n_fills=n,
    )


def nominal_trial_count(sharpes: np.ndarray) -> float:
    """NOMINAL (raw) number of candidate trials, i.e. the count of finite
    Sharpe estimates. This is *not* a deflation: it is the headcount of the
    slate before any effective-trials shrinkage, used only to label the size
    of the search so that "strategies x windows" is not silently inflated.

    History / honesty note: an earlier version of this function carried a
    "variance-of-trials heuristic" docstring and computed the Sharpe variance,
    but then returned the raw count whenever the variance was > 0 -- so the
    variance was dead code and the returned value was always the nominal count.
    It was therefore never a deflated-Sharpe effective-trials estimate. We
    keep returning the nominal count (the honest description of the value) and
    have removed the misleading variance computation. A genuine
    correlation-based effective-trials estimator is available separately as
    ``effective_trials_from_correlation`` for callers that want a real
    deflation; the MM corpus does not need one, because there are 0 survivors
    under BH and BHY before any deflation is applied (the deflation is moot)."""
    sharpes = np.asarray(sharpes, dtype=np.float64)
    sharpes = sharpes[~np.isnan(sharpes)]
    return float(sharpes.size)


# Backwards-compatible alias. The name "dsr_effective_trials" is retained so
# existing call sites and saved-run JSON keys keep working, but it returns the
# NOMINAL trial count (see nominal_trial_count); it is not a deflation. New
# code should call nominal_trial_count (or, for a real deflation,
# effective_trials_from_correlation).
def dsr_effective_trials(sharpes: np.ndarray) -> float:
    """Deprecated alias for :func:`nominal_trial_count`. Returns the NOMINAL
    (raw) trial count, not a deflated-Sharpe effective-trials estimate. Kept
    for compatibility with saved-run JSON keys. See ``nominal_trial_count``."""
    return nominal_trial_count(sharpes)


def effective_trials_from_correlation(
    trial_returns: np.ndarray, *, eps: float = 1e-12
) -> float:
    """Genuine effective number of independent trials from the trial-return
    correlation matrix (the deflation an honest DSR would use). Given an
    ``(n_obs, n_trials)`` matrix of per-window (or per-bar) returns for each
    candidate strategy, we form the trial-by-trial correlation matrix R and
    return Kaiser's effective dimensionality

        N_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2 = (trace R)^2 / ||R||_F^2

    where lambda_i are the eigenvalues of R. N_eff = n_trials when the trials
    are mutually uncorrelated and collapses toward 1 as they become perfectly
    correlated. This is provided for callers that need a real deflation; it is
    NOT used by the MM quoting/OFI corpora, where no cell is significant before
    any deflation, so the effective-trials count is reported nominally and
    noted as moot. Returns NaN if fewer than two trials are usable."""
    x = np.asarray(trial_returns, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        return float("nan")
    # drop trials that are constant / all-NaN (undefined correlation)
    finite = np.all(np.isfinite(x), axis=0)
    x = x[:, finite]
    if x.shape[1] < 2:
        return float(x.shape[1])
    std = x.std(axis=0)
    keep = std > eps
    x = x[:, keep]
    if x.shape[1] < 2:
        return float(x.shape[1])
    R = np.corrcoef(x, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    fro2 = float(np.sum(R * R))
    if fro2 <= eps:
        return float(x.shape[1])
    trace = float(np.trace(R))
    return float(trace * trace / fro2)


def tail_guarded_rrr(pnl_per_window: np.ndarray) -> dict:
    """Tail-guarded reward/risk on per-window net PnL: full RRR, RRR with
    the best window removed, worst window, and skew. The H7 PASS condition
    requires the ex-best-window RRR to stay positive on the held-out slice."""
    x = np.asarray(pnl_per_window, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"rrr": float("nan"), "rrr_ex_best": float("nan"),
                "worst": float("nan"), "skew": float("nan")}
    def _rrr(v):
        wins = v[v > 0].sum()
        losses = -v[v < 0].sum()
        return float(wins / losses) if losses > 0 else float("inf")
    ex_best = np.delete(x, int(np.argmax(x))) if x.size > 1 else x
    m = x.mean()
    sd = x.std()
    skew = float(np.mean(((x - m) / sd) ** 3)) if sd > 0 else 0.0
    return {"rrr": _rrr(x), "rrr_ex_best": _rrr(ex_best),
            "worst": float(x.min()), "skew": skew}


__all__ = [
    "permutation_null", "PermNullResult", "nominal_trial_count",
    "dsr_effective_trials", "effective_trials_from_correlation",
    "tail_guarded_rrr", "_perm_null_reference", "_perm_null_numba",
]
