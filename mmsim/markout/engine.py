"""Markout engine: per-fill mid lookups + spread decomposition.

Two implementations of the hot loop (per-fill, per-horizon mid lookup
over a sorted snapshot timeline):
  - ``_markout_kernel_reference``  : pure Python / numpy, the spec.
  - ``_markout_kernel_numba``      : numba @njit, the production path.
They are bit-identical (verified in tests/test_markout.py and by the
parity script). ``compute_markout`` dispatches to numba when available
and falls back to the reference otherwise.

Causality / no-lookahead: mid(t+tau) is the most-recent snapshot mid
at-or-before fill_ts + tau. It deliberately MAY read snapshots after
the fill (that is the whole point of a forward markout), but never any
data beyond the requested horizon. The L4 pollution test pins that
appending snapshots beyond fill_ts+60s cannot change any markout at a
fill whose 60s horizon ends before them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_SEC_NS = 1_000_000_000
HORIZONS_NS = (1 * _SEC_NS, 10 * _SEC_NS, 60 * _SEC_NS)  # 1s, 10s, 60s

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


@dataclass
class MarkoutResult:
    """Column arrays aligned to the input fill order (length = n_fills)."""
    mid0: np.ndarray
    mid_1s: np.ndarray
    mid_10s: np.ndarray
    mid_60s: np.ndarray
    markout_1s: np.ndarray
    markout_10s: np.ndarray
    markout_60s: np.ndarray
    realised_spread_1s: np.ndarray
    realised_spread_10s: np.ndarray
    realised_spread_60s: np.ndarray
    adverse_1s: np.ndarray
    adverse_10s: np.ndarray
    adverse_60s: np.ndarray


def _mid_at_or_before_scalar(ts_arr, mid_arr, t):
    """Reference helper using numpy searchsorted. NaN before first snap."""
    idx = int(np.searchsorted(ts_arr, t, side="right")) - 1
    if idx < 0:
        return np.nan
    return mid_arr[idx]


def _markout_kernel_reference(
    fill_ts, fill_px, fill_side, snap_ts, snap_mid, horizons,
):
    """Pure-Python/numpy reference. Returns a (n_fills, 4) array of mids:
    [mid0, mid_h0, mid_h1, mid_h2] and is the source of truth."""
    n = fill_ts.shape[0]
    out = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        t = fill_ts[i]
        out[i, 0] = _mid_at_or_before_scalar(snap_ts, snap_mid, t)
        for k in range(3):
            out[i, k + 1] = _mid_at_or_before_scalar(
                snap_ts, snap_mid, t + horizons[k])
    return out


if _HAVE_NUMBA:
    @njit(cache=True)
    def _searchsorted_le(ts_arr, t):
        # index of last element <= t (right-side searchsorted - 1)
        lo = 0
        hi = ts_arr.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if ts_arr[mid] <= t:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    @njit(cache=True)
    def _markout_kernel_numba(fill_ts, fill_px, fill_side, snap_ts, snap_mid, horizons):
        n = fill_ts.shape[0]
        out = np.empty((n, 4), dtype=np.float64)
        nan = np.nan
        for i in range(n):
            t = fill_ts[i]
            idx0 = _searchsorted_le(snap_ts, t)
            out[i, 0] = nan if idx0 < 0 else snap_mid[idx0]
            for k in range(3):
                idxk = _searchsorted_le(snap_ts, t + horizons[k])
                out[i, k + 1] = nan if idxk < 0 else snap_mid[idxk]
        return out
else:  # pragma: no cover
    _markout_kernel_numba = None


def _decompose(mids, fill_px, fill_side):
    """Given the (n,4) mid matrix, produce the markout / realised-spread /
    adverse-selection columns. Vectorized; identical for both kernels."""
    mid0 = mids[:, 0]
    px = fill_px
    s = fill_side.astype(np.float64)
    res = {}
    for k, name in enumerate(("1s", "10s", "60s")):
        mtau = mids[:, k + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            markout = s * (mtau - px) / mid0
            realised = s * (px - mtau) / mid0
            adverse = s * (mtau - mid0) / mid0
        res[f"mid_{name}"] = mtau
        res[f"markout_{name}"] = markout
        res[f"realised_spread_{name}"] = realised
        res[f"adverse_{name}"] = adverse
    res["mid0"] = mid0
    return res


def compute_markout(fills, snap_ts, snap_mid, *, use_numba: Optional[bool] = None) -> MarkoutResult:
    """Compute markout columns for an iterable of Fill records.

    ``snap_ts`` / ``snap_mid`` are the sorted snapshot mid timeline
    (build via mmsim.ledger.writer.build_mid_timeline). ``use_numba``
    None => auto (numba if available)."""
    fill_ts = np.fromiter((int(f.ts_ns) for f in fills), dtype=np.int64)
    # second pass needs price/side; re-fetch via list to avoid exhausting
    # an iterator twice.
    fills = list(fills) if not hasattr(fills, "__len__") else fills
    fill_px = np.fromiter((float(f.price) for f in fills), dtype=np.float64)
    fill_side = np.fromiter((int(f.side) for f in fills), dtype=np.int64)
    if fill_ts.shape[0] != fill_px.shape[0]:
        # iterator was consumed by the first fromiter; rebuild from list
        fill_ts = np.array([int(f.ts_ns) for f in fills], dtype=np.int64)
    horizons = np.array(HORIZONS_NS, dtype=np.int64)

    if use_numba is None:
        use_numba = _HAVE_NUMBA
    if use_numba and _markout_kernel_numba is not None:
        mids = _markout_kernel_numba(fill_ts, fill_px, fill_side, snap_ts, snap_mid, horizons)
    else:
        mids = _markout_kernel_reference(fill_ts, fill_px, fill_side, snap_ts, snap_mid, horizons)

    d = _decompose(mids, fill_px, fill_side)
    return MarkoutResult(
        mid0=d["mid0"],
        mid_1s=d["mid_1s"], mid_10s=d["mid_10s"], mid_60s=d["mid_60s"],
        markout_1s=d["markout_1s"], markout_10s=d["markout_10s"], markout_60s=d["markout_60s"],
        realised_spread_1s=d["realised_spread_1s"], realised_spread_10s=d["realised_spread_10s"],
        realised_spread_60s=d["realised_spread_60s"],
        adverse_1s=d["adverse_1s"], adverse_10s=d["adverse_10s"], adverse_60s=d["adverse_60s"],
    )


def compute_markout_reference(fills, snap_ts, snap_mid) -> MarkoutResult:
    """Force the pure-Python reference kernel (for parity checks)."""
    return compute_markout(fills, snap_ts, snap_mid, use_numba=False)


__all__ = [
    "MarkoutResult", "compute_markout", "compute_markout_reference",
    "HORIZONS_NS", "_markout_kernel_reference", "_markout_kernel_numba",
]
