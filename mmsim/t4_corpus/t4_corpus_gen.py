"""T4 corpus generator.

Walks (combos × assets) calling :func:`run_t4_combo` for each
combination and writing batched parquets + sidecar JSONs to
``{output_root}/{asset}/{combo_hash}.parquet`` (plus ``.json``).

**Discipline**: real LOB fixtures only; resumable via skip-if-exists;
single-process serial by default. Default ``n_combos=10`` so the
generator is provably usable in seconds without burning hours.

Per-combo output:
  - parquet with one row per fill (columns per :data:`LEG_COLS`)
  - sidecar JSON describing the combo + IS params + cost-aware metrics
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from mmsim.ingest.lob import EventStream, load_lob

from .combos import (
    Combo, combo_to_dict, combo_to_strategy_name, pre_prune_combos, sample_combos,
)
from .search_space import search_space_t4
from .t4_runner import LEG_COLS, T4RunResult, run_t4_combo


# Default corpus root; user can override via the T4_MM_CORPUS_ROOT env var.
DEFAULT_CORPUS_ROOT = Path(os.environ.get("T4_MM_CORPUS_ROOT", "data/T4_MM_Corpus"))


@dataclass
class T4CorpusResult:
    """One generator output entry — per (combo, asset) pair."""
    asset: str
    combo_name: str
    combo_dict: dict
    params: dict
    metrics: dict
    n_fills: int
    n_maker_fills: int
    n_taker_fills: int
    parquet_path: str
    sidecar_path: str
    elapsed_s: float


def _combo_hash(combo_dict: dict, params: dict, asset: str) -> str:
    """Stable 12-char hash uniquely identifying a (combo, params, asset)
    output. Sort keys deterministically before hashing."""
    blob = json.dumps({"combo": combo_dict, "params": params, "asset": asset},
                       sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _write_legs(leg_rows: list[tuple], out_path: Path) -> None:
    """Write per-fill rows to parquet. An empty fills list still writes a
    well-formed empty parquet so skip-if-exists works correctly."""
    if not leg_rows:
        df = pd.DataFrame(columns=list(LEG_COLS))
    else:
        df = pd.DataFrame(leg_rows, columns=list(LEG_COLS))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")


def _write_sidecar(
    out_path: Path,
    *,
    asset: str,
    combo_dict: dict,
    params: dict,
    metrics: dict,
    n_fills: int,
) -> None:
    payload = {
        "asset": asset,
        "combo": combo_dict,
        "params": params,
        "metrics": metrics,
        "n_fills": n_fills,
        "schema_version": 1,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def _resolve_fixture_paths(
    asset: str,
    fixture_root: Optional[Path],
) -> tuple[Path, Path]:
    """Resolve (snapshots_parquet, trades_parquet) for ``asset``.

    For the smoke + parity surface we use the DS-LOB-1H BTCUSDT fixture
    bundled in mmsim's tests. Future assets would extend this resolver
    to look up the canonical capture path per asset.
    """
    if fixture_root is None:
        # Default: use the mmsim test fixtures.
        fixture_root = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    snap = fixture_root / "lob_btcusdt_60min_snapshots.parquet"
    trade = fixture_root / "lob_btcusdt_60min_trades.parquet"
    return snap, trade


def generate_t4_corpus(
    *,
    n_combos: int = 10,
    seed: int = 2026,
    assets: Optional[List[str]] = None,
    output_root: Path = DEFAULT_CORPUS_ROOT,
    fixture_root: Optional[Path] = None,
    is_params: Optional[dict] = None,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> List[T4CorpusResult]:
    """Generate one corpus slice (combos × assets).

    Parameters
    ----------
    n_combos:
        Number of structural combos to sample. Default ``10`` keeps the
        smoke surface fast (seconds, not minutes).
    seed:
        RNG seed for combo sampling. Default 2026.
    assets:
        List of asset tags to drive. Default ``["BTCUSDT"]`` — the
        DS-LOB-1H fixture's only asset.
    output_root:
        Root directory for parquet + JSON outputs.
    fixture_root:
        Override for the LOB fixture directory. Defaults to the mmsim
        test fixtures (DS-LOB-1H / DS-LOB-30s).
    is_params:
        IS-axis params dict shared across all combos. Pass ``None`` to
        use the composer's bundled defaults.
    skip_if_exists:
        When True, skip combos whose parquet output already exists.
    verbose:
        Print progress to stdout.

    Returns
    -------
    List[T4CorpusResult]
        One entry per (combo, asset) produced.
    """
    if assets is None:
        assets = ["BTCUSDT"]
    if is_params is None:
        is_params = {}
    output_root = Path(output_root)

    raw_combos = sample_combos(n_combos, seed=seed)
    combos = pre_prune_combos(raw_combos)

    results: List[T4CorpusResult] = []
    t0 = time.time()

    if verbose:
        print(f"=== T4 corpus gen — n_combos={n_combos} (post-prune={len(combos)}), "
              f"seed={seed}, assets={assets} ===")

    # Per-asset stream cache: load once, reuse across all combos for the
    # same asset.
    stream_cache: dict[str, EventStream] = {}

    for asset in assets:
        if asset not in stream_cache:
            snap, trade = _resolve_fixture_paths(asset, fixture_root)
            if not snap.exists() or not trade.exists():
                if verbose:
                    print(f"  SKIP {asset}: fixture not found "
                          f"(snap={snap.exists()}, trade={trade.exists()})")
                continue
            stream_cache[asset] = load_lob(snap, trade)
            if verbose:
                print(f"  loaded {asset}: n_events={len(stream_cache[asset])}")
        stream = stream_cache[asset]

        for combo in combos:
            combo_d = combo_to_dict(combo)
            h = _combo_hash(combo_d, is_params, asset)
            out_dir = output_root / asset
            p_path = out_dir / f"{h}.parquet"
            s_path = out_dir / f"{h}.json"
            if skip_if_exists and p_path.exists() and s_path.exists():
                if verbose:
                    print(f"    SKIP existing: {asset}/{h}")
                continue
            t_combo = time.time()
            try:
                rr = run_t4_combo(combo, is_params, stream, asset=asset)
            except Exception as e:
                if verbose:
                    print(f"    ERR {asset}/{h}: {type(e).__name__}: {e}")
                continue
            _write_legs(rr.leg_rows, p_path)
            _write_sidecar(s_path, asset=asset, combo_dict=combo_d,
                            params=dict(is_params), metrics=rr.metrics,
                            n_fills=rr.n_fills)
            elapsed = time.time() - t_combo
            results.append(T4CorpusResult(
                asset=asset,
                combo_name=combo_to_strategy_name(combo),
                combo_dict=combo_d,
                params=dict(is_params),
                metrics=rr.metrics,
                n_fills=rr.n_fills,
                n_maker_fills=rr.n_maker_fills,
                n_taker_fills=rr.n_taker_fills,
                parquet_path=str(p_path),
                sidecar_path=str(s_path),
                elapsed_s=elapsed,
            ))
            if verbose:
                print(f"    {asset}/{h}  fills={rr.n_fills:>4}  "
                      f"cost={rr.metrics['total_cost']:.4f}  "
                      f"elapsed={elapsed:.2f}s")

    if verbose:
        elapsed = time.time() - t0
        n_results = len(results)
        n_fills = sum(r.n_fills for r in results)
        print(f"=== DONE: {n_results} runs written, "
              f"{n_fills:,} fills in {elapsed:.1f}s ===")

    return results


__all__ = [
    "generate_t4_corpus", "T4CorpusResult", "DEFAULT_CORPUS_ROOT",
]
