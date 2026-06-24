"""Smoke tests for the T4 corpus generator (resumable / skip-if-exists)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mmsim.t4_corpus.t4_corpus_gen import generate_t4_corpus
from mmsim.t4_corpus.t4_runner import LEG_COLS


def test_generate_t4_corpus_smoke_n10(tmp_path: Path, fixture_root: Path):
    out_root = tmp_path / "t4_smoke"
    results = generate_t4_corpus(
        n_combos=10, seed=2026,
        assets=["BTCUSDT"],
        output_root=out_root,
        fixture_root=fixture_root,
        verbose=False,
    )
    # 10 combos × 1 asset = up to 10 results (some may error, none should
    # at the smoke surface — but we accept >= 8 in case of edge composes).
    assert len(results) >= 8
    # Every result has both files on disk.
    for r in results:
        assert Path(r.parquet_path).exists(), r.parquet_path
        assert Path(r.sidecar_path).exists(), r.sidecar_path


def test_resumable_skip_if_exists(tmp_path: Path, fixture_root: Path):
    out_root = tmp_path / "t4_resume"
    r1 = generate_t4_corpus(
        n_combos=3, seed=2026,
        assets=["BTCUSDT"],
        output_root=out_root,
        fixture_root=fixture_root,
        verbose=False,
    )
    n1 = len(r1)
    assert n1 >= 2
    # Second pass with skip_if_exists=True returns zero new results.
    r2 = generate_t4_corpus(
        n_combos=3, seed=2026,
        assets=["BTCUSDT"],
        output_root=out_root,
        fixture_root=fixture_root,
        verbose=False,
        skip_if_exists=True,
    )
    assert len(r2) == 0


def test_parquet_has_expected_columns(tmp_path: Path, fixture_root: Path):
    out_root = tmp_path / "t4_cols"
    results = generate_t4_corpus(
        n_combos=2, seed=2026,
        assets=["BTCUSDT"],
        output_root=out_root,
        fixture_root=fixture_root,
        verbose=False,
    )
    # Find at least one result with > 0 fills to inspect its columns.
    inspected = False
    for r in results:
        df = pd.read_parquet(r.parquet_path)
        # Empty parquets are written for no-fill runs; we want a non-empty.
        if df.empty:
            continue
        for col in LEG_COLS:
            assert col in df.columns, f"missing column {col}"
        # Every row's net_pnl ~= gross_pnl - fee - slippage.
        for _, row in df.iterrows():
            expected = row["gross_pnl"] - row["fee"] - row["slippage"]
            assert abs(row["net_pnl"] - expected) < 1e-9
        inspected = True
        break
    # Don't fail if every smoke combo happened to produce 0 fills; the
    # column check above is the load-bearing test on non-empty parquets.
    assert inspected or all(pd.read_parquet(r.parquet_path).empty for r in results)


def test_sidecar_records_combo_and_metrics(tmp_path: Path, fixture_root: Path):
    out_root = tmp_path / "t4_sidecar"
    results = generate_t4_corpus(
        n_combos=2, seed=2026,
        assets=["BTCUSDT"],
        output_root=out_root,
        fixture_root=fixture_root,
        verbose=False,
    )
    assert results, "expected at least one result"
    for r in results:
        sidecar = json.loads(Path(r.sidecar_path).read_text())
        assert sidecar["asset"] == "BTCUSDT"
        for axis in ("quoting_model", "inventory_penalty", "adverse_filter",
                      "hedge_mode", "reference_price", "quote_shape",
                      "refresh_trigger"):
            assert axis in sidecar["combo"]
        for k in ("n_fills", "total_cost", "total_fees", "total_slippage",
                   "gross_pnl", "net_pnl"):
            assert k in sidecar["metrics"]
