"""Shared pytest fixtures for T4 corpus tests.

Real LOB fixtures only — no synthetic streams. The 30sec fixture is the
default for smokes (~600 events, ~50ms load); the 60min fixture is the
parity / cost-identity reference (~70k events).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mmsim.ingest.lob import EventStream, load_lob

# Resolve fixtures relative to the mmsim test fixtures dir.
_FIX_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
SNAP_30S = _FIX_DIR / "lob_btcusdt_30sec_snapshots.parquet"
TRADE_30S = _FIX_DIR / "lob_btcusdt_30sec_trades.parquet"
SNAP_1H = _FIX_DIR / "lob_btcusdt_60min_snapshots.parquet"
TRADE_1H = _FIX_DIR / "lob_btcusdt_60min_trades.parquet"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"missing fixture: {path}")
    return path


@pytest.fixture(scope="session")
def stream_30s() -> EventStream:
    """DS-LOB-30sec real LOB event stream. Fast smoke surface."""
    return load_lob(_require(SNAP_30S), _require(TRADE_30S))


@pytest.fixture(scope="session")
def stream_1h() -> EventStream:
    """DS-LOB-1H real LOB event stream. Parity + cost-identity surface."""
    return load_lob(_require(SNAP_1H), _require(TRADE_1H))


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    """Root for the corpus_gen fixture resolver."""
    return _FIX_DIR
