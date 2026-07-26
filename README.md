# quant-mm-simulator

> The simulator used in "Adverse Selection Consumes the Touch" by Daniel Gatto ([SSRN 7022599](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7022599)). Method and results: [daru.finance/research/market-making](https://daru.finance/research/market-making).

An event-driven market-making simulator: L2 book + tape replay, queue-position-aware fills,
continuous fractional inventory, a quoter contract with pluggable quoting models, a hedge engine,
and a costed multi-leg trade ledger with post-fill markout decomposition.

Sibling project to [`quant-mm-simulator-rs`](https://github.com/DaruFinance/quant-mm-simulator-rs)
(the Rust port) and [`quant-research-framework`](https://github.com/DaruFinance/quant-research-framework).
Every fill emitted is a `Leg` row in the framework's trade-log v1 schema, so the framework's
multi-leg aggregation and audit harness operate on simulator output unchanged.

## Layout
```
mmsim/
├── ingest/     L2 + tape ingestion
├── sim/        event loop, fills, queue position, inventory
├── quoter/     quoter contract, quote shapes, refresh triggers, reference prices, adverse filter
├── models/     quoting-model library (Avellaneda-Stoikov, Cartea-Jaimungal, GLFT, …)
├── hedge/      hedge engine
├── ledger/     costed per-fill ledger (streaming parquet writer)
├── markout/    post-fill mid-drift markout + realised-spread decomposition
├── research/   walk-forward adapter, gating policy, permutation null
└── cli/        replay CLI: clone → replay → emit inventory / PnL / markout JSON
scripts/        end-to-end run drivers
tests/          unit + parity tests (fixtures under tests/fixtures/)
docs/schemas/   trade-log + aux-stream schemas
```

## Install
```bash
pip install -e .
```

## Usage
```bash
python -m mmsim.cli.replay \
    --snapshots tests/fixtures/lob_btcusdt_60min_snapshots.parquet \
    --trades    tests/fixtures/lob_btcusdt_60min_trades.parquet \
    --horizon 10s --json
```
Emits fill counts, inventory-path extrema, net/gross/cost PnL, and the markout decomposition
(markout / realised spread / adverse selection). The markout and permutation hot loops have a
numba kernel verified bit-identical to a pure-Python reference.

## Cross-language parity
Each component lands here first, then is mirrored in `quant-mm-simulator-rs`; parity scripts live
under `tools/`.

## License
MIT — see `LICENSE`.
