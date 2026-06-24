"""Event-driven market-making simulator.

Sibling project to `quant-research-framework`. Consumes the trade-log
v1 schema verbatim — every emitted fill is a `Leg` row consumable by
`backtester.ledger.aggregate_legs`.

Subpackages stay lazy-importing so a consumer that only needs, say,
ingest doesn't have to pay for the quoter/models import cost.
"""
__version__ = "0.0.1"
