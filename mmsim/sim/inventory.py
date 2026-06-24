"""Continuous fractional inventory tracker.

Tracks the sim's net inventory as an ``f64`` (not integer shares —
crypto venues all support fractional sizes; the ``MIN_NOTIONAL`` /
``LOT_SIZE`` filters are far below the per-fill granularity we
observe in DS-LOB-1H).

## Convention

`inv_t = Σ signed_fill.size for every fill with ts_ns <= t`, where
`signed_fill.size = fill.size * fill.side` (so a bid fill of 0.001
is +0.001 inventory; an ask fill of 0.001 is -0.001).

The convention is symmetric for both maker and taker fills — both
move inventory in the same direction since they net the same
position change for us.

## Causality

Every state mutation reads only the current Fill record. Polluting
fills past T cannot change `inv` at any T' <= T. (Leak-freedom at the
*Fill* layer is the fill model's responsibility; here we only bind the
inventory accumulator to that.)

## API

```
tracker = InventoryTracker()
for fill in fills:
    tracker.observe(fill)
print(tracker.inv)        # signed inventory
print(tracker.peak_long)  # max inv ever held
print(tracker.peak_short) # min inv ever held (most negative)
```

Or as a one-shot pure function:
```
trace = inventory_path(fills)  # InventoryTrace
```

The trace records `(ts_ns, inv)` after every fill, plus the path
extrema so the test gate can pin "max long / max short" as
regression targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

from mmsim.sim.loop import Fill


@dataclass(frozen=True)
class InventorySample:
    ts_ns: int
    inv: float
    fill_id: int        # the fill that produced this sample
    fill_signed_size: float


@dataclass(frozen=True)
class InventoryTrace:
    samples: List[InventorySample]
    final_inv: float
    peak_long: float          # most positive inv ever
    peak_short: float         # most negative inv ever
    n_fills: int


class InventoryTracker:
    """Stateful, fill-by-fill inventory accumulator."""

    def __init__(self):
        self.inv: float = 0.0
        self.peak_long: float = 0.0
        self.peak_short: float = 0.0
        self.samples: List[InventorySample] = []

    def observe(self, fill: Fill) -> None:
        signed = float(fill.size) * float(fill.side)
        self.inv += signed
        if self.inv > self.peak_long:
            self.peak_long = self.inv
        if self.inv < self.peak_short:
            self.peak_short = self.inv
        self.samples.append(InventorySample(
            ts_ns=int(fill.ts_ns),
            inv=self.inv,
            fill_id=int(fill.fill_id),
            fill_signed_size=signed,
        ))

    def trace(self) -> InventoryTrace:
        return InventoryTrace(
            samples=list(self.samples),
            final_inv=self.inv,
            peak_long=self.peak_long,
            peak_short=self.peak_short,
            n_fills=len(self.samples),
        )


def inventory_path(fills: Iterable[Fill]) -> InventoryTrace:
    """One-shot convenience: feed an iterable of fills, return trace."""
    tr = InventoryTracker()
    for f in fills:
        tr.observe(f)
    return tr.trace()


__all__ = [
    "InventorySample", "InventoryTrace", "InventoryTracker",
    "inventory_path",
]
