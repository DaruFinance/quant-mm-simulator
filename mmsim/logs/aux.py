"""Auxiliary log streams.

Three sidecar logs that ride alongside the trade ledger for each sim
run and feed downstream analytics:

  - **FillRateLogger** — per-second fill-rate buckets `(n_fills,
    total_filled_qty)`.  Required by the `fill-rate > floor` constraint
    in the IS objective.
  - **InventoryLogger** — per-second inventory snapshots.  Required by
    the `Var(inventory)` term in the IS objective.
  - **QueuePosLogger** — per-snapshot queue-position observations for
    every active resting order.  Required by post-hoc latency / fill
    attribution analysis.

All three are **downstream-only**: every emitted row at time `t`
derives from sim state at or before `t`.  No look-ahead — observe()
mutations only depend on the current event being passed in.

## Convention

Buckets are computed by `t_ns // 1_000_000_000`.  For an event stream
spanning `[t0, t1]`, the fill-rate / inventory streams expose one row
per second-bucket between `floor(t0 / 1e9)` and `floor(t1 / 1e9)`
inclusive, *carrying forward* the last-observed inventory into seconds
with no fills.  Empty fill-rate buckets get `(n_fills=0, total_qty=0)`.

QueuePosLogger emits one row per (snapshot, active_order) pair — its
row count depends on the active order count over time.

## Causality

Each logger holds only running state; observe_*(event) reads only
the event passed in.  No log row at time `t` reflects any event with
ts_ns > t.  The 5-snapshot reconciliation test in
`tests/test_logs_aux.py` proves this by hand-deriving the expected
fill-rate / inv / queue-pos at 5 bucket timestamps from the raw fill
list and comparing to the CSV output.

## Output

`AuxLogs.to_csvs(out_dir)` writes three files:
  - `fill_rate.csv`  — `bucket_s,t_ns_start,n_fills,total_qty`
  - `inventory.csv`  — `bucket_s,t_ns_start,inv,n_fills_so_far`
  - `queue_pos.csv`  — `t_ns,order_id,side,price,queue_pos,frozen`
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from mmsim.sim.loop import Fill


NS_PER_S: int = 1_000_000_000


# --------------------------------------------------------------------- #
# FillRateLogger
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class FillRateBucket:
    bucket_s: int           # t_ns // 1_000_000_000
    t_ns_start: int         # bucket_s * 1_000_000_000
    n_fills: int
    total_qty: float


class FillRateLogger:
    """Per-second fill-rate accumulator.

    `observe_fill(fill)` increments the bucket keyed by
    `fill.ts_ns // 1_000_000_000`.  `buckets()` returns a contiguous
    list of buckets between (optionally specified) start/end seconds,
    filling empty seconds with zero rows.
    """

    def __init__(self):
        self._n: Dict[int, int] = {}
        self._qty: Dict[int, float] = {}
        self._min_bucket: Optional[int] = None
        self._max_bucket: Optional[int] = None

    def observe_fill(self, fill: Fill) -> None:
        b = int(fill.ts_ns) // NS_PER_S
        self._n[b] = self._n.get(b, 0) + 1
        self._qty[b] = self._qty.get(b, 0.0) + float(fill.size)
        if self._min_bucket is None or b < self._min_bucket:
            self._min_bucket = b
        if self._max_bucket is None or b > self._max_bucket:
            self._max_bucket = b

    def buckets(
        self,
        start_s: Optional[int] = None,
        end_s: Optional[int] = None,
    ) -> List[FillRateBucket]:
        """Contiguous bucket list, filling empties with zero rows.

        `start_s` / `end_s` default to the observed range; pass them
        explicitly when you want a known grid (e.g., to match the
        sim-time grid even if the boundary seconds had no fills)."""
        if start_s is None:
            start_s = self._min_bucket if self._min_bucket is not None else 0
        if end_s is None:
            end_s = self._max_bucket if self._max_bucket is not None else start_s - 1
        out: List[FillRateBucket] = []
        for b in range(int(start_s), int(end_s) + 1):
            out.append(FillRateBucket(
                bucket_s=b,
                t_ns_start=b * NS_PER_S,
                n_fills=self._n.get(b, 0),
                total_qty=self._qty.get(b, 0.0),
            ))
        return out


# --------------------------------------------------------------------- #
# InventoryLogger
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class InventoryBucket:
    bucket_s: int
    t_ns_start: int
    inv: float              # carry-forward from last observed inv
    n_fills_so_far: int     # cumulative fill count at end of bucket


class InventoryLogger:
    """Per-second inventory snapshots.

    Tracks running inventory (signed sum of `fill.size * fill.side`)
    and exposes per-second snapshots.  Buckets with no fills carry
    forward the most-recent inventory.  Easiest cadence: every fill
    pushes inv update; bucket query takes the last value at or
    before `bucket_end_ns`.
    """

    def __init__(self):
        self._inv: float = 0.0
        self._cum_fills: int = 0
        # End-of-bucket inv: the inv after the last fill within
        # bucket_s, OR carry-forward from earlier if no fill landed
        # this bucket.
        self._end_of_bucket_inv: Dict[int, float] = {}
        self._end_of_bucket_count: Dict[int, int] = {}
        self._min_bucket: Optional[int] = None
        self._max_bucket: Optional[int] = None

    def observe_fill(self, fill: Fill) -> None:
        signed = float(fill.size) * float(fill.side)
        self._inv += signed
        self._cum_fills += 1
        b = int(fill.ts_ns) // NS_PER_S
        self._end_of_bucket_inv[b] = self._inv
        self._end_of_bucket_count[b] = self._cum_fills
        if self._min_bucket is None or b < self._min_bucket:
            self._min_bucket = b
        if self._max_bucket is None or b > self._max_bucket:
            self._max_bucket = b

    def buckets(
        self,
        start_s: Optional[int] = None,
        end_s: Optional[int] = None,
    ) -> List[InventoryBucket]:
        if start_s is None:
            start_s = self._min_bucket if self._min_bucket is not None else 0
        if end_s is None:
            end_s = self._max_bucket if self._max_bucket is not None else start_s - 1
        out: List[InventoryBucket] = []
        # Carry-forward state walks across buckets in order.
        cur_inv = 0.0
        cur_count = 0
        for b in range(int(start_s), int(end_s) + 1):
            if b in self._end_of_bucket_inv:
                cur_inv = self._end_of_bucket_inv[b]
                cur_count = self._end_of_bucket_count[b]
            out.append(InventoryBucket(
                bucket_s=b,
                t_ns_start=b * NS_PER_S,
                inv=cur_inv,
                n_fills_so_far=cur_count,
            ))
        return out


# --------------------------------------------------------------------- #
# QueuePosLogger
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class QueuePosSample:
    ts_ns: int
    order_id: int
    side: int
    price: float
    queue_pos: float
    frozen: bool


class QueuePosLogger:
    """Per-(snapshot, active-order) queue-position log.

    `observe_queue_pos(t_ns, order_id, side, price, queue_pos, frozen)`
    appends one row.  The driver calls this once per active order at
    every snapshot, after the fill model has applied that snapshot's
    state update.
    """

    def __init__(self):
        self._samples: List[QueuePosSample] = []

    def observe_queue_pos(
        self,
        ts_ns: int,
        order_id: int,
        side: int,
        price: float,
        queue_pos: float,
        frozen: bool,
    ) -> None:
        self._samples.append(QueuePosSample(
            ts_ns=int(ts_ns),
            order_id=int(order_id),
            side=int(side),
            price=float(price),
            queue_pos=float(queue_pos),
            frozen=bool(frozen),
        ))

    def samples(self) -> List[QueuePosSample]:
        return list(self._samples)


# --------------------------------------------------------------------- #
# AuxLogs aggregate + CSV writer
# --------------------------------------------------------------------- #

@dataclass
class AuxLogs:
    """Bundle of the three aux log streams.  Each stream is the
    canonical logger from this module; the aggregate exists so the
    sim driver can hand back a single object and the CSV writer can
    walk the three uniformly."""
    fill_rate: FillRateLogger = field(default_factory=FillRateLogger)
    inventory: InventoryLogger = field(default_factory=InventoryLogger)
    queue_pos: QueuePosLogger = field(default_factory=QueuePosLogger)

    def observe_fill(self, fill: Fill) -> None:
        """Convenience: dispatch a fill to both fill-rate and
        inventory loggers.  Queue-pos requires per-snapshot calls,
        which the driver issues directly."""
        self.fill_rate.observe_fill(fill)
        self.inventory.observe_fill(fill)

    def to_csvs(
        self,
        out_dir: Path,
        start_s: Optional[int] = None,
        end_s: Optional[int] = None,
    ) -> Tuple[Path, Path, Path]:
        """Write the three CSVs to `out_dir`.  Returns
        `(fill_rate_path, inventory_path, queue_pos_path)`.

        `start_s`/`end_s` pin the per-second grid for fill-rate and
        inventory.  When unset, each logger uses its observed range
        (which may skip leading/trailing empty seconds)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fr_path = out_dir / "fill_rate.csv"
        with fr_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["bucket_s", "t_ns_start", "n_fills", "total_qty"])
            for b in self.fill_rate.buckets(start_s, end_s):
                w.writerow([b.bucket_s, b.t_ns_start, b.n_fills,
                            f"{b.total_qty:.12f}"])

        inv_path = out_dir / "inventory.csv"
        with inv_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["bucket_s", "t_ns_start", "inv", "n_fills_so_far"])
            for b in self.inventory.buckets(start_s, end_s):
                w.writerow([b.bucket_s, b.t_ns_start,
                            f"{b.inv:.12f}", b.n_fills_so_far])

        qp_path = out_dir / "queue_pos.csv"
        with qp_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ts_ns", "order_id", "side", "price",
                        "queue_pos", "frozen"])
            for s in self.queue_pos.samples():
                w.writerow([s.ts_ns, s.order_id, s.side,
                            f"{s.price:.12f}", f"{s.queue_pos:.12f}",
                            "1" if s.frozen else "0"])
        return fr_path, inv_path, qp_path


# --------------------------------------------------------------------- #
# Sim driver
# --------------------------------------------------------------------- #

def run_sim_with_aux_logs(
    events,
    quoter,
    fill_model,
) -> Tuple["object", AuxLogs]:
    """Drive the event stream through the sim loop while collecting
    the three aux log streams.

    Returns `(sim_result, aux_logs)`.

    Mechanics:
      - Wrap the user's quoter so we can intercept SnapshotEvents and
        log queue positions of every active order *after* the model
        applied the snapshot (mirrors the QueueAwareFillModel
        on_snapshot semantics — cancel attribution happens first,
        then we log the post-snapshot queue_pos).
      - At the end, walk `result.fills` once to drive fill-rate +
        inventory loggers.

    Causality: fill-rate / inventory observe only `Fill` records,
    which are produced by the loop in ts-monotonic order.  Queue-pos
    is observed inside the quoter's snapshot callback, *before* the
    quoter places new orders — so logged queue_pos at t reflects only
    events with ts_ns <= t (the snapshot at t, plus all preceding
    trades / cancels).

    This driver requires a `QueueAwareFillModel` (its `_trackers`
    map is read for queue-pos logging).  Pass any FillModelProtocol
    for fill-rate + inventory only; queue-pos rows will be empty.
    """
    from mmsim.ingest.lob import SnapshotEvent
    from mmsim.sim.fills import QueueAwareFillModel
    from mmsim.sim.loop import run_sim

    aux = AuxLogs()
    qa = isinstance(fill_model, QueueAwareFillModel)

    # Snapshot the active-order list AS THE MODEL SEES IT at each
    # snapshot.  The model's on_snapshot has already fired (the loop
    # calls it before the quoter); we read trackers' queue_pos here.
    # We use a wrapping quoter that, before delegating, logs the
    # queue positions of the *current* trackers.
    def wrapped_quoter(book, active_orders, t_ns):
        # If the underlying model is QueueAware, log every active
        # tracker's current queue_pos here.  Active orders are the
        # orders that were resting going into this snapshot — the
        # model's on_snapshot just updated their trackers.
        if qa:
            trackers = fill_model._trackers
            for o in active_orders:
                tr = trackers.get(o.order_id)
                if tr is None:
                    continue
                aux.queue_pos.observe_queue_pos(
                    ts_ns=int(t_ns),
                    order_id=int(o.order_id),
                    side=int(o.side),
                    price=float(o.price),
                    queue_pos=float(tr.queue_pos),
                    frozen=bool(tr.frozen),
                )
        return quoter(book, active_orders, t_ns)

    res = run_sim(events, wrapped_quoter, fill_model)

    # After the run, walk fills for fill-rate + inventory.
    for f in res.fills:
        aux.observe_fill(f)

    return res, aux


__all__ = [
    "NS_PER_S",
    "FillRateBucket", "FillRateLogger",
    "InventoryBucket", "InventoryLogger",
    "QueuePosSample", "QueuePosLogger",
    "AuxLogs",
    "run_sim_with_aux_logs",
]
