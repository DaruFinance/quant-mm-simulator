"""Quoting-model library.

Eight concrete quoting models, each implementing the `Quoter`
Protocol. The library is the capstone composition over the quoting
primitives: it builds on top of refprices, shapes, inventory-penalty,
and adverse filters, all of which are verified individually.

Models
------
| Class                       | Family             | Stateful? |
|-----------------------------|--------------------|-----------|
| `SymmetricQuoter`           | fixed half-spread  | No        |
| `LadderQuoter`              | multi-level ladder | No        |
| `MicropriceSkewQuoter`      | book-imbalance ref | No        |
| `FairAnchoredQuoter`        | EWMA-fair anchor   | Yes (EWMA)|
| `AvellanedaStoikovQuoter`   | AS closed-form     | Yes (σ)   |
| `CarteaJaimungalQuoter`     | CJ variant         | Yes (σ)   |
| `GLFTQuoter`                | GLFT closed-form   | Yes (σ)   |
| `HoStollQuoter`             | classical (1981)   | Yes (σ)   |

All 8 implement `quote(book, inv, t_ns) -> List[QuoteRequest | TakerRequest]`
and therefore plug directly into `run_sim` (the loop's `isinstance(
quoter, Quoter)` branch handles the Protocol path).

Helpers
-------
`RollingSigma(window_ns)` — shared trailing-window log-return std
tracker (mid-fed) used by the AS-family models (AS, CJ, GLFT,
Ho-Stoll).  Deterministic, leak-free, and parity-friendly.
"""
from __future__ import annotations

from ._vol import RollingSigma
from .avellaneda_stoikov import AvellanedaStoikovQuoter
from .cartea_jaimungal import CarteaJaimungalQuoter
from .fair_anchored import FairAnchoredQuoter
from .glft import GLFTQuoter
from .ho_stoll import HoStollQuoter
from .ladder import LadderQuoter
from .microprice_skew import MicropriceSkewQuoter
from .symmetric import SymmetricQuoter

__all__ = [
    "SymmetricQuoter",
    "LadderQuoter",
    "MicropriceSkewQuoter",
    "FairAnchoredQuoter",
    "AvellanedaStoikovQuoter",
    "CarteaJaimungalQuoter",
    "GLFTQuoter",
    "HoStollQuoter",
    "RollingSigma",
]
