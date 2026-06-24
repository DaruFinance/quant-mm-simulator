"""Quoter contract + reference implementations.

The single integration point for every quoting model (Avellaneda-Stoikov,
Cartea-Jaimungal, GLFT, Ho-Stoll, microprice-skew, fair-anchored).  Each
model implements the ``Quoter`` Protocol and plugs into
``run_sim_with_model`` unmodified.

Exports:
  - Quoter (Protocol)
  - ConstantQuoter, TopOfBookQuoter, BracketQuoter (reference impls)

This subpackage also provides:
  - quote-shape primitives (single, paired, ladder, …)
  - refresh-trigger primitives (time / mid-move / …)
  - reference-price primitives (top-mid, weighted, microprice, …)
  - adverse-selection filters
  - inventory-penalty primitives
"""
from __future__ import annotations

from .base import Quoter
from .builtin import ConstantQuoter, TopOfBookQuoter, BracketQuoter

__all__ = [
    "Quoter",
    "ConstantQuoter", "TopOfBookQuoter", "BracketQuoter",
]
