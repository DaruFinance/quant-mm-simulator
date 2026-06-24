"""Make the tests/ directory importable so sibling test modules can be
imported by name (e.g. ``from test_sim_fills import BracketQuoter``), which is
this suite's established convention. Without this, pytest's default "prepend"
import mode only puts the repo root on sys.path and a full-suite collection
(``pytest tests/``) fails to resolve those sibling imports.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
