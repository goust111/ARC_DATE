# Variant of 2b83f449
# Dimension: D3 — 令牌移动方式 (path2)
# Change: 令牌沿单出口管段垂直下落，每次穿过一面墙（2行），直到无出口可走 -> 令牌上升而非下落

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[2] / "_variant_lib.py"
_SPEC = importlib.util.spec_from_file_location("_variant_lib_2b83f449", _LIB_PATH)
_LIB = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_LIB)
main, generate_input = _LIB.build_functions("D3", 0)
