# Variant of 2b83f449
# Dimension: D2 — 门打开后的颜色变化 (transform)
# Change: 门颜色(7 teal)变为垂直柱颜色(6 magenta)，即 recolor 从门色到开门色 -> 多色门对应多色柱

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[2] / "_variant_lib.py"
_SPEC = importlib.util.spec_from_file_location("_variant_lib_2b83f449", _LIB_PATH)
_LIB = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_LIB)
main, generate_input = _LIB.build_functions("D2", 1)
