# Variant of 2b83f449
# Dimension: D7 — 单出口约束 (meta)
# Change: 每个管段最多有1个向下出口和1个从上入口，保证令牌路径确定性 -> 允许多出口+选择规则

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[2] / "_variant_lib.py"
_SPEC = importlib.util.spec_from_file_location("_variant_lib_2b83f449", _LIB_PATH)
_LIB = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_LIB)
main, generate_input = _LIB.build_functions("D7", 0)
