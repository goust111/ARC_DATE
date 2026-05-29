# Variant of 2b83f449
# Dimension: D12 — 令牌下落深度与路径复杂度 (meta)
# Change: 令牌下落0-6层（穿过0-6面墙），生成时要求至少一个令牌下落>=3层 -> 路径交叉检测

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[2] / "_variant_lib.py"
_SPEC = importlib.util.spec_from_file_location("_variant_lib_2b83f449", _LIB_PATH)
_LIB = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_LIB)
main, generate_input = _LIB.build_functions("D12", 2)
