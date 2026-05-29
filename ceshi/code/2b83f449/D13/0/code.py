# Variant of 2b83f449
# Dimension: D13 — 网格边缘令牌的存在规则 (meta)
# Change: 最后一行管道行(最底行)没有令牌，其余管道行左右各有一个令牌 -> 图例行指定哪些管道行有令牌

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[2] / "_variant_lib.py"
_SPEC = importlib.util.spec_from_file_location("_variant_lib_2b83f449", _LIB_PATH)
_LIB = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_LIB)
main, generate_input = _LIB.build_functions("D13", 0)
