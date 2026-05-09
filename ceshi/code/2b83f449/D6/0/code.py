# Variant of 2b83f449
# Dimension: D6 — 管道网络结构 (grid_shape)
# Change: 交替的管道行(背景色)和墙行(黑色)，网格高度=2N+1(N=5-9)，宽度16-19 -> 垂直管道+水平墙

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[2] / "_variant_lib.py"
_SPEC = importlib.util.spec_from_file_location("_variant_lib_2b83f449", _LIB_PATH)
_LIB = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_LIB)
main, generate_input = _LIB.build_functions("D6", 0)
